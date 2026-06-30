"""Prepare bounded steady restart branches for steady-restart diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from numpy.typing import NDArray

from aeromap.cfd.dictionaries import header
from aeromap.cfd.patch_surface import article_patch_names
from aeromap.cfd.postprocess import (
    _coefficient_rows,
    _flow_rows,
    _force_rows,
    _residual_report,
    _steady_diagnostics,
)
from aeromap.cfd.region_mapping import RegionMappingError, map_wall_regions_analytically_to_vtp
from aeromap.cfd.schema import CfdConfig
from aeromap.cfd.spatial_loads import integrate_spatial_loads
from aeromap.constants import REF
from aeromap.io import atomic_write_json, atomic_write_text
from aeromap.parameters import AeroParams

DEFAULT_RESTART_ITERATIONS = 120
DEFAULT_WRITE_INTERVAL = 5
DEFAULT_MOMENTUM_RELAXATION = 0.70
REQUIRED_RESTART_FIELDS = ("U", "p", "k", "omega", "nut")
OPENFOAM_TIME_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
MIN_CORRELATION_SAMPLES = 2
NUMERIC_EPSILON = 1.0e-15
MIN_COMPARISON_REPORTS = 2
DEFAULT_PHASE_MAX_LAG_SAMPLES = 6
PHYSICAL_UNSTEADINESS_CORRELATION_THRESHOLD = 0.70
NUMERICS_SENSITIVE_MEAN_SHIFT_THRESHOLD = 0.05
NUMERICS_SENSITIVE_AMPLITUDE_SHIFT_THRESHOLD = 0.25
NUMERICS_SENSITIVE_ABSOLUTE_AMPLITUDE_SHIFT_THRESHOLD = 1.0e-3

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SteadyRestartBranchArtifacts:
    branch_name: str
    branch_dir: Path
    manifest_path: Path
    control_dict_path: Path
    fv_schemes_path: Path
    fv_solution_path: Path
    run_script_path: Path


@dataclass(frozen=True)
class SteadyRestartPlanArtifacts:
    plan_id: str
    plan_dir: Path
    branches: tuple[SteadyRestartBranchArtifacts, ...]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _repo_relative_required(path: Path, *, label: str) -> str:
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as exc:
        message = f"{label} must be inside project root for Docker Compose: {path}"
        raise ValueError(message) from exc
    if rel in {"", "."} or rel.startswith("../") or "/../" in rel or rel.endswith("/.."):
        message = f"{label} resolved to an unsafe repository-relative path: {rel}"
        raise ValueError(message)
    return rel


def _shell_literal(value: str) -> str:
    return shlex.quote(value)


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"expected JSON object in {path}"
        raise TypeError(message)
    return loaded


def _data_rows_from_files(
    paths: list[Path],
    parser: Callable[[Path], list[dict[str, Any]]],
    *,
    start_time: float,
    end_time: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(parser(path))
    return [
        row
        for row in rows
        if start_time < float(row.get("time", float("nan"))) <= end_time + 1.0e-9
    ]


def _unique_times(rows: list[dict[str, Any]]) -> list[float]:
    return sorted({float(row["time"]) for row in rows if "time" in row})


def _postprocessing_file_paths(case_dir: Path, function_name: str, file_name: str) -> list[Path]:
    return sorted((case_dir / "openfoam" / "postProcessing" / function_name).glob(f"*/{file_name}"))


def _latest_numeric_time(openfoam_dir: Path) -> str:
    numeric_times: list[tuple[float, str]] = []
    for child in openfoam_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            numeric_times.append((float(child.name), child.name))
        except ValueError:
            continue
    if not numeric_times:
        message = f"no numeric OpenFOAM time directories found in {openfoam_dir}"
        raise FileNotFoundError(message)
    return max(numeric_times, key=lambda item: item[0])[1]


def _validate_openfoam_time_name(value: str) -> str:
    if not OPENFOAM_TIME_RE.fullmatch(value):
        message = f"steady_time must be a numeric OpenFOAM time directory name: {value}"
        raise ValueError(message)
    return value


def _validate_restart_fields(source_case: Path, steady_time: str) -> None:
    missing = [
        str(source_case / "openfoam" / steady_time / field)
        for field in REQUIRED_RESTART_FIELDS
        if not (source_case / "openfoam" / steady_time / field).exists()
    ]
    if missing:
        message = "source case is missing required steady restart fields: " + ", ".join(missing)
        raise FileNotFoundError(message)
    mesh_dir = source_case / "openfoam" / "constant" / "polyMesh"
    if not mesh_dir.exists():
        message = f"source case is missing OpenFOAM mesh directory: {mesh_dir}"
        raise FileNotFoundError(message)
    for name in ("fvSchemes", "fvSolution"):
        path = source_case / "openfoam" / "system" / name
        if not path.exists():
            message = f"source case is missing OpenFOAM system file: {path}"
            raise FileNotFoundError(message)


def _patch_list(config: CfdConfig) -> str:
    patches = article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
    return "(" + " ".join(patches) + ")"


def _control_dict_steady_restart(
    config: CfdConfig,
    *,
    start_time: float,
    end_time: float,
    write_interval: int,
) -> str:
    patches = _patch_list(config)
    return (
        header("dictionary", "system", "controlDict")
        + f"""
solver          incompressibleFluid;

startFrom       latestTime;
startTime       {start_time:g};
stopAt          endTime;
endTime         {end_time:g};
deltaT          1;

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         {patches};
        rho             rhoInf;
        rhoInf          {REF.rho_kg_m3:g};
        CofR            (1 0 0);
        writeControl    timeStep;
        writeInterval   1;
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         {patches};
        rho             rhoInf;
        rhoInf          {REF.rho_kg_m3:g};
        liftDir         (0 0 -1);
        dragDir         (1 0 0);
        CofR            (1 0 0);
        pitchAxis       (0 1 0);
        magUInf         {REF.u_inf_m_s:g};
        lRef            {REF.l_ref_m:g};
        Aref            {REF.a_ref_m2:g};
        writeControl    timeStep;
        writeInterval   1;
    }}

    inletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           inlet;
        fields          (phi);
        operation       sum;
    }}

    outletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           outlet;
        fields          (phi);
        operation       sum;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         {patches};
        writeControl    writeTime;
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
}}
"""
    )


def _replace_momentum_relaxation(fv_solution: str, *, relaxation: float) -> str:
    marker = fv_solution.rfind("relaxationFactors")
    if marker < 0:
        message = "fvSolution has no relaxationFactors block"
        raise ValueError(message)
    return (
        fv_solution[:marker]
        + f"""relaxationFactors
{{
    equations
    {{
        p       0.9;
        pcorr   0.9;
        k       0.9;
        omega   0.9;
        U       {relaxation:g};
    }}
}}
"""
    )


def _run_script(
    *,
    source_case_rel: str,
    branch_dir_rel: str,
    work_case_rel: str,
    steady_time: str,
    branch_name: str,
) -> str:
    source_case_literal = _shell_literal(source_case_rel)
    branch_dir_literal = _shell_literal(branch_dir_rel)
    work_case_literal = _shell_literal(work_case_rel)
    steady_time_literal = _shell_literal(steady_time)
    branch_name_literal = _shell_literal(branch_name)
    return f"""#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${{AEROMAP_REPO_ROOT:-$(pwd)}}"
SOURCE_CASE_REL={source_case_literal}
BRANCH_DIR_REL={branch_dir_literal}
WORK_CASE_REL={work_case_literal}
STEADY_TIME={steady_time_literal}
BRANCH_NAME={branch_name_literal}

validate_repo_relative() {{
    local name="$1"
    local value="$2"
    if [[ -z "$value" || "$value" == /* || "$value" == "." || "$value" == ".." \\
        || "$value" == ../* || "$value" == */.. || "$value" == */../* ]]; then
        echo "Unsafe $name repository-relative path: $value" >&2
        exit 2
    fi
}}

validate_repo_relative SOURCE_CASE_REL "$SOURCE_CASE_REL"
validate_repo_relative BRANCH_DIR_REL "$BRANCH_DIR_REL"
validate_repo_relative WORK_CASE_REL "$WORK_CASE_REL"
case "$WORK_CASE_REL" in
    artifacts/campaign/steady_restart_runs/*) ;;
    *)
        echo "Steady restart work case outside generated run directory: $WORK_CASE_REL" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
cd "$REPO_ROOT"

SOURCE_CASE="$REPO_ROOT/$SOURCE_CASE_REL"
BRANCH_DIR="$REPO_ROOT/$BRANCH_DIR_REL"
WORK_CASE="$REPO_ROOT/$WORK_CASE_REL"
WORK_PARENT="$REPO_ROOT/artifacts/campaign/steady_restart_runs"
mkdir -p "$WORK_PARENT"
WORK_PARENT_REAL="$(cd "$WORK_PARENT" && pwd -P)"
WORK_CASE_PARENT_REAL="$(cd "$(dirname "$WORK_CASE")" && pwd -P)"
if [[ "$WORK_CASE_PARENT_REAL" != "$WORK_PARENT_REAL" ]]; then
    echo "Refusing to write steady restart outside generated run directory: $WORK_CASE" >&2
    exit 2
fi

if [[ -e "$WORK_CASE" && "${{AEROMAP_STEADY_RESTART_OVERWRITE:-0}}" != "1" ]]; then
    echo "Refusing to overwrite existing steady restart work case: $WORK_CASE" >&2
    echo "Set AEROMAP_STEADY_RESTART_OVERWRITE=1 to replace this generated work case." >&2
    exit 2
fi

rm -rf "$WORK_CASE"
mkdir -p "$WORK_CASE/openfoam" "$WORK_CASE/logs" "$WORK_CASE/quality" "$WORK_CASE/outputs"
cp -a "$SOURCE_CASE/openfoam/constant" "$WORK_CASE/openfoam/constant"
cp -a "$SOURCE_CASE/openfoam/$STEADY_TIME" "$WORK_CASE/openfoam/$STEADY_TIME"
rm -rf "$WORK_CASE/openfoam/$STEADY_TIME/uniform"
mkdir -p "$WORK_CASE/openfoam/system"
cp "$BRANCH_DIR/system/controlDict" "$WORK_CASE/openfoam/system/controlDict"
cp "$BRANCH_DIR/system/fvSchemes" "$WORK_CASE/openfoam/system/fvSchemes"
cp "$BRANCH_DIR/system/fvSolution" "$WORK_CASE/openfoam/system/fvSolution"
cp "$SOURCE_CASE/manifest.json" "$WORK_CASE/source_steady_manifest.json"
cp "$BRANCH_DIR/manifest.json" "$WORK_CASE/steady_restart_manifest.json"

CONTAINER_WORK_DIR="/work/$WORK_CASE_REL/openfoam"
CONTAINER_WORK_DIR_ESCAPED="$(printf '%q' "$CONTAINER_WORK_DIR")"
container_cmd="set +u"
container_cmd="$container_cmd; source /opt/openfoam13/etc/bashrc"
container_cmd="$container_cmd; set -euo pipefail"
container_cmd="$container_cmd; cd $CONTAINER_WORK_DIR_ESCAPED"
container_cmd="$container_cmd; foamRun -solver incompressibleFluid"
container_cmd="$container_cmd > ../logs/foamRun_steady_restart_${{BRANCH_NAME}}.log 2>&1"
container_cmd="$container_cmd; postProcess -func yPlus -latestTime"
container_cmd="$container_cmd > ../logs/yPlus_restart_${{BRANCH_NAME}}.log 2>&1 || true"
container_cmd="$container_cmd; postProcess -func wallShearStress -latestTime"
container_cmd="$container_cmd > ../logs/wss_restart_${{BRANCH_NAME}}.log 2>&1 || true"
container_cmd="$container_cmd; foamToVTK -latestTime"
container_cmd="$container_cmd > ../logs/vtk_restart_${{BRANCH_NAME}}.log 2>&1 || true"
docker compose run --rm cfd "$container_cmd"
"""


def _plan_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "steady_restart_" + hashlib.sha256(encoded).hexdigest()[:16]


def prepare_steady_restart_branches(  # noqa: PLR0915
    *,
    source_case: Path,
    out_dir: Path,
    steady_time: str | None = None,
    iterations: int = DEFAULT_RESTART_ITERATIONS,
    write_interval: int = DEFAULT_WRITE_INTERVAL,
    momentum_relaxation: float = DEFAULT_MOMENTUM_RELAXATION,
) -> SteadyRestartPlanArtifacts:
    """Prepare unchanged and momentum-relaxation steady restart diagnostic branches."""

    source_case = source_case.resolve()
    out_dir = out_dir.resolve()
    if iterations <= 0:
        message = "iterations must be positive"
        raise ValueError(message)
    if write_interval <= 0:
        message = "write_interval must be positive"
        raise ValueError(message)
    if not 0.0 < momentum_relaxation <= 1.0:
        message = "momentum_relaxation must be in (0, 1]"
        raise ValueError(message)

    manifest = _load_json_if_present(source_case / "manifest.json")
    if not manifest:
        message = f"source case has no manifest: {source_case / 'manifest.json'}"
        raise FileNotFoundError(message)
    config = CfdConfig.model_validate(manifest["cfd_config"])
    steady_time = _validate_openfoam_time_name(
        steady_time or _latest_numeric_time(source_case / "openfoam"),
    )
    _validate_restart_fields(source_case, steady_time)

    start_time = float(steady_time)
    end_time = start_time + float(iterations)
    source_case_rel = _repo_relative_required(source_case, label="source_case")
    status = _load_json_if_present(source_case / "quality" / "status.json")
    source_fv_solution = (source_case / "openfoam" / "system" / "fvSolution").read_text(
        encoding="utf-8",
    )
    source_fv_schemes = (source_case / "openfoam" / "system" / "fvSchemes").read_text(
        encoding="utf-8",
    )
    branch_specs = (
        {
            "branch_name": "unchanged",
            "description": (
                "same fvSolution as source case; only restart window/output cadence change"
            ),
            "momentum_relaxation": None,
        },
        {
            "branch_name": "momentum_relaxation_0p70",
            "description": (
                "identical to unchanged branch except equation relaxation factors are explicit "
                "and U equation relaxation is set to 0.70"
            ),
            "momentum_relaxation": momentum_relaxation,
        },
    )
    plan_payload = {
        "source_simulation_id": manifest.get("simulation_id"),
        "source_case": source_case_rel,
        "steady_time": steady_time,
        "iterations": iterations,
        "write_interval": write_interval,
        "momentum_relaxation": momentum_relaxation,
        "branches": branch_specs,
    }
    plan_id = _plan_id(plan_payload)
    plan_dir = out_dir / plan_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    branches: list[SteadyRestartBranchArtifacts] = []
    for spec in branch_specs:
        branch_name = str(spec["branch_name"])
        branch_dir = plan_dir / branch_name
        system_dir = branch_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        branch_dir_rel = _repo_relative_required(branch_dir, label="branch_dir")
        work_case_rel = f"artifacts/campaign/steady_restart_runs/{plan_id}_{branch_name}"

        fv_solution = source_fv_solution
        momentum_value = spec["momentum_relaxation"]
        if isinstance(momentum_value, float):
            fv_solution = _replace_momentum_relaxation(
                source_fv_solution,
                relaxation=momentum_value,
            )

        control_dict_path = system_dir / "controlDict"
        fv_schemes_path = system_dir / "fvSchemes"
        fv_solution_path = system_dir / "fvSolution"
        manifest_path = branch_dir / "manifest.json"
        run_script_path = branch_dir / "run_steady_restart.sh"

        atomic_write_text(
            control_dict_path,
            _control_dict_steady_restart(
                config,
                start_time=start_time,
                end_time=end_time,
                write_interval=write_interval,
            ),
        )
        atomic_write_text(fv_schemes_path, source_fv_schemes)
        atomic_write_text(fv_solution_path, fv_solution)
        atomic_write_text(
            run_script_path,
            _run_script(
                source_case_rel=source_case_rel,
                branch_dir_rel=branch_dir_rel,
                work_case_rel=work_case_rel,
                steady_time=steady_time,
                branch_name=branch_name,
            ),
        )
        run_script_path.chmod(0o755)

        branch_manifest = {
            "schema_version": "steady_restart_branch_v0.1.0",
            "plan_id": plan_id,
            "branch_name": branch_name,
            "status": "PREPARED_NOT_RUN",
            "accepted": False,
            "training_eligible": False,
            "source": {
                "case_dir": source_case_rel,
                "simulation_id": manifest.get("simulation_id"),
                "case_id": manifest.get("case_id"),
                "geometry_id": manifest.get("geometry_id"),
                "state_id": manifest.get("state_id"),
                "steady_time": steady_time,
                "status": status.get("status"),
                "accepted": status.get("accepted"),
                "training_eligible": status.get("training_eligible"),
            },
            "run_setup": {
                "solver": "foamRun -solver incompressibleFluid",
                "start_from": "latestTime",
                "source_start_time": start_time,
                "end_time": end_time,
                "delta_t": 1.0,
                "intended_simple_iterations": iterations,
                "write_interval_iterations": write_interval,
                "force_write_interval_iterations": 1,
                "wall_field_write_interval_iterations": write_interval,
            },
            "branch_perturbation": {
                "description": spec["description"],
                "momentum_relaxation": spec["momentum_relaxation"],
                "unchanged_controls": [
                    "mesh",
                    "geometry",
                    "fvSchemes",
                    "turbulence model",
                    "boundary conditions",
                    "force references",
                ],
            },
            "generated_files": {
                "controlDict": _repo_relative(control_dict_path),
                "fvSchemes": _repo_relative(fv_schemes_path),
                "fvSolution": _repo_relative(fv_solution_path),
                "run_script": _repo_relative(run_script_path),
            },
            "work_case": {
                "path": work_case_rel,
                "overwrite_guard": (
                    "Set AEROMAP_STEADY_RESTART_OVERWRITE=1 to replace the generated work case."
                ),
                "initialization": f"copy source openfoam/{steady_time} fields into restart case",
            },
            "run_command": _repo_relative(run_script_path),
            "classification_rules": {
                "not_a_label": True,
                "steady_iteration_is_not_physical_time": True,
                "required_before_claims": (
                    "post-run force, moment, residual, mass-flow, wall-field and spatial-load "
                    "time-series diagnostics"
                ),
            },
        }
        atomic_write_json(manifest_path, branch_manifest)
        branches.append(
            SteadyRestartBranchArtifacts(
                branch_name=branch_name,
                branch_dir=branch_dir,
                manifest_path=manifest_path,
                control_dict_path=control_dict_path,
                fv_schemes_path=fv_schemes_path,
                fv_solution_path=fv_solution_path,
                run_script_path=run_script_path,
            ),
        )

    atomic_write_json(
        plan_dir / "manifest.json",
        {
            "schema_version": "steady_restart_plan_v0.1.0",
            "plan_id": plan_id,
            "status": "PREPARED_NOT_RUN",
            "accepted": False,
            "training_eligible": False,
            "source_case": source_case_rel,
            "steady_time": steady_time,
            "iterations": iterations,
            "write_interval": write_interval,
            "branches": {
                branch.branch_name: {
                    "manifest": _repo_relative(branch.manifest_path),
                    "run_command": _repo_relative(branch.run_script_path),
                }
                for branch in branches
            },
        },
    )
    return SteadyRestartPlanArtifacts(
        plan_id=plan_id,
        plan_dir=plan_dir,
        branches=tuple(branches),
    )


def analyze_steady_restart_branch(
    case_dir: Path,
    *,
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Summarize one generated steady restart branch after it has run."""

    manifest_path = case_dir / "steady_restart_manifest.json"
    manifest = _load_json_if_present(manifest_path)
    if not manifest:
        message = f"steady restart work case has no manifest: {manifest_path}"
        raise FileNotFoundError(message)
    run_setup = manifest["run_setup"]
    start_time = float(run_setup["source_start_time"])
    end_time = float(run_setup["end_time"])
    intended_iterations = int(run_setup["intended_simple_iterations"])

    coeff_paths = _postprocessing_file_paths(case_dir, "forceCoeffs", "forceCoeffs.dat")
    force_paths = _postprocessing_file_paths(case_dir, "forces", "forces.dat")
    if not coeff_paths:
        message = f"steady restart has no forceCoeffs history under {case_dir}"
        raise FileNotFoundError(message)
    if not force_paths:
        message = f"steady restart has no forces history under {case_dir}"
        raise FileNotFoundError(message)

    coeff_rows = _data_rows_from_files(
        coeff_paths,
        _coefficient_rows,
        start_time=start_time,
        end_time=end_time,
    )
    force_rows = _data_rows_from_files(
        force_paths,
        _force_rows,
        start_time=start_time,
        end_time=end_time,
    )
    coeff_times = _unique_times(coeff_rows)
    force_times = _unique_times(force_rows)
    completed_iterations = len(coeff_times)
    status = "COMPLETE" if completed_iterations == intended_iterations else "INCOMPLETE"
    diagnostics = _steady_diagnostics(coeff_rows=coeff_rows, force_rows=force_rows)
    inlet_rows = _data_rows_from_files(
        _postprocessing_file_paths(case_dir, "inletFlowRate", "surfaceFieldValue.dat"),
        _flow_rows,
        start_time=start_time,
        end_time=end_time,
    )
    outlet_rows = _data_rows_from_files(
        _postprocessing_file_paths(case_dir, "outletFlowRate", "surfaceFieldValue.dat"),
        _flow_rows,
        start_time=start_time,
        end_time=end_time,
    )
    written_times = _unique_times(
        [
            {"time": float(path.name)}
            for path in (case_dir / "openfoam").iterdir()
            if path.is_dir() and OPENFOAM_TIME_RE.fullmatch(path.name)
        ],
    )

    report = {
        "schema_version": "steady_restart_branch_analysis_v0.1.0",
        "case_dir": str(case_dir),
        "manifest": str(manifest_path),
        "plan_id": manifest.get("plan_id"),
        "branch_name": manifest.get("branch_name"),
        "status": status,
        "accepted": False,
        "training_eligible": False,
        "time_window": {
            "source_start_time": start_time,
            "end_time": end_time,
            "intended_iterations": intended_iterations,
            "completed_force_coefficient_iterations": completed_iterations,
            "first_force_coefficient_time": coeff_times[0] if coeff_times else None,
            "last_force_coefficient_time": coeff_times[-1] if coeff_times else None,
            "force_history_times_match": coeff_times == force_times,
            "written_field_times": written_times,
        },
        "steady_diagnostics": diagnostics,
        "residuals": _residual_report(case_dir),
        "mass_flow": {
            "inlet_sample_count": len(inlet_rows),
            "outlet_sample_count": len(outlet_rows),
            "latest_inlet_flow_m3_s": inlet_rows[-1]["flow_m3_s"] if inlet_rows else None,
            "latest_outlet_flow_m3_s": outlet_rows[-1]["flow_m3_s"] if outlet_rows else None,
        },
        "source_paths": {
            "forceCoeffs": [str(path) for path in coeff_paths],
            "forces": [str(path) for path in force_paths],
        },
        "classification_note": (
            "This branch remains a numerical diagnostic only; steady iteration index is not "
            "physical time and the result is not a training label."
        ),
    }
    output = out_json or case_dir / "quality" / "steady_restart_diagnostics.json"
    atomic_write_json(output, report)
    return report


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _restart_wall_vtk_command(
    *,
    case_dir: Path,
    config: CfdConfig,
    times: list[str],
) -> str:
    _ = article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
    time_arg = ",".join(times)
    return (
        'docker compose run --rm cfd "set +u; source /opt/openfoam13/etc/bashrc; '
        "set -euo pipefail; "
        f"cd /work/{_repo_relative_required(case_dir / 'openfoam', label='case_openfoam')}; "
        "foamToVTK -noInternal -useTimeName "
        "-fields '(p wallShearStress yPlus)' "
        f"-time '{time_arg}'\""
    )


def _available_wall_vtk_times(case_dir: Path, config: CfdConfig) -> list[str]:
    first_patch = article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)[0]
    patch_dir = case_dir / "openfoam" / "VTK" / first_patch
    if not patch_dir.exists():
        return []
    prefix = f"{first_patch}_"
    suffixes: list[tuple[float, str]] = []
    for path in patch_dir.glob(f"{prefix}*.vtk"):
        suffix = path.stem.removeprefix(prefix)
        if OPENFOAM_TIME_RE.fullmatch(suffix):
            suffixes.append((float(suffix), suffix))
    return [suffix for _, suffix in sorted(suffixes, key=lambda item: item[0])]


def _expected_written_times(manifest: dict[str, Any]) -> list[str]:
    setup = manifest["run_setup"]
    start = int(float(setup["source_start_time"]))
    end = int(float(setup["end_time"]))
    interval = int(setup.get("wall_field_write_interval_iterations", DEFAULT_WRITE_INTERVAL))
    return [str(time) for time in range(start + interval, end + 1, interval)]


def _requested_wall_times(
    *,
    case_dir: Path,
    manifest: dict[str, Any],
    config: CfdConfig,
    times: tuple[str, ...] | None,
) -> list[str]:
    if times is not None:
        requested = [_validate_openfoam_time_name(time) for time in times]
    else:
        requested = _available_wall_vtk_times(case_dir, config)
        if not requested:
            requested = _expected_written_times(manifest)
    if not requested:
        message = "no wall-field times requested or discoverable"
        raise ValueError(message)
    return requested


def _combine_restart_wall_vtks(
    *,
    case_dir: Path,
    config: CfdConfig,
    time_name: str,
) -> Path:
    vtk_dir = case_dir / "openfoam" / "VTK"
    outputs = case_dir / "outputs"
    outputs.mkdir(exist_ok=True)
    wall_patches = article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
    wall_polys: list[pv.PolyData] = []
    missing: list[str] = []
    for patch in wall_patches:
        wall_vtk = vtk_dir / patch / f"{patch}_{time_name}.vtk"
        if not wall_vtk.exists():
            missing.append(str(wall_vtk))
            continue
        loaded = pv.read(wall_vtk)
        wall_polys.append(loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface())
    if missing:
        message = "wall-field VTK export missing required patch files: " + ", ".join(missing)
        raise FileNotFoundError(message)
    wall = wall_polys[0] if len(wall_polys) == 1 else pv.MultiBlock(wall_polys).combine()
    wall_poly = wall if isinstance(wall, pv.PolyData) else wall.extract_surface()
    out = outputs / f"article_wall_restart_{time_name}.vtp"
    wall_poly.save(out)
    return out


def _load_restart_source_inputs(
    case_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, AeroParams, CfdConfig]:
    manifest = _load_json_if_present(case_dir / "steady_restart_manifest.json")
    source_manifest = _load_json_if_present(case_dir / "source_steady_manifest.json")
    if not manifest:
        message = (
            f"steady restart work case has no manifest: {case_dir / 'steady_restart_manifest.json'}"
        )
        raise FileNotFoundError(message)
    if not source_manifest:
        message = (
            "steady restart work case has no source manifest: "
            f"{case_dir / 'source_steady_manifest.json'}"
        )
        raise FileNotFoundError(message)
    source_case = _resolve_project_path(str(manifest["source"]["case_dir"]))
    params = AeroParams.model_validate(source_manifest["params"])
    config = CfdConfig.model_validate(source_manifest["cfd_config"])
    return manifest, source_manifest, source_case, params, config


def _map_restart_wall_snapshot(
    *,
    case_dir: Path,
    source_case: Path,
    params: AeroParams,
    config: CfdConfig,
    wall_vtp: Path,
    time_name: str,
) -> tuple[Path, dict[str, Any]]:
    source_mesh = trimesh.load_mesh(source_case / "cfd_surface" / "article.stl", process=True)
    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh = trimesh.util.concatenate(tuple(source_mesh.geometry.values()))
    source_regions = json.loads(
        (source_case / "cfd_surface" / "surface_regions.json").read_text(encoding="utf-8"),
    )
    output = case_dir / "outputs" / f"article_wall_regions_restart_{time_name}.vtp"
    report_path = case_dir / "quality" / f"region_mapping_restart_{time_name}.json"
    result = map_wall_regions_analytically_to_vtp(
        source_mesh=source_mesh,
        source_regions=source_regions,
        target_surface=pv.read(wall_vtp),
        params=params,
        output_vtp_path=output,
        report_path=report_path,
        max_distance_face_scale=config.quality.region_mapping_max_distance_face_scale,
        min_coverage=config.quality.region_mapping_min_coverage,
    )
    return output, result.as_dict()


def _series_stats(values: FloatArray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "sample_count": 0,
            "mean": None,
            "std": None,
            "rms": None,
            "min": None,
            "max": None,
            "peak_to_peak": None,
            "coefficient_of_variation": None,
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "sample_count": len(finite),
        "mean": mean,
        "std": std,
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "peak_to_peak": float(np.max(finite) - np.min(finite)),
        "coefficient_of_variation": abs(std / mean) if abs(mean) > NUMERIC_EPSILON else None,
    }


def _detrended(values: FloatArray) -> FloatArray:
    finite = np.asarray(values, dtype=np.float64)
    if len(finite) < MIN_CORRELATION_SAMPLES:
        return finite - np.mean(finite) if len(finite) else finite
    x = np.arange(len(finite), dtype=np.float64)
    coefficients = np.asarray(np.polyfit(x, finite, deg=1), dtype=np.float64)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    return finite - (slope * x + intercept)


def _pearson(a: FloatArray, b: FloatArray) -> float | None:
    if len(a) < MIN_CORRELATION_SAMPLES or len(b) < MIN_CORRELATION_SAMPLES or len(a) != len(b):
        return None
    if float(np.std(a)) <= NUMERIC_EPSILON or float(np.std(b)) <= NUMERIC_EPSILON:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _phase_relation(
    lhs: list[float],
    rhs: list[float],
    *,
    max_lag_samples: int = DEFAULT_PHASE_MAX_LAG_SAMPLES,
) -> dict[str, Any]:
    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    mask = np.isfinite(lhs_array) & np.isfinite(rhs_array)
    lhs_d = _detrended(lhs_array[mask])
    rhs_d = _detrended(rhs_array[mask])
    zero_lag = _pearson(lhs_d, rhs_d)
    best: tuple[int, float | None] = (0, zero_lag)
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag == 0:
            continue
        if lag < 0:
            a = lhs_d[:lag]
            b = rhs_d[-lag:]
        else:
            a = lhs_d[lag:]
            b = rhs_d[:-lag]
        corr = _pearson(a, b)
        if corr is None:
            continue
        if best[1] is None or abs(corr) > abs(best[1]):
            best = (lag, corr)
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "max_lag_samples": max_lag_samples,
        "zero_lag_correlation": zero_lag,
        "best_lag_samples": best[0],
        "best_lag_correlation": best[1],
        "detrending": "linear least-squares trend removed before correlation",
    }


def _coeff_series(rows: list[dict[str, Any]], *, group: str, coefficient: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if group == "total":
            coeffs = row["total"]["coefficients"]
        elif group == "critical_underfloor":
            coeffs = row["critical_underfloor"]["coefficients"]
        else:
            coeffs = row["named_groups"][group]["coefficients"]
        values.append(float(coeffs[coefficient]))
    return values


def _branch_phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_cdf = _coeff_series(rows, group="left_tunnel_y_negative", coefficient="c_df")
    right_cdf = _coeff_series(rows, group="right_tunnel_y_positive", coefficient="c_df")
    diffuser_cdf = _coeff_series(rows, group="diffuser_ramp", coefficient="c_df")
    total_cdf = _coeff_series(rows, group="total", coefficient="c_df")
    total_cd = _coeff_series(rows, group="total", coefficient="c_d")
    total_pitch = _coeff_series(rows, group="total", coefficient="c_m_pitch")
    return {
        "total_c_d": _series_stats(np.asarray(total_cd, dtype=np.float64)),
        "total_c_df": _series_stats(np.asarray(total_cdf, dtype=np.float64)),
        "total_c_m_pitch": _series_stats(np.asarray(total_pitch, dtype=np.float64)),
        "left_tunnel_c_df": _series_stats(np.asarray(left_cdf, dtype=np.float64)),
        "right_tunnel_c_df": _series_stats(np.asarray(right_cdf, dtype=np.float64)),
        "diffuser_ramp_c_df": _series_stats(np.asarray(diffuser_cdf, dtype=np.float64)),
        "left_right_c_df_phase": _phase_relation(left_cdf, right_cdf),
        "diffuser_total_c_df_phase": _phase_relation(diffuser_cdf, total_cdf),
    }


def write_steady_restart_wall_series_report(
    *,
    case_dir: Path,
    out_json: Path | None = None,
    times: tuple[str, ...] | None = None,
    streamwise_bins: int = 16,
) -> dict[str, Any]:
    """Map and integrate a compact wall-field time series for one restart branch."""

    if streamwise_bins < 1:
        message = "streamwise_bins must be positive"
        raise ValueError(message)
    manifest, source_manifest, source_case, params, config = _load_restart_source_inputs(case_dir)
    requested_times = _requested_wall_times(
        case_dir=case_dir,
        manifest=manifest,
        config=config,
        times=times,
    )
    rows: list[dict[str, Any]] = []
    mapping_reports: list[dict[str, Any]] = []
    for time_name in requested_times:
        wall_vtp = _combine_restart_wall_vtks(case_dir=case_dir, config=config, time_name=time_name)
        try:
            mapped_vtp, mapping = _map_restart_wall_snapshot(
                case_dir=case_dir,
                source_case=source_case,
                params=params,
                config=config,
                wall_vtp=wall_vtp,
                time_name=time_name,
            )
        except RegionMappingError as exc:
            message = f"restart wall mapping failed for time {time_name}: {exc}"
            raise RegionMappingError(message) from exc
        loads = integrate_spatial_loads(
            mapped_vtp,
            params=params,
            streamwise_bins=streamwise_bins,
        )
        mapping_reports.append(
            {
                "time": float(time_name),
                "area_coverage": mapping["area_coverage"],
                "cross_check_area_coverage": (mapping.get("cross_check", {}) or {}).get(
                    "area_coverage"
                ),
                "unmapped_faces": mapping["unmapped_faces"],
                "missing_regions": mapping["missing_regions"],
                "report_path": mapping["report_path"],
                "output_vtp_path": mapping["output_vtp_path"],
            },
        )
        rows.append(
            {
                "time": float(time_name),
                "mapped_wall_vtp": str(mapped_vtp),
                "total": loads["total"],
                "critical_underfloor": loads["critical_underfloor"],
                "named_groups": loads["named_groups"]["loads"],
                "streamwise_bins": loads["streamwise_bins"]["critical_underfloor"],
            },
        )

    phase_summary = _branch_phase_summary(rows)
    report = {
        "schema_version": "steady_restart_wall_series_v0.1.0",
        "case_dir": str(case_dir),
        "source_case": str(source_case),
        "source_simulation_id": source_manifest.get("simulation_id"),
        "branch_name": manifest.get("branch_name"),
        "plan_id": manifest.get("plan_id"),
        "accepted": False,
        "training_eligible": False,
        "time_series_note": (
            "Steady iteration samples are numerical diagnostics, not physical time samples."
        ),
        "requested_times": requested_times,
        "streamwise_bins": streamwise_bins,
        "foam_to_vtk_command_if_missing": _restart_wall_vtk_command(
            case_dir=case_dir,
            config=config,
            times=requested_times,
        ),
        "mapping_summary": {
            "min_area_coverage": min(item["area_coverage"] for item in mapping_reports),
            "min_cross_check_area_coverage": min(
                item["cross_check_area_coverage"]
                for item in mapping_reports
                if item["cross_check_area_coverage"] is not None
            )
            if any(item["cross_check_area_coverage"] is not None for item in mapping_reports)
            else None,
            "max_unmapped_faces": max(item["unmapped_faces"] for item in mapping_reports),
            "per_time": mapping_reports,
        },
        "phase_summary": phase_summary,
        "samples": rows,
        "classification_note": (
            "This report can support steady-restart diagnostic classification only. It is not an "
            "accepted CFD label and must not be used for training."
        ),
    }
    output = out_json or case_dir / "quality" / "steady_restart_wall_series.json"
    atomic_write_json(output, report)
    return report


def _relative_shift(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    if abs(reference) <= NUMERIC_EPSILON:
        return None
    return float((candidate - reference) / reference)


def _is_coherent_branch(summary: dict[str, Any]) -> bool:
    left_right = summary["left_right_c_df_phase"]["zero_lag_correlation"]
    diffuser_total = summary["diffuser_total_c_df_phase"]["zero_lag_correlation"]
    return (
        left_right is not None
        and abs(float(left_right)) >= PHYSICAL_UNSTEADINESS_CORRELATION_THRESHOLD
    ) or (
        diffuser_total is not None
        and abs(float(diffuser_total)) >= PHYSICAL_UNSTEADINESS_CORRELATION_THRESHOLD
    )


def compare_steady_restart_wall_series(
    *,
    reports: tuple[Path, ...],
    out_json: Path,
) -> dict[str, Any]:
    """Compare restart wall-series diagnostics across bounded steady branches."""

    if len(reports) < MIN_COMPARISON_REPORTS:
        message = "at least two wall-series reports are required for branch comparison"
        raise ValueError(message)
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    reference_summary = loaded[0]["phase_summary"]
    branch_summaries: list[dict[str, Any]] = []
    mean_shifts: dict[str, list[float]] = {
        "total_c_d": [],
        "total_c_df": [],
        "total_c_m_pitch": [],
    }
    amplitude_shifts: dict[str, list[float]] = {
        "total_c_d": [],
        "total_c_df": [],
        "total_c_m_pitch": [],
    }
    absolute_amplitude_shifts: dict[str, list[float]] = {
        "total_c_d": [],
        "total_c_df": [],
        "total_c_m_pitch": [],
    }
    for report_index, report in enumerate(loaded):
        summary = report["phase_summary"]
        branch_summaries.append(
            {
                "branch_name": report["branch_name"],
                "case_dir": report["case_dir"],
                "sample_count": len(report["samples"]),
                "mapping_min_area_coverage": report["mapping_summary"]["min_area_coverage"],
                "mapping_min_cross_check_area_coverage": report["mapping_summary"][
                    "min_cross_check_area_coverage"
                ],
                "phase_summary": summary,
                "coherent_wall_load_motion": _is_coherent_branch(summary),
            },
        )
        if report_index == 0:
            continue
        for key in ("total_c_d", "total_c_df", "total_c_m_pitch"):
            mean_shift = _relative_shift(
                reference_summary[key]["mean"],
                summary[key]["mean"],
            )
            amp_shift = _relative_shift(
                reference_summary[key]["peak_to_peak"],
                summary[key]["peak_to_peak"],
            )
            absolute_amp_shift = None
            if (
                reference_summary[key]["peak_to_peak"] is not None
                and summary[key]["peak_to_peak"] is not None
            ):
                absolute_amp_shift = abs(
                    float(summary[key]["peak_to_peak"])
                    - float(reference_summary[key]["peak_to_peak"]),
                )
            if mean_shift is not None:
                mean_shifts[key].append(abs(mean_shift))
            if amp_shift is not None:
                amplitude_shifts[key].append(abs(amp_shift))
            if absolute_amp_shift is not None:
                absolute_amplitude_shifts[key].append(absolute_amp_shift)

    max_mean_shift = max(
        (value for values in mean_shifts.values() for value in values),
        default=0.0,
    )
    max_amplitude_shift = max(
        (value for values in amplitude_shifts.values() for value in values),
        default=0.0,
    )
    max_abs_amplitude_shift = max(
        (value for values in absolute_amplitude_shifts.values() for value in values),
        default=0.0,
    )
    if max_mean_shift > NUMERICS_SENSITIVE_MEAN_SHIFT_THRESHOLD or (
        max_amplitude_shift > NUMERICS_SENSITIVE_AMPLITUDE_SHIFT_THRESHOLD
        and max_abs_amplitude_shift > NUMERICS_SENSITIVE_ABSOLUTE_AMPLITUDE_SHIFT_THRESHOLD
    ):
        classification = "NUMERICS_SENSITIVE"
    elif all(item["coherent_wall_load_motion"] for item in branch_summaries):
        classification = "PHYSICAL_UNSTEADINESS_CANDIDATE"
    else:
        classification = "UNRESOLVED_NEEDS_URANS_RECONNAISSANCE"
    comparison = {
        "schema_version": "steady_restart_wall_series_comparison_v0.1.0",
        "reports": [str(path) for path in reports],
        "accepted": False,
        "training_eligible": False,
        "classification": classification,
        "classification_inputs": {
            "max_total_mean_relative_shift": max_mean_shift,
            "max_total_peak_to_peak_relative_shift": max_amplitude_shift,
            "max_total_peak_to_peak_absolute_shift": max_abs_amplitude_shift,
            "mean_shift_threshold": NUMERICS_SENSITIVE_MEAN_SHIFT_THRESHOLD,
            "amplitude_shift_threshold": NUMERICS_SENSITIVE_AMPLITUDE_SHIFT_THRESHOLD,
            "absolute_amplitude_shift_threshold": (
                NUMERICS_SENSITIVE_ABSOLUTE_AMPLITUDE_SHIFT_THRESHOLD
            ),
            "coherence_threshold_abs_correlation": PHYSICAL_UNSTEADINESS_CORRELATION_THRESHOLD,
        },
        "branch_summaries": branch_summaries,
        "limitations": [
            "steady iteration index is not physical time",
            "classification remains diagnostic and does not make a label-grade CFD claim",
            "wall shear and separation metrics remain ineligible until regional y+ evidence passes",
        ],
    }
    atomic_write_json(out_json, comparison)
    return comparison

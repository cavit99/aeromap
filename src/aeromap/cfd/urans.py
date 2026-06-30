"""Prepare bounded URANS audit cases without running the expensive solve."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeromap.cfd.dictionaries import header
from aeromap.cfd.patch_surface import article_patch_names
from aeromap.cfd.postprocess import _coefficient_rows, _force_rows
from aeromap.cfd.schema import CfdConfig
from aeromap.constants import REF
from aeromap.io import atomic_write_json, atomic_write_text

DEFAULT_INITIAL_DELTA_T_S = 1.0e-5
DEFAULT_MAX_DELTA_T_S = 2.5e-5
DEFAULT_END_TIME_S = 0.02
DEFAULT_WRITE_INTERVAL_S = 5.0e-4
DEFAULT_PURGE_WRITE = 0
DEFAULT_MAX_CO = 1.0
DEFAULT_OUTER_CORRECTORS = 3
DEFAULT_PRESSURE_CORRECTORS = 2
DEFAULT_NON_ORTHOGONAL_CORRECTORS = 1
REQUIRED_RESTART_FIELDS = ("U", "p", "k", "omega", "nut")
BYTES_PER_SIZE_UNIT = 1024.0
EXPECTED_TRANSIENT_GZIP_FIELDS = (
    "U.gz",
    "UMean.gz",
    "UPrime2Mean.gz",
    "k.gz",
    "kMean.gz",
    "kPrime2Mean.gz",
    "nut.gz",
    "omega.gz",
    "omegaMean.gz",
    "omegaPrime2Mean.gz",
    "p.gz",
    "pMean.gz",
    "pPrime2Mean.gz",
    "phi.gz",
    "wallShearStress.gz",
    "yPlus.gz",
)
DUPLICATE_PREVIEW_LIMIT = 100
COMPACT_DUPLICATE_PREVIEW_LIMIT = 3
OPENFOAM_TIME_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
URANS_AUDIT_PURPOSES = frozenset(
    {
        "fine_shakedown",
        "medium_reconnaissance",
        "timestep_halving",
        "general_reconnaissance",
    },
)


@dataclass(frozen=True)
class UransAuditArtifacts:
    audit_id: str
    audit_dir: Path
    manifest_path: Path
    control_dict_path: Path
    fv_schemes_path: Path
    fv_solution_path: Path
    run_script_path: Path


@dataclass(frozen=True)
class UransForceHistoryArtifacts:
    report_path: Path


@dataclass(frozen=True)
class UransCheckpointArtifacts:
    report_path: Path
    latest_complete_time_s: float


@dataclass(frozen=True)
class UransPruneArtifacts:
    manifest_path: Path
    dry_run: bool
    candidate_count: int
    candidate_bytes: int


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


def _numeric_start_dir(path: Path) -> float:
    try:
        return float(path.parent.name)
    except ValueError:
        return float("-inf")


def _postprocessing_file_paths(case_dir: Path, function_name: str, file_name: str) -> list[Path]:
    post_dir = case_dir / "openfoam" / "postProcessing" / function_name
    if not post_dir.exists():
        return []
    return sorted(
        post_dir.glob(f"*/{file_name}"),
        key=lambda path: (_numeric_start_dir(path), path.as_posix()),
    )


def _time_key(value: float) -> str:
    return f"{value:.12g}"


def _merge_restart_rows(
    rows_by_file: list[tuple[Path, list[dict[str, Any]]]],
    *,
    max_time_s: float | None = None,
) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    excluded_by_max_time = 0
    for path, rows in rows_by_file:
        source_start = _numeric_start_dir(path)
        for row_index, row in enumerate(rows):
            if "time" not in row:
                continue
            row_time = float(row["time"])
            if max_time_s is not None and row_time > max_time_s:
                excluded_by_max_time += 1
                continue
            key = _time_key(row_time)
            tagged = {
                **row,
                "source_file": _repo_relative(path),
                "source_start_time_s": source_start,
                "source_row_index": row_index,
            }
            previous = merged.get(key)
            if previous is not None:
                duplicates.append(
                    {
                        "time_s": float(row["time"]),
                        "kept_source_file": tagged["source_file"],
                        "kept_source_start_time_s": source_start,
                        "replaced_source_file": previous["source_file"],
                        "replaced_source_start_time_s": previous["source_start_time_s"],
                    },
                )
            merged[key] = tagged
    merged_rows = sorted(merged.values(), key=lambda row: float(row["time"]))
    times = [float(row["time"]) for row in merged_rows]
    return {
        "rows": merged_rows,
        "summary": {
            "row_count": len(merged_rows),
            "first_time_s": times[0] if times else None,
            "last_time_s": times[-1] if times else None,
            "duplicate_time_count": len(duplicates),
            "duplicate_policy": (
                "later numeric postProcessing start directory replaces earlier rows at the same "
                "OpenFOAM time"
            ),
            "duplicates": duplicates[:DUPLICATE_PREVIEW_LIMIT],
            "duplicates_truncated": len(duplicates) > DUPLICATE_PREVIEW_LIMIT,
            "max_time_filter_s": max_time_s,
            "rows_excluded_by_max_time": excluded_by_max_time,
        },
    }


def _shell_literal(value: str) -> str:
    return shlex.quote(value)


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


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _directory_size_human(path: Path) -> str:
    size = float(_directory_size_bytes(path))
    units = ("B", "K", "M", "G", "T")
    for unit in units:
        if size < BYTES_PER_SIZE_UNIT or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}B"
            return f"{size:.1f}{unit}"
        size /= BYTES_PER_SIZE_UNIT
    return f"{size:.1f}{units[-1]}"


def _numeric_time_dirs(openfoam_dir: Path) -> list[Path]:
    time_dirs: list[Path] = []
    for child in openfoam_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            float(child.name)
        except ValueError:
            continue
        time_dirs.append(child)
    return sorted(time_dirs, key=lambda path: float(path.name))


def _numeric_time_dir_value(path: Path) -> float:
    return float(path.name)


def _validate_gzip_time_dir(
    time_dir: Path,
    *,
    expected_fields: tuple[str, ...] = EXPECTED_TRANSIENT_GZIP_FIELDS,
) -> dict[str, Any]:
    missing: list[str] = []
    bad: list[dict[str, str]] = []
    for name in expected_fields:
        path = time_dir / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            with gzip.open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
        except (OSError, EOFError) as exc:
            bad.append({"field": name, "error": str(exc)})
    return {
        "time_dir": time_dir.name,
        "time_s": float(time_dir.name),
        "missing_expected_fields": missing,
        "bad_gzip_fields": bad,
    }


def _parse_urans_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    max_cos: list[float] = []
    mean_cos: list[float] = []
    times: list[float] = []
    execution_times: list[float] = []
    for line in text.splitlines():
        courant_match = re.search(
            r"Courant Number mean:\s*([0-9.eE+-]+)\s+max:\s*([0-9.eE+-]+)",
            line,
        )
        if courant_match:
            mean_cos.append(float(courant_match.group(1)))
            max_cos.append(float(courant_match.group(2)))
        time_match = re.search(r"^Time =\s*([0-9.eE+-]+)s?\s*$", line)
        if time_match:
            times.append(float(time_match.group(1)))
        execution_match = re.search(r"ExecutionTime =\s*([0-9.eE+-]+)\s*s", line)
        if execution_match:
            execution_times.append(float(execution_match.group(1)))
    fatal_tokens = ("FOAM FATAL ERROR", "FOAM exiting", "Segmentation fault")
    return {
        "path": _repo_relative(path),
        "line_count": len(text.splitlines()),
        "contains_fatal_error": any(token in text for token in fatal_tokens),
        "sigfpe_trap_unsupported_banner_present": (
            "Floating point exception trapping - not supported on this platform" in text
        ),
        "ends_cleanly": text.rstrip().endswith("End"),
        "time_count": len(times),
        "first_time_s": times[0] if times else None,
        "last_time_s": times[-1] if times else None,
        "max_logged_time_s": max(times) if times else None,
        "last_execution_time_s": execution_times[-1] if execution_times else None,
        "max_courant_observed": max(max_cos) if max_cos else None,
        "last_mean_courant": mean_cos[-1] if mean_cos else None,
        "last_max_courant": max_cos[-1] if max_cos else None,
    }


def _compact_merge_summary(summary: dict[str, Any]) -> dict[str, Any]:
    duplicates = summary.get("duplicates", [])
    return {
        "duplicate_policy": summary.get("duplicate_policy"),
        "duplicate_time_count": summary.get("duplicate_time_count"),
        "duplicate_preview": duplicates[:COMPACT_DUPLICATE_PREVIEW_LIMIT],
        "duplicates_truncated": bool(
            summary.get("duplicates_truncated")
            or len(duplicates) > COMPACT_DUPLICATE_PREVIEW_LIMIT,
        ),
        "first_time_s": summary.get("first_time_s"),
        "last_time_s": summary.get("last_time_s"),
        "max_time_filter_s": summary.get("max_time_filter_s"),
        "row_count": summary.get("row_count"),
        "rows_excluded_by_max_time": summary.get("rows_excluded_by_max_time"),
    }


def _coefficient_short_stats(
    rows: list[dict[str, Any]],
    key: str,
    *,
    tail_count: int = 80,
) -> dict[str, Any]:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and math.isfinite(float(row[key]))
    ]
    tail = values[-min(tail_count, len(values)) :]
    return {
        "rows": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean_all": statistics.fmean(values) if values else None,
        "mean_tail_last_80_or_fewer": statistics.fmean(tail) if tail else None,
    }


def _validate_openfoam_time_name(value: str) -> str:
    if not OPENFOAM_TIME_RE.fullmatch(value):
        message = f"steady_time must be a numeric OpenFOAM time directory name: {value}"
        raise ValueError(message)
    return value


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"expected JSON object in {path}"
        raise TypeError(message)
    return loaded


def _dominant_period_iterations(steady_diagnostics: dict[str, Any]) -> float | None:
    try:
        period = steady_diagnostics["coefficients"]["c_df"]["spectrum"][
            "dominant_period_iterations"
        ]
    except KeyError:
        return None
    return float(period) if period is not None else None


def _patch_list(config: CfdConfig) -> str:
    patches = article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
    return "(" + " ".join(patches) + ")"


def _control_dict_transient(
    config: CfdConfig,
    *,
    initial_delta_t_s: float,
    max_delta_t_s: float,
    end_time_s: float,
    write_interval_s: float,
    purge_write: int,
    write_compression: bool,
    max_co: float,
) -> str:
    patches = _patch_list(config)
    compression = "on" if write_compression else "off"
    return (
        header("dictionary", "system", "controlDict")
        + f"""
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time_s:g};
deltaT          {initial_delta_t_s:g};

adjustTimeStep  yes;
maxCo           {max_co:g};
maxDeltaT       {max_delta_t_s:g};

writeControl    adjustableRunTime;
writeInterval   {write_interval_s:g};
purgeWrite      {purge_write};
writeFormat     ascii;
writePrecision  7;
writeCompression {compression};
timeFormat      general;
timePrecision   8;
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

    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            k
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            omega
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
        );
    }}
}}
"""
    )


def _fv_schemes_transient() -> str:
    return (
        header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
    )


def _fv_solution_transient(
    *,
    n_outer_correctors: int,
    n_correctors: int,
    n_non_orthogonal_correctors: int,
) -> str:
    return (
        header("dictionary", "system", "fvSolution")
        + f"""
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.05;
        smoother        GaussSeidel;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    pcorr
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0;
        smoother        GaussSeidel;
    }}

    pcorrFinal
    {{
        $pcorr;
    }}

    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.05;
    }}

    "(U|k|omega)Final"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }}
}}

PIMPLE
{{
    momentumPredictor yes;
    nOuterCorrectors {n_outer_correctors};
    nCorrectors {n_correctors};
    nNonOrthogonalCorrectors {n_non_orthogonal_correctors};
}}
"""
    )


def _run_script(
    *,
    source_case_rel: str,
    audit_dir_rel: str,
    work_case_rel: str,
    steady_time: str,
) -> str:
    source_case_literal = _shell_literal(source_case_rel)
    audit_dir_literal = _shell_literal(audit_dir_rel)
    work_case_literal = _shell_literal(work_case_rel)
    steady_time_literal = _shell_literal(steady_time)
    return f"""#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${{AEROMAP_REPO_ROOT:-$(pwd)}}"
SOURCE_CASE_REL={source_case_literal}
AUDIT_DIR_REL={audit_dir_literal}
WORK_CASE_REL={work_case_literal}
STEADY_TIME={steady_time_literal}

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
validate_repo_relative AUDIT_DIR_REL "$AUDIT_DIR_REL"
validate_repo_relative WORK_CASE_REL "$WORK_CASE_REL"
case "$WORK_CASE_REL" in
    artifacts/campaign/urans_audit_runs/*) ;;
    *)
        echo "URANS work case outside generated run directory: $WORK_CASE_REL" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"

SOURCE_CASE="$REPO_ROOT/$SOURCE_CASE_REL"
AUDIT_DIR="$REPO_ROOT/$AUDIT_DIR_REL"
WORK_CASE="$REPO_ROOT/$WORK_CASE_REL"
WORK_PARENT="$REPO_ROOT/artifacts/campaign/urans_audit_runs"
mkdir -p "$WORK_PARENT"
WORK_PARENT_REAL="$(cd "$WORK_PARENT" && pwd -P)"
WORK_CASE_PARENT_REAL="$(cd "$(dirname "$WORK_CASE")" && pwd -P)"
if [[ "$WORK_CASE_PARENT_REAL" != "$WORK_PARENT_REAL" ]]; then
    echo "Refusing to write URANS work case outside generated run directory: $WORK_CASE" >&2
    exit 2
fi

LOCK_DIR="$WORK_CASE.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another URANS run appears to be using this work case: $WORK_CASE" >&2
    echo "Remove the lock only after verifying no OpenFOAM/Docker process is active: $LOCK_DIR" >&2
    exit 2
fi
cleanup_lock() {{
    rm -f "$LOCK_DIR/host_pid" "$LOCK_DIR/started_at_utc"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}}
trap cleanup_lock EXIT
printf '%s\\n' "$$" > "$LOCK_DIR/host_pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK_DIR/started_at_utc"

if [[ -e "$WORK_CASE" && "${{AEROMAP_URANS_OVERWRITE:-0}}" != "1" ]]; then
    echo "Refusing to overwrite existing URANS work case: $WORK_CASE" >&2
    echo "Set AEROMAP_URANS_OVERWRITE=1 to replace this generated work case." >&2
    exit 2
fi

rm -rf "$WORK_CASE"
mkdir -p "$WORK_CASE/openfoam" "$WORK_CASE/logs" "$WORK_CASE/quality" "$WORK_CASE/outputs"
cp -a "$SOURCE_CASE/openfoam/constant" "$WORK_CASE/openfoam/constant"
cp -a "$SOURCE_CASE/openfoam/$STEADY_TIME" "$WORK_CASE/openfoam/0"
rm -rf "$WORK_CASE/openfoam/0/uniform"
mkdir -p "$WORK_CASE/openfoam/system"
cp "$AUDIT_DIR/system/controlDict" "$WORK_CASE/openfoam/system/controlDict"
cp "$AUDIT_DIR/system/fvSchemes" "$WORK_CASE/openfoam/system/fvSchemes"
cp "$AUDIT_DIR/system/fvSolution" "$WORK_CASE/openfoam/system/fvSolution"
cp "$SOURCE_CASE/manifest.json" "$WORK_CASE/source_steady_manifest.json"
cp "$AUDIT_DIR/manifest.json" "$WORK_CASE/urans_audit_manifest.json"

CONTAINER_WORK_DIR="/work/$WORK_CASE_REL/openfoam"
CONTAINER_WORK_DIR_ESCAPED="$(printf '%q' "$CONTAINER_WORK_DIR")"
container_cmd="set +u"
container_cmd="$container_cmd; source /opt/openfoam13/etc/bashrc"
container_cmd="$container_cmd; set -euo pipefail"
container_cmd="$container_cmd; cd $CONTAINER_WORK_DIR_ESCAPED"
container_cmd="$container_cmd; foamRun -solver incompressibleFluid"
container_cmd="$container_cmd > ../logs/foamRun_urans_recon.log 2>&1"
container_cmd="$container_cmd; postProcess -func yPlus -latestTime"
container_cmd="$container_cmd > ../logs/yPlus_urans_latest.log 2>&1 || true"
container_cmd="$container_cmd; postProcess -func wallShearStress -latestTime"
container_cmd="$container_cmd > ../logs/wallShearStress_urans_latest.log 2>&1 || true"
container_cmd="$container_cmd; foamToVTK -latestTime"
container_cmd="$container_cmd > ../logs/foamToVTK_urans_latest.log 2>&1 || true"
docker compose run --rm cfd "$container_cmd"
"""


def _audit_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "urans_audit_" + hashlib.sha256(encoded).hexdigest()[:16]


def _validate_restart_fields(source_case: Path, steady_time: str) -> None:
    missing = [
        str(source_case / "openfoam" / steady_time / field)
        for field in REQUIRED_RESTART_FIELDS
        if not (source_case / "openfoam" / steady_time / field).exists()
    ]
    if missing:
        message = "source case is missing required URANS restart fields: " + ", ".join(missing)
        raise FileNotFoundError(message)
    mesh_dir = source_case / "openfoam" / "constant" / "polyMesh"
    if not mesh_dir.exists():
        message = f"source case is missing OpenFOAM mesh directory: {mesh_dir}"
        raise FileNotFoundError(message)


def prepare_urans_audit(
    *,
    source_case: Path,
    out_dir: Path,
    audit_purpose: str = "general_reconnaissance",
    steady_time: str | None = None,
    initial_delta_t_s: float = DEFAULT_INITIAL_DELTA_T_S,
    max_delta_t_s: float = DEFAULT_MAX_DELTA_T_S,
    end_time_s: float = DEFAULT_END_TIME_S,
    write_interval_s: float = DEFAULT_WRITE_INTERVAL_S,
    purge_write: int = DEFAULT_PURGE_WRITE,
    write_compression: bool = False,
    max_co: float = DEFAULT_MAX_CO,
    n_outer_correctors: int = DEFAULT_OUTER_CORRECTORS,
    n_correctors: int = DEFAULT_PRESSURE_CORRECTORS,
    n_non_orthogonal_correctors: int = DEFAULT_NON_ORTHOGONAL_CORRECTORS,
) -> UransAuditArtifacts:
    """Prepare a bounded transient audit plan from an existing steady OpenFOAM case."""

    source_case = source_case.resolve()
    out_dir = out_dir.resolve()
    if initial_delta_t_s <= 0.0 or max_delta_t_s <= 0.0 or end_time_s <= 0.0:
        message = "URANS time settings must be positive"
        raise ValueError(message)
    if initial_delta_t_s > max_delta_t_s:
        message = "initial_delta_t_s must be <= max_delta_t_s"
        raise ValueError(message)
    if write_interval_s <= 0.0 or write_interval_s > end_time_s:
        message = "write_interval_s must be positive and <= end_time_s"
        raise ValueError(message)
    if purge_write < 0:
        message = "purge_write must be >= 0"
        raise ValueError(message)
    if audit_purpose not in URANS_AUDIT_PURPOSES:
        allowed = ", ".join(sorted(URANS_AUDIT_PURPOSES))
        message = f"audit_purpose must be one of: {allowed}"
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
    status = _load_json_if_present(source_case / "quality" / "status.json")
    steady_diagnostics = _load_json_if_present(source_case / "quality" / "steady_diagnostics.json")

    source_case_rel = _repo_relative_required(source_case, label="source_case")
    audit_payload = {
        "audit_purpose": audit_purpose,
        "source_simulation_id": manifest.get("simulation_id"),
        "source_case": source_case_rel,
        "steady_time": steady_time,
        "initial_delta_t_s": initial_delta_t_s,
        "max_delta_t_s": max_delta_t_s,
        "end_time_s": end_time_s,
        "write_interval_s": write_interval_s,
        "purge_write": purge_write,
        "write_compression": write_compression,
        "max_co": max_co,
        "n_outer_correctors": n_outer_correctors,
        "n_correctors": n_correctors,
        "n_non_orthogonal_correctors": n_non_orthogonal_correctors,
    }
    audit_id = _audit_id(audit_payload)
    audit_dir = out_dir / audit_id
    audit_dir_rel = _repo_relative_required(audit_dir, label="out_dir")
    system_dir = audit_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    work_case_rel = f"artifacts/campaign/urans_audit_runs/{audit_id}"
    control_dict_path = system_dir / "controlDict"
    fv_schemes_path = system_dir / "fvSchemes"
    fv_solution_path = system_dir / "fvSolution"
    run_script_path = audit_dir / "run_urans_audit.sh"
    manifest_path = audit_dir / "manifest.json"

    atomic_write_text(
        control_dict_path,
        _control_dict_transient(
            config,
            initial_delta_t_s=initial_delta_t_s,
            max_delta_t_s=max_delta_t_s,
            end_time_s=end_time_s,
            write_interval_s=write_interval_s,
            purge_write=purge_write,
            write_compression=write_compression,
            max_co=max_co,
        ),
    )
    atomic_write_text(fv_schemes_path, _fv_schemes_transient())
    atomic_write_text(
        fv_solution_path,
        _fv_solution_transient(
            n_outer_correctors=n_outer_correctors,
            n_correctors=n_correctors,
            n_non_orthogonal_correctors=n_non_orthogonal_correctors,
        ),
    )
    atomic_write_text(
        run_script_path,
        _run_script(
            source_case_rel=source_case_rel,
            audit_dir_rel=audit_dir_rel,
            work_case_rel=work_case_rel,
            steady_time=steady_time,
        ),
    )
    run_script_path.chmod(0o755)

    prepared_manifest = {
        "schema_version": "urans_audit_plan_v0.1.0",
        "audit_id": audit_id,
        "audit_purpose": audit_purpose,
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
        "time_setup": {
            "initial_delta_t_s": initial_delta_t_s,
            "max_delta_t_s": max_delta_t_s,
            "end_time_s": end_time_s,
            "write_interval_s": write_interval_s,
            "purge_write": purge_write,
            "write_compression": write_compression,
            "max_co": max_co,
            "output_budget_note": (
                "purgeWrite and compression affect retained field output only; high-frequency "
                "force and mass-flow postProcessing histories remain written every time step."
            ),
            "basis": (
                "Bounded reconnaissance setup for physical-time URANS. The steady SIMPLE "
                "oscillation period is recorded only as an iteration-sequence diagnostic "
                "and is not converted into physical time."
            ),
        },
        "steady_iteration_cycle": {
            "c_df_dominant_period_iterations": _dominant_period_iterations(steady_diagnostics),
            "note": "Steady SIMPLE iteration index is not physical time.",
        },
        "pimple": {
            "n_outer_correctors": n_outer_correctors,
            "n_correctors": n_correctors,
            "n_non_orthogonal_correctors": n_non_orthogonal_correctors,
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
                "Set AEROMAP_URANS_OVERWRITE=1 to replace the generated work case."
            ),
            "initialization": f"copy source openfoam/{steady_time} fields to generated openfoam/0",
        },
        "run_command": f"{_repo_relative(run_script_path)}",
        "classification_rules": {
            "audit_purpose": audit_purpose,
            "provisional": (
                "Any result from this plan remains provisional until finite fields, timestep "
                "sensitivity, stationarity, repeatability, wall treatment, mass balance and "
                "independent force integration are reviewed."
            ),
            "never_use_phase_snapshot_as_label": True,
        },
    }
    atomic_write_json(manifest_path, prepared_manifest)

    return UransAuditArtifacts(
        audit_id=audit_id,
        audit_dir=audit_dir,
        manifest_path=manifest_path,
        control_dict_path=control_dict_path,
        fv_schemes_path=fv_schemes_path,
        fv_solution_path=fv_solution_path,
        run_script_path=run_script_path,
    )


def write_urans_force_history_report(
    work_case: Path,
    *,
    out_json: Path | None = None,
    max_time_s: float | None = None,
) -> UransForceHistoryArtifacts:
    """Merge restart-overlapping URANS force histories with explicit provenance."""

    work_case = work_case.resolve()
    coeff_paths = _postprocessing_file_paths(work_case, "forceCoeffs", "forceCoeffs.dat")
    force_paths = _postprocessing_file_paths(work_case, "forces", "forces.dat")
    if not coeff_paths:
        message = f"URANS work case has no forceCoeffs history under {work_case}"
        raise FileNotFoundError(message)
    if not force_paths:
        message = f"URANS work case has no forces history under {work_case}"
        raise FileNotFoundError(message)

    if max_time_s is not None and max_time_s < 0.0:
        message = "max_time_s must be non-negative"
        raise ValueError(message)

    coeff = _merge_restart_rows(
        [(path, _coefficient_rows(path)) for path in coeff_paths],
        max_time_s=max_time_s,
    )
    forces = _merge_restart_rows(
        [(path, _force_rows(path)) for path in force_paths],
        max_time_s=max_time_s,
    )
    coeff_times = [row["time"] for row in coeff["rows"]]
    force_times = [row["time"] for row in forces["rows"]]
    output = out_json or work_case / "quality" / "transient_force_history.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "urans_restart_aware_force_history_v0.1.0",
        "work_case": _repo_relative(work_case),
        "status": "MERGED_RESTART_OVERLAPS_FOR_ANALYSIS_INPUT",
        "accepted": False,
        "training_eligible": False,
        "analysis_limits": [
            (
                "This report merges duplicate OpenFOAM restart rows; it does not establish "
                "stationarity, repeatability, timestep sensitivity or label eligibility."
            ),
            (
                "Use only as an input to later transient diagnostics, not as a campaign "
                "reference result by itself."
            ),
        ],
        "max_time_filter_s": max_time_s,
        "force_coefficients": coeff,
        "forces": forces,
        "time_alignment": {
            "coefficient_row_count": len(coeff_times),
            "force_row_count": len(force_times),
            "times_match_exactly": coeff_times == force_times,
        },
        "source_files": {
            "forceCoeffs": [_repo_relative(path) for path in coeff_paths],
            "forces": [_repo_relative(path) for path in force_paths],
        },
    }
    atomic_write_json(output, report)
    return UransForceHistoryArtifacts(report_path=output)


def write_urans_checkpoint_report(
    work_case: Path,
    *,
    out_json: Path,
    planned_end_time_s: float,
    source_case: Path | None = None,
    audit_id: str | None = None,
    audit_purpose: str = "medium_reconnaissance",
    status: str = "PARTIAL_MEDIUM_RECONNAISSANCE_RETAINED_RESTART_CHECKPOINT",
) -> UransCheckpointArtifacts:
    """Write compact URANS restart evidence from retained fields and logs."""

    work_case = work_case.resolve()
    openfoam_dir = work_case / "openfoam"
    if not openfoam_dir.exists():
        message = f"URANS work case has no OpenFOAM directory: {work_case}"
        raise FileNotFoundError(message)
    if planned_end_time_s <= 0.0:
        message = "planned_end_time_s must be positive"
        raise ValueError(message)

    validations = [
        _validate_gzip_time_dir(time_dir) for time_dir in _numeric_time_dirs(openfoam_dir)
    ]
    complete = [
        validation
        for validation in validations
        if not validation["missing_expected_fields"] and not validation["bad_gzip_fields"]
    ]
    if not complete:
        message = f"URANS work case has no complete gzip-readable transient field time: {work_case}"
        raise FileNotFoundError(message)
    latest = complete[-1]
    latest_time_s = float(latest["time_s"])

    force_history = write_urans_force_history_report(
        work_case,
        max_time_s=latest_time_s,
    )
    force_report = json.loads(force_history.report_path.read_text(encoding="utf-8"))
    spatial_history_path = work_case / "quality" / "urans_spatial_load_history.json"
    spatial_history: dict[str, Any] | None = None
    if spatial_history_path.exists():
        spatial_report = json.loads(spatial_history_path.read_text(encoding="utf-8"))
        time_dirs = spatial_report.get("time_dirs", [])
        spatial_history = {
            "schema": spatial_report.get("schema_version"),
            "report_path": _repo_relative(spatial_history_path),
            "accepted": spatial_report.get("accepted"),
            "training_eligible": spatial_report.get("training_eligible"),
            "row_count": spatial_report.get("row_count"),
            "first_time_dir": time_dirs[0] if time_dirs else None,
            "last_time_dir": time_dirs[-1] if time_dirs else None,
            "patches": spatial_report.get("patches"),
            "regions_present": spatial_report.get("regions_present"),
            "phase_relation": spatial_report.get("phase_relation"),
            "analysis_limits": spatial_report.get("analysis_limits"),
        }
    force_coefficients = force_report["force_coefficients"]
    forces = force_report["forces"]
    logs = [
        _parse_urans_log(path)
        for path in sorted((work_case / "logs").glob("foamRun_urans_recon*.log"))
    ]
    max_courant_values = [
        float(log["max_courant_observed"])
        for log in logs
        if log["max_courant_observed"] is not None
    ]

    checkpoint = {
        "schema": "aerocliff_urans_checkpoint_summary_v0.1.0",
        "status": status,
        "accepted": False,
        "training_eligible": False,
        "audit_id": audit_id or work_case.name,
        "audit_purpose": audit_purpose,
        "source_case": _repo_relative(source_case) if source_case is not None else None,
        "work_case": _repo_relative(work_case),
        "planned_end_time_s": planned_end_time_s,
        "latest_complete_written_time_dir": latest["time_dir"],
        "latest_complete_written_time_s": latest_time_s,
        "written_time_directories": [path.name for path in _numeric_time_dirs(openfoam_dir)],
        "time_directory_validation": validations,
        "gzip_validation": {
            "expected_fields": list(EXPECTED_TRANSIENT_GZIP_FIELDS),
            "latest_complete_time_dir": latest["time_dir"],
            "all_expected_gzip_fields_readable": True,
            "missing_expected_fields": [],
            "bad_gzip_fields": [],
        },
        "fields_at_latest_complete_time": list(EXPECTED_TRANSIENT_GZIP_FIELDS),
        "logs": logs,
        "overall_max_courant_observed": max(max_courant_values) if max_courant_values else None,
        "openfoam_end_reached_in_latest_log": logs[-1]["ends_cleanly"] if logs else False,
        "sigfpe_banner_note": (
            "OpenFOAM reports that floating point exception trapping is unsupported on this "
            "platform; this is a startup banner, not a solver crash."
        ),
        "transient_force_history": {
            "schema": force_report.get("schema_version"),
            "status": force_report.get("status"),
            "accepted": force_report.get("accepted"),
            "training_eligible": force_report.get("training_eligible"),
            "report_path": _repo_relative(force_history.report_path),
            "max_time_filter_s": force_report.get("max_time_filter_s"),
            "source_files": force_report.get("source_files"),
            "time_alignment": force_report.get("time_alignment"),
            "force_coefficients_summary": _compact_merge_summary(force_coefficients["summary"]),
            "forces_summary": _compact_merge_summary(forces["summary"]),
            "coefficient_short_stats": {
                "c_d": _coefficient_short_stats(force_coefficients["rows"], "c_d"),
                "c_df": _coefficient_short_stats(force_coefficients["rows"], "c_df"),
                "c_m_pitch": _coefficient_short_stats(force_coefficients["rows"], "c_m_pitch"),
            },
            "force_row_count": len(forces["rows"]),
            "coefficient_row_count": len(force_coefficients["rows"]),
        },
        "spatial_load_history": spatial_history
        if spatial_history is not None
        else {
            "status": "MISSING",
            "reason": (
                "No compact offline transient left/right or streamwise wall-load history found."
            ),
        },
        "output_budget": {
            "purge_write": 8,
            "write_compression": True,
            "retained_openfoam_restart_fields": True,
            "regenerateable_vtk_removed_to_save_space": True,
            "current_work_case_size_bytes": _directory_size_bytes(work_case),
            "current_work_case_size": _directory_size_human(work_case),
        },
        "claims_established": [
            "The corrected medium transient run can be resumed from retained OpenFOAM fields.",
            (
                "Complete gzip-readable transient field checkpoints now exist through "
                f"{latest_time_s:g} s."
            ),
            f"The latest {latest_time_s:g} s continuation ended cleanly in OpenFOAM.",
            "Max Courant stayed below 1 across original and resumed partial logs.",
            (
                "A restart-aware force-history merge exists and is capped at the latest "
                "complete field time for this checkpoint."
            ),
        ]
        + (
            [
                (
                    "A compact offline transient left/right and streamwise wall-load history "
                    "has been extracted from retained OpenFOAM boundary fields."
                ),
            ]
            if spatial_history is not None
            else []
        ),
        "claims_not_established": [
            "The planned 0.10-0.20 s medium reconnaissance is not complete.",
            (
                "The capped force history is not long enough for stationarity, period, "
                "frequency or mean-load acceptance."
            ),
            (
                "No URANS mean, repeatability, timestep sensitivity or label eligibility "
                "is established."
            ),
        ]
        + (
            [
                (
                    "The compact offline spatial-load history is diagnostic only and does "
                    "not establish physical phase, stationarity or mean-load acceptance."
                ),
            ]
            if spatial_history is not None
            else [
                (
                    "No offline transient left/right or streamwise load series has been "
                    "computed from this partial run."
                ),
            ]
        ),
        "analysis_limits": [
            (
                "This report merges duplicate OpenFOAM restart rows; it does not establish "
                "stationarity, repeatability, timestep sensitivity or label eligibility."
            ),
            (
                "Use only as an input to later transient diagnostics, not as a campaign "
                "reference result by itself."
            ),
        ],
        "git_sha_at_recording": _git_sha(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, checkpoint)
    return UransCheckpointArtifacts(
        report_path=out_json,
        latest_complete_time_s=latest_time_s,
    )


def write_urans_retained_field_prune_manifest(
    work_case: Path,
    *,
    keep_latest: int,
    out_json: Path | None = None,
    dry_run: bool = True,
) -> UransPruneArtifacts:
    """Plan or apply bounded pruning of retained URANS OpenFOAM time directories."""

    if keep_latest < 1:
        message = "keep_latest must be at least 1"
        raise ValueError(message)
    work_case = work_case.resolve()
    lock_dir = Path(f"{work_case}.lock")
    if lock_dir.exists():
        message = f"refusing to prune while URANS work-case lock exists: {lock_dir}"
        raise RuntimeError(message)
    openfoam_dir = work_case / "openfoam"
    if not openfoam_dir.exists():
        message = f"URANS work case has no OpenFOAM directory: {work_case}"
        raise FileNotFoundError(message)

    time_dirs = _numeric_time_dirs(openfoam_dir)
    initial_dirs = [path for path in time_dirs if _numeric_time_dir_value(path) == 0.0]
    transient_dirs = [path for path in time_dirs if _numeric_time_dir_value(path) > 0.0]
    kept_transient = transient_dirs[-keep_latest:]
    kept = sorted(
        [*initial_dirs, *kept_transient],
        key=_numeric_time_dir_value,
    )
    kept_set = {path.resolve() for path in kept}
    candidates = [path for path in transient_dirs if path.resolve() not in kept_set]
    candidate_sizes = {path.name: _directory_size_bytes(path) for path in candidates}
    candidate_bytes = sum(candidate_sizes.values())

    manifest = {
        "schema_version": "aerocliff_urans_retained_field_prune_v0.1.0",
        "work_case": _repo_relative(work_case),
        "dry_run": dry_run,
        "applied": not dry_run,
        "accepted": False,
        "training_eligible": False,
        "keep_latest": keep_latest,
        "policy": (
            "Keep OpenFOAM time 0 and the latest N positive transient field directories. "
            "Post-processing force histories are not pruned by this command."
        ),
        "kept_time_directories": [path.name for path in kept],
        "candidate_time_directories": [path.name for path in candidates],
        "candidate_count": len(candidates),
        "candidate_bytes": candidate_bytes,
        "candidate_size_by_time_dir_bytes": candidate_sizes,
        "warnings": [
            (
                "Do not apply pruning before extracting any required compact transient "
                "wall-load or field diagnostics from candidate directories."
            ),
            (
                "Pruning old field directories preserves restartability from the latest kept "
                "time but removes those raw volume/wall fields from local disk."
            ),
        ],
    }

    if not dry_run:
        for path in candidates:
            resolved = path.resolve()
            if resolved.parent != openfoam_dir.resolve():
                message = f"refusing to prune path outside OpenFOAM directory: {path}"
                raise ValueError(message)
            if not OPENFOAM_TIME_RE.fullmatch(path.name):
                message = f"refusing to prune non-time directory: {path}"
                raise ValueError(message)
            shutil.rmtree(path)

    output = out_json or work_case / "quality" / (
        "retained_field_prune_preview.json" if dry_run else "retained_field_prune_manifest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, manifest)
    return UransPruneArtifacts(
        manifest_path=output,
        dry_run=dry_run,
        candidate_count=len(candidates),
        candidate_bytes=candidate_bytes,
    )

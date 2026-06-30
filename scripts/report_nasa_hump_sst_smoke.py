"""Summarise the local NASA/TMR hump OpenFOAM SST smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from aeromap.cfd.nasa_hump import (
    HUMP_NU,
    HUMP_REYNOLDS_NUMBER,
    HUMP_U_INF,
    SST_SMOKE_CLASSIFICATION,
    methodology_mesh_policy,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = ROOT / "artifacts/methodology/nasa_hump/sst_smoke_case"
DEFAULT_EVIDENCE_PATH = ROOT / "docs/evidence/methodology/nasa_hump_sst_smoke_v0_1.json"
DEFAULT_REPORT_PATH = ROOT / "docs/reports/nasa_hump_sst_smoke_v0_1.md"
FLOAT_RE = r"[-+]?(?:\d*\.?\d+(?:[Ee][-+]?\d+)?|nan|inf)"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: str) -> float:
    return float(value.lower())


def _finite(value: float) -> bool:
    return math.isfinite(value)


def parse_checkmesh_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    failed = re.search(r"Failed\s+(\d+)\s+mesh checks", text)
    cells = re.search(r"cells:\s+(\d+)", text)
    patches = re.search(r"boundary patches:\s+(\d+)", text)
    aspect = re.search(r"High aspect ratio cells found, Max aspect ratio:\s+([0-9.Ee+-]+)", text)
    determinant = re.search(r"Cells with small determinant .* number of cells:\s+(\d+)", text)
    determinant_faces = re.search(
        r"faces on cells with determinant < [0-9.Ee+-]+\s+:\s+(\d+)",
        text,
    )
    return {
        "log_path": _relative(path),
        "cells": int(cells.group(1)) if cells else None,
        "boundary_patches": int(patches.group(1)) if patches else None,
        "failed_mesh_checks": int(failed.group(1)) if failed else 0 if "Mesh OK" in text else None,
        "max_aspect_ratio": float(aspect.group(1)) if aspect else None,
        "small_determinant_cells": int(determinant.group(1)) if determinant else 0,
        "determinant_warning_faces": int(determinant_faces.group(1)) if determinant_faces else 0,
        "mesh_ok": "Mesh OK" in text and "Failed" not in text,
    }


def parse_potential_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    residual = re.search(
        rf"Solving for Phi, Initial residual = ({FLOAT_RE}), "
        rf"Final residual = ({FLOAT_RE}), No Iterations (\d+)",
        text,
        flags=re.IGNORECASE,
    )
    continuity = re.search(rf"Continuity error = ({FLOAT_RE})", text, flags=re.IGNORECASE)
    final_residual = _as_float(residual.group(2)) if residual else math.nan
    continuity_error = _as_float(continuity.group(1)) if continuity else math.nan
    return {
        "log_path": _relative(path),
        "completed": "End" in text and "FOAM FATAL" not in text,
        "phi_initial_residual": _as_float(residual.group(1)) if residual else None,
        "phi_final_residual": final_residual if _finite(final_residual) else None,
        "phi_iterations": int(residual.group(3)) if residual else None,
        "continuity_error": continuity_error if _finite(continuity_error) else None,
    }


def parse_foamrun_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    times = [
        float(item) for item in re.findall(rf"^Time = ({FLOAT_RE})s", text, flags=re.MULTILINE)
    ]
    continuity_matches = re.findall(
        rf"time step continuity errors : sum local = ({FLOAT_RE}), global = ({FLOAT_RE}), "
        rf"cumulative = ({FLOAT_RE})",
        text,
        flags=re.IGNORECASE,
    )
    final_continuity = None
    if continuity_matches:
        local, global_error, cumulative = continuity_matches[-1]
        final_continuity = {
            "sum_local": _as_float(local),
            "global": _as_float(global_error),
            "cumulative": _as_float(cumulative),
        }
    nonfinite = re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE) is not None
    return {
        "log_path": _relative(path),
        "completed": "End" in text and "FOAM FATAL" not in text,
        "final_time": times[-1] if times else None,
        "iteration_count": len(times),
        "nonfinite_residuals_detected": nonfinite,
        "final_continuity": final_continuity,
    }


def parse_wall_shear_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"min/max\(hump_wall\) = \(({FLOAT_RE}) ({FLOAT_RE}) ({FLOAT_RE})\), "
        rf"\(({FLOAT_RE}) ({FLOAT_RE}) ({FLOAT_RE})\)",
        text,
        flags=re.IGNORECASE,
    )
    return {
        "log_path": _relative(path),
        "completed": "End" in text and "FOAM FATAL" not in text,
        "hump_wall_min": [_as_float(item) for item in match.groups()[:3]] if match else None,
        "hump_wall_max": [_as_float(item) for item in match.groups()[3:]] if match else None,
    }


def _field_stats(values: list[float], components: int) -> dict[str, Any]:
    tuples = len(values) // components
    if components == 1:
        finite = [value for value in values if _finite(value)]
        return {
            "components": components,
            "tuple_count": tuples,
            "min": min(finite),
            "max": max(finite),
            "mean": sum(finite) / len(finite),
        }
    component_values = [
        [values[row * components + col] for row in range(tuples)] for col in range(components)
    ]
    magnitudes = [
        math.sqrt(sum(values[row * components + col] ** 2 for col in range(components)))
        for row in range(tuples)
    ]
    return {
        "components": components,
        "tuple_count": tuples,
        "component_min": [min(column) for column in component_values],
        "component_max": [max(column) for column in component_values],
        "magnitude_min": min(magnitudes),
        "magnitude_max": max(magnitudes),
        "magnitude_mean": sum(magnitudes) / len(magnitudes),
    }


def parse_vtk_cell_fields(path: Path) -> dict[str, Any]:
    tokens = path.read_text(encoding="utf-8", errors="replace").split()
    cell_idx = tokens.index("CELL_DATA")
    cell_count = int(tokens[cell_idx + 1])
    field_idx = tokens.index("FIELD", cell_idx)
    field_count = int(tokens[field_idx + 2])
    cursor = field_idx + 3
    fields: dict[str, Any] = {}
    for _ in range(field_count):
        name = tokens[cursor]
        components = int(tokens[cursor + 1])
        tuples = int(tokens[cursor + 2])
        cursor += 4
        value_count = components * tuples
        values = [_as_float(value) for value in tokens[cursor : cursor + value_count]]
        cursor += value_count
        if name in {"p", "wallShearStress", "U"}:
            fields[name] = _field_stats(values, components)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "cell_count": cell_count,
        "fields": fields,
    }


def latest_hump_wall_vtk(case_dir: Path) -> Path:
    """Return the latest exported hump-wall VTK file from a smoke case."""

    candidates = sorted((case_dir / "VTK" / "hump_wall").glob("hump_wall_*.vtk"))
    if not candidates:
        msg = f"no hump_wall_*.vtk files found under {case_dir / 'VTK' / 'hump_wall'}"
        raise FileNotFoundError(msg)

    def time_value(path: Path) -> float:
        try:
            return float(path.stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            msg = f"could not parse VTK time from {path.name}"
            raise ValueError(msg) from exc

    return max(candidates, key=time_value)


def build_payload(case_dir: Path) -> dict[str, Any]:
    logs = case_dir / "logs"
    checkmesh_log = (
        logs / "checkMesh_meshQuality.log"
        if (logs / "checkMesh_meshQuality.log").exists()
        else logs / "checkMesh.log"
    )
    checkmesh = parse_checkmesh_log(checkmesh_log)
    potential = parse_potential_log(logs / "potentialFoam.log")
    foamrun = parse_foamrun_log(logs / "foamRun.log")
    wall_shear = parse_wall_shear_log(logs / "wallShearStress.log")
    wall_vtk = parse_vtk_cell_fields(latest_hump_wall_vtk(case_dir))
    mesh_policy = methodology_mesh_policy(checkmesh)
    smoke_passed = (
        potential["completed"]
        and foamrun["completed"]
        and not foamrun["nonfinite_residuals_detected"]
        and wall_shear["completed"]
        and wall_vtk["cell_count"] > 0
    )
    return {
        "schema_version": "nasa_hump_sst_smoke_v0.1.0",
        "classification": (
            SST_SMOKE_CLASSIFICATION if smoke_passed else f"{SST_SMOKE_CLASSIFICATION}_BLOCKED"
        ),
        "selected_case": "NASA/TMR 2D wall-mounted hump, no-flow-control, 103 x 28 no-plenum grid",
        "case_dir": _relative(case_dir),
        "run_mode": "local Docker/OpenFOAM v13 smoke",
        "flow_setup": {
            "solver": "incompressibleFluid",
            "initialisation": "potentialFoam",
            "turbulence_model": "kOmegaSST",
            "reynolds_number": HUMP_REYNOLDS_NUMBER,
            "u_inf": HUMP_U_INF,
            "chord": 1.0,
            "nu": HUMP_NU,
        },
        "mesh": {
            "checkmesh": checkmesh,
            "methodology_policy": mesh_policy,
        },
        "solver": {
            "potential_foam": potential,
            "foam_run": foamrun,
            "smoke_passed": smoke_passed,
        },
        "postprocess": {
            "wall_shear": wall_shear,
            "hump_wall_vtk": wall_vtk,
        },
        "claim_boundary": {
            "openfoam_sst_smoke_run": smoke_passed,
            "hump_wall_pressure_and_shear_exported": smoke_passed,
            "openfoam_vs_experiment_correlation": False,
            "turbulence_model_recommendation": False,
            "grid_convergence_claim": False,
            "production_cfd_accuracy": False,
        },
        "next_step": (
            "Use the same recipe on medium/fine NASA/TMR grids before any Cp/Cf "
            "correlation or turbulence-model statement."
        ),
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    mesh = payload["mesh"]["checkmesh"]
    solver = payload["solver"]["foam_run"]
    continuity = solver["final_continuity"] or {}
    vtk = payload["postprocess"]["hump_wall_vtk"]
    fields = vtk["fields"]
    wall = payload["postprocess"]["wall_shear"]
    lines = [
        "# NASA Hump SST Smoke v0.1",
        "",
        "## Result",
        "",
        (
            "A bounded OpenFOAM v13 SST smoke run now completes locally on the NASA/TMR "
            "wall-mounted hump `103 x 28` no-plenum grid after potential-flow "
            "initialisation."
        ),
        "",
        "| Item | Status |",
        "|---|---|",
        f"| Classification | `{payload['classification']}` |",
        f"| Solver smoke passed | `{payload['solver']['smoke_passed']}` |",
        f"| Final iteration | `{solver['final_time']}` |",
        f"| Non-finite residuals detected | `{solver['nonfinite_residuals_detected']}` |",
        f"| Final local continuity | `{continuity.get('sum_local')}` |",
        f"| Hump-wall VTK cells | `{vtk['cell_count']}` |",
        "",
        "## Setup",
        "",
        "- Solver: OpenFOAM Foundation v13 `incompressibleFluid`.",
        "- Turbulence model: `kOmegaSST`.",
        "- Initialisation: `potentialFoam` before `foamRun`.",
        "- Reynolds number: `936000` with `U_inf = 1`, chord `1`, `nu = 1 / 936000`.",
        "- Grid: NASA/TMR no-plenum `103 x 28` PLOT3D grid converted with `plot3dToFoam`.",
        "",
        "## Mesh Gate",
        "",
        (
            "The official tiny boundary-layer grid is still treated under the NASA/TMR "
            "methodology gate, not the global AeroMap production mesh gate."
        ),
        "",
        f"- Cells: `{mesh['cells']}`.",
        f"- Boundary patches: `{mesh['boundary_patches']}`.",
        f"- Failed mesh checks: `{mesh['failed_mesh_checks']}`.",
        f"- Max aspect ratio: `{mesh['max_aspect_ratio']}`.",
        f"- Determinant-warning faces: `{mesh['determinant_warning_faces']}`.",
        f"- Mesh policy: `{payload['mesh']['methodology_policy']['mesh_quality']}`.",
        "",
        "## Field Export",
        "",
        f"- Hump-wall pressure range: `{fields['p']['min']:.6g}` to `{fields['p']['max']:.6g}`.",
        (
            "- Hump-wall wall-shear-stress x range: "
            f"`{fields['wallShearStress']['component_min'][0]:.6g}` to "
            f"`{fields['wallShearStress']['component_max'][0]:.6g}`."
        ),
        (
            "- `foamPostProcess -solver incompressibleFluid -func wallShearStress` "
            f"min/max: `{wall['hump_wall_min']}` to `{wall['hump_wall_max']}`."
        ),
        "",
        "## Claim Boundary",
        "",
        "- Established: local OpenFOAM SST smoke execution and hump-wall field export.",
        (
            "- Not established: NASA correlation, turbulence-model recommendation, "
            "grid convergence or production CFD accuracy."
        ),
        "",
        "## Evidence",
        "",
        f"- JSON: `{DEFAULT_EVIDENCE_PATH.relative_to(ROOT)}`",
        f"- Generated case directory: `{payload['case_dir']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    payload = build_payload(args.case_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, args.report)
    print(json.dumps({"evidence": str(args.out), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()

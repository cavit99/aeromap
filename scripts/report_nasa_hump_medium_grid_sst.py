"""Report the NASA/TMR medium-grid OpenFOAM SST candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_nasa_hump_cp_cf import (  # noqa: E402
    Curve,
    _load_references,
    _openfoam_curve_error,
    extract_openfoam_wall_curve,
    latest_hump_wall_vtk,
    zero_crossings,
)
from scripts.prepare_nasa_hump_methodology import grid_summary  # noqa: E402
from scripts.report_nasa_hump_sst_smoke import (  # noqa: E402
    parse_checkmesh_log,
    parse_foamrun_log,
    parse_potential_log,
    parse_vtk_cell_fields,
    parse_wall_shear_log,
)

CLASSIFICATION = "OPENFOAM_NASA_HUMP_MEDIUM_GRID_SST_V0_1"
DEFAULT_CASE_DIR = ROOT / "artifacts/methodology/nasa_hump/sst_medium_grid_case"
DEFAULT_GRID = ROOT / "artifacts/methodology/nasa_hump/raw/hump2newtop_noplenumZ409x109.p2dfmt.gz"
RAW_DIR = ROOT / "artifacts/methodology/nasa_hump/raw"
EVIDENCE_PATH = ROOT / "docs/evidence/methodology/nasa_hump_medium_grid_sst_v0_1.json"
REPORT_PATH = ROOT / "docs/reports/nasa_hump_medium_grid_sst_v0_1.md"
FIGURE_PATH = ROOT / "docs/assets/methodology/nasa_hump_medium_grid_sst_cp_cf_overlay_v0_1.png"


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


def _curve_payload(curve: Curve) -> dict[str, list[float]]:
    return {
        "x": [float(item) for item in curve.x],
        "y": [float(item) for item in curve.y],
    }


def _plot_overlay(
    *,
    cp_curve: Curve,
    cf_curve: Curve,
    references: dict[str, Curve],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    axes[0].plot(references["cfl3d_sa_cp"].x, references["cfl3d_sa_cp"].y, label="CFL3D SA")
    axes[0].plot(references["cfl3d_sst_cp"].x, references["cfl3d_sst_cp"].y, label="CFL3D SST")
    axes[0].scatter(
        references["experimental_cp"].x,
        references["experimental_cp"].y,
        s=12,
        label="Experiment",
        zorder=4,
    )
    axes[0].plot(cp_curve.x, cp_curve.y, color="black", linewidth=1.8, label="OpenFOAM SST 409x109")
    axes[0].set_ylabel("$C_p$")
    axes[0].invert_yaxis()
    axes[0].grid(visible=True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(references["cfl3d_sa_cf"].x, references["cfl3d_sa_cf"].y, label="CFL3D SA")
    axes[1].plot(references["cfl3d_sst_cf"].x, references["cfl3d_sst_cf"].y, label="CFL3D SST")
    axes[1].scatter(
        references["experimental_cf"].x,
        references["experimental_cf"].y,
        s=12,
        label="Experiment",
        zorder=4,
    )
    axes[1].plot(cf_curve.x, cf_curve.y, color="black", linewidth=1.8, label="OpenFOAM SST 409x109")
    axes[1].axhline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("$C_f$")
    axes[1].grid(visible=True, alpha=0.25)

    for axis in axes:
        axis.set_xlim(-1.0, 2.2)
    fig.suptitle("NASA hump 409 x 109 OpenFOAM SST candidate - not grid-converged")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metrics(references: dict[str, Curve], cp_curve: Curve, cf_curve: Curve) -> dict[str, Any]:
    return {
        "experiment": {
            "cp": _openfoam_curve_error(references["experimental_cp"], cp_curve),
            "cf": _openfoam_curve_error(references["experimental_cf"], cf_curve),
        },
        "cfl3d_sst": {
            "cp": _openfoam_curve_error(references["cfl3d_sst_cp"], cp_curve),
            "cf": _openfoam_curve_error(references["cfl3d_sst_cf"], cf_curve),
        },
        "cfl3d_sa": {
            "cp": _openfoam_curve_error(references["cfl3d_sa_cp"], cp_curve),
            "cf": _openfoam_curve_error(references["cfl3d_sa_cf"], cf_curve),
        },
    }


def build_payload(*, case_dir: Path, grid_path: Path, figure_path: Path) -> dict[str, Any]:
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
    wall_vtk_path = latest_hump_wall_vtk(case_dir)
    wall_vtk = parse_vtk_cell_fields(wall_vtk_path)

    tangent_curve, tangent_audit = extract_openfoam_wall_curve(
        wall_vtk_path,
        cf_mode="wall_tangent",
    )
    global_curve, global_audit = extract_openfoam_wall_curve(wall_vtk_path, cf_mode="global_x")
    cp_curve = Curve(x=tangent_curve.x, y=tangent_curve.cp)
    cf_curve = Curve(x=tangent_curve.x, y=tangent_curve.cf)
    global_cf_curve = Curve(x=global_curve.x, y=global_curve.cf)
    references = _load_references(RAW_DIR)
    metrics = _metrics(references, cp_curve, cf_curve)
    _plot_overlay(cp_curve=cp_curve, cf_curve=cf_curve, references=references, path=figure_path)

    solver_completed = (
        potential["completed"]
        and foamrun["completed"]
        and not foamrun["nonfinite_residuals_detected"]
        and wall_shear["completed"]
        and wall_vtk["cell_count"] > 0
    )
    cf_zero_crossings = zero_crossings(cf_curve)
    candidate_plausible = solver_completed and len(cf_zero_crossings) > 0
    return {
        "schema_version": "nasa_hump_medium_grid_sst_v0.1.0",
        "classification": CLASSIFICATION,
        "selected_case": "NASA/TMR 2D wall-mounted hump, no-flow-control, 409 x 109 no-plenum grid",
        "case_dir": _relative(case_dir),
        "grid": {
            "path": _relative(grid_path),
            "sha256": _sha256(grid_path),
            "summary": grid_summary(grid_path),
        },
        "run_mode": "local Docker/OpenFOAM v13 medium-grid SST candidate",
        "solver": {
            "potential_foam": potential,
            "foam_run": foamrun,
            "completed_without_nonfinite_residuals": solver_completed,
        },
        "mesh": {
            "checkmesh": checkmesh,
            "mesh_gate_status": (
                "accepted_for_medium_grid_candidate"
                if checkmesh["failed_mesh_checks"] in (0, 2)
                else "blocked_or_unclassified"
            ),
        },
        "postprocess": {
            "wall_shear": wall_shear,
            "wall_vtk": wall_vtk,
        },
        "coefficient_contract": {
            "cp": "Cp = p / (0.5 * U_inf^2)",
            "cf_main": tangent_audit["cf_sign_audit"]["sign_transform_applied"],
            "cf_diagnostic_global_x": global_audit["cf_sign_audit"]["sign_transform_applied"],
            "main_cf_mode": "wall_tangent",
        },
        "cf_sign_audit": tangent_audit["cf_sign_audit"],
        "cf_global_x_diagnostic_audit": global_audit["cf_sign_audit"],
        "surface_coordinate_audit": tangent_audit["surface_coordinate_audit"],
        "openfoam_medium_curve": {
            "cp": _curve_payload(cp_curve),
            "cf_wall_tangent": _curve_payload(cf_curve),
            "cf_global_x_diagnostic": _curve_payload(global_cf_curve),
        },
        "zero_crossings": {
            "cf_wall_tangent": cf_zero_crossings,
            "cf_global_x_diagnostic": zero_crossings(global_cf_curve),
        },
        "medium_grid_candidate_error_metrics": metrics,
        "figure": {
            "path": _relative(figure_path),
            "sha256": _sha256(figure_path),
        },
        "candidate_assessment": {
            "solver_completed": solver_completed,
            "cf_zero_crossing_detected": len(cf_zero_crossings) > 0,
            "correlation_plausible_before_sa": candidate_plausible,
            "interpretation": (
                "Medium-grid SST is correlation-plausible enough to justify an SA/SST branch."
                if candidate_plausible
                else (
                    "Medium-grid SST is not yet correlation-plausible; further grid, boundary "
                    "condition or numerics work is needed before model comparison."
                )
            ),
        },
        "claim_boundary": {
            "medium_grid_candidate_overlay": True,
            "wall_tangent_cf_projection": True,
            "nasa_validation_accuracy": False,
            "grid_convergence": False,
            "turbulence_model_recommendation": False,
            "production_cfd_methodology_result": False,
            "f1_specific_accuracy": False,
        },
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    metrics = payload["medium_grid_candidate_error_metrics"]
    checkmesh = payload["mesh"]["checkmesh"]
    foamrun = payload["solver"]["foam_run"]
    assessment = payload["candidate_assessment"]
    lines = [
        "# NASA Hump Medium-Grid SST Candidate v0.1",
        "",
        "## Result",
        "",
        (
            "The OpenFOAM SST pipeline has been rerun on the official 409 x 109 "
            "NASA/TMR no-plenum grid and processed through the same Cp/Cf overlay "
            "machinery. This is a medium-grid candidate, not a validation result."
        ),
        "",
        "| Item | Status |",
        "|---|---|",
        f"| Classification | `{payload['classification']}` |",
        f"| Case directory | `{payload['case_dir']}` |",
        f"| Grid | `{payload['grid']['path']}` |",
        f"| Cells | `{checkmesh['cells']}` |",
        f"| Failed mesh checks | `{checkmesh['failed_mesh_checks']}` |",
        f"| foamRun final time | `{foamrun['final_time']}` |",
        f"| OpenFOAM `C_f` zero crossings | `{payload['zero_crossings']['cf_wall_tangent']}` |",
        (
            "| Correlation-plausible before SA branch | "
            f"`{assessment['correlation_plausible_before_sa']}` |"
        ),
        "",
        "## Coefficient Contract",
        "",
        "- `C_p = p / (0.5 * U_inf^2)`.",
        f"- Main `C_f`: `{payload['coefficient_contract']['cf_main']}`.",
        "- Main `C_f` uses wall-shear projection onto the local downstream wall tangent.",
        "- Global-x `C_f` is retained only as a diagnostic compatibility curve.",
        "",
        "## Medium-Grid Candidate Overlay Metrics",
        "",
        "These are medium-grid candidate metrics. They are not validation-quality metrics.",
        "",
        "| Reference | Cp RMSE | Cp MAE | Cf RMSE | Cf MAE |",
        "|---|---:|---:|---:|---:|",
        *[
            (
                f"| {name} | {values['cp']['rmse']:.5f} | {values['cp']['mae']:.5f} | "
                f"{values['cf']['rmse']:.6f} | {values['cf']['mae']:.6f} |"
            )
            for name, values in [
                ("Experiment", metrics["experiment"]),
                ("CFL3D SST", metrics["cfl3d_sst"]),
                ("CFL3D SA", metrics["cfl3d_sa"]),
            ]
        ],
        "",
        "## Assessment",
        "",
        assessment["interpretation"],
        "",
        "## Figure",
        "",
        (
            "![NASA hump medium-grid Cp/Cf overlay]"
            f"(../assets/methodology/{Path(payload['figure']['path']).name})"
        ),
        "",
        "## Claim Boundary",
        "",
        "- Established: the medium grid can be run through the local SST overlay pipeline.",
        "- Established: wall-tangent `C_f` projection is implemented for the medium candidate.",
        (
            "- Not established: NASA validation accuracy, grid convergence, "
            "turbulence-model recommendation or production CFD methodology."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--out", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()

    payload = build_payload(case_dir=args.case_dir, grid_path=args.grid, figure_path=args.figure)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, args.report)
    print(
        json.dumps(
            {"evidence": str(args.out), "report": str(args.report), "figure": str(args.figure)},
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()

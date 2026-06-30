"""Extract smoke-grid NASA/TMR hump Cp/Cf curves from OpenFOAM wall fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from aeromap.cfd.nasa_hump import HUMP_U_INF

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts/methodology/nasa_hump/raw"
SMOKE_EVIDENCE_PATH = ROOT / "docs/evidence/methodology/nasa_hump_sst_smoke_v0_1.json"
EVIDENCE_PATH = ROOT / "docs/evidence/methodology/nasa_hump_cp_cf_extraction_v0_1.json"
REPORT_PATH = ROOT / "docs/reports/nasa_hump_cp_cf_extraction_v0_1.md"
FIGURE_PATH = ROOT / "docs/assets/methodology/nasa_hump_cp_cf_overlay_v0_1.png"
CLASSIFICATION = "OPENFOAM_NASA_HUMP_CP_CF_EXTRACTION_V0_1"
ATTACHED_REGION_X_MAX = -2.0
MIN_ATTACHED_SIGN_SAMPLES = 5
MIN_TANGENT_POINTS = 2
TANGENT_NORM_EPSILON = 1e-14


@dataclass(frozen=True)
class WallCurve:
    x: np.ndarray
    cp: np.ndarray
    cf: np.ndarray
    raw_wall_shear_x: np.ndarray
    raw_wall_shear_tangent: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray


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


def parse_curve(path: Path, y_column: int = 1) -> Curve:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith(("variables", "zone")):
            continue
        values = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", stripped)
        if len(values) <= y_column:
            continue
        rows.append((float(values[0]), float(values[y_column])))
    if not rows:
        msg = f"no numeric curve rows found in {path}"
        raise ValueError(msg)
    arr = np.asarray(rows, dtype=np.float64)
    return Curve(x=arr[:, 0], y=arr[:, 1])


def curve_error(reference: Curve, candidate: Curve) -> dict[str, float]:
    lo = max(float(reference.x.min()), float(candidate.x.min()))
    hi = min(float(reference.x.max()), float(candidate.x.max()))
    mask = (reference.x >= lo) & (reference.x <= hi)
    x = reference.x[mask]
    y_ref = reference.y[mask]
    y_pred = np.interp(x, candidate.x, candidate.y)
    err = y_pred - y_ref
    return {
        "comparison_points": int(x.size),
        "x_min": lo,
        "x_max": hi,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def zero_crossings(curve: Curve) -> list[float]:
    crossings: list[float] = []
    for idx in range(len(curve.y) - 1):
        y0 = float(curve.y[idx])
        y1 = float(curve.y[idx + 1])
        if y0 == 0:
            crossings.append(float(curve.x[idx]))
        elif y0 * y1 < 0:
            fraction = abs(y0) / (abs(y0) + abs(y1))
            crossings.append(float(curve.x[idx] + fraction * (curve.x[idx + 1] - curve.x[idx])))
    return crossings


def latest_hump_wall_vtk(case_dir: Path) -> Path:
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


def _wall_tangent(face_points: np.ndarray) -> np.ndarray:
    """Return a downstream-oriented wall tangent for a boundary face."""

    xy = np.unique(np.round(face_points[:, :2], decimals=12), axis=0)
    if xy.shape[0] >= MIN_TANGENT_POINTS:
        deltas = xy[:, None, :] - xy[None, :, :]
        distances = np.sum(deltas**2, axis=2)
        i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
        vector_2d = xy[j] - xy[i]
    else:
        vector_2d = np.array([1.0, 0.0], dtype=np.float64)
    if (
        abs(float(vector_2d[0])) < TANGENT_NORM_EPSILON
        and abs(float(vector_2d[1])) < TANGENT_NORM_EPSILON
    ):
        vector_2d = np.array([1.0, 0.0], dtype=np.float64)
    if vector_2d[0] < 0:
        vector_2d = -vector_2d
    tangent = np.array([vector_2d[0], vector_2d[1], 0.0], dtype=np.float64)
    return tangent / float(np.linalg.norm(tangent))


def _parse_vtk_polydata(path: Path) -> dict[str, Any]:
    tokens = path.read_text(encoding="utf-8", errors="replace").split()

    points_idx = tokens.index("POINTS")
    point_count = int(tokens[points_idx + 1])
    cursor = points_idx + 3
    point_values = [_as_float(item) for item in tokens[cursor : cursor + 3 * point_count]]
    points = np.asarray(point_values, dtype=np.float64).reshape(point_count, 3)
    cursor += 3 * point_count

    polygons_idx = tokens.index("POLYGONS", cursor)
    polygon_count = int(tokens[polygons_idx + 1])
    cursor = polygons_idx + 3
    polygons: list[list[int]] = []
    for _ in range(polygon_count):
        vertex_count = int(tokens[cursor])
        cursor += 1
        polygon = [int(item) for item in tokens[cursor : cursor + vertex_count]]
        polygons.append(polygon)
        cursor += vertex_count

    cell_data_idx = tokens.index("CELL_DATA", cursor)
    cell_count = int(tokens[cell_data_idx + 1])
    field_idx = tokens.index("FIELD", cell_data_idx)
    field_count = int(tokens[field_idx + 2])
    cursor = field_idx + 3
    fields: dict[str, np.ndarray] = {}
    for _ in range(field_count):
        name = tokens[cursor]
        components = int(tokens[cursor + 1])
        tuples = int(tokens[cursor + 2])
        cursor += 4
        values = [_as_float(item) for item in tokens[cursor : cursor + components * tuples]]
        cursor += components * tuples
        fields[name] = np.asarray(values, dtype=np.float64).reshape(tuples, components)

    centers = np.asarray([points[polygon].mean(axis=0) for polygon in polygons], dtype=np.float64)
    tangents = np.asarray(
        [_wall_tangent(points[polygon]) for polygon in polygons],
        dtype=np.float64,
    )
    return {
        "points": points,
        "polygons": polygons,
        "cell_count": cell_count,
        "centers": centers,
        "wall_tangents": tangents,
        "fields": fields,
    }


def _curve_summary(curve: Curve) -> dict[str, float | int]:
    return {
        "points": int(curve.x.size),
        "x_min": float(curve.x.min()),
        "x_max": float(curve.x.max()),
        "y_min": float(curve.y.min()),
        "y_max": float(curve.y.max()),
    }


def _openfoam_curve_error(reference: Curve, candidate: Curve) -> dict[str, float | int]:
    metrics = curve_error(reference, candidate)
    return {
        "comparison_points": metrics["comparison_points"],
        "x_min": metrics["x_min"],
        "x_max": metrics["x_max"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "max_abs": metrics["max_abs"],
    }


def _cf_formula(
    *,
    cf_mode: Literal["global_x", "wall_tangent"],
    sign_multiplier: float,
) -> str:
    if cf_mode == "global_x":
        prefix = "-" if sign_multiplier < 0 else ""
        return f"Cf = {prefix}wallShearStress_x / (0.5 * U_inf^2)"
    prefix = "-" if sign_multiplier < 0 else ""
    return f"Cf = {prefix}dot(wallShearStress, wall_tangent) / (0.5 * U_inf^2)"


def extract_openfoam_wall_curve(
    vtk_path: Path,
    *,
    u_inf: float = HUMP_U_INF,
    cf_mode: Literal["global_x", "wall_tangent"] = "global_x",
) -> tuple[WallCurve, dict[str, Any]]:
    vtk = _parse_vtk_polydata(vtk_path)
    fields = vtk["fields"]
    if "p" not in fields or "wallShearStress" not in fields:
        msg = f"{vtk_path} does not contain both p and wallShearStress cell fields"
        raise ValueError(msg)

    centers = vtk["centers"]
    pressure = fields["p"][:, 0]
    raw_wall_shear = fields["wallShearStress"]
    raw_wall_shear_x = raw_wall_shear[:, 0]
    raw_wall_shear_tangent = np.einsum("ij,ij->i", raw_wall_shear, vtk["wall_tangents"])
    raw_shear_for_cf = raw_wall_shear_x if cf_mode == "global_x" else raw_wall_shear_tangent
    x = centers[:, 0]
    order = np.argsort(x)

    x_sorted = x[order]
    pressure_sorted = pressure[order]
    shear_x_sorted = raw_wall_shear_x[order]
    shear_tangent_sorted = raw_wall_shear_tangent[order]
    shear_for_cf_sorted = raw_shear_for_cf[order]
    q_inf = 0.5 * u_inf**2

    upstream_mask = x_sorted < ATTACHED_REGION_X_MAX
    if int(upstream_mask.sum()) < MIN_ATTACHED_SIGN_SAMPLES:
        upstream_mask = np.arange(x_sorted.size) < max(
            MIN_ATTACHED_SIGN_SAMPLES,
            x_sorted.size // 5,
        )
    upstream_median = float(np.median(shear_for_cf_sorted[upstream_mask]))
    sign_multiplier = -1.0 if upstream_median < 0 else 1.0

    cp = pressure_sorted / q_inf
    cf = sign_multiplier * shear_for_cf_sorted / q_inf
    non_monotonic_steps = int(np.sum(np.diff(x_sorted) <= 0))
    duplicate_x_count = int(x_sorted.size - np.unique(np.round(x_sorted, decimals=10)).size)
    sign_audit = {
        "audit_status": "accepted_for_smoke_overlay_only",
        "positive_cf_convention": (
            "positive in the local downstream wall-tangent direction of the external flow"
        ),
        "cf_mode": cf_mode,
        "raw_openfoam_wall_shear_field": (
            "wallShearStress_x from hump_wall cell data"
            if cf_mode == "global_x"
            else "wallShearStress projected onto downstream-oriented local wall tangent"
        ),
        "raw_attached_region_definition": (
            "hump-wall cells with x < -2.0, falling back to first 20 percent if needed"
        ),
        "raw_attached_region_sample_count": int(upstream_mask.sum()),
        "openfoam_wall_shear_x_raw_attached_region_median": upstream_median,
        "raw_attached_region_sign": "negative" if upstream_median < 0 else "positive",
        "sign_transform_applied": _cf_formula(cf_mode=cf_mode, sign_multiplier=sign_multiplier),
        "later_correlation_note": (
            "This extraction records the shear component convention explicitly. "
            "Medium/fine correlation candidates should use local wall-tangent projection."
        ),
    }
    curve = WallCurve(
        x=x_sorted,
        cp=cp,
        cf=cf,
        raw_wall_shear_x=shear_x_sorted,
        raw_wall_shear_tangent=shear_tangent_sorted,
        pressure=pressure_sorted,
    )
    geometry_audit = {
        "sample_count": int(x_sorted.size),
        "x_min": float(x_sorted.min()),
        "x_max": float(x_sorted.max()),
        "non_monotonic_steps_after_sort": non_monotonic_steps,
        "duplicate_x_count_after_rounding_1e_10": duplicate_x_count,
    }
    return curve, {"cf_sign_audit": sign_audit, "surface_coordinate_audit": geometry_audit}


def _load_references(raw_dir: Path) -> dict[str, Curve]:
    return {
        "experimental_cp": parse_curve(raw_dir / "experimental_cp.dat"),
        "experimental_cf": parse_curve(raw_dir / "experimental_cf.dat"),
        "cfl3d_sst_cp": parse_curve(raw_dir / "cfl3d_sst_cp.dat"),
        "cfl3d_sst_cf": parse_curve(raw_dir / "cfl3d_sst_cf.dat"),
        "cfl3d_sa_cp": parse_curve(raw_dir / "cfl3d_sa_cp.dat"),
        "cfl3d_sa_cf": parse_curve(raw_dir / "cfl3d_sa_cf.dat"),
    }


def _reference_sources(raw_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: {
            "path": _relative(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(raw_dir.glob("*.dat"))
    }


def _curve_to_payload(curve: WallCurve) -> dict[str, list[float]]:
    return {
        "x": [float(item) for item in curve.x],
        "cp": [float(item) for item in curve.cp],
        "cf": [float(item) for item in curve.cf],
        "raw_wall_shear_x": [float(item) for item in curve.raw_wall_shear_x],
        "raw_wall_shear_tangent": [float(item) for item in curve.raw_wall_shear_tangent],
        "pressure": [float(item) for item in curve.pressure],
    }


def _plot_overlay(*, curve: WallCurve, references: dict[str, Curve], path: Path) -> None:
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
    axes[0].plot(curve.x, curve.cp, color="black", linewidth=1.8, label="OpenFOAM SST smoke")
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
    axes[1].plot(curve.x, curve.cf, color="black", linewidth=1.8, label="OpenFOAM SST smoke")
    axes[1].axhline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("$C_f$")
    axes[1].grid(visible=True, alpha=0.25)

    for axis in axes:
        axis.set_xlim(-1.0, 2.2)
    fig.suptitle("NASA hump 103 x 28 OpenFOAM SST smoke - not grid-converged")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_payload(
    *,
    case_dir: Path,
    raw_dir: Path,
    figure_path: Path,
    cf_mode: Literal["global_x", "wall_tangent"] = "global_x",
) -> dict[str, Any]:
    smoke_evidence = json.loads(SMOKE_EVIDENCE_PATH.read_text(encoding="utf-8"))
    vtk_path = latest_hump_wall_vtk(case_dir)
    curve, audits = extract_openfoam_wall_curve(vtk_path, cf_mode=cf_mode)
    references = _load_references(raw_dir)
    openfoam_cp = Curve(x=curve.x, y=curve.cp)
    openfoam_cf = Curve(x=curve.x, y=curve.cf)
    metrics = {
        "experiment": {
            "cp": _openfoam_curve_error(references["experimental_cp"], openfoam_cp),
            "cf": _openfoam_curve_error(references["experimental_cf"], openfoam_cf),
        },
        "cfl3d_sst": {
            "cp": _openfoam_curve_error(references["cfl3d_sst_cp"], openfoam_cp),
            "cf": _openfoam_curve_error(references["cfl3d_sst_cf"], openfoam_cf),
        },
        "cfl3d_sa": {
            "cp": _openfoam_curve_error(references["cfl3d_sa_cp"], openfoam_cp),
            "cf": _openfoam_curve_error(references["cfl3d_sa_cf"], openfoam_cf),
        },
    }
    _plot_overlay(curve=curve, references=references, path=figure_path)
    return {
        "schema_version": "nasa_hump_cp_cf_extraction_v0.1.0",
        "classification": CLASSIFICATION,
        "purpose": (
            "Convert OpenFOAM SST smoke wall pressure/shear fields into NASA/TMR-style "
            "Cp/Cf curves and smoke-only overlay metrics."
        ),
        "source_smoke_evidence": {
            "path": _relative(SMOKE_EVIDENCE_PATH),
            "sha256": _sha256(SMOKE_EVIDENCE_PATH),
            "classification": smoke_evidence["classification"],
        },
        "selected_hump_wall_vtk": {
            "path": _relative(vtk_path),
            "sha256": _sha256(vtk_path),
        },
        "coefficient_formulas": {
            "dynamic_pressure_kinematic": "q = 0.5 * U_inf^2",
            "cp": "Cp = p / (0.5 * U_inf^2)",
            "cf": audits["cf_sign_audit"]["sign_transform_applied"],
            "u_inf": HUMP_U_INF,
            "pressure_units": "OpenFOAM incompressible kinematic pressure",
            "wall_shear_units": "OpenFOAM kinematic wall shear stress",
        },
        "cf_sign_audit": audits["cf_sign_audit"],
        "surface_coordinate_audit": audits["surface_coordinate_audit"],
        "openfoam_smoke_curve": _curve_to_payload(curve),
        "reference_sources": _reference_sources(raw_dir),
        "reference_curve_summary": {
            name: _curve_summary(curve_item) for name, curve_item in references.items()
        },
        "openfoam_smoke_zero_crossings": {
            "cf": zero_crossings(openfoam_cf),
        },
        "single_grid_smoke_error_metrics": metrics,
        "figure": {
            "path": _relative(figure_path),
            "sha256": _sha256(figure_path),
        },
        "claim_boundary": {
            "cp_cf_extraction": True,
            "smoke_grid_overlay_metrics": True,
            "nasa_validation_accuracy": False,
            "grid_convergence": False,
            "turbulence_model_recommendation": False,
            "production_cfd_methodology_result": False,
            "f1_specific_accuracy": False,
        },
        "next_step": (
            "Run the same extraction on a medium NASA/TMR grid before any correlation "
            "or turbulence-model statement."
        ),
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    sign = payload["cf_sign_audit"]
    surface = payload["surface_coordinate_audit"]
    metrics = payload["single_grid_smoke_error_metrics"]
    figure = payload["figure"]["path"]
    lines = [
        "# NASA Hump Cp/Cf Extraction v0.1",
        "",
        "## Result",
        "",
        (
            "The OpenFOAM SST smoke wall fields can be converted into NASA/TMR-style "
            "`C_p(x)` and `C_f(x)` curves. This is an extraction and overlay check, "
            "not a validation result."
        ),
        "",
        "| Item | Status |",
        "|---|---|",
        f"| Classification | `{payload['classification']}` |",
        f"| Wall samples | `{surface['sample_count']}` |",
        f"| x-range | `{surface['x_min']:.5g}` to `{surface['x_max']:.5g}` |",
        f"| OpenFOAM VTK | `{payload['selected_hump_wall_vtk']['path']}` |",
        f"| Figure | `{figure}` |",
        "",
        "## Coefficient Contract",
        "",
        "- OpenFOAM pressure is treated as incompressible kinematic pressure.",
        "- `C_p = p / (0.5 * U_inf^2)`.",
        f"- `C_f`: `{sign['sign_transform_applied']}`.",
        "- `U_inf = 1.0` for this nondimensional smoke setup.",
        "",
        "## C_f Sign Audit",
        "",
        (
            "Positive `C_f` is defined as wall shear acting in the local downstream "
            "wall-tangent direction of the external flow."
        ),
        "",
        f"- Raw attached-region definition: `{sign['raw_attached_region_definition']}`.",
        f"- Raw attached-region samples: `{sign['raw_attached_region_sample_count']}`.",
        (
            "- Raw OpenFOAM `wallShearStress_x` attached-region median: "
            f"`{sign['openfoam_wall_shear_x_raw_attached_region_median']:.6g}`."
        ),
        f"- Raw sign: `{sign['raw_attached_region_sign']}`.",
        f"- Audit status: `{sign['audit_status']}`.",
        "",
        "This v0.1 smoke extraction uses the global x-component after sign audit. A "
        "medium/fine correlation branch should project wall shear onto the local wall tangent.",
        "",
        "## Smoke-Only Overlay Metrics",
        "",
        "These are single-grid smoke metrics. They are not correlation-quality metrics.",
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
        "## Figure",
        "",
        f"![NASA hump Cp/Cf overlay](../assets/methodology/{Path(figure).name})",
        "",
        "## Claim Boundary",
        "",
        "- Established: smoke-grid wall fields can be converted into `C_p(x)` and `C_f(x)` curves.",
        (
            "- Established: experiment and CFL3D overlays/metrics can be generated "
            "with fixed formulas."
        ),
        (
            "- Not established: NASA validation accuracy, grid convergence, "
            "turbulence-model recommendation or production CFD methodology."
        ),
        "",
        "## Evidence",
        "",
        f"- JSON: `{EVIDENCE_PATH.relative_to(ROOT)}`",
        f"- Figure: `{figure}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=ROOT / "artifacts/methodology/nasa_hump/sst_smoke_case",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    parser.add_argument("--cf-mode", choices=["global_x", "wall_tangent"], default="global_x")
    args = parser.parse_args()

    payload = build_payload(
        case_dir=args.case_dir,
        raw_dir=args.raw_dir,
        figure_path=args.figure,
        cf_mode=args.cf_mode,
    )
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

"""Prepare the NASA wall-mounted hump CFD methodology preflight artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from aeromap.cfd.nasa_hump import (
    CLASSIFICATION,
    PATCH_CONTRACT,
    RECORDED_103X28_CHECKMESH,
    correlation_eligibility,
    methodology_mesh_policy,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts/methodology/nasa_hump/raw"
EVIDENCE_PATH = ROOT / "docs/evidence/methodology/nasa_hump_methodology_preflight_v0_1.json"
REPORT_PATH = ROOT / "docs/reports/nasa_hump_cfd_methodology_preflight_v0_1.md"
GRID_ZIP_URL = "https://www.nasa.gov/wp-content/uploads/2026/02/nasahump-grids.zip"
GRID_MEMBER = (
    "u/piyer/nasa_tmr/gitlab/turbmodels/Nasahump_grids/hump2newtop_noplenumZ103x28.p2dfmt.gz"
)

SOURCES = {
    "tmr_case_intro": "https://tmbwg.github.io/turbmodels/nasahump_val.html",
    "tmr_sst_reference": "https://tmbwg.github.io/turbmodels/nasahump_val_sst.html",
    "tmr_sa_reference": "https://tmbwg.github.io/turbmodels/nasahump_val_sa.html",
    "experimental_cp": ("https://tmbwg.github.io/turbmodels/Nasahump_validation/noflow_cp.exp.dat"),
    "experimental_cf": ("https://tmbwg.github.io/turbmodels/Nasahump_validation/noflow_cf.exp.dat"),
    "cfl3d_sst_cp": (
        "https://tmbwg.github.io/turbmodels/Nasahump_validation/nasahump_cfl3d_cp_noplenum_sst.dat"
    ),
    "cfl3d_sst_cf": (
        "https://tmbwg.github.io/turbmodels/Nasahump_validation/nasahump_cfl3d_cf_noplenum_sst.dat"
    ),
    "cfl3d_sa_cp": (
        "https://tmbwg.github.io/turbmodels/Nasahump_validation/nasahump_cfl3d_cp_noplenum_sa.dat"
    ),
    "cfl3d_sa_cf": (
        "https://tmbwg.github.io/turbmodels/Nasahump_validation/nasahump_cfl3d_cf_noplenum_sa.dat"
    ),
    "grid_nmf_103x28": (
        "https://tmbwg.github.io/turbmodels/Nasahump_grids/hump2newtop_noplenumZ103x28.nmf"
    ),
    "mapbc": (
        "https://tmbwg.github.io/turbmodels/Nasahump_grids/hump2newtop_noplenumZ_all.hex.mapbc"
    ),
}


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, *, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def _download_grid_from_zip(path: Path, *, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    response = requests.get(GRID_ZIP_URL, timeout=90)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(GRID_MEMBER))


def fetch_inputs(raw_dir: Path, *, refresh: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, url in SOURCES.items():
        suffix = ".html" if name.startswith("tmr_") else Path(url).suffix
        if suffix == "":
            suffix = ".dat"
        path = raw_dir / f"{name}{suffix}"
        _download(url, path, refresh=refresh)
        paths[name] = path
    grid_path = raw_dir / "hump2newtop_noplenumZ103x28.p2dfmt.gz"
    _download_grid_from_zip(grid_path, refresh=refresh)
    paths["grid_p2d_103x28"] = grid_path
    return paths


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
    x = curve.x
    y = curve.y
    for idx in range(len(y) - 1):
        y0 = float(y[idx])
        y1 = float(y[idx + 1])
        if y0 == 0:
            crossings.append(float(x[idx]))
        elif y0 * y1 < 0:
            fraction = abs(y0) / (abs(y0) + abs(y1))
            crossings.append(float(x[idx] + fraction * (x[idx + 1] - x[idx])))
    return crossings


def grid_summary(path: Path) -> dict[str, Any]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    values = [float(token) for token in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", text)]
    block_count = int(values[0])
    i_dim = int(values[1])
    j_dim = int(values[2])
    point_count = i_dim * j_dim
    x = np.asarray(values[3 : 3 + point_count], dtype=np.float64).reshape(j_dim, i_dim)
    y = np.asarray(values[3 + point_count : 3 + 2 * point_count], dtype=np.float64).reshape(
        j_dim,
        i_dim,
    )
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    streamwise_spacing = np.sqrt(dx**2 + dy**2)
    dxn = np.diff(x, axis=0)
    dyn = np.diff(y, axis=0)
    wall_normal_spacing = np.sqrt(dxn**2 + dyn**2)
    return {
        "block_count": block_count,
        "i_dim": i_dim,
        "j_dim": j_dim,
        "expected_2d_cells": int((i_dim - 1) * (j_dim - 1)),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "minimum_streamwise_spacing": float(np.min(streamwise_spacing)),
        "minimum_wall_normal_spacing": float(np.min(wall_normal_spacing)),
    }


def parse_checkmesh_log(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    failed = re.search(r"Failed\s+(\d+)\s+mesh checks", text)
    cells = re.search(r"cells:\s+(\d+)", text)
    patches = re.search(r"boundary patches:\s+(\d+)", text)
    determinant = re.search(r"Cells with small determinant .* number of cells:\s+(\d+)", text)
    aspect = re.search(r"High aspect ratio cells found, Max aspect ratio:\s+([0-9.Ee+-]+)", text)
    return {
        "log_path": str(path.relative_to(ROOT)),
        "failed_mesh_checks": int(failed.group(1)) if failed else None,
        "cells": int(cells.group(1)) if cells else None,
        "boundary_patches": int(patches.group(1)) if patches else None,
        "small_determinant_cells": int(determinant.group(1)) if determinant else 0,
        "max_aspect_ratio": float(aspect.group(1)) if aspect else None,
        "mesh_ok": "Mesh OK" in text and "Failed" not in text,
    }


def build_payload(paths: dict[str, Path]) -> dict[str, Any]:
    cp_exp = parse_curve(paths["experimental_cp"])
    cf_exp = parse_curve(paths["experimental_cf"])
    cp_sst = parse_curve(paths["cfl3d_sst_cp"])
    cf_sst = parse_curve(paths["cfl3d_sst_cf"])
    cp_sa = parse_curve(paths["cfl3d_sa_cp"])
    cf_sa = parse_curve(paths["cfl3d_sa_cf"])
    source_files = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in sorted(paths.items())
    }
    smoke_dir = ROOT / "artifacts/methodology/nasa_hump/smoke_case_p2d_noblank"
    checkmesh = parse_checkmesh_log(smoke_dir / "checkMesh_split.log") or RECORDED_103X28_CHECKMESH
    mesh_policy = methodology_mesh_policy(checkmesh)
    return {
        "schema_version": "nasa_hump_methodology_preflight_v0.1.0",
        "classification": CLASSIFICATION,
        "selected_case": "NASA/TMR 2D wall-mounted hump, no-flow-control, no-plenum grid",
        "purpose": (
            "CFD methodology preflight: public separated-flow reference, turbulence-model "
            "correlation targets, OpenFOAM ingest feasibility and mesh-gate policy."
        ),
        "case_context": {
            "nominal_dimensionality": "2D validation treatment",
            "reynolds_number": 936000,
            "validation_focus": [
                "smooth-body separation",
                "reattachment",
                "boundary-layer recovery",
            ],
            "upper_boundary": "contoured/slip-style tunnel top from NASA/TMR setup",
        },
        "source_urls": SOURCES | {"grid_zip": GRID_ZIP_URL},
        "source_files": source_files,
        "source_file_retention": (
            "Raw NASA/TMR downloads live under ignored artifacts/methodology when the "
            "preflight script is run. The committed artifact stores source URLs and hashes."
        ),
        "reference_curve_summary": {
            "experimental_cp_points": int(cp_exp.x.size),
            "experimental_cf_points": int(cf_exp.x.size),
            "experimental_cf_zero_crossings": zero_crossings(cf_exp),
            "cfl3d_sa_cf_zero_crossings": zero_crossings(cf_sa),
            "cfl3d_sst_cf_zero_crossings": zero_crossings(cf_sst),
        },
        "published_reference_correlation": {
            "note": (
                "These compare NASA/TMR-published CFL3D SA/SST curves against the "
                "experimental data. They validate the metric pipeline; they are not "
                "AeroMap/OpenFOAM results."
            ),
            "sa": {
                "cp": curve_error(cp_exp, cp_sa),
                "cf": curve_error(cf_exp, cf_sa),
            },
            "sst": {
                "cp": curve_error(cp_exp, cp_sst),
                "cf": curve_error(cf_exp, cf_sst),
            },
        },
        "grid_summary": grid_summary(paths["grid_p2d_103x28"]),
        "openfoam_ingest_smoke": {
            "plot3d_to_foam_command": (
                "plot3dToFoam -noBlank -2D 0.1 hump2newtop_noplenumZ103x28.p2dfmt"
            ),
            "patch_split_status": "prototype_successful",
            "patches": PATCH_CONTRACT,
            "strict_mesh_gate": checkmesh,
            "solver_run": False,
            "solver_blocker": (
                "The global AeroMap mesh gate does not pass because the tiny official "
                "boundary-layer grid reports high-aspect-ratio and low-determinant "
                "warnings. The NASA/TMR methodology gate permits smoke work with an "
                "explicit warning, but not correlation claims."
            ),
        },
        "mesh_policy": mesh_policy,
        "correlation_eligibility": correlation_eligibility(),
        "claim_boundary": {
            "openfoam_correlation_result": False,
            "turbulence_model_recommendation": False,
            "solver_ran": False,
            "reference_data_ingestion": True,
            "openfoam_grid_ingest_feasible": True,
        },
        "next_step": (
            "Use scripts/convert_tmr_nasa_hump_to_openfoam.py to materialise a local "
            "conversion scaffold. The follow-up SST smoke artifact records the bounded "
            "single-grid OpenFOAM run; medium/fine grid correlation remains future work."
        ),
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    corr = payload["published_reference_correlation"]
    check = payload["openfoam_ingest_smoke"]["strict_mesh_gate"] or {}
    policy = payload["mesh_policy"]
    eligibility = payload["correlation_eligibility"]
    lines = [
        "# NASA Hump CFD Methodology Preflight v0.1",
        "",
        "## Question",
        "",
        (
            "Can AeroMap add a compact CFD-methodology slice around a recognised "
            "separated-flow validation case without turning the public repo into a "
            "large CFD validation framework?"
        ),
        "",
        "## Current Answer",
        "",
        (
            "Yes for preflight. The NASA/TMR wall-mounted hump gives a compact "
            "separated-flow validation target with experimental data and published "
            "SA/SST reference curves. The small no-plenum grid can enter the OpenFOAM "
            "workflow, but it is not correlation-eligible yet."
        ),
        "",
        "## Case",
        "",
        "- Source: NASA/TMR 2D wall-mounted hump, no flow control.",
        "- Validation focus: smooth-body separation, reattachment and recovery.",
        "- Reynolds number: `936000`.",
        "- Reference data: experimental `Cp` and `Cf` curves.",
        "- Published CFD comparison data: CFL3D SA and SST curves on the no-plenum grid.",
        "- Small grid inspected: `hump2newtop_noplenumZ103x28.p2dfmt.gz`.",
        "",
        "## Reference Correlation Metrics",
        "",
        (
            "These numbers compare NASA/TMR-published CFL3D curves against the "
            "experimental curves. They validate the correlation-metric plumbing; they "
            "are not AeroMap/OpenFOAM results."
        ),
        "",
        "| Published curve | Cp RMSE | Cp MAE | Cf RMSE | Cf MAE |",
        "|---|---:|---:|---:|---:|",
        (
            f"| CFL3D SA | {corr['sa']['cp']['rmse']:.5f} | "
            f"{corr['sa']['cp']['mae']:.5f} | {corr['sa']['cf']['rmse']:.6f} | "
            f"{corr['sa']['cf']['mae']:.6f} |"
        ),
        (
            f"| CFL3D SST | {corr['sst']['cp']['rmse']:.5f} | "
            f"{corr['sst']['cp']['mae']:.5f} | {corr['sst']['cf']['rmse']:.6f} | "
            f"{corr['sst']['cf']['mae']:.6f} |"
        ),
        "",
        "## OpenFOAM Ingest Smoke",
        "",
        "- `plot3dToFoam -noBlank -2D 0.1` reads the `103 x 28` grid correctly.",
        "- Prototype patch split produced `front/back/inlet/outlet/top_slip/hump_wall`.",
        f"- Converted mesh cells: `{check.get('cells')}`.",
        f"- Boundary patches after split: `{check.get('boundary_patches')}`.",
        f"- Failed mesh checks: `{check.get('failed_mesh_checks')}`.",
        f"- Max aspect ratio: `{check.get('max_aspect_ratio')}`.",
        f"- Small-determinant cells: `{check.get('small_determinant_cells')}`.",
        f"- Methodology gate: `{policy['mesh_quality']}`.",
        "",
        "## Mesh Policy",
        "",
        (
            "Do not weaken the global AeroMap mesh gate. The NASA/TMR hump uses a "
            "separate methodology gate for official boundary-layer validation grids. "
            "The `103 x 28` grid is accepted only for conversion, boundary-condition "
            "and single-solver smoke work with explicit warnings. It is not accepted "
            "for headline correlation or turbulence-model recommendation."
        ),
        "",
        "| Gate result | Status |",
        "|---|---|",
        f"| Global AeroMap gate | `{policy['global_aeromap_gate_passed']}` |",
        f"| NASA/TMR methodology gate | `{policy['methodology_gate_passed']}` |",
        f"| Mesh quality class | `{policy['mesh_quality']}` |",
        "",
        "## Correlation Eligibility",
        "",
        "| Requirement | Status |",
        "|---|---|",
        *[f"| `{row['requirement']}` | {row['status']} |" for row in eligibility],
        "",
        "## Next Step",
        "",
        (
            "The follow-up SST smoke and Cp/Cf extraction artifacts show that the "
            "OpenFOAM wall-field overlay pipeline works. The 409 x 109 SST candidate "
            "now tests the next methodology question and is not correlation-plausible "
            "yet, so a model recommendation should wait for boundary-condition, grid "
            "or numerics improvements."
        ),
        "",
        "## Claim Boundary",
        "",
        (
            "- Established: NASA/TMR reference ingestion, correlation metrics and "
            "OpenFOAM grid-ingest feasibility."
        ),
        (
            "- Not established: OpenFOAM hump correlation, SA/SST recommendation, "
            "production CFD accuracy."
        ),
        "",
        "## Evidence",
        "",
        f"- JSON: `{EVIDENCE_PATH.relative_to(ROOT)}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=EVIDENCE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    paths = fetch_inputs(args.raw_dir, refresh=args.refresh)
    payload = build_payload(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, args.report)
    print(json.dumps({"evidence": str(args.out), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()

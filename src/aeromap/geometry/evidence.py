"""Bounded geometry evidence generation."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

import pyvista as pv

from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.geometry.generator import generate_geometry
from aeromap.geometry.schema import GeometryArtifacts
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams, corner_params, sobol_params

IMAGE_SIZE = [1400, 900]
OVERVIEW_CAMERA = [(2.7, -2.0, 1.0), (1.0, 0.0, 0.10), (0.0, 0.0, 1.0)]
CUTAWAY_CAMERA = [(2.4, -1.5, 0.8), (1.0, 0.0, 0.09), (0.0, 0.0, 1.0)]
UNDERSIDE_CAMERA = [(1.0, -3.0, -0.55), (1.0, 0.0, 0.06), (0.0, 0.0, 1.0)]
EVIDENCE_MARKERS = ("manifest.json", "geometry_smoke_summary.json")


def _case_record(
    label: str, kind: str, params: AeroParams, artifacts: GeometryArtifacts
) -> dict[str, Any]:
    metrics = artifacts.validation.metrics
    return {
        "label": label,
        "kind": kind,
        "case_id": artifacts.case_id,
        "geometry_family_id": params.geometry_family_id(),
        "params": params.model_dump(),
        "valid": artifacts.validation.valid,
        "reasons": list(artifacts.validation.reasons),
        "metrics": None
        if metrics is None
        else {
            "vertex_count": metrics.vertex_count,
            "face_count": metrics.face_count,
            "min_ground_clearance_m": metrics.min_ground_clearance_m,
            "throat_x_m": metrics.throat_x_m,
            "diffuser_exit_roof_height_m": metrics.diffuser_exit_roof_height_m,
            "surface_area_m2": metrics.surface_area_m2,
            "volume_m3": metrics.volume_m3,
        },
    }


def _copy_canonical_files(artifacts: GeometryArtifacts, target_dir: Path) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "article_body_datum.step": artifacts.step_path,
        "article.stl": artifacts.stl_path,
        "params.yaml": artifacts.params_yaml_path,
        "params.json": artifacts.params_path,
        "hashes.json": artifacts.hashes_path,
        "validation.json": artifacts.validation_path,
        "geometry_metrics.json": artifacts.metrics_path,
        "surface_regions.json": artifacts.regions_json_path,
        "surface_regions.vtp": artifacts.regions_vtp_path,
    }
    for name, source in copies.items():
        shutil.copy2(source, target_dir / name)
    return {name: sha256_file(target_dir / name) for name in copies}


def _render_image(mesh: pv.PolyData, path: Path, camera: list[tuple[float, float, float]]) -> None:
    plotter: Any = pv.Plotter(off_screen=True, window_size=IMAGE_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="#8fb7c9", smooth_shading=True, specular=0.25)
    plotter.add_axes(line_width=2)
    plotter.camera_position = camera
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(path), auto_close=True)


def _render_canonical_images(stl_path: Path, images_dir: Path) -> dict[str, str]:
    mesh = pv.read(stl_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    cutaway = mesh.clip(normal=(0.0, -1.0, 0.0), origin=(0.0, 0.0, 0.0))
    images = {
        "overview.png": (mesh, OVERVIEW_CAMERA),
        "cutaway_y0.png": (cutaway, CUTAWAY_CAMERA),
        "underside.png": (mesh, UNDERSIDE_CAMERA),
    }
    hashes: dict[str, str] = {}
    for name, (image_mesh, camera) in images.items():
        path = images_dir / name
        _render_image(image_mesh, path, camera)
        hashes[name] = sha256_file(path)
    return hashes


def _write_evidence_readme(out: Path, summary: dict[str, Any]) -> None:
    content = f"""# AeroCliff Geometry Evidence

This directory contains the compact committed geometry evidence for the original AeroCliff
CadQuery article.

- Canonical case: `{summary["canonical_case_id"]}`
- Generator version: `{GEOMETRY_GENERATOR_VERSION}`
- Valid geometries in bounded smoke: `{summary["valid_count"]}/{summary["geometries_generated"]}`
- Surface-region labels: diffuser, tunnel roofs, underfloor, keel, floor edges, upper body

![Canonical overview](canonical/images/overview.png)

![Longitudinal cutaway](canonical/images/cutaway_y0.png)

![Underside](canonical/images/underside.png)
"""
    atomic_write_text(out / "README.md", content)


def _prepare_staging_dir(out: Path) -> Path:
    resolved = out.resolve()
    unsafe_targets = {Path.cwd().resolve(), Path.home().resolve(), Path("/").resolve()}
    if resolved in unsafe_targets:
        message = f"refusing to write geometry evidence to unsafe directory: {out}"
        raise ValueError(message)
    if out.exists() and not out.is_dir():
        message = f"geometry evidence output must be a directory: {out}"
        raise ValueError(message)
    if (
        out.exists()
        and any(out.iterdir())
        and not any((out / marker).exists() for marker in EVIDENCE_MARKERS)
    ):
        message = f"refusing to overwrite non-evidence directory: {out}"
        raise ValueError(message)

    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{out.name}.staging.", dir=out.parent))


def _replace_output_dir(staging: Path, out: Path) -> None:
    backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.previous.", dir=out.parent))
    backup.rmdir()
    if out.exists():
        out.rename(backup)
    try:
        staging.rename(out)
    except OSError:
        if out.exists():
            shutil.rmtree(out)
        if backup.exists():
            backup.rename(out)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _progress(message: str, stream: TextIO) -> None:
    print(message, file=stream, flush=True)


def build_geometry_evidence(
    out: Path,
    *,
    samples: int = 100,
    seed: int = 1729,
    progress_stream: TextIO | None = sys.stderr,
) -> dict[str, Any]:
    """Build compact committed geometry evidence without retaining full smoke artifacts."""

    if samples <= 0:
        message = "samples must be positive"
        raise ValueError(message)

    staging = _prepare_staging_dir(out)

    records: list[dict[str, Any]] = []
    canonical_hashes: dict[str, str]
    image_hashes: dict[str, str]
    with tempfile.TemporaryDirectory(prefix="aerocliff_geometry_evidence_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        canonical_params = AeroParams.canonical()
        if progress_stream is not None:
            _progress("geometry evidence: generating canonical", progress_stream)
        canonical = generate_geometry(canonical_params, tmp_root)
        canonical_hashes = _copy_canonical_files(canonical, staging / "canonical")
        image_hashes = _render_canonical_images(
            staging / "canonical" / "article.stl", staging / "canonical" / "images"
        )
        records.append(_case_record("canonical", "canonical", canonical_params, canonical))
        if progress_stream is not None:
            _progress(
                f"geometry evidence: canonical {canonical.case_id} "
                f"valid={canonical.validation.valid}",
                progress_stream,
            )

        for label, params in corner_params().items():
            if progress_stream is not None:
                _progress(f"geometry evidence: generating corner {label}", progress_stream)
            artifacts = generate_geometry(params, tmp_root)
            records.append(_case_record(label, "corner", params, artifacts))
            shutil.rmtree(tmp_root / artifacts.case_id, ignore_errors=True)
            if progress_stream is not None:
                _progress(
                    f"geometry evidence: corner {label} {artifacts.case_id} "
                    f"valid={artifacts.validation.valid}",
                    progress_stream,
                )

        for index, params in enumerate(sobol_params(samples, seed=seed)):
            if progress_stream is not None:
                _progress(
                    f"geometry evidence: generating sobol {index + 1}/{samples}",
                    progress_stream,
                )
            artifacts = generate_geometry(params, tmp_root)
            records.append(_case_record(f"sobol_{index:03d}", "sobol", params, artifacts))
            shutil.rmtree(tmp_root / artifacts.case_id, ignore_errors=True)
            if progress_stream is not None:
                _progress(
                    f"geometry evidence: sobol {index + 1}/{samples} {artifacts.case_id} "
                    f"valid={artifacts.validation.valid}",
                    progress_stream,
                )

    valid_count = sum(record["valid"] for record in records)
    summary: dict[str, Any] = {
        "samples_requested": samples,
        "seed": seed,
        "corner_cases_generated": len(corner_params()),
        "geometries_generated": len(records),
        "valid_count": valid_count,
        "valid_fraction": valid_count / len(records),
        "canonical_case_id": records[0]["case_id"],
        "canonical_hashes": canonical_hashes,
        "canonical_image_hashes": image_hashes,
        "invalid": [
            {"label": record["label"], "case_id": record["case_id"], "reasons": record["reasons"]}
            for record in records
            if not record["valid"]
        ],
        "cases": records,
    }
    atomic_write_json(staging / "geometry_smoke_summary.json", summary)
    _write_evidence_readme(staging, summary)
    atomic_write_json(staging / "manifest.json", summary)
    _replace_output_dir(staging, out)
    return summary

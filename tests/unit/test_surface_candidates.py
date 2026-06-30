from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from aeromap.geometry.surface_candidates import (
    GMSH_SURFACE_VARIANTS,
    _candidate_name,
    _gmsh_executable,
    _load_exported_mesh,
    _quality_histogram,
    _surface_metrics,
    _write_candidate_region_sidecars,
    _write_gmsh_candidate,
    surface_candidate_id,
)
from aeromap.parameters import AeroParams


def test_surface_candidate_id_is_stable_for_payload_order() -> None:
    first = surface_candidate_id({"kind": "cadquery_fixed_tolerance", "linear": 0.001})
    second = surface_candidate_id({"linear": 0.001, "kind": "cadquery_fixed_tolerance"})
    changed = surface_candidate_id({"kind": "cadquery_fixed_tolerance", "linear": 0.002})

    assert first == second
    assert changed != first
    assert first.startswith("surface_candidate_")


def test_gmsh_variants_are_bounded_g0_g1_only() -> None:
    variants = {item["variant"]: item for item in GMSH_SURFACE_VARIANTS}

    assert set(variants) == {"g0_no_healing", "g1_conservative_autofix"}
    assert variants["g0_no_healing"]["occ_options"]["Geometry.OCCFixSmallEdges"] == 0
    assert variants["g0_no_healing"]["occ_options"]["Geometry.OCCSewFaces"] == 0
    assert variants["g1_conservative_autofix"]["occ_options"]["Geometry.OCCAutoFix"] == 1
    assert variants["g1_conservative_autofix"]["occ_options"]["Geometry.OCCFixSmallFaces"] == 0


def test_gmsh_candidate_name_includes_variant() -> None:
    name = _candidate_name({"kind": "gmsh_occ_surface_remesh", "variant": "g0_no_healing"})

    assert name == "gmsh_occ_g0_no_healing"


def test_explicit_gmsh_path_must_be_executable_file(tmp_path: Path) -> None:
    regular_file = tmp_path / "gmsh"
    regular_file.write_text("#!/bin/sh\n", encoding="utf-8")

    assert _gmsh_executable(tmp_path) is None
    assert _gmsh_executable(regular_file) is None
    regular_file.chmod(0o755)
    assert _gmsh_executable(regular_file) == regular_file


def test_gmsh_candidate_records_subprocess_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> object:
        raise OSError("cannot execute gmsh")

    monkeypatch.setattr("aeromap.geometry.surface_candidates.subprocess.run", fail_run)
    gmsh = tmp_path / "gmsh"
    gmsh.write_text("#!/bin/sh\n", encoding="utf-8")
    step_path = tmp_path / "article.step"
    step_path.write_text("STEP", encoding="utf-8")
    reference_mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.1))
    payload = {
        "kind": "gmsh_occ_surface_remesh",
        "variant": "g0_no_healing",
        "mesh_size_min_m": 0.001,
        "mesh_size_max_m": 0.012,
        "mesh_optimize": 1,
        "mesh_algorithm": "front2d",
        "occ_options": {},
    }

    result = _write_gmsh_candidate(
        gmsh_path=gmsh,
        step_path=step_path,
        params=AeroParams.canonical(),
        out_dir=tmp_path,
        payload=payload,
        cad_sample_centroids=np.asarray(reference_mesh.triangles_center, dtype=np.float64),
        reference_mesh=reference_mesh,
    )

    assert result.status == "GMSH_FAILED"
    assert result.stl_path is None
    assert "cannot execute gmsh" in result.metrics_path.read_text(encoding="utf-8")


def test_quality_histogram_counts_all_triangles() -> None:
    histogram = _quality_histogram(np.array([0.0, 5e-7, 0.02, 0.08, 0.9]))

    assert sum(bin_["count"] for bin_ in histogram) == 5
    assert histogram[0]["count"] == 2
    assert histogram[-1]["count"] == 1
    assert histogram[-1]["upper"] == pytest.approx(1.01)


def test_surface_metrics_report_raw_cleaned_delta(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.1))
    duplicated = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=np.vstack([mesh.faces, [0, 0, 0]]),
        process=False,
    )
    stl_path = tmp_path / "duplicate.stl"
    duplicated.export(stl_path)
    cleaned = _load_exported_mesh(stl_path)

    metrics = _surface_metrics(
        mesh=cleaned,
        body_mesh=cleaned,
        stl_path=stl_path,
        params=AeroParams.canonical(),
        cad_sample_centroids=np.asarray(cleaned.triangles_center, dtype=np.float64),
        reference_mesh=cleaned,
    )

    assert metrics["raw_export_metrics"]["face_count"] > metrics["face_count"]
    assert metrics["cleaned_mesh_delta"]["removed_faces"] > 0


def test_candidate_region_sidecars_match_candidate_faces(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.1))

    artifacts = _write_candidate_region_sidecars(
        mesh=mesh,
        classification_mesh=mesh,
        params=AeroParams.canonical(),
        candidate_dir=tmp_path,
    )

    regions_json = tmp_path / "surface_regions.json"
    regions_vtp = tmp_path / "surface_regions.vtp"
    assert artifacts["surface_regions_json_path"] == str(regions_json)
    assert artifacts["surface_regions_vtp_path"] == str(regions_vtp)
    assert regions_json.exists()
    assert regions_vtp.exists()

    payload = json.loads(regions_json.read_text(encoding="utf-8"))
    face_regions = payload["face_regions"]
    face_indices = [item["face_index"] for item in face_regions]
    per_region_counts: dict[str, int] = {}
    for item in face_regions:
        region = str(item["region"])
        per_region_counts[region] = per_region_counts.get(region, 0) + 1
    assert len(face_regions) == len(mesh.faces)
    assert len(set(face_indices)) == len(mesh.faces)
    assert sorted(face_indices) == list(range(len(mesh.faces)))
    assert all("region" in item for item in face_regions)
    assert sum(per_region_counts.values()) == len(mesh.faces)

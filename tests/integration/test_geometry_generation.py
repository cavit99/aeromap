from __future__ import annotations

import json
from pathlib import Path

import pyvista as pv
import yaml

from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.geometry.generator import generate_geometry
from aeromap.geometry.validate import validate_stl
from aeromap.io import sha256_file
from aeromap.parameters import AeroParams


def test_canonical_geometry_generates_valid_artifacts(tmp_path: Path) -> None:
    artifacts = generate_geometry(AeroParams.canonical(), tmp_path)
    assert artifacts.validation.valid, artifacts.validation.reasons
    assert artifacts.step_path.exists()
    assert artifacts.stl_path.exists()
    assert artifacts.params_yaml_path.exists()
    assert artifacts.hashes_path.exists()
    assert artifacts.regions_json_path.exists()
    assert artifacts.regions_vtp_path.exists()
    assert artifacts.preview_glb_path.exists()
    assert artifacts.preview_html_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.step_path.stat().st_size > 0
    assert artifacts.regions_vtp_path.stat().st_size > 0
    assert artifacts.preview_glb_path.stat().st_size > 0
    assert "<html" in artifacts.preview_html_path.read_text(encoding="utf-8").lower()

    validation = validate_stl(artifacts.stl_path)
    assert validation.valid, validation.reasons
    assert validation.metrics is not None
    assert validation.metrics.min_ground_clearance_m >= 0.015
    assert validation.metrics.generator_version == GEOMETRY_GENERATOR_VERSION

    params_yaml = yaml.safe_load(artifacts.params_yaml_path.read_text(encoding="utf-8"))
    assert params_yaml["diffuser_angle_deg"] == AeroParams.canonical().diffuser_angle_deg

    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["generator_version"] == GEOMETRY_GENERATOR_VERSION

    hashes = json.loads(artifacts.hashes_path.read_text(encoding="utf-8"))
    assert set(hashes) == {
        "article_body_datum.step",
        "article.stl",
        "geometry_metrics.json",
        "params.json",
        "params.yaml",
        "surface_regions.json",
        "surface_regions.vtp",
        "validation.json",
    }
    for filename, digest in hashes.items():
        path = artifacts.stl_path.parent / filename
        if filename in {"params.json", "params.yaml"}:
            path = artifacts.stl_path.parent.parent / filename
        assert sha256_file(path) == digest

    regions = json.loads(artifacts.regions_json_path.read_text(encoding="utf-8"))
    assert regions["classification_frame"] == "body_local"
    region_vtp = pv.read(artifacts.regions_vtp_path)
    assert "region_id" in region_vtp.cell_data

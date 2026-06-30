from __future__ import annotations

import json
from pathlib import Path

from aeromap.attempts import stable_id, write_attempt_manifest
from aeromap.io import sha256_file


def test_stable_id_is_deterministic_and_order_independent() -> None:
    first = stable_id("attempt", {"b": 2, "a": 1})
    second = stable_id("attempt", {"a": 1, "b": 2})

    assert first == second
    assert first.startswith("attempt_")


def test_attempt_manifest_hashes_existing_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "surfaceCheck.log"
    artifact.write_text("surface ok\n", encoding="utf-8")
    missing = tmp_path / "missing.log"

    manifest_path = write_attempt_manifest(
        attempt_dir=tmp_path,
        attempt_id="attempt_demo",
        geometry_id="geometry_demo",
        surface_export_id="surface_export_demo",
        mesh_config={"surface_level": [1, 2]},
        openfoam_image_digest="image@sha256:demo",
        parent_attempt_id=None,
        configuration_diff={"probe": "current"},
        status="FAILED_CHECKMESH",
        artifacts={"surface_check_log": artifact, "missing": missing},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["attempt_id"] == "attempt_demo"
    assert manifest["artifacts"]["surface_check_log"] == str(artifact)
    assert manifest["artifact_hashes"] == {"surface_check_log": sha256_file(artifact)}

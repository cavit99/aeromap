"""Immutable attempt identifiers and manifests for CFD retry evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aeromap.io import atomic_write_json, sha256_file

ATTEMPT_SCHEMA_VERSION = "attempt_manifest_v0.1.0"


def stable_id(prefix: str, payload: dict[str, Any], *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:length]}"


def existing_file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items()) if path.exists()}


def write_attempt_manifest(
    *,
    attempt_dir: Path,
    attempt_id: str,
    geometry_id: str,
    surface_export_id: str,
    mesh_config: dict[str, Any],
    openfoam_image_digest: str,
    parent_attempt_id: str | None,
    configuration_diff: dict[str, Any],
    status: str,
    artifacts: dict[str, Path],
    metrics_path: Path | None = None,
) -> Path:
    manifest_path = attempt_dir / "attempt_manifest.json"
    payload: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "geometry_id": geometry_id,
        "surface_export_id": surface_export_id,
        "mesh_config": mesh_config,
        "openfoam_image_digest": openfoam_image_digest,
        "parent_attempt_id": parent_attempt_id,
        "configuration_diff": configuration_diff,
        "status": status,
        "artifacts": {name: str(path) for name, path in sorted(artifacts.items())},
        "artifact_hashes": existing_file_hashes(artifacts),
        "metrics_json": str(metrics_path) if metrics_path is not None else None,
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path

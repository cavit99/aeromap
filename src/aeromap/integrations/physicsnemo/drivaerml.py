"""External DrivAerML smoke asset preparation."""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

from aeromap.io import atomic_write_json, sha256_file

DRIVAERML_STL_URL = (
    "https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/drivaer_1.stl"
)
DRIVAERML_LICENSE = "CC BY-SA 4.0"
MIN_STL_BYTES = 84


def _valid_stl_smoke(path: Path) -> bool:
    """Return true for a minimally plausible STL file.

    This is intentionally a lightweight install-smoke guard, not a geometry-quality check.
    Binary STL requires at least an 80-byte header plus a 4-byte triangle count.
    """

    return path.is_file() and path.stat().st_size >= MIN_STL_BYTES


def write_drivaerml_external_manifest(
    out_dir: Path,
    *,
    download: bool = False,
    stl_url: str = DRIVAERML_STL_URL,
) -> Path:
    """Prepare a manifest that keeps DrivAerML external to AeroCliff datasets."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stl_path = out_dir / "drivaer_1_external_cc_by_sa_4_0.stl"
    if download:
        with tempfile.NamedTemporaryFile(
            prefix=f".{stl_path.name}.",
            suffix=".download",
            dir=out_dir,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        try:
            urllib.request.urlretrieve(stl_url, tmp_path)  # noqa: S310
            if not _valid_stl_smoke(tmp_path):
                msg = "downloaded DrivAerML STL is missing or too small"
                raise ValueError(msg)
            tmp_path.replace(stl_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    manifest = {
        "schema": "aerocliff_external_drivaerml_smoke_v0.1.0",
        "evidence_provenance": {
            "producer": (
                "aeromap.integrations.physicsnemo.drivaerml.write_drivaerml_external_manifest"
            ),
            "synthetic_fixture": False,
        },
        "source": "DrivAerML",
        "source_url": stl_url,
        "license": DRIVAERML_LICENSE,
        "external_geometry": True,
        "aerocliff_geometry": False,
        "permitted_use": "DoMINO installation/connectivity smoke only",
        "prohibited_use": (
            "Do not mix into AeroCliff campaign CFD, training data, benchmark results, "
            "or demo claims."
        ),
        "stl_path": str(stl_path),
        "stl_present": _valid_stl_smoke(stl_path),
        "stl_sha256": sha256_file(stl_path) if _valid_stl_smoke(stl_path) else "",
        "stl_validation": "min_84_bytes" if _valid_stl_smoke(stl_path) else "missing_or_too_small",
    }
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path

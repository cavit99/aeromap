"""Optional DrivAerML real-asset integration smoke.

This test is skipped unless explicitly enabled because it downloads selected
external benchmark files from Hugging Face.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aeromap.benchmarks.drivaerml import (
    DrivAerMLBenchmarkConfig,
    validate_drivaerml_asset_manifest,
    write_drivaerml_asset_manifest,
    write_drivaerml_benchmark_plan,
    write_drivaerml_sample_manifest,
)


@pytest.mark.skipif(
    os.environ.get("AEROMAP_RUN_DRIVAERML_NETWORK_SMOKE") != "1",
    reason="real DrivAerML download smoke is opt-in",
)
def test_drivaerml_two_case_real_asset_smoke(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    asset_manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=tmp_path / "cache",
        splits=("initial_labelled",),
        max_cases=2,
        dry_run=False,
    )
    validation = validate_drivaerml_asset_manifest(asset_manifest_path)
    assert validation["ok"] is True
    assert validation["data_ready"] is True

    sample_manifest_path = write_drivaerml_sample_manifest(
        asset_manifest_path,
        tmp_path / "samples.json",
    )
    assert sample_manifest_path.exists()

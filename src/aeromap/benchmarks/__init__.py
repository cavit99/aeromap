"""External benchmark helpers."""

from __future__ import annotations

from aeromap.benchmarks.aeromap import (
    AeroMapConfig,
    build_aeromap_plan,
    build_airfrans_geometry_dataset,
    build_airfrans_scalar_dataset,
    extract_airfrans_archive,
    write_active_learning_replay,
    write_aeromap_plan,
    write_airfrans_feasibility,
    write_airfrans_v02_audit,
    write_decision_replay_v02,
    write_decision_replay_v03,
    write_fixture_dataset,
    write_model_baselines_v02,
)
from aeromap.benchmarks.aeromap3d import (
    build_drivaerml_scalar_bridge_dataset,
    build_drivaerml_scalar_bridge_dataset_from_paths,
    write_aeromap3d_metadata_triage,
    write_geometry_readiness_sample,
)
from aeromap.benchmarks.drivaerml import (
    DrivAerMLBenchmarkConfig,
    DrivAerMLSplitConfig,
    write_drivaerml_asset_manifest,
    write_drivaerml_benchmark_plan,
    write_drivaerml_cuda_bundle,
    write_drivaerml_sample_manifest,
    write_drivaerml_sampling_manifest,
)

__all__ = [
    "AeroMapConfig",
    "DrivAerMLBenchmarkConfig",
    "DrivAerMLSplitConfig",
    "build_aeromap_plan",
    "build_airfrans_geometry_dataset",
    "build_airfrans_scalar_dataset",
    "build_drivaerml_scalar_bridge_dataset",
    "build_drivaerml_scalar_bridge_dataset_from_paths",
    "extract_airfrans_archive",
    "write_active_learning_replay",
    "write_aeromap3d_metadata_triage",
    "write_aeromap_plan",
    "write_airfrans_feasibility",
    "write_airfrans_v02_audit",
    "write_decision_replay_v02",
    "write_decision_replay_v03",
    "write_drivaerml_asset_manifest",
    "write_drivaerml_benchmark_plan",
    "write_drivaerml_cuda_bundle",
    "write_drivaerml_sample_manifest",
    "write_drivaerml_sampling_manifest",
    "write_fixture_dataset",
    "write_geometry_readiness_sample",
    "write_model_baselines_v02",
]

"""Immutable CFD data conversion and loading."""

from aeromap.data.converter import convert_case_to_sample
from aeromap.data.loader import (
    CampaignBatch,
    CampaignSampleDataset,
    LoadedCampaignSample,
    TrainingEligibilityError,
    VariableBatch,
    aggregate_source_cell_field,
    batch_samples,
    build_campaign_dataloader,
    collate_variable_samples,
    load_sample,
)
from aeromap.data.sampling import DataSamplingConfig, SampleSelection, build_sample_selection
from aeromap.data.schema import DataSample, DataSampleArtifacts, DataSampleManifest
from aeromap.data.splits import (
    GroupedSplitEntry,
    GroupedSplitManifest,
    build_grouped_split_manifest,
)
from aeromap.data.volume import VolumeReductionError
from aeromap.data.vtk_workflow import load_geometry_stl, load_volume_vtu, load_wall_vtp

__all__ = [
    "CampaignBatch",
    "CampaignSampleDataset",
    "DataSample",
    "DataSampleArtifacts",
    "DataSampleManifest",
    "DataSamplingConfig",
    "GroupedSplitEntry",
    "GroupedSplitManifest",
    "LoadedCampaignSample",
    "SampleSelection",
    "TrainingEligibilityError",
    "VariableBatch",
    "VolumeReductionError",
    "aggregate_source_cell_field",
    "batch_samples",
    "build_campaign_dataloader",
    "build_grouped_split_manifest",
    "build_sample_selection",
    "collate_variable_samples",
    "convert_case_to_sample",
    "load_geometry_stl",
    "load_sample",
    "load_volume_vtu",
    "load_wall_vtp",
]

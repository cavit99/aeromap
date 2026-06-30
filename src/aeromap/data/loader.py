"""Device-neutral sample loading with campaign-only training guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

from aeromap.data.sampling import DataSamplingConfig, SampleSelection, build_sample_selection
from aeromap.data.schema import DataSample, DataSampleManifest
from aeromap.data.volume import VolumeReductionError, aggregate_by_cell_id
from aeromap.io import sha256_file


class TrainingEligibilityError(RuntimeError):
    """Raised when a non-campaign sample is loaded for training."""


@dataclass(frozen=True)
class VariableBatch:
    """Variable-size collated samples with explicit offsets and local cellID semantics."""

    sample_ids: tuple[str, ...]
    identifiers: dict[str, tuple[object, ...]]
    arrays: dict[str, np.ndarray]
    offsets: dict[str, np.ndarray]
    semantics: dict[str, str]


@dataclass(frozen=True)
class LoadedCampaignSample:
    """One sample loaded through the production dataset API."""

    sample: DataSample
    selection: SampleSelection | None


@dataclass(frozen=True)
class CampaignBatch:
    """Variable batch plus optional deterministic sampling selections."""

    variable: VariableBatch
    selections: tuple[SampleSelection | None, ...]


def _arrays_path(sample_dir: Path, manifest: DataSampleManifest) -> Path:
    path = Path(manifest.arrays_path)
    return path if path.is_absolute() else sample_dir / path


def load_sample(sample_dir: Path, *, allow_non_campaign: bool = False) -> DataSample:
    """Load one converted sample, rejecting non-campaign samples by default."""

    sample_dir = sample_dir.resolve()
    manifest = DataSampleManifest.model_validate(
        json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8")),
    )
    if not allow_non_campaign and (
        manifest.case_class != "CAMPAIGN_REFERENCE_CFD" or not manifest.training_eligible
    ):
        msg = (
            f"sample {manifest.sample_id} is {manifest.case_class} and is not training eligible; "
            "pass allow_non_campaign=True only for interface tests or demos"
        )
        raise TrainingEligibilityError(msg)
    with np.load(_arrays_path(sample_dir, manifest), allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    missing = sorted(set(manifest.array_names) - set(arrays))
    if missing:
        msg = f"sample {manifest.sample_id} is missing arrays: {missing}"
        raise ValueError(msg)
    if manifest.arrays_sha256:
        digest = sha256_file(_arrays_path(sample_dir, manifest))
        if digest != manifest.arrays_sha256:
            msg = f"sample {manifest.sample_id} arrays sha256 mismatch"
            raise ValueError(msg)
    return DataSample(manifest=manifest, arrays=arrays)


def batch_samples(samples: list[DataSample]) -> dict[str, np.ndarray]:
    """Stack same-shaped sample arrays for tiny correctness and interface tests."""

    if not samples:
        msg = "at least one sample is required"
        raise ValueError(msg)
    keys = set(samples[0].arrays)
    for sample in samples[1:]:
        if set(sample.arrays) != keys:
            msg = "all samples in a batch must expose the same array names"
            raise ValueError(msg)

    batch: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [sample.arrays[key] for sample in samples]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            msg = f"array {key!r} has variable shapes and cannot be stacked"
            raise ValueError(msg)
        batch[key] = np.stack(arrays, axis=0)
    return batch


def aggregate_source_cell_field(
    sample: DataSample,
    array_name: str,
    *,
    reducer: str = "mean",
) -> np.ndarray:
    """Reduce one exported VTU cell field to source OpenFOAM cells through ``cellID``."""

    values = sample.arrays[array_name]
    cell_ids = np.asarray(sample.arrays["volume_cell_id"], dtype=np.int64)
    if values.shape[0] != cell_ids.shape[0]:
        msg = f"{array_name!r} is not a cell-associated volume field"
        raise VolumeReductionError(msg)
    return aggregate_by_cell_id(
        values,
        cell_ids,
        source_cell_count=sample.manifest.volume_provenance.source_openfoam_cell_count,
        reducer=reducer,
    )


def _offsets(lengths: list[int]) -> np.ndarray:
    starts = np.cumsum([0, *lengths[:-1]], dtype=np.int64)
    ends = starts + np.asarray(lengths, dtype=np.int64)
    return np.column_stack([starts, ends])


def _offset_vtk_connectivity(cells: np.ndarray, point_offset: int) -> np.ndarray:
    stream = np.asarray(cells, dtype=np.int64).reshape(-1)
    shifted: list[int] = []
    cursor = 0
    while cursor < len(stream):
        vertex_count = int(stream[cursor])
        if vertex_count <= 0 or cursor + vertex_count >= len(stream):
            msg = "invalid VTK connectivity stream"
            raise ValueError(msg)
        shifted.append(vertex_count)
        shifted.extend((stream[cursor + 1 : cursor + 1 + vertex_count] + point_offset).tolist())
        cursor += vertex_count + 1
    if cursor != len(stream):
        msg = "invalid VTK connectivity stream"
        raise ValueError(msg)
    return np.asarray(shifted, dtype=np.int64)


def _concat(samples: list[DataSample], key: str) -> np.ndarray:
    return np.concatenate([sample.arrays[key] for sample in samples], axis=0)


def _association(sample: DataSample, key: str) -> str:
    value = np.asarray(sample.arrays[key]).reshape(-1)
    if value.shape[0] != 1:
        msg = f"{key!r} must contain one association value"
        raise ValueError(msg)
    return str(value[0])


def _require_same_array_names(samples: list[DataSample]) -> set[str]:
    keys = set(samples[0].arrays)
    for sample in samples[1:]:
        if set(sample.arrays) != keys:
            msg = "all samples in a variable batch must expose the same array names"
            raise ValueError(msg)
    return keys


def collate_variable_samples(samples: list[DataSample]) -> VariableBatch:
    """Collate variable-size AeroCliff samples by concatenating arrays and recording offsets."""

    if not samples:
        msg = "at least one sample is required"
        raise ValueError(msg)
    _require_same_array_names(samples)

    surface_point_lengths = [int(sample.manifest.counts["surface_points"]) for sample in samples]
    surface_face_lengths = [int(sample.manifest.counts["surface_faces"]) for sample in samples]
    volume_point_lengths = [int(sample.manifest.counts["volume_points"]) for sample in samples]
    volume_cell_lengths = [
        int(sample.manifest.volume_provenance.exported_vtu_cell_count) for sample in samples
    ]
    volume_source_lengths = [
        int(sample.manifest.volume_provenance.source_openfoam_cell_count) for sample in samples
    ]
    surface_point_offsets = _offsets(surface_point_lengths)
    surface_face_value_offsets = _offsets(surface_face_lengths)
    surface_face_streams = [
        _offset_vtk_connectivity(sample.arrays["surface_faces"], int(offset[0]))
        for sample, offset in zip(samples, surface_point_offsets, strict=True)
    ]
    surface_face_stream_offsets = _offsets(
        [int(stream.shape[0]) for stream in surface_face_streams],
    )
    volume_point_offsets = _offsets(volume_point_lengths)
    volume_exported_cell_offsets = _offsets(volume_cell_lengths)
    volume_source_cell_offsets = _offsets(volume_source_lengths)

    pressure_associations = {
        _association(sample, "volume_pressure_association") for sample in samples
    }
    velocity_associations = {
        _association(sample, "volume_velocity_association") for sample in samples
    }
    if len(pressure_associations) != 1 or len(velocity_associations) != 1:
        msg = "variable batches require consistent volume field associations"
        raise ValueError(msg)
    pressure_association = next(iter(pressure_associations))
    velocity_association = next(iter(velocity_associations))

    arrays: dict[str, np.ndarray] = {
        "surface_points_m": _concat(samples, "surface_points_m"),
        "surface_points_nd": _concat(samples, "surface_points_nd"),
        "surface_faces": np.concatenate(surface_face_streams),
        "surface_cell_normals": _concat(samples, "surface_cell_normals"),
        "surface_pressure_kinematic": _concat(samples, "surface_pressure_kinematic"),
        "surface_cp": _concat(samples, "surface_cp"),
        "surface_wall_shear_kinematic": _concat(samples, "surface_wall_shear_kinematic"),
        "surface_cf": _concat(samples, "surface_cf"),
        "surface_region_id": _concat(samples, "surface_region_id"),
        "surface_region_name": _concat(samples, "surface_region_name"),
        "surface_local_face_area_m2": _concat(samples, "surface_local_face_area_m2"),
        "volume_points_m": _concat(samples, "volume_points_m"),
        "volume_points_nd": _concat(samples, "volume_points_nd"),
        "volume_cells": np.concatenate(
            [
                _offset_vtk_connectivity(sample.arrays["volume_cells"], int(offset[0]))
                for sample, offset in zip(samples, volume_point_offsets, strict=True)
            ],
        ),
        "volume_celltypes": _concat(samples, "volume_celltypes"),
        "volume_cell_id": _concat(samples, "volume_cell_id"),
        "volume_global_cell_id": np.concatenate(
            [
                np.asarray(sample.arrays["volume_cell_id"], dtype=np.int64) + int(offset[0])
                for sample, offset in zip(samples, volume_source_cell_offsets, strict=True)
            ],
        ),
        "volume_pressure_association": np.asarray([pressure_association]),
        "volume_velocity_association": np.asarray([velocity_association]),
        "surface_point_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            surface_point_lengths,
        ),
        "surface_face_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            surface_face_lengths,
        ),
        "volume_point_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            volume_point_lengths,
        ),
        "volume_exported_cell_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            volume_cell_lengths,
        ),
        "volume_pressure_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            volume_point_lengths if pressure_association == "point" else volume_cell_lengths,
        ),
        "volume_velocity_sample_index": np.repeat(
            np.arange(len(samples), dtype=np.int64),
            volume_point_lengths if velocity_association == "point" else volume_cell_lengths,
        ),
    }
    for key in ("volume_pressure_kinematic", "volume_cp"):
        arrays[key] = _concat(samples, key)
    for key in ("volume_velocity", "volume_velocity_nd"):
        arrays[key] = _concat(samples, key)

    offsets = {
        "surface_points": surface_point_offsets,
        "surface_faces": surface_face_stream_offsets,
        "surface_face_values": surface_face_value_offsets,
        "volume_points": volume_point_offsets,
        "volume_exported_vtu_cells": volume_exported_cell_offsets,
        "volume_source_openfoam_cells": volume_source_cell_offsets,
        "volume_pressure_values": (
            volume_point_offsets
            if pressure_association == "point"
            else volume_exported_cell_offsets
        ),
        "volume_velocity_values": (
            volume_point_offsets
            if velocity_association == "point"
            else volume_exported_cell_offsets
        ),
    }
    semantics = {
        "cellID": (
            "volume_cell_id values remain local to each sample; volume_global_cell_id offsets "
            "them into the concatenated source-cell space."
        ),
        "volume_global_cell_id": "Global source OpenFOAM cell identity within this batch.",
        "volume_cells": "VTK connectivity indices are offset into the concatenated point arrays.",
        "surface_faces": (
            "VTK connectivity indices are offset into the concatenated surface point arrays; "
            "offsets['surface_faces'] slices the connectivity stream."
        ),
        "surface_face_values": (
            "surface_face_values offsets select face-associated arrays such as pressure, "
            "normals, regions, and local area."
        ),
        "volume_field_offsets": (
            "volume_pressure_values and volume_velocity_values select point or exported-cell "
            "offsets according to the recorded field association."
        ),
        "volume_reduction": (
            "Raw exported cells are preserved. Source-cell reductions must be performed per "
            "sample through aggregate_source_cell_field or over volume_global_cell_id for a "
            "collated batch."
        ),
    }
    return VariableBatch(
        sample_ids=tuple(sample.manifest.sample_id for sample in samples),
        identifiers={
            "sample_id": tuple(sample.manifest.sample_id for sample in samples),
            "geometry_id": tuple(sample.manifest.geometry_id for sample in samples),
            "state_id": tuple(sample.manifest.state_id for sample in samples),
            "simulation_id": tuple(sample.manifest.simulation_id for sample in samples),
            "attempt_id": tuple(sample.manifest.attempt_id for sample in samples),
            "case_class": tuple(sample.manifest.case_class for sample in samples),
            "training_eligible": tuple(sample.manifest.training_eligible for sample in samples),
        },
        arrays=arrays,
        offsets=offsets,
        semantics=semantics,
    )


def collate_campaign_samples(items: list[LoadedCampaignSample]) -> CampaignBatch:
    """Collate production dataset items into a variable-size campaign batch."""

    if not items:
        msg = "at least one dataset item is required"
        raise ValueError(msg)
    return CampaignBatch(
        variable=collate_variable_samples([item.sample for item in items]),
        selections=tuple(item.selection for item in items),
    )


class CampaignSampleDataset(Dataset[LoadedCampaignSample]):
    """PyTorch-compatible campaign sample dataset with smoke rejection by default."""

    def __init__(
        self,
        sample_dirs: list[Path] | tuple[Path, ...],
        *,
        sampling: DataSamplingConfig | None = None,
        allow_non_campaign_for_tests: bool = False,
    ) -> None:
        if not sample_dirs:
            msg = "at least one sample directory is required"
            raise ValueError(msg)
        self.sample_dirs = tuple(Path(path) for path in sample_dirs)
        self.sampling = sampling
        self.allow_non_campaign_for_tests = allow_non_campaign_for_tests

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> LoadedCampaignSample:
        sample = load_sample(
            self.sample_dirs[index],
            allow_non_campaign=self.allow_non_campaign_for_tests,
        )
        return LoadedCampaignSample(
            sample=sample,
            selection=build_sample_selection(sample, self.sampling),
        )


def build_campaign_dataloader(
    sample_dirs: list[Path] | tuple[Path, ...],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    sampling: DataSamplingConfig | None = None,
    allow_non_campaign_for_tests: bool = False,
) -> DataLoader[LoadedCampaignSample]:
    """Build a PyTorch ``DataLoader`` for production campaign samples.

    Non-campaign samples are rejected by default. The override name is deliberately
    test-scoped so smoke artifacts are not admitted to production training by accident.
    """

    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if num_workers < 0:
        msg = "num_workers must be non-negative"
        raise ValueError(msg)
    dataset = CampaignSampleDataset(
        sample_dirs,
        sampling=sampling,
        allow_non_campaign_for_tests=allow_non_campaign_for_tests,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_campaign_samples,
        num_workers=num_workers,
    )

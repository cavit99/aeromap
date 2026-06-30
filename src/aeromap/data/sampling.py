"""Deterministic sampling plans for variable-size AeroCliff samples."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from aeromap.data.schema import DataSample


@dataclass(frozen=True)
class DataSamplingConfig:
    """Optional deterministic index budgets for campaign-scale arrays.

    The plan returns indices into immutable sample arrays instead of rewriting the
    converted sample. That keeps provenance and cellID semantics intact while
    allowing training code to consume bounded point, face, and cell subsets.
    """

    seed: int = 0
    surface_point_count: int | None = None
    surface_face_count: int | None = None
    volume_point_count: int | None = None
    volume_exported_cell_count: int | None = None
    volume_source_cell_count: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("surface_point_count", self.surface_point_count),
            ("surface_face_count", self.surface_face_count),
            ("volume_point_count", self.volume_point_count),
            ("volume_exported_cell_count", self.volume_exported_cell_count),
            ("volume_source_cell_count", self.volume_source_cell_count),
        ):
            if value is not None and value <= 0:
                msg = f"{name} must be positive when supplied"
                raise ValueError(msg)
        if (
            self.volume_exported_cell_count is not None
            and self.volume_source_cell_count is not None
        ):
            msg = "sample either exported VTU cells or source OpenFOAM cells, not both"
            raise ValueError(msg)


@dataclass(frozen=True)
class SampleSelection:
    """Deterministic local indices for one immutable sample."""

    sample_id: str
    surface_point_indices: np.ndarray | None
    surface_face_indices: np.ndarray | None
    volume_point_indices: np.ndarray | None
    volume_exported_cell_indices: np.ndarray | None
    volume_source_cell_ids: np.ndarray | None
    semantics: dict[str, str]


def _seed(sample_id: str, user_seed: int, scope: str) -> int:
    digest = hashlib.blake2b(
        f"{sample_id}:{user_seed}:{scope}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _indices(
    length: int, count: int | None, *, sample_id: str, seed: int, scope: str
) -> np.ndarray | None:
    if count is None:
        return None
    if length <= 0:
        msg = f"cannot sample {scope} from an empty array"
        raise ValueError(msg)
    if count >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(_seed(sample_id, seed, scope))
    return np.sort(rng.choice(length, size=count, replace=False).astype(np.int64))


def build_sample_selection(
    sample: DataSample,
    config: DataSamplingConfig | None,
) -> SampleSelection | None:
    """Build deterministic point/face/cell indices for a sample.

    Source-cell sampling selects source OpenFOAM ``cellID`` values first, then
    returns every exported VTU child cell whose ``cellID`` belongs to that set.
    This preserves foamToVTK child-cell reduction semantics.
    """

    if config is None:
        return None
    sample_id = sample.manifest.sample_id
    cell_ids = np.asarray(sample.arrays["volume_cell_id"], dtype=np.int64).reshape(-1)
    exported_cell_indices = _indices(
        int(sample.arrays["volume_celltypes"].shape[0]),
        config.volume_exported_cell_count,
        sample_id=sample_id,
        seed=config.seed,
        scope="volume_exported_cell",
    )
    source_cell_ids = _indices(
        sample.manifest.volume_provenance.source_openfoam_cell_count,
        config.volume_source_cell_count,
        sample_id=sample_id,
        seed=config.seed,
        scope="volume_source_cell",
    )
    if source_cell_ids is not None:
        mask = np.isin(cell_ids, source_cell_ids)
        exported_cell_indices = np.flatnonzero(mask).astype(np.int64)

    return SampleSelection(
        sample_id=sample_id,
        surface_point_indices=_indices(
            int(sample.arrays["surface_points_m"].shape[0]),
            config.surface_point_count,
            sample_id=sample_id,
            seed=config.seed,
            scope="surface_point",
        ),
        surface_face_indices=_indices(
            int(sample.arrays["surface_pressure_kinematic"].shape[0]),
            config.surface_face_count,
            sample_id=sample_id,
            seed=config.seed,
            scope="surface_face",
        ),
        volume_point_indices=_indices(
            int(sample.arrays["volume_points_m"].shape[0]),
            config.volume_point_count,
            sample_id=sample_id,
            seed=config.seed,
            scope="volume_point",
        ),
        volume_exported_cell_indices=exported_cell_indices,
        volume_source_cell_ids=source_cell_ids,
        semantics={
            "surface": "indices reference immutable VTP-derived point and face arrays",
            "volume_exported_cells": "indices reference exported VTU cells, not source cells",
            "volume_source_cells": (
                "source OpenFOAM cell sampling uses volume_cell_id/cellID and includes all "
                "exported child cells for selected source IDs"
            ),
        },
    )

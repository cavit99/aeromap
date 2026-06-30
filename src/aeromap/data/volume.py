"""Source-cell-aware volume helpers for foamToVTK exports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class VolumeReductionError(ValueError):
    """Raised when a volume reduction would ignore source-cell provenance."""


def aggregate_by_cell_id(
    values: np.ndarray,
    cell_ids: IntArray,
    *,
    source_cell_count: int,
    reducer: str = "mean",
) -> np.ndarray:
    """Aggregate exported VTU cell data back to source OpenFOAM cells through ``cellID``.

    ``foamToVTK`` can split one OpenFOAM source cell into multiple exported child cells. This
    helper is the required path for reductions over cell fields; callers that need raw exported
    cells should keep the VTU arrays unchanged.
    """

    if reducer != "mean":
        msg = f"unsupported source-cell reducer: {reducer}"
        raise VolumeReductionError(msg)
    values_array = np.asarray(values)
    ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    if source_cell_count <= 0:
        msg = "source_cell_count must be positive"
        raise VolumeReductionError(msg)
    if values_array.shape[0] != ids.shape[0]:
        msg = "cell field length must match cellID length"
        raise VolumeReductionError(msg)
    if np.any(ids < 0) or np.any(ids >= source_cell_count):
        msg = "cellID values are outside the source OpenFOAM cell range"
        raise VolumeReductionError(msg)

    flat = values_array.reshape(values_array.shape[0], -1).astype(np.float64, copy=False)
    sums = np.zeros((source_cell_count, flat.shape[1]), dtype=np.float64)
    counts = np.bincount(ids, minlength=source_cell_count)
    np.add.at(sums, ids, flat)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        msg = f"cellID mapping is missing {len(missing)} source OpenFOAM cells"
        raise VolumeReductionError(msg)
    reduced = sums / counts[:, None]
    return reduced.reshape((source_cell_count, *values_array.shape[1:]))


def duplicate_group_max_spread(values: np.ndarray, cell_ids: IntArray) -> float:
    """Return the maximum within-source spread among duplicated exported child cells."""

    values_array = np.asarray(values, dtype=np.float64)
    ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    if values_array.shape[0] != ids.shape[0]:
        msg = "cell field length must match cellID length"
        raise VolumeReductionError(msg)
    counts = np.bincount(ids)
    duplicated_ids = np.flatnonzero(counts > 1)
    if len(duplicated_ids) == 0:
        return 0.0
    max_spread = 0.0
    flat = values_array.reshape(values_array.shape[0], -1)
    for source_id in duplicated_ids:
        group = flat[ids == source_id]
        spread = float(np.max(np.ptp(group, axis=0))) if len(group) else 0.0
        max_spread = max(max_spread, spread)
    return max_spread

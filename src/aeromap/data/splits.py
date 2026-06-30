"""Grouped split metadata for geometry-family held-out evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeromap.data.schema import DataSample

SPLIT_SCHEMA_VERSION = "aerocliff_grouped_split_v0.1.0"
SplitName = Literal[
    "train",
    "pool",
    "calibration",
    "test",
    "boundary_audit",
    "mesh_audit",
]


class GroupedSplitEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    geometry_id: str
    state_id: str
    simulation_id: str
    attempt_id: str
    case_class: Literal["NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"]
    training_eligible: bool
    split: SplitName
    group_id: str | None = None

    @property
    def geometry_family_id(self) -> str:
        return self.group_id or self.geometry_id


class GroupedSplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SPLIT_SCHEMA_VERSION
    split_dimension: Literal["geometry_family"] = "geometry_family"
    entries: list[GroupedSplitEntry] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _geometry_families_do_not_leak(self) -> Self:
        split_by_group: dict[str, str] = {}
        for entry in self.entries:
            group = entry.geometry_family_id
            existing = split_by_group.setdefault(group, entry.split)
            if existing != entry.split:
                msg = f"geometry family {group!r} appears in both {existing!r} and {entry.split!r}"
                raise ValueError(msg)
            if entry.split in {"train", "pool"} and not entry.training_eligible:
                msg = (
                    f"sample {entry.sample_id} is assigned to {entry.split!r} "
                    "but is not training eligible"
                )
                raise ValueError(msg)
        return self


def build_grouped_split_manifest(
    samples: Sequence[DataSample],
    split_by_geometry_id: Mapping[str, SplitName],
    *,
    notes: Sequence[str] = (),
) -> GroupedSplitManifest:
    """Create split metadata where complete geometry families stay together."""

    if not samples:
        msg = "at least one sample is required"
        raise ValueError(msg)
    entries: list[GroupedSplitEntry] = []
    for sample in samples:
        manifest = sample.manifest
        try:
            split = split_by_geometry_id[manifest.geometry_id]
        except KeyError as exc:
            msg = f"missing split assignment for geometry_id {manifest.geometry_id!r}"
            raise ValueError(msg) from exc
        entries.append(
            GroupedSplitEntry(
                sample_id=manifest.sample_id,
                geometry_id=manifest.geometry_id,
                state_id=manifest.state_id,
                simulation_id=manifest.simulation_id,
                attempt_id=manifest.attempt_id,
                case_class=manifest.case_class,
                training_eligible=manifest.training_eligible,
                split=split,
                group_id=manifest.geometry_id,
            ),
        )
    return GroupedSplitManifest(entries=entries, notes=list(notes))

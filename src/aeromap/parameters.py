"""Parameter schema, canonical hashing, and deterministic sample generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import qmc

from aeromap.constants import CFD_CONFIG_VERSION, GEOMETRY_GENERATOR_VERSION, REF


def _stable_id(prefix: str, payload: Mapping[str, Any], *, length: int = 16) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    return f"{prefix}_{digest[:length]}"


class AeroParams(BaseModel):
    """Fixed six-dimensional design/state vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    geometry_family: Literal["advanced_challenge", "stable_reference"] = "advanced_challenge"
    ride_height_mm: float = Field(ge=25.0, le=110.0)
    pitch_deg: float = Field(ge=-1.0, le=1.5)
    yaw_deg: float = Field(ge=-3.0, le=3.0)
    throat_offset_mm: float = Field(ge=20.0, le=50.0)
    diffuser_angle_deg: float = Field(ge=0.5, le=2.0)
    edge_radius_mm: float = Field(ge=5.0, le=25.0)

    ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "ride_height_mm": (25.0, 75.0),
        "pitch_deg": (-1.0, 1.5),
        "yaw_deg": (-3.0, 3.0),
        "throat_offset_mm": (20.0, 50.0),
        "diffuser_angle_deg": (1.0, 2.0),
        "edge_radius_mm": (5.0, 25.0),
    }
    stable_reference_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "ride_height_mm": (90.0, 110.0),
        "pitch_deg": (0.0, 0.0),
        "yaw_deg": (0.0, 0.0),
        "throat_offset_mm": (35.0, 35.0),
        "diffuser_angle_deg": (0.5, 0.75),
        "edge_radius_mm": (12.0, 12.0),
    }

    @classmethod
    def canonical(cls) -> AeroParams:
        return cls(
            ride_height_mm=40.0,
            pitch_deg=0.4,
            yaw_deg=0.0,
            throat_offset_mm=35.0,
            diffuser_angle_deg=1.25,
            edge_radius_mm=12.0,
        )

    @classmethod
    def stable_reference(cls) -> AeroParams:
        """Return the deliberately mild MVP reference fixture."""

        return cls(
            geometry_family="stable_reference",
            ride_height_mm=100.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            throat_offset_mm=35.0,
            diffuser_angle_deg=0.6,
            edge_radius_mm=12.0,
        )

    @model_validator(mode="after")
    def _family_bounds_are_respected(self) -> Self:
        bounds = (
            self.stable_reference_ranges
            if self.geometry_family == "stable_reference"
            else self.ranges
        )
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not (low <= value <= high):
                message = (
                    f"{name}={value:g} is outside the {self.geometry_family} "
                    f"range [{low:g}, {high:g}]"
                )
                raise ValueError(message)
        return self

    def canonical_dict(self) -> dict[str, float]:
        return {key: round(float(getattr(self, key)), 8) for key in self.ranges}

    def canonical_json(self, *, extra: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "params": self.canonical_dict(),
            "geometry_generator_version": GEOMETRY_GENERATOR_VERSION,
            "cfd_config_version": CFD_CONFIG_VERSION,
        }
        if self.geometry_family != "advanced_challenge":
            payload["geometry_family"] = self.geometry_family
        if extra:
            payload.update(extra)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def case_id(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"case_{digest[:16]}"

    def geometry_family_id(self) -> str:
        return self.geometry_id()

    def geometry_payload(self) -> dict[str, Any]:
        """Physical geometry identity, excluding operating state."""

        floor_family = (
            "stable_reference_underfloor_fixture"
            if self.geometry_family == "stable_reference"
            else "venturi_underfloor_research_benchmark"
        )
        return {
            "floor_family": floor_family,
            "throat_offset_mm": self.canonical_dict()["throat_offset_mm"],
            "actual_diffuser_angle_deg": self.canonical_dict()["diffuser_angle_deg"],
            "edge_radius_mm": self.canonical_dict()["edge_radius_mm"],
            "geometry_family": self.geometry_family,
            "geometry_generator_version": GEOMETRY_GENERATOR_VERSION,
        }

    def geometry_id(self) -> str:
        return _stable_id("geometry", self.geometry_payload())

    def state_payload(self, *, inlet_speed_m_s: float = REF.u_inf_m_s) -> dict[str, Any]:
        """Operating-state identity, excluding physical geometry parameters."""

        return {
            "ride_height_mm": self.canonical_dict()["ride_height_mm"],
            "pitch_deg": self.canonical_dict()["pitch_deg"],
            "yaw_deg": self.canonical_dict()["yaw_deg"],
            "inlet_speed_m_s": round(float(inlet_speed_m_s), 8),
        }

    def state_id(self, *, inlet_speed_m_s: float = REF.u_inf_m_s) -> str:
        return _stable_id("state", self.state_payload(inlet_speed_m_s=inlet_speed_m_s))

    def simulation_payload(
        self,
        *,
        mesh_config: Mapping[str, Any],
        surface_export_config: Mapping[str, Any] | None = None,
        solver_config: Mapping[str, Any],
        quality_config: Mapping[str, Any] | None = None,
        openfoam_version: str,
        inlet_speed_m_s: float = REF.u_inf_m_s,
    ) -> dict[str, Any]:
        """Full simulation identity for immutable CFD records."""

        return {
            "geometry_id": self.geometry_id(),
            "state_id": self.state_id(inlet_speed_m_s=inlet_speed_m_s),
            "mesh_config": dict(mesh_config),
            "surface_export_config": dict(surface_export_config or {}),
            "solver_config": dict(solver_config),
            "quality_config": dict(quality_config or {}),
            "openfoam_version": openfoam_version,
            "cfd_config_version": CFD_CONFIG_VERSION,
        }

    def simulation_id(
        self,
        *,
        mesh_config: Mapping[str, Any],
        surface_export_config: Mapping[str, Any] | None = None,
        solver_config: Mapping[str, Any],
        quality_config: Mapping[str, Any] | None = None,
        openfoam_version: str,
        inlet_speed_m_s: float = REF.u_inf_m_s,
    ) -> str:
        return _stable_id(
            "simulation",
            self.simulation_payload(
                mesh_config=mesh_config,
                surface_export_config=surface_export_config,
                solver_config=solver_config,
                quality_config=quality_config,
                openfoam_version=openfoam_version,
                inlet_speed_m_s=inlet_speed_m_s,
            ),
        )


def corner_params() -> dict[str, AeroParams]:
    """Deterministic geometry/operating corner cases for validation and CFD smoke."""

    base = AeroParams.canonical().model_dump()
    return {
        "min_ride_height": AeroParams(**{**base, "ride_height_mm": 25.0, "pitch_deg": -1.0}),
        "max_ride_height": AeroParams(**{**base, "ride_height_mm": 75.0, "pitch_deg": 1.5}),
        "forward_throat_low_diffuser_tight_edge": AeroParams(
            **{
                **base,
                "ride_height_mm": 25.0,
                "pitch_deg": -1.0,
                "throat_offset_mm": 20.0,
                "diffuser_angle_deg": 1.0,
                "edge_radius_mm": 5.0,
            },
        ),
        "rear_throat_high_diffuser_large_edge": AeroParams(
            **{
                **base,
                "ride_height_mm": 75.0,
                "pitch_deg": 1.5,
                "throat_offset_mm": 50.0,
                "diffuser_angle_deg": 2.0,
                "edge_radius_mm": 25.0,
            },
        ),
        "worst_clearance": AeroParams(
            **{
                **base,
                "ride_height_mm": 25.0,
                "pitch_deg": -1.0,
                "throat_offset_mm": 50.0,
                "diffuser_angle_deg": 2.0,
                "edge_radius_mm": 25.0,
            },
        ),
    }


def sobol_params(count: int, *, seed: int = 1729) -> list[AeroParams]:
    """Generate deterministic Sobol samples over the fixed parameter ranges."""

    if count <= 0:
        message = "count must be positive"
        raise ValueError(message)

    dimensions = len(AeroParams.ranges)
    power = int(np.ceil(np.log2(count)))
    sampler = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
    unit_samples = sampler.random_base2(m=power)[:count]

    names = list(AeroParams.ranges)
    lower = np.array([AeroParams.ranges[name][0] for name in names], dtype=float)
    upper = np.array([AeroParams.ranges[name][1] for name in names], dtype=float)
    scaled = qmc.scale(unit_samples, lower, upper)

    return [AeroParams(**dict(zip(names, row, strict=True))) for row in scaled]

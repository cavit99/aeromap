"""Central physical constants and sign conventions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConditions:
    """Fixed AeroCliff reference conditions in SI units."""

    l_ref_m: float = 2.0
    w_ref_m: float = 1.0
    u_inf_m_s: float = 40.0
    rho_kg_m3: float = 1.225
    nu_m2_s: float = 1.5e-5
    p_inf_pa: float = 0.0

    @property
    def a_ref_m2(self) -> float:
        return self.l_ref_m * self.w_ref_m

    @property
    def q_inf_pa(self) -> float:
        return 0.5 * self.rho_kg_m3 * self.u_inf_m_s**2


REF = ReferenceConditions()

GEOMETRY_GENERATOR_VERSION = "cadquery_article_v0.10.0"
CFD_CONFIG_VERSION = "openfoam_v13_smoke_v0.1.0"

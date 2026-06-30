"""Structured AeroCliff Core / Venturi Lab case generation."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pyvista as pv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeromap.attempts import stable_id
from aeromap.cfd.dictionaries import (
    header,
    mesh_quality_dict,
    momentum_transport,
    physical_properties,
    turbulence_values,
    vec,
)
from aeromap.constants import REF
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file

VENTURI_CORE_SCHEMA = "aerocliff_core_venturi_lab_v0.1.0"
VENTURI_CORE_CLASSIFICATION: Literal["AEROCLIFF_CORE_VENTURI_LAB"] = "AEROCLIFF_CORE_VENTURI_LAB"
OPENFOAM_VERSION = "OpenFOAM Foundation v13"
PROFILE_SEGMENT_COUNT = 5
CORE_METRICS_SCHEMA = "aerocliff_core_case_metrics_v0.2.0"
CORE_GRID_VALIDATION_SCHEMA = "aerocliff_core_grid_validation_v0.2.0"
MASS_IMBALANCE_LIMIT = 0.005
FORCE_CV_LIMIT = 0.005
SUCTION_GRID_DIFF_LIMIT = 0.03
DRAG_GRID_DIFF_LIMIT = 0.05
PRESSURE_RECOVERY_GRID_DIFF_LIMIT = 0.05
F_SEP_CLIFF_THRESHOLD = 0.10
WALL_SHEAR_SIGN_AUDIT_SCHEMA = "aerocliff_core_wallshear_sign_audit_v0.1.0"


class VenturiCoreGeometryConfig(BaseModel):
    """2.5D Venturi-underfloor geometry contract."""

    model_config = ConfigDict(frozen=True)

    ride_height_mm: float = Field(default=60.0, ge=20.0, le=100.0)
    diffuser_angle_deg: float = Field(default=1.5, ge=0.5, le=14.0)
    throat_ratio: float = Field(default=0.7, gt=0.25, lt=1.0)
    inlet_length_m: float = Field(default=0.4, gt=0.0)
    contraction_length_m: float = Field(default=0.55, gt=0.0)
    throat_length_m: float = Field(default=0.55, gt=0.0)
    diffuser_length_m: float = Field(default=1.2, gt=0.0)
    outlet_recovery_length_m: float = Field(default=0.4, gt=0.0)
    span_m: float = Field(default=0.12, gt=0.0)
    inlet_gap_ratio: float = Field(default=1.25, gt=1.0)

    @property
    def ride_height_m(self) -> float:
        return self.ride_height_mm / 1000.0

    @property
    def throat_height_m(self) -> float:
        return self.ride_height_m * self.throat_ratio

    @property
    def diffuser_exit_height_m(self) -> float:
        return self.throat_height_m + math.tan(math.radians(self.diffuser_angle_deg)) * (
            self.diffuser_length_m
        )


class VenturiCoreMeshConfig(BaseModel):
    """Structured medium/fine mesh controls."""

    model_config = ConfigDict(frozen=True)

    grid: Literal["coarse", "medium", "fine"] = "medium"
    x_cells_per_segment: tuple[int, ...] = (16, 24, 24, 40, 16)
    span_cells: int = Field(default=8, gt=0)
    wall_normal_cells: int = Field(default=36, gt=4)
    wall_normal_grading: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _segments_match_profile(self) -> Self:
        if len(self.x_cells_per_segment) != PROFILE_SEGMENT_COUNT:
            message = "x_cells_per_segment must contain five entries for the six profile stations"
            raise ValueError(message)
        if any(value <= 0 for value in self.x_cells_per_segment):
            message = "x_cells_per_segment entries must be positive"
            raise ValueError(message)
        return self


class VenturiCoreSolverConfig(BaseModel):
    """OpenFOAM steady solver controls for the core benchmark."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=250, gt=0)
    write_interval: int = Field(default=50, gt=0)
    force_window: int = Field(default=50, gt=0)
    u_inf_m_s: float = Field(default=REF.u_inf_m_s, gt=0.0)
    rho_kg_m3: float = Field(default=REF.rho_kg_m3, gt=0.0)


class VenturiCoreConfig(BaseModel):
    """Top-level structured Venturi lab configuration."""

    model_config = ConfigDict(frozen=True)

    classification: Literal["AEROCLIFF_CORE_VENTURI_LAB"] = VENTURI_CORE_CLASSIFICATION
    geometry: VenturiCoreGeometryConfig = VenturiCoreGeometryConfig()
    mesh: VenturiCoreMeshConfig = VenturiCoreMeshConfig()
    solver: VenturiCoreSolverConfig = VenturiCoreSolverConfig()


@dataclass(frozen=True)
class VenturiCoreArtifacts:
    """Generated structured Venturi case artifacts."""

    case_id: str
    case_dir: Path
    openfoam_dir: Path
    manifest_path: Path
    profile_path: Path
    run_mesh_script_path: Path
    run_solver_script_path: Path


@dataclass(frozen=True)
class _Station:
    x: float
    gap: float
    label: str


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def profile_stations(config: VenturiCoreGeometryConfig) -> tuple[dict[str, float | str], ...]:
    """Return the deterministic x-gap Venturi profile."""

    x0 = -config.inlet_length_m
    x1 = 0.0
    x2 = x1 + config.contraction_length_m
    x3 = x2 + config.throat_length_m
    x4 = x3 + config.diffuser_length_m
    x5 = x4 + config.outlet_recovery_length_m
    inlet_gap = config.ride_height_m * config.inlet_gap_ratio
    stations = (
        _Station(x=x0, gap=inlet_gap, label="inlet_plenum"),
        _Station(x=x1, gap=config.ride_height_m, label="floor_entry"),
        _Station(x=x2, gap=config.throat_height_m, label="throat_start"),
        _Station(x=x3, gap=config.throat_height_m, label="diffuser_start"),
        _Station(x=x4, gap=config.diffuser_exit_height_m, label="diffuser_exit"),
        _Station(x=x5, gap=config.diffuser_exit_height_m, label="outlet_recovery"),
    )
    return tuple({"x_m": item.x, "gap_m": item.gap, "label": item.label} for item in stations)


def _station_x(config: VenturiCoreGeometryConfig, label: str) -> float:
    for station in profile_stations(config):
        if station["label"] == label:
            return float(station["x_m"])
    message = f"unknown Venturi profile station: {label}"
    raise KeyError(message)


def _gap_at_x(config: VenturiCoreGeometryConfig, x_values: np.ndarray) -> np.ndarray:
    stations = profile_stations(config)
    station_x = np.asarray([float(station["x_m"]) for station in stations], dtype=np.float64)
    station_gap = np.asarray([float(station["gap_m"]) for station in stations], dtype=np.float64)
    return np.asarray(np.interp(x_values, station_x, station_gap), dtype=np.float64)


def _profile_payload(config: VenturiCoreConfig) -> dict[str, Any]:
    stations = profile_stations(config.geometry)
    min_gap = min(float(station["gap_m"]) for station in stations)
    max_gap = max(float(station["gap_m"]) for station in stations)
    cell_count = int(
        sum(config.mesh.x_cells_per_segment)
        * config.mesh.span_cells
        * config.mesh.wall_normal_cells,
    )
    return {
        "schema_version": VENTURI_CORE_SCHEMA,
        "classification": config.classification,
        "geometry": config.geometry.model_dump(mode="json"),
        "mesh": config.mesh.model_dump(mode="json"),
        "solver": config.solver.model_dump(mode="json"),
        "profile_stations": list(stations),
        "derived": {
            "minimum_gap_m": min_gap,
            "maximum_gap_m": max_gap,
            "throat_height_m": config.geometry.throat_height_m,
            "diffuser_exit_height_m": config.geometry.diffuser_exit_height_m,
            "estimated_blockmesh_cells": cell_count,
            "reference_area_m2_per_span": REF.l_ref_m * config.geometry.span_m,
            "dynamic_pressure_pa": 0.5 * config.solver.rho_kg_m3 * config.solver.u_inf_m_s**2,
        },
        "claim_boundary": {
            "full_3d_aerocliff_accuracy": False,
            "f1_floor_accuracy": False,
            "domino_accuracy": False,
            "training_eligible_before_validation": False,
            "structured_venturi_underfloor_benchmark": True,
        },
    }


def _format_vertices(config: VenturiCoreConfig) -> tuple[str, list[tuple[int, int, int, int]]]:
    y_min = -0.5 * config.geometry.span_m
    y_max = 0.5 * config.geometry.span_m
    stations = profile_stations(config.geometry)
    lines: list[str] = []
    ids: list[tuple[int, int, int, int]] = []
    vertex_id = 0
    for station in stations:
        x = float(station["x_m"])
        gap = float(station["gap_m"])
        current = (vertex_id, vertex_id + 1, vertex_id + 2, vertex_id + 3)
        ids.append(current)
        lines.extend(
            [
                f"    ({x:.10g} {y_min:.10g} 0)",
                f"    ({x:.10g} {y_max:.10g} 0)",
                f"    ({x:.10g} {y_min:.10g} {gap:.10g})",
                f"    ({x:.10g} {y_max:.10g} {gap:.10g})",
            ],
        )
        vertex_id += 4
    return "\n".join(lines), ids


def block_mesh_dict(config: VenturiCoreConfig) -> str:
    """Render a blockMesh-only structured Venturi channel."""

    vertices, ids = _format_vertices(config)
    blocks: list[str] = []
    for index, x_cells in enumerate(config.mesh.x_cells_per_segment):
        b0, b1, t0, t1 = ids[index]
        nb0, nb1, nt0, nt1 = ids[index + 1]
        blocks.append(
            "    hex "
            f"({b0} {nb0} {nb1} {b1} {t0} {nt0} {nt1} {t1}) "
            f"({x_cells} {config.mesh.span_cells} {config.mesh.wall_normal_cells}) "
            f"simpleGrading (1 1 {config.mesh.wall_normal_grading:g})",
        )
    ground_faces: list[str] = []
    floor_faces: list[str] = []
    side_min_faces: list[str] = []
    side_max_faces: list[str] = []
    for index in range(len(ids) - 1):
        b0, b1, t0, t1 = ids[index]
        nb0, nb1, nt0, nt1 = ids[index + 1]
        ground_faces.append(f"            ({b0} {nb0} {nb1} {b1})")
        floor_faces.append(f"            ({t0} {t1} {nt1} {nt0})")
        side_min_faces.append(f"            ({b0} {t0} {nt0} {nb0})")
        side_max_faces.append(f"            ({b1} {nb1} {nt1} {t1})")
    inlet_b0, inlet_b1, inlet_t0, inlet_t1 = ids[0]
    outlet_b0, outlet_b1, outlet_t0, outlet_t1 = ids[-1]
    return (
        header("dictionary", "system", "blockMeshDict")
        + f"""
scale 1;

vertices
(
{vertices}
);

blocks
(
{chr(10).join(blocks)}
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            ({inlet_b0} {inlet_t0} {inlet_t1} {inlet_b1})
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            ({outlet_b0} {outlet_b1} {outlet_t1} {outlet_t0})
        );
    }}
    ground
    {{
        type wall;
        faces
        (
{chr(10).join(ground_faces)}
        );
    }}
    floor
    {{
        type wall;
        faces
        (
{chr(10).join(floor_faces)}
        );
    }}
    side_y_min
    {{
        type symmetryPlane;
        faces
        (
{chr(10).join(side_min_faces)}
        );
    }}
    side_y_max
    {{
        type symmetryPlane;
        faces
        (
{chr(10).join(side_max_faces)}
        );
    }}
);

mergePatchPairs
(
);
"""
    )


def control_dict(config: VenturiCoreConfig) -> str:
    area = REF.l_ref_m * config.geometry.span_m
    return (
        header("dictionary", "system", "controlDict")
        + f"""
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {config.solver.max_iterations};
deltaT          1;

writeControl    timeStep;
writeInterval   {config.solver.write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         (floor);
        rho             rhoInf;
        rhoInf          {config.solver.rho_kg_m3:g};
        CofR            (1 0 0);
        writeControl    timeStep;
        writeInterval   1;
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (floor);
        rho             rhoInf;
        rhoInf          {config.solver.rho_kg_m3:g};
        liftDir         (0 0 -1);
        dragDir         (1 0 0);
        CofR            (1 0 0);
        pitchAxis       (0 1 0);
        magUInf         {config.solver.u_inf_m_s:g};
        lRef            {REF.l_ref_m:g};
        Aref            {area:g};
        writeControl    timeStep;
        writeInterval   1;
    }}

    inletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           inlet;
        fields          (phi);
        operation       sum;
    }}

    outletFlowRate
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           outlet;
        fields          (phi);
        operation       sum;
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         (floor ground);
        writeControl    writeTime;
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
}}
"""
    )


def fv_schemes() -> str:
    return (
        header("dictionary", "system", "fvSchemes")
        + """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
    )


def fv_solution() -> str:
    return (
        header("dictionary", "system", "fvSolution")
        + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    pcorr
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0;
        smoother        GaussSeidel;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent yes;
    residualControl
    {
        p       1e-2;
        U       1e-3;
        "(k|omega)" 1e-3;
    }
}

relaxationFactors
{
    equations
    {
        U       0.7;
        ".*"    0.7;
    }
}
"""
    )


def _field_boundary(default_entry: str, *, include_inlet_outlet: str) -> str:
    return f"""
    inlet       {{ {include_inlet_outlet} }}
    outlet      {{ type zeroGradient; }}
    ground      {{ {default_entry} }}
    floor       {{ {default_entry} }}
    side_y_min  {{ type symmetryPlane; }}
    side_y_max  {{ type symmetryPlane; }}"""


def field_u(config: VenturiCoreConfig) -> str:
    u = (config.solver.u_inf_m_s, 0.0, 0.0)
    return (
        header("volVectorField", "0", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform {vec(u)};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {vec(u)}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type fixedValue; value uniform {vec(u)}; }}
    floor       {{ type noSlip; }}
    side_y_min  {{ type symmetryPlane; }}
    side_y_max  {{ type symmetryPlane; }}
}}
"""
    )


def field_p() -> str:
    return (
        header("volScalarField", "0", "p")
        + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    ground      { type zeroGradient; }
    floor       { type zeroGradient; }
    side_y_min  { type symmetryPlane; }
    side_y_max  { type symmetryPlane; }
}
"""
    )


def field_k() -> str:
    k, _omega = turbulence_values()
    return (
        header("volScalarField", "0", "k")
        + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k:.10g};

boundaryField
{{{
            _field_boundary(
                f"type kqRWallFunction; value uniform {k:.10g};",
                include_inlet_outlet=f"type fixedValue; value uniform {k:.10g};",
            )
        }
}}
"""
    )


def field_omega() -> str:
    _k, omega = turbulence_values()
    return (
        header("volScalarField", "0", "omega")
        + f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega:.10g};

boundaryField
{{{
            _field_boundary(
                f"type omegaWallFunction; value uniform {omega:.10g};",
                include_inlet_outlet=f"type fixedValue; value uniform {omega:.10g};",
            )
        }
}}
"""
    )


def field_nut() -> str:
    return (
        header("volScalarField", "0", "nut")
        + f"""
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{{{
            _field_boundary(
                "type nutkWallFunction; value uniform 0;",
                include_inlet_outlet="type calculated; value uniform 0;",
            )
        }
}}
"""
    )


def _write_openfoam_file(path: Path, content: str) -> None:
    atomic_write_text(path, content.strip() + "\n")


def _case_id(config: VenturiCoreConfig) -> str:
    return stable_id("venturi_core", config.model_dump(mode="json"))


def _write_profile_svg(config: VenturiCoreConfig, path: Path) -> None:
    stations = profile_stations(config.geometry)
    width = 900
    height = 260
    padding = 50
    x_values = [float(station["x_m"]) for station in stations]
    z_values = [float(station["gap_m"]) for station in stations]
    x_min, x_max = min(x_values), max(x_values)
    z_max = max(z_values) * 1.2

    def sx(value: float) -> float:
        return padding + (value - x_min) / (x_max - x_min) * (width - 2 * padding)

    def sz(value: float) -> float:
        return height - padding - value / z_max * (height - 2 * padding)

    top_points = " ".join(
        f"{sx(x):.1f},{sz(z):.1f}" for x, z in zip(x_values, z_values, strict=True)
    )
    ground_y = sz(0.0)
    svg = "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '  <rect width="100%" height="100%" fill="#ffffff"/>',
            (
                f'  <line x1="{padding}" y1="{ground_y:.1f}" '
                f'x2="{width - padding}" y2="{ground_y:.1f}" '
                'stroke="#222" stroke-width="3"/>'
            ),
            (f'  <polyline points="{top_points}" fill="none" stroke="#0f766e" stroke-width="4"/>'),
            (
                f'  <text x="{padding}" y="28" font-family="Arial" '
                'font-size="20" fill="#111">AeroCliff Core / Venturi Lab profile</text>'
            ),
            (
                f'  <text x="{padding}" y="{height - 12}" font-family="Arial" '
                'font-size="13" fill="#444">moving ground below; stationary floor '
                "underside above; 2.5D structured extrusion</text>"
            ),
            "</svg>",
            "",
        ],
    )
    atomic_write_text(path, svg)


def _write_case_files(case_dir: Path, config: VenturiCoreConfig) -> None:
    openfoam_dir = case_dir / "openfoam"
    for path in [
        openfoam_dir / "0",
        openfoam_dir / "constant",
        openfoam_dir / "system",
        case_dir / "logs",
        case_dir / "quality",
        case_dir / "outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _write_openfoam_file(openfoam_dir / "system" / "blockMeshDict", block_mesh_dict(config))
    _write_openfoam_file(openfoam_dir / "system" / "controlDict", control_dict(config))
    _write_openfoam_file(openfoam_dir / "system" / "fvSchemes", fv_schemes())
    _write_openfoam_file(openfoam_dir / "system" / "fvSolution", fv_solution())
    _write_openfoam_file(openfoam_dir / "system" / "meshQualityDict", mesh_quality_dict())
    _write_openfoam_file(openfoam_dir / "constant" / "physicalProperties", physical_properties())
    _write_openfoam_file(openfoam_dir / "constant" / "momentumTransport", momentum_transport())
    _write_openfoam_file(openfoam_dir / "0" / "U", field_u(config))
    _write_openfoam_file(openfoam_dir / "0" / "p", field_p())
    _write_openfoam_file(openfoam_dir / "0" / "k", field_k())
    _write_openfoam_file(openfoam_dir / "0" / "omega", field_omega())
    _write_openfoam_file(openfoam_dir / "0" / "nut", field_nut())
    _write_openfoam_file(case_dir / "run_core_mesh.sh", _mesh_script())
    _write_openfoam_file(case_dir / "run_core_solver.sh", _solver_script())
    (case_dir / "run_core_mesh.sh").chmod(0o755)
    (case_dir / "run_core_solver.sh").chmod(0o755)


def _load_manifest_config(case_dir: Path) -> VenturiCoreConfig:
    manifest_path = case_dir / "manifest.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"manifest is not a JSON object: {manifest_path}"
        raise TypeError(message)
    return VenturiCoreConfig.model_validate(loaded["config"])


def _numeric_time_dirs(openfoam_dir: Path) -> list[int]:
    return sorted(
        int(path.name) for path in openfoam_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )


def _force_columns(path: Path) -> list[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Time"):
            return line[2:].split()
    message = f"forceCoeffs header not found: {path}"
    raise ValueError(message)


def _load_force_coefficients(path: Path) -> dict[str, np.ndarray]:
    columns = _force_columns(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(item) for item in stripped.split()])
    data = np.asarray(rows, dtype=np.float64)
    return {name: data[:, index] for index, name in enumerate(columns)}


def _window_stats(values: np.ndarray) -> dict[str, float | None]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": abs(std / mean) if not math.isclose(mean, 0.0, abs_tol=1.0e-15) else None,
        "relative_drift": (
            float((values[-1] - values[0]) / mean)
            if not math.isclose(mean, 0.0, abs_tol=1.0e-15)
            else None
        ),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _load_surface_flow(path: Path) -> float:
    columns: list[str] | None = None
    last_row: list[float] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Time"):
            columns = line[2:].split()
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        last_row = [float(item) for item in stripped.split()]
    if columns is None or last_row is None:
        message = f"surfaceFieldValue data not found: {path}"
        raise ValueError(message)
    return dict(zip(columns, last_row, strict=True))["sum(phi)"]


def _mesh_log_status(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")
    failed_match = re.search(r"Failed\s+(\d+)\s+mesh checks?", text)
    return {
        "mesh_ok": "Mesh OK." in text and (failed_match is None or int(failed_match.group(1)) == 0),
        "failed_checks": int(failed_match.group(1)) if failed_match else 0,
        "sha256": sha256_file(log_path),
    }


def _patch_boundary_block(field_text: str, patch_name: str) -> str:
    match = re.search(rf"\b{re.escape(patch_name)}\s*\{{(?P<body>.*?)\}}", field_text, re.DOTALL)
    if match is None:
        message = f"patch {patch_name!r} not found in OpenFOAM field"
        raise ValueError(message)
    return match.group("body")


def _field_entry_value(block: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s+([^;]+);", block)
    return match.group(1).strip() if match else None


def _boundary_condition_audit(openfoam_dir: Path, config: VenturiCoreConfig) -> dict[str, Any]:
    u_text = (openfoam_dir / "0" / "U").read_text(encoding="utf-8")
    ground_block = _patch_boundary_block(u_text, "ground")
    floor_block = _patch_boundary_block(u_text, "floor")
    expected_ground = vec((config.solver.u_inf_m_s, 0.0, 0.0))
    ground_type = _field_entry_value(ground_block, "type")
    ground_value = _field_entry_value(ground_block, "value")
    floor_type = _field_entry_value(floor_block, "type")
    ground_explicit_moving_belt_ok = (
        ground_type == "fixedValue" and ground_value == f"uniform {expected_ground}"
    )
    floor_stationary_no_slip_ok = floor_type == "noSlip"
    return {
        "ground_patch": {
            "type": ground_type,
            "value": ground_value,
            "expected_value": f"uniform {expected_ground}",
            "explicit_moving_belt_ok": ground_explicit_moving_belt_ok,
            "uses_movingWallVelocity": ground_type == "movingWallVelocity",
        },
        "floor_patch": {
            "type": floor_type,
            "stationary_no_slip_ok": floor_stationary_no_slip_ok,
        },
        "passed": ground_explicit_moving_belt_ok and floor_stationary_no_slip_ok,
        "interpretation": (
            "Stationary mesh moving-belt setup: ground must impose explicit fixedValue "
            "freestream velocity and the floor/body patch remains stationary no-slip."
        ),
    }


def _area_weighted_mean(values: np.ndarray, areas: np.ndarray) -> float:
    return float(np.sum(values * areas) / np.sum(areas))


def _area_fraction(mask: np.ndarray, areas: np.ndarray) -> float:
    if areas.size == 0 or math.isclose(float(np.sum(areas)), 0.0):
        return 0.0
    return float(np.sum(areas[mask]) / np.sum(areas))


def _sign_label(value: float) -> Literal["negative", "positive", "zero"]:
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def _patch_normal_summary(mesh: pv.DataSet, areas: np.ndarray) -> dict[str, Any]:
    normal_mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
    )
    normals = np.asarray(normal_mesh.cell_data["Normals"], dtype=np.float64)
    mean = np.sum(normals * areas[:, None], axis=0) / np.sum(areas)
    return {
        "area_weighted_mean": [float(value) for value in mean],
        "median": [float(value) for value in np.median(normals, axis=0)],
    }


def _floor_region_masks(
    *,
    centers: np.ndarray,
    config: VenturiCoreConfig,
) -> dict[str, np.ndarray]:
    floor_entry = _station_x(config.geometry, "floor_entry")
    throat_start = _station_x(config.geometry, "throat_start")
    diffuser_start = _station_x(config.geometry, "diffuser_start")
    diffuser_exit = _station_x(config.geometry, "diffuser_exit")
    outlet_recovery = _station_x(config.geometry, "outlet_recovery")
    return {
        "attached_reference": (centers[:, 0] >= floor_entry) & (centers[:, 0] <= throat_start),
        "throat": (centers[:, 0] >= throat_start) & (centers[:, 0] <= diffuser_start),
        "diffuser": (centers[:, 0] >= diffuser_start) & (centers[:, 0] <= diffuser_exit),
        "exit_recovery": (centers[:, 0] >= diffuser_exit) & (centers[:, 0] <= outlet_recovery),
    }


def _near_wall_internal_velocity_metrics(
    *,
    internal_vtk_path: Path,
    config: VenturiCoreConfig,
) -> dict[str, Any]:
    mesh = pv.read(internal_vtk_path).compute_cell_sizes(length=False, area=False, volume=True)
    centers = mesh.cell_centers().points
    volumes = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    velocity = np.asarray(mesh.cell_data["U"], dtype=np.float64)
    local_gap = _gap_at_x(config.geometry, centers[:, 0])
    first_layer_height = local_gap / config.mesh.wall_normal_cells
    near_floor = (local_gap - centers[:, 2]) <= first_layer_height * 1.05
    near_ground = centers[:, 2] <= first_layer_height * 1.05
    region_masks = _floor_region_masks(centers=centers, config=config)

    def collect(mask: np.ndarray) -> dict[str, Any]:
        region_volumes = volumes[mask]
        region_u = velocity[mask]
        if region_volumes.size == 0:
            return {
                "cell_count": 0,
                "mean_u_x": None,
                "median_u_x": None,
                "min_u_x": None,
                "max_u_x": None,
                "reverse_flow_cell_fraction_u_x_lt_0": None,
            }
        return {
            "cell_count": int(np.count_nonzero(mask)),
            "mean_u_x": _area_weighted_mean(region_u[:, 0], region_volumes),
            "median_u_x": float(np.median(region_u[:, 0])),
            "min_u_x": float(np.min(region_u[:, 0])),
            "max_u_x": float(np.max(region_u[:, 0])),
            "reverse_flow_cell_fraction_u_x_lt_0": float(np.mean(region_u[:, 0] < 0.0)),
        }

    return {
        "internal_vtk_path": str(internal_vtk_path),
        "first_cell_selection": "cell centres within 1.05 * local_gap / wall_normal_cells",
        "near_floor": {
            name: collect(region_mask & near_floor) for name, region_mask in region_masks.items()
        },
        "near_ground": {
            name: collect(region_mask & near_ground) for name, region_mask in region_masks.items()
        },
    }


def _floor_region_metrics(
    *,
    floor_vtk_path: Path,
    internal_vtk_path: Path,
    config: VenturiCoreConfig,
) -> dict[str, Any]:
    mesh = pv.read(floor_vtk_path).compute_cell_sizes(length=False, area=True, volume=False)
    centers = mesh.cell_centers().points
    cell_data = mesh.cell_data
    areas = np.asarray(cell_data["Area"], dtype=np.float64)
    p = np.asarray(cell_data["p"], dtype=np.float64)
    tau = np.asarray(cell_data["wallShearStress"], dtype=np.float64)
    y_plus = np.asarray(cell_data["yPlus"], dtype=np.float64)

    dynamic_pressure_kinematic = 0.5 * config.solver.u_inf_m_s**2
    region_masks = _floor_region_masks(centers=centers, config=config)
    regions: dict[str, Any] = {}
    for name, mask in region_masks.items():
        region_areas = areas[mask]
        region_p = p[mask]
        region_tau = tau[mask]
        region_y_plus = y_plus[mask]
        regions[name] = {
            "cell_count": int(np.count_nonzero(mask)),
            "area_m2": float(np.sum(region_areas)),
            "mean_p_kinematic": _area_weighted_mean(region_p, region_areas),
            "mean_cp": _area_weighted_mean(region_p, region_areas) / dynamic_pressure_kinematic,
            "mean_wall_shear_x": _area_weighted_mean(region_tau[:, 0], region_areas),
            "raw_addendum_area_fraction_tau_x_lt_0": _area_fraction(
                region_tau[:, 0] < 0.0,
                region_areas,
            ),
            "y_plus_mean": _area_weighted_mean(region_y_plus, region_areas),
            "y_plus_max": float(np.max(region_y_plus)),
        }
    attached_reference_tau_x = regions["attached_reference"]["mean_wall_shear_x"]
    diffuser_mask = region_masks["diffuser"]
    diffuser_areas = areas[diffuser_mask]
    diffuser_tau_x = tau[diffuser_mask, 0]
    corrected_f_sep = _area_fraction(
        diffuser_tau_x * attached_reference_tau_x < 0.0,
        diffuser_areas,
    )
    pressure_recovery = regions["exit_recovery"]["mean_cp"] - regions["throat"]["mean_cp"]
    near_wall_velocity = _near_wall_internal_velocity_metrics(
        internal_vtk_path=internal_vtk_path,
        config=config,
    )
    near_wall_diffuser = near_wall_velocity["near_floor"]["diffuser"]
    return {
        "floor_vtk_path": str(floor_vtk_path),
        "patch_normal": _patch_normal_summary(mesh, areas),
        "raw_addendum_f_sep_definition": "area(tau_w dot x_hat < 0) / diffuser_ramp_area",
        "corrected_f_sep_definition": (
            "area(diffuser tau_x has opposite sign to upstream attached-reference tau_x) "
            "/ diffuser_ramp_area"
        ),
        "f_sep_attached_threshold": F_SEP_CLIFF_THRESHOLD,
        "attached_reference": {
            "region": "floor_entry_to_throat_start",
            "mean_wall_shear_x": attached_reference_tau_x,
            "sign": _sign_label(attached_reference_tau_x),
        },
        "regions": regions,
        "throat_pressure_mean_cp": regions["throat"]["mean_cp"],
        "diffuser_exit_pressure_mean_cp": regions["exit_recovery"]["mean_cp"],
        "pressure_recovery_cp_exit_minus_cp_throat": pressure_recovery,
        "diffuser_raw_addendum_f_sep_tau_x_lt_0": regions["diffuser"][
            "raw_addendum_area_fraction_tau_x_lt_0"
        ],
        "diffuser_f_sep": corrected_f_sep,
        "diffuser_f_sep_regime": (
            "cliff_or_separated"
            if corrected_f_sep >= F_SEP_CLIFF_THRESHOLD
            else "attached_pre_cliff"
        ),
        "near_wall_velocity": near_wall_velocity,
        "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0": near_wall_diffuser[
            "reverse_flow_cell_fraction_u_x_lt_0"
        ],
    }


def _wall_surface_summary(path: Path) -> dict[str, Any]:
    mesh = pv.read(path).compute_cell_sizes(length=False, area=True, volume=False)
    cell_data = mesh.cell_data
    areas = np.asarray(cell_data["Area"], dtype=np.float64)
    tau = np.asarray(cell_data["wallShearStress"], dtype=np.float64)
    y_plus = np.asarray(cell_data["yPlus"], dtype=np.float64)
    return {
        "vtk_path": str(path),
        "area_m2": float(np.sum(areas)),
        "patch_normal": _patch_normal_summary(mesh, areas),
        "mean_wall_shear_x": _area_weighted_mean(tau[:, 0], areas),
        "raw_addendum_area_fraction_tau_x_lt_0": _area_fraction(tau[:, 0] < 0.0, areas),
        "y_plus_mean": _area_weighted_mean(y_plus, areas),
        "y_plus_max": float(np.max(y_plus)),
    }


def write_venturi_core_case_metrics(case_dir: Path, *, out: Path | None = None) -> Path:
    """Extract compact Core pressure/load/cliff metrics from a completed case."""

    config = _load_manifest_config(case_dir)
    openfoam_dir = case_dir / "openfoam"
    latest_time = max(_numeric_time_dirs(openfoam_dir))
    post_dir = openfoam_dir / "postProcessing"
    force_path = post_dir / "forceCoeffs" / "0" / "forceCoeffs.dat"
    force_data = _load_force_coefficients(force_path)
    force_window = config.solver.force_window
    force_metrics = {
        key: _window_stats(force_data[key][-force_window:]) for key in ("Cd", "Cl", "Cm")
    }
    inlet_phi = _load_surface_flow(post_dir / "inletFlowRate" / "0" / "surfaceFieldValue.dat")
    outlet_phi = _load_surface_flow(post_dir / "outletFlowRate" / "0" / "surfaceFieldValue.dat")
    floor_vtk = openfoam_dir / "VTK" / "floor" / f"floor_{latest_time}.vtk"
    ground_vtk = openfoam_dir / "VTK" / "ground" / f"ground_{latest_time}.vtk"
    internal_vtk = openfoam_dir / "VTK" / f"openfoam_{latest_time}.vtk"
    floor_metrics = _floor_region_metrics(
        floor_vtk_path=floor_vtk,
        internal_vtk_path=internal_vtk,
        config=config,
    )
    ground_metrics = _wall_surface_summary(ground_vtk)
    mesh_quality = _mesh_log_status(case_dir / "logs" / "checkMesh_meshQuality.log")
    extended_mesh = _mesh_log_status(case_dir / "logs" / "checkMesh.log")
    boundary_conditions = _boundary_condition_audit(openfoam_dir, config)
    payload: dict[str, Any] = {
        "schema_version": CORE_METRICS_SCHEMA,
        "classification": "AEROCLIFF_CORE_CASE_METRICS",
        "case_id": case_dir.name,
        "case_dir": str(case_dir),
        "grid": config.mesh.grid,
        "accepted": False,
        "training_eligible": False,
        "latest_time": latest_time,
        "mesh": {
            "fatal_mesh_quality": mesh_quality,
            "extended_mesh": extended_mesh,
            "estimated_cells": (
                sum(config.mesh.x_cells_per_segment)
                * config.mesh.span_cells
                * config.mesh.wall_normal_cells
            ),
        },
        "solve": {
            "force_rows": int(force_data["Time"].size),
            "final_window_rows": int(force_window),
        },
        "mass_balance": {
            "inlet_phi": inlet_phi,
            "outlet_phi": outlet_phi,
            "relative_imbalance": abs(inlet_phi + outlet_phi) / abs(inlet_phi),
        },
        "boundary_conditions": boundary_conditions,
        "force_coefficients_final_window": force_metrics,
        "floor_metrics": floor_metrics,
        "ground_metrics": ground_metrics,
        "wall_treatment": {
            "target_y_plus": "around 1 preferred for wall-resolved Core separation evidence",
            "generated_wall_functions": [
                "nutkWallFunction",
                "kqRWallFunction",
                "omegaWallFunction",
            ],
            "separation_metric_strength": (
                "f_sep is a Core regime metric; cliff labels require near-cliff validation and "
                "wall-treatment disclosure."
            ),
        },
        "claim_boundary": {
            "core_case_metrics": True,
            "accepted_core_label": False,
            "requires_grid_validation": True,
            "cliff_label_requires_near_cliff_validation": True,
            "full_3d_aerocliff_accuracy": False,
            "f1_floor_accuracy": False,
            "domino_accuracy": False,
        },
        "artifacts": {
            "forceCoeffs": str(force_path),
            "floor_vtk": str(floor_vtk),
            "internal_vtk": str(internal_vtk),
            "checkMesh_meshQuality_log": str(case_dir / "logs" / "checkMesh_meshQuality.log"),
            "checkMesh_extended_log": str(case_dir / "logs" / "checkMesh.log"),
        },
    }
    output = out or (case_dir / "outputs" / "core_metrics.json")
    atomic_write_json(output, payload)
    return output


def _load_case_metrics(case_dir: Path) -> dict[str, Any]:
    metrics_path = case_dir / "outputs" / "core_metrics.json"
    if not metrics_path.exists():
        write_venturi_core_case_metrics(case_dir)
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"Core metrics are not a JSON object: {metrics_path}"
        raise TypeError(message)
    return loaded


def write_venturi_core_wallshear_sign_audit(
    *,
    coarse_case: Path,
    medium_case: Path,
    fine_case: Path,
    out: Path,
) -> Path:
    """Write a compact sign-convention audit for Core wall-shear separation metrics."""

    metrics = {
        "coarse": _load_case_metrics(coarse_case),
        "medium": _load_case_metrics(medium_case),
        "fine": _load_case_metrics(fine_case),
    }

    cases: dict[str, Any] = {}
    for grid, item in metrics.items():
        floor = item["floor_metrics"]
        ground = item["ground_metrics"]
        near_wall = floor["near_wall_velocity"]["near_floor"]
        cases[grid] = {
            "case_id": item["case_id"],
            "grid": item["grid"],
            "floor_patch_normal_area_weighted": floor["patch_normal"]["area_weighted_mean"],
            "ground_patch_normal_area_weighted": ground["patch_normal"]["area_weighted_mean"],
            "floor_attached_reference": floor["attached_reference"],
            "floor_regions": {
                name: {
                    "mean_wall_shear_x": region["mean_wall_shear_x"],
                    "raw_addendum_area_fraction_tau_x_lt_0": region[
                        "raw_addendum_area_fraction_tau_x_lt_0"
                    ],
                    "near_wall_mean_u_x": near_wall[name]["mean_u_x"],
                    "near_wall_median_u_x": near_wall[name]["median_u_x"],
                    "near_wall_reverse_flow_cell_fraction_u_x_lt_0": near_wall[name][
                        "reverse_flow_cell_fraction_u_x_lt_0"
                    ],
                    "wall_shear_sign": _sign_label(region["mean_wall_shear_x"]),
                    "near_wall_u_x_sign": _sign_label(near_wall[name]["mean_u_x"]),
                }
                for name, region in floor["regions"].items()
            },
            "raw_addendum_diffuser_f_sep_tau_x_lt_0": floor[
                "diffuser_raw_addendum_f_sep_tau_x_lt_0"
            ],
            "corrected_diffuser_f_sep_reference_sign": floor["diffuser_f_sep"],
            "corrected_diffuser_regime": floor["diffuser_f_sep_regime"],
            "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0": floor[
                "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"
            ],
            "ground_patch": {
                "mean_wall_shear_x": ground["mean_wall_shear_x"],
                "raw_addendum_area_fraction_tau_x_lt_0": ground[
                    "raw_addendum_area_fraction_tau_x_lt_0"
                ],
                "mean_u_x_boundary_value": item["boundary_conditions"]["ground_patch"]["value"],
            },
        }

    payload = {
        "schema_version": WALL_SHEAR_SIGN_AUDIT_SCHEMA,
        "classification": "AEROCLIFF_CORE_WALLSHEAR_SIGN_CONVENTION_AUDIT",
        "cases": cases,
        "finding": (
            "OpenFOAM exported wallShearStress_x is negative on the floor diffuser while "
            "first-cell internal U_x remains positive. Therefore raw tau_x < 0 is not a "
            "safe universal reverse-flow rule for this patch convention."
        ),
        "corrected_metric": {
            "definition": (
                "f_sep = area(diffuser tau_x has opposite sign to upstream attached-reference "
                "floor tau_x) / diffuser_ramp_area"
            ),
            "reference_region": "floor_entry_to_throat_start",
            "near_wall_velocity_cross_check": (
                "report first-cell diffuser U_x < 0 fraction; do not accept separation/cliff "
                "labels from wallShearStress alone while y+ remains wall-function range"
            ),
        },
        "target_split_decision": {
            "pressure_load_targets_may_use_corrected_attached_anchor": True,
            "separation_fraction_label_eligible": False,
            "cliff_boundary_label_eligible": False,
            "wall_shear_magnitude_label_eligible": False,
        },
        "claim_boundary": {
            "core_sign_convention_resolved_for_attached_pressure_load_anchor": True,
            "core_separation_or_cliff_label": False,
            "full_3d_aerocliff_accuracy": False,
            "f1_floor_accuracy": False,
            "active_learning_claim": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _relative_difference(reference: float, candidate: float) -> float:
    denominator = max(abs(reference), abs(candidate), 1.0e-15)
    return abs(candidate - reference) / denominator


def write_venturi_core_grid_validation(
    *,
    coarse_case: Path,
    medium_case: Path,
    fine_case: Path,
    out: Path,
) -> Path:
    """Compare coarse/medium/fine Core cases and classify the canonical reference."""

    metrics = {
        "coarse": _load_case_metrics(coarse_case),
        "medium": _load_case_metrics(medium_case),
        "fine": _load_case_metrics(fine_case),
    }
    medium = metrics["medium"]
    fine = metrics["fine"]
    comparisons = {
        "medium_fine_cd_relative_difference": _relative_difference(
            medium["force_coefficients_final_window"]["Cd"]["mean"],
            fine["force_coefficients_final_window"]["Cd"]["mean"],
        ),
        "medium_fine_suction_relative_difference": _relative_difference(
            medium["force_coefficients_final_window"]["Cl"]["mean"],
            fine["force_coefficients_final_window"]["Cl"]["mean"],
        ),
        "medium_fine_pressure_recovery_relative_difference": _relative_difference(
            medium["floor_metrics"]["pressure_recovery_cp_exit_minus_cp_throat"],
            fine["floor_metrics"]["pressure_recovery_cp_exit_minus_cp_throat"],
        ),
        "medium_fine_f_sep_absolute_difference": abs(
            medium["floor_metrics"]["diffuser_f_sep"] - fine["floor_metrics"]["diffuser_f_sep"],
        ),
        "medium_fine_raw_addendum_f_sep_absolute_difference": abs(
            medium["floor_metrics"]["diffuser_raw_addendum_f_sep_tau_x_lt_0"]
            - fine["floor_metrics"]["diffuser_raw_addendum_f_sep_tau_x_lt_0"],
        ),
        "medium_fine_f_sep_same_regime": (
            medium["floor_metrics"]["diffuser_f_sep_regime"]
            == fine["floor_metrics"]["diffuser_f_sep_regime"]
        ),
        "medium_fine_near_wall_reverse_flow_fraction_absolute_difference": abs(
            medium["floor_metrics"]["near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"]
            - fine["floor_metrics"]["near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"],
        ),
    }
    boundary_ok = all(item["boundary_conditions"]["passed"] for item in metrics.values())
    mesh_ok = all(
        item["mesh"]["fatal_mesh_quality"]["mesh_ok"] and item["mesh"]["extended_mesh"]["mesh_ok"]
        for item in metrics.values()
    )
    mass_ok = all(
        item["mass_balance"]["relative_imbalance"] < MASS_IMBALANCE_LIMIT
        for item in metrics.values()
    )
    force_ok = all(
        item["force_coefficients_final_window"]["Cd"]["cv"] < FORCE_CV_LIMIT
        and item["force_coefficients_final_window"]["Cl"]["cv"] < FORCE_CV_LIMIT
        for item in metrics.values()
    )
    grid_ok = (
        comparisons["medium_fine_suction_relative_difference"] <= SUCTION_GRID_DIFF_LIMIT
        and comparisons["medium_fine_cd_relative_difference"] <= DRAG_GRID_DIFF_LIMIT
        and comparisons["medium_fine_pressure_recovery_relative_difference"]
        <= PRESSURE_RECOVERY_GRID_DIFF_LIMIT
        and comparisons["medium_fine_f_sep_same_regime"]
    )
    attached_anchor_ok = all(
        item["floor_metrics"]["diffuser_f_sep"] < F_SEP_CLIFF_THRESHOLD
        and item["floor_metrics"]["near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"]
        < F_SEP_CLIFF_THRESHOLD
        for item in metrics.values()
    )
    accepted = bool(
        boundary_ok and mesh_ok and mass_ok and force_ok and grid_ok and attached_anchor_ok
    )
    payload = {
        "schema_version": CORE_GRID_VALIDATION_SCHEMA,
        "classification": (
            "AEROCLIFF_CORE_ATTACHED_PRESSURE_LOAD_REFERENCE_V0"
            if accepted
            else "AEROCLIFF_CORE_ATTACHED_PRESSURE_LOAD_VALIDATION_FAILED"
        ),
        "accepted": accepted,
        "training_eligible": accepted,
        "training_eligible_scope": (
            "only explicit Core attached pressure/load targets; separation, cliff and wall-shear "
            "magnitude remain ineligible"
        ),
        "eligible_targets": {
            "core_attached_suction_downforce": accepted,
            "core_attached_drag": accepted,
            "core_attached_pressure_recovery": accepted,
            "core_attached_pre_cliff_regime": accepted,
            "core_separation_fraction": False,
            "core_cliff_boundary": False,
            "core_near_cliff_separation": False,
            "wall_shear_magnitude": False,
            "full_3d_aerocliff": False,
        },
        "thresholds": {
            "mass_imbalance": MASS_IMBALANCE_LIMIT,
            "force_cv": FORCE_CV_LIMIT,
            "suction_grid_difference": SUCTION_GRID_DIFF_LIMIT,
            "drag_grid_difference": DRAG_GRID_DIFF_LIMIT,
            "pressure_recovery_grid_difference": PRESSURE_RECOVERY_GRID_DIFF_LIMIT,
            "f_sep_cliff_threshold": F_SEP_CLIFF_THRESHOLD,
        },
        "comparisons": comparisons,
        "gate_checks": {
            "boundary_ok": boundary_ok,
            "mesh_ok": mesh_ok,
            "mass_ok": mass_ok,
            "force_ok": force_ok,
            "grid_ok": grid_ok,
            "attached_anchor_ok": attached_anchor_ok,
        },
        "cases": {
            grid: {
                "case_id": item["case_id"],
                "case_dir": item["case_dir"],
                "grid": item["grid"],
                "cells": item["mesh"]["estimated_cells"],
                "cd_mean": item["force_coefficients_final_window"]["Cd"]["mean"],
                "suction_mean": item["force_coefficients_final_window"]["Cl"]["mean"],
                "pressure_recovery": item["floor_metrics"][
                    "pressure_recovery_cp_exit_minus_cp_throat"
                ],
                "raw_addendum_f_sep_tau_x_lt_0": item["floor_metrics"][
                    "diffuser_raw_addendum_f_sep_tau_x_lt_0"
                ],
                "attached_reference_tau_x": item["floor_metrics"]["attached_reference"][
                    "mean_wall_shear_x"
                ],
                "attached_reference_sign": item["floor_metrics"]["attached_reference"]["sign"],
                "diffuser_f_sep": item["floor_metrics"]["diffuser_f_sep"],
                "diffuser_f_sep_regime": item["floor_metrics"]["diffuser_f_sep_regime"],
                "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0": item["floor_metrics"][
                    "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"
                ],
                "diffuser_y_plus_mean": item["floor_metrics"]["regions"]["diffuser"]["y_plus_mean"],
                "diffuser_y_plus_max": item["floor_metrics"]["regions"]["diffuser"]["y_plus_max"],
                "ground_y_plus_mean": item["ground_metrics"]["y_plus_mean"],
                "ground_y_plus_max": item["ground_metrics"]["y_plus_max"],
                "mass_imbalance": item["mass_balance"]["relative_imbalance"],
                "ground_boundary_type": item["boundary_conditions"]["ground_patch"]["type"],
                "ground_boundary_value": item["boundary_conditions"]["ground_patch"]["value"],
                "ground_boundary_ok": item["boundary_conditions"]["ground_patch"][
                    "explicit_moving_belt_ok"
                ],
                "floor_boundary_type": item["boundary_conditions"]["floor_patch"]["type"],
            }
            for grid, item in metrics.items()
        },
        "claim_boundary": {
            "core_structured_attached_anchor": accepted,
            "core_pressure_load_reference": accepted,
            "core_cliff_boundary_label": False,
            "near_cliff_case_validation_required": True,
            "separation_fraction_label": False,
            "full_3d_aerocliff_accuracy": False,
            "f1_floor_accuracy": False,
            "domino_accuracy": False,
            "active_learning_claim": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _mesh_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

if [[ ! -r /opt/openfoam13/etc/bashrc ]]; then
  echo "OpenFOAM v13 environment file not found: /opt/openfoam13/etc/bashrc" >&2
  exit 2
fi
set +eu
source /opt/openfoam13/etc/bashrc
source_status=$?
if [[ "${source_status}" -ne 0 ]]; then
  echo "failed to source OpenFOAM v13 environment" >&2
  exit 2
fi
set -euo pipefail
cd "$(dirname "$0")/openfoam"
mkdir -p ../logs ../quality ../outputs

blockMesh > ../logs/blockMesh.log 2>&1
checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log 2>&1
mesh_quality_failed_checks=$(
  awk '/Failed [0-9]+ mesh check/ {print $2}' ../logs/checkMesh_meshQuality.log | tail -n 1
)
mesh_quality_failed_checks="${mesh_quality_failed_checks:-0}"
printf "%s\\n" "${mesh_quality_failed_checks}" > ../quality/checkMesh_meshQuality.failed_checks
if [[ "${mesh_quality_failed_checks}" -gt 0 ]]; then
  echo "fatal checkMesh -meshQuality reported Failed ${mesh_quality_failed_checks} mesh checks" >&2
  exit 3
fi
checkMesh -allGeometry -allTopology -writeSurfaces -writeSets \
  -surfaceFormat vtk -setFormat vtk > ../logs/checkMesh.log 2>&1
extended_failed_checks=$(
  awk '/Failed [0-9]+ mesh check/ {print $2}' ../logs/checkMesh.log | tail -n 1
)
extended_failed_checks="${extended_failed_checks:-0}"
printf "%s\\n" "${extended_failed_checks}" > ../quality/checkMesh_extended.failed_checks
if [[ "${extended_failed_checks}" -gt 0 ]]; then
  echo "extended checkMesh reported Failed ${extended_failed_checks} mesh checks" >&2
  exit 4
fi
printf "%s\\n" "MESH_OK" > ../quality/status.txt
"""


def _solver_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

"$(dirname "$0")/run_core_mesh.sh"
set +eu
source /opt/openfoam13/etc/bashrc
source_status=$?
if [[ "${source_status}" -ne 0 ]]; then
  echo "failed to source OpenFOAM v13 environment" >&2
  exit 2
fi
set -euo pipefail
cd "$(dirname "$0")/openfoam"
foamRun -solver incompressibleFluid > ../logs/solver.log 2>&1
postProcess -func yPlus -latestTime > ../logs/yPlus.log 2>&1 || true
postProcess -func wallShearStress -latestTime > ../logs/wallShearStress.log 2>&1 || true
foamToVTK -latestTime > ../logs/foamToVTK.log 2>&1 || true
printf "%s\\n" "SOLVER_COMPLETED" > ../quality/status.txt
"""


def build_venturi_core_case(
    config: VenturiCoreConfig,
    *,
    cases_dir: Path,
    overwrite: bool = False,
) -> VenturiCoreArtifacts:
    """Build a deterministic structured Venturi lab case without snappy/STL."""

    case_id = _case_id(config)
    case_dir = cases_dir / case_id
    if case_dir.exists():
        if not overwrite:
            message = f"Venturi Core case already exists: {case_dir}"
            raise FileExistsError(message)
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    _write_case_files(case_dir, config)
    profile_payload = _profile_payload(config)
    profile_path = case_dir / "profile.json"
    atomic_write_json(profile_path, profile_payload)
    _write_profile_svg(config, case_dir / "profile.svg")
    manifest_path = case_dir / "manifest.json"
    file_hashes = {
        str(path.relative_to(case_dir)): sha256_file(path)
        for path in sorted(case_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_write_json(
        manifest_path,
        {
            "schema_version": VENTURI_CORE_SCHEMA,
            "classification": VENTURI_CORE_CLASSIFICATION,
            "case_id": case_id,
            "git_sha": _git_sha(),
            "openfoam_version": OPENFOAM_VERSION,
            "config": config.model_dump(mode="json"),
            "profile_path": str(profile_path.relative_to(case_dir)),
            "profile_svg_path": "profile.svg",
            "run_mesh_script_path": "run_core_mesh.sh",
            "run_solver_script_path": "run_core_solver.sh",
            "status": "BUILT_NOT_RUN",
            "file_hashes": file_hashes,
            "claim_boundary": profile_payload["claim_boundary"],
        },
    )
    return VenturiCoreArtifacts(
        case_id=case_id,
        case_dir=case_dir,
        openfoam_dir=case_dir / "openfoam",
        manifest_path=manifest_path,
        profile_path=profile_path,
        run_mesh_script_path=case_dir / "run_core_mesh.sh",
        run_solver_script_path=case_dir / "run_core_solver.sh",
    )


def write_venturi_core_design_report(*, config: VenturiCoreConfig, out: Path) -> Path:
    """Write the committed design/claim-boundary report for the Core tier."""

    payload = _profile_payload(config)
    atomic_write_json(out.with_suffix(".json"), payload)
    lines = [
        "# AeroCliff Core / Venturi Lab",
        "",
        (
            "AeroCliff Core is a controlled 2.5D Venturi-underfloor benchmark. "
            "It keeps the original cliff-loop idea but removes the full 3D "
            "snappyHexMesh/CAD fragility from the first accepted-label path."
        ),
        "",
        "## Classification",
        "",
        f"- classification: `{VENTURI_CORE_CLASSIFICATION}`",
        "- cloud: not required",
        "- NIM/DoMINO: not used",
        "- full 3D AeroCliff: paused",
        "",
        "## Geometry",
        "",
        f"- ride height: `{config.geometry.ride_height_mm:g} mm`",
        f"- diffuser angle: `{config.geometry.diffuser_angle_deg:g} deg`",
        f"- throat ratio: `{config.geometry.throat_ratio:g}`",
        f"- span: `{config.geometry.span_m:g} m`",
        "",
        "Profile stations:",
        "",
        "| station | x [m] | gap [m] |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        [
            (
                f"| {station['label']} | {float(station['x_m']):.4f} | "
                f"{float(station['gap_m']):.5f} |"
            )
            for station in profile_stations(config.geometry)
        ],
    )
    lines.extend(
        [
            "",
            "## Mesh Strategy",
            "",
            "- `blockMesh` structured hexahedral extrusion.",
            "- No STL.",
            "- No `snappyHexMesh`.",
            "- Symmetry/slip span boundaries for a 2.5D channel.",
            "- Moving ground and stationary floor underside.",
            f"- estimated cells: `{payload['derived']['estimated_blockmesh_cells']}`.",
            "",
            "## Intended Outputs",
            "",
            "- suction/downforce coefficient on the floor patch;",
            "- pressure recovery and mass balance diagnostics;",
            "- centre-of-pressure diagnostic from force/moment outputs where available;",
            "- reverse-flow/separation fraction in a later bounded postprocessor;",
            "- cliff flag only after validated cases exist.",
            "",
            "## Claim Boundary",
            "",
            (
                "- Allowed: structured Venturi-underfloor benchmark and validated Core "
                "labels once mesh/solve checks pass."
            ),
            (
                "- Not allowed: full F1 floor accuracy, full 3D AeroCliff accuracy, "
                "DoMINO accuracy, or training eligibility before validation."
            ),
        ],
    )
    atomic_write_text(out, "\n".join(lines) + "\n")
    return out

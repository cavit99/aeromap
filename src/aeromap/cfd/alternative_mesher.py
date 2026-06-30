"""Alternative campaign volume-mesher input generation.

This module prepares the authorised post-snappy fallback lane without claiming
that a campaign mesh has been accepted. The first adapter is CLI-driven Gmsh so
the project does not add a Python gmsh dependency before the fallback proves
useful.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeromap.attempts import stable_id
from aeromap.constants import GEOMETRY_GENERATOR_VERSION, REF
from aeromap.geometry.generator import generate_geometry
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams

ALTERNATIVE_MESHER_SCHEMA_VERSION = "alternative_volume_mesher_v0.1.0"


class MeshCellTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    minimum: int = Field(gt=0)
    maximum: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.minimum > self.maximum:
            message = "target cell minimum must be <= maximum"
            raise ValueError(message)
        return self


class FarfieldDomainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bounds_min_m: tuple[float, float, float] = (-3.0, -1.5, 0.0)
    bounds_max_m: tuple[float, float, float] = (5.0, 1.5, 1.5)
    patch_selection_tolerance_m: float = Field(default=1.0e-6, gt=0.0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if any(lo >= hi for lo, hi in zip(self.bounds_min_m, self.bounds_max_m, strict=True)):
            message = "farfield bounds_min_m entries must be below bounds_max_m entries"
            raise ValueError(message)
        return self


class GmshBoundaryLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    first_height_m: float = Field(default=5.0e-4, gt=0.0)
    ratio: float = Field(default=1.2, ge=1.0)
    layers: int = Field(default=3, ge=0)
    apply_to_article_boundary: bool = False

    @model_validator(mode="after")
    def _enabled_layers_are_positive(self) -> Self:
        if self.enabled and self.layers <= 0:
            message = "enabled boundary layers require at least one layer"
            raise ValueError(message)
        return self


class GmshVolumeMesherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    gmsh_path: Path | None = None
    mesh_size_min_m: float = Field(default=0.004, gt=0.0)
    mesh_size_max_m: float = Field(default=0.08, gt=0.0)
    article_mesh_size_m: float = Field(default=0.012, gt=0.0)
    underfloor_mesh_size_m: float = Field(default=0.006, gt=0.0)
    volume_algorithm: Literal["delaunay", "hxt"] = "delaunay"
    msh_file_version: Literal["2.2", "4.1"] = "2.2"
    optimize: bool = True
    domain: FarfieldDomainConfig = FarfieldDomainConfig()
    boundary_layer: GmshBoundaryLayerConfig = GmshBoundaryLayerConfig()
    target_cells: MeshCellTarget = MeshCellTarget(minimum=800_000, maximum=1_200_000)

    @model_validator(mode="after")
    def _mesh_sizes_are_ordered(self) -> Self:
        if self.mesh_size_min_m > self.mesh_size_max_m:
            message = "mesh_size_min_m must be <= mesh_size_max_m"
            raise ValueError(message)
        if self.underfloor_mesh_size_m > self.article_mesh_size_m:
            message = "underfloor_mesh_size_m must be <= article_mesh_size_m"
            raise ValueError(message)
        if self.article_mesh_size_m > self.mesh_size_max_m:
            message = "article_mesh_size_m must be <= mesh_size_max_m"
            raise ValueError(message)
        return self


class AlternativeVolumeMesherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: str = "campaign_gmsh_medium"
    schema_version: Literal["alternative_volume_mesher_v0.1.0"] = "alternative_volume_mesher_v0.1.0"
    case_class: Literal["CAMPAIGN_REFERENCE_CFD"] = "CAMPAIGN_REFERENCE_CFD"
    geometry_version: Literal["cadquery_article_v0.10.0"] = "cadquery_article_v0.10.0"
    mesher: Literal["gmsh_occ_volume"] = "gmsh_occ_volume"
    openfoam_solver: Literal["OpenFOAM Foundation v13"] = "OpenFOAM Foundation v13"
    gmsh: GmshVolumeMesherConfig = GmshVolumeMesherConfig()
    run_locally: bool = False

    @model_validator(mode="after")
    def _geometry_matches_source(self) -> Self:
        if self.geometry_version != GEOMETRY_GENERATOR_VERSION:
            message = "alternative mesher config must match the active geometry generator"
            raise ValueError(message)
        return self


class AlternativeMeshArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    out_dir: Path
    manifest_path: Path
    geo_path: Path
    step_path: Path
    params_path: Path
    status: Literal[
        "INPUTS_WRITTEN_NOT_MESHED",
        "BLOCKED_GMSH_NOT_FOUND",
        "BLOCKED_GMSH_PARSE_FAILED",
    ]


def _gmsh_executable(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    discovered = shutil.which("gmsh")
    return Path(discovered) if discovered else None


def _gmsh_version(executable: Path) -> str:
    completed = subprocess.run(  # noqa: S603
        [str(executable), "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip() or completed.stdout.strip()}"
    return completed.stdout.strip()


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _gmsh_parse_preflight(executable: Path, geo_path: Path, log_dir: Path) -> dict[str, Any]:
    command = [str(executable), "-parse_and_exit", str(geo_path)]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        returncode: int | str = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = "TIMEOUT"
        stdout = _subprocess_text(exc.stdout)
        stderr = _subprocess_text(exc.stderr)

    stdout_path = log_dir / "gmsh_parse.stdout.log"
    stderr_path = log_dir / "gmsh_parse.stderr.log"
    atomic_write_text(stdout_path, stdout)
    atomic_write_text(stderr_path, stderr)
    return {
        "command": " ".join(command),
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _geo_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def _volume_algorithm_code(name: Literal["delaunay", "hxt"]) -> int:
    return {"delaunay": 1, "hxt": 10}[name]


def _gmsh_format(version: Literal["2.2", "4.1"]) -> str:
    return {"2.2": "msh2", "4.1": "msh41"}[version]


def render_gmsh_volume_geo(
    *,
    step_path: Path,
    msh_path: Path,
    config: AlternativeVolumeMesherConfig,
) -> str:
    """Render a deterministic Gmsh/OpenCASCADE volume-mesh input file."""

    gmsh = config.gmsh
    domain = gmsh.domain
    x0, y0, z0 = domain.bounds_min_m
    x1, y1, z1 = domain.bounds_max_m
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    algorithm = _volume_algorithm_code(gmsh.volume_algorithm)
    optimize = "1" if gmsh.optimize else "0"
    eps = domain.patch_selection_tolerance_m

    boundary_layer_lines = [
        "// Boundary-layer extrusion is intentionally disabled in the first generated .geo.",
        "// AeroCliff will enable it only after the unlayered Gmsh/OpenFOAM path is accepted.",
    ]
    if gmsh.boundary_layer.enabled:
        boundary_layer_lines.extend(
            [
                (
                    "// Requested future first layer height: "
                    f"{gmsh.boundary_layer.first_height_m:.12g} m"
                ),
                f"// Requested future expansion ratio: {gmsh.boundary_layer.ratio:.12g}",
                f"// Requested future layer count: {gmsh.boundary_layer.layers}",
                (
                    "// apply_to_article_boundary=false is deliberate until article boundary "
                    "classification is verified."
                ),
            ],
        )

    return "\n".join(
        [
            "// Auto-generated by AeroCliff. Do not treat this file as accepted campaign CFD.",
            f"// schema_version: {ALTERNATIVE_MESHER_SCHEMA_VERSION}",
            f"// profile: {config.profile}",
            'SetFactory("OpenCASCADE");',
            'Geometry.OCCTargetUnit = "M";',
            f'Merge "{_geo_path(step_path)}";',
            "",
            "// Imported STEP should be one watertight article volume.",
            "article_volumes[] = Volume{:};",
            "farfield = newv;",
            (
                f"Box(farfield) = {{{x0:.12g}, {y0:.12g}, {z0:.12g}, "
                f"{dx:.12g}, {dy:.12g}, {dz:.12g}}};"
            ),
            "fluid_volumes[] = BooleanDifference{ Volume{farfield}; Delete; }"
            "{ Volume{article_volumes[]}; Delete; };",
            "// Do not run Coherence here: it is an implicit topology operation and produced",
            "// a nonfatal BooleanFragments warning during parse-only preflight.",
            "",
            "// Physical groups are conservative: domain planes are explicit; the remaining",
            "// fluid boundary is exported for later audited article-surface classification.",
            "fluid_boundary[] = CombinedBoundary{ Volume{fluid_volumes[]}; };",
            (
                f"inlet[] = Surface In BoundingBox{{{x0 - eps:.12g}, {y0 - eps:.12g}, "
                f"{z0 - eps:.12g}, {x0 + eps:.12g}, {y1 + eps:.12g}, {z1 + eps:.12g}}};"
            ),
            (
                f"outlet[] = Surface In BoundingBox{{{x1 - eps:.12g}, {y0 - eps:.12g}, "
                f"{z0 - eps:.12g}, {x1 + eps:.12g}, {y1 + eps:.12g}, {z1 + eps:.12g}}};"
            ),
            (
                f"ground[] = Surface In BoundingBox{{{x0 - eps:.12g}, {y0 - eps:.12g}, "
                f"{z0 - eps:.12g}, {x1 + eps:.12g}, {y1 + eps:.12g}, {z0 + eps:.12g}}};"
            ),
            (
                f"farfield[] = Surface In BoundingBox{{{x0 - eps:.12g}, {y0 - eps:.12g}, "
                f"{z0 - eps:.12g}, {x1 + eps:.12g}, {y0 + eps:.12g}, {z1 + eps:.12g}}};"
            ),
            (
                f"farfield[] += Surface In BoundingBox{{{x0 - eps:.12g}, {y1 - eps:.12g}, "
                f"{z0 - eps:.12g}, {x1 + eps:.12g}, {y1 + eps:.12g}, {z1 + eps:.12g}}};"
            ),
            (
                f"farfield[] += Surface In BoundingBox{{{x0 - eps:.12g}, {y0 - eps:.12g}, "
                f"{z1 - eps:.12g}, {x1 + eps:.12g}, {y1 + eps:.12g}, {z1 + eps:.12g}}};"
            ),
            'Physical Volume("fluid") = {fluid_volumes[]};',
            'Physical Surface("fluid_boundary_unclassified") = {fluid_boundary[]};',
            'Physical Surface("inlet") = {inlet[]};',
            'Physical Surface("outlet") = {outlet[]};',
            'Physical Surface("ground") = {ground[]};',
            'Physical Surface("farfield") = {farfield[]};',
            "",
            f"Mesh.MeshSizeMin = {gmsh.mesh_size_min_m:.12g};",
            f"Mesh.MeshSizeMax = {gmsh.mesh_size_max_m:.12g};",
            "Mesh.MeshSizeFromCurvature = 20;",
            "Mesh.MeshSizeExtendFromBoundary = 1;",
            f"Mesh.Algorithm3D = {algorithm};",
            f"Mesh.Optimize = {optimize};",
            f"Mesh.MshFileVersion = {gmsh.msh_file_version};",
            "",
            *boundary_layer_lines,
            "",
            f"// Planned mesh output: {_geo_path(msh_path)}",
            "// Use the manifest command with '-o <path>' so parse-only checks do not write it.",
            "",
        ],
    )


def _manifest_payload(
    *,
    params: AeroParams,
    config: AlternativeVolumeMesherConfig,
    out_dir: Path,
    geo_path: Path,
    msh_path: Path,
    gmsh_path: Path | None,
    gmsh_version: str | None,
    stale_outputs_removed: list[str],
    status: str,
) -> dict[str, Any]:
    plan_id_inputs = {
        "params": params.canonical_dict(),
        "geometry_version": config.geometry_version,
        "mesher": config.mesher,
        "gmsh": config.gmsh.model_dump(mode="json"),
    }
    payload: dict[str, Any] = {
        "schema_version": ALTERNATIVE_MESHER_SCHEMA_VERSION,
        "profile": config.profile,
        "case_class": config.case_class,
        "training_eligible": False,
        "mesher": config.mesher,
        "geometry_version": config.geometry_version,
        "geometry_id": params.geometry_id(),
        "state_id": params.state_id(),
        "plan_id_inputs": plan_id_inputs,
        "status": status,
        "out_dir": str(out_dir),
        "geo_path": str(geo_path),
        "planned_msh_path": str(msh_path),
        "stale_outputs_removed": stale_outputs_removed,
        "gmsh_executable": str(gmsh_path) if gmsh_path is not None else None,
        "gmsh_version": gmsh_version,
        "openfoam_solver": config.openfoam_solver,
        "reference_conditions": {
            "u_inf_m_s": REF.u_inf_m_s,
            "rho_kg_m3": REF.rho_kg_m3,
            "nu_m2_s": REF.nu_m2_s,
            "l_ref_m": REF.l_ref_m,
            "a_ref_m2": REF.a_ref_m2,
        },
        "configured_commands": {
            "gmsh_volume": (
                f"gmsh -3 {geo_path} -format {_gmsh_format(config.gmsh.msh_file_version)} "
                f"-o {msh_path}"
            ),
            "openfoam_import": f"gmshToFoam {msh_path.name}",
            "fatal_mesh_quality": "checkMesh -meshQuality",
            "extended_diagnostics": (
                "checkMesh -allGeometry -allTopology -writeSurfaces -writeSets "
                "-surfaceFormat vtk -setFormat vtk"
            ),
        },
        "established_claims": [
            "A typed alternative campaign volume-mesher input contract exists.",
            "The generated STEP input is the active AeroCliff geometry, not DrivAerML.",
            "The first fallback adapter is CLI-driven Gmsh/OpenCASCADE to avoid adding an "
            "unpinned Python gmsh dependency.",
        ],
        "unestablished_claims": [
            "No Gmsh volume mesh has been accepted.",
            "No gmshToFoam import has been accepted.",
            "No CAMPAIGN_REFERENCE_CFD solve has been run from this fallback.",
            "No regional y+, mass-balance, load-stability, independent-force, or mesh-sensitivity "
            "campaign evidence exists from this fallback.",
        ],
    }
    payload["plan_id"] = stable_id("altmesh", plan_id_inputs)
    return payload


def build_alternative_mesh_inputs(
    *,
    out_dir: Path,
    config: AlternativeVolumeMesherConfig,
    params: AeroParams | None = None,
) -> AlternativeMeshArtifacts:
    """Write a compact campaign alternative-mesher plan and Gmsh input files."""

    active_params = params or AeroParams.canonical()
    geometry_artifacts = generate_geometry(active_params, out_dir / "geometry")
    step_path = geometry_artifacts.step_path
    geo_path = out_dir / "gmsh" / f"{config.profile}.geo"
    msh_path = out_dir / "gmsh" / f"{config.profile}.msh"
    stale_outputs_removed: list[str] = []
    if msh_path.exists():
        msh_path.unlink()
        stale_outputs_removed.append(str(msh_path))

    geo = render_gmsh_volume_geo(step_path=step_path, msh_path=msh_path, config=config)
    atomic_write_text(geo_path, geo)

    gmsh_path = _gmsh_executable(config.gmsh.gmsh_path)
    gmsh_version = _gmsh_version(gmsh_path) if gmsh_path is not None else None
    parse_preflight = (
        _gmsh_parse_preflight(gmsh_path, geo_path, out_dir / "logs")
        if gmsh_path is not None
        else None
    )
    status: Literal[
        "INPUTS_WRITTEN_NOT_MESHED",
        "BLOCKED_GMSH_NOT_FOUND",
        "BLOCKED_GMSH_PARSE_FAILED",
    ] = "INPUTS_WRITTEN_NOT_MESHED" if gmsh_path is not None else "BLOCKED_GMSH_NOT_FOUND"
    if parse_preflight is not None and parse_preflight["returncode"] != 0:
        status = "BLOCKED_GMSH_PARSE_FAILED"
    manifest = _manifest_payload(
        params=active_params,
        config=config,
        out_dir=out_dir,
        geo_path=geo_path,
        msh_path=msh_path,
        gmsh_path=gmsh_path,
        gmsh_version=gmsh_version,
        stale_outputs_removed=stale_outputs_removed,
        status=status,
    )
    manifest["artifacts"] = {
        "step": str(step_path),
        "geo": str(geo_path),
        "params": str(geometry_artifacts.params_path),
        "geometry_validation": str(geometry_artifacts.validation_path),
        "geometry_hashes": str(geometry_artifacts.hashes_path),
    }
    manifest["artifact_hashes"] = {
        "step": sha256_file(step_path),
        "geo": sha256_file(geo_path),
        "params": sha256_file(geometry_artifacts.params_path),
        "geometry_validation": sha256_file(geometry_artifacts.validation_path),
        "geometry_hashes": sha256_file(geometry_artifacts.hashes_path),
    }
    manifest["gmsh_parse_preflight"] = parse_preflight
    if parse_preflight is not None and parse_preflight["returncode"] == 0:
        manifest["established_claims"].append(
            "Gmsh parse-only preflight accepted the generated OpenCASCADE .geo input.",
        )
    manifest_path = out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    return AlternativeMeshArtifacts(
        plan_id=str(manifest["plan_id"]),
        out_dir=out_dir,
        manifest_path=manifest_path,
        geo_path=geo_path,
        step_path=step_path,
        params_path=geometry_artifacts.params_path,
        status=status,
    )

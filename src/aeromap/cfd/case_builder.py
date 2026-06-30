"""Build deterministic OpenFOAM case directories."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import trimesh

from aeromap.attempts import stable_id
from aeromap.cfd import dictionaries as foam
from aeromap.cfd.patch_surface import is_multi_patch_mode, write_openfoam_patch_surface
from aeromap.cfd.schema import CfdCaseArtifacts, CfdConfig
from aeromap.geometry.generator import generate_geometry
from aeromap.geometry.schema import GeometryArtifacts
from aeromap.geometry.surface_candidates import write_gmsh_g0_surface_export
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams


@dataclass(frozen=True)
class SurfaceSource:
    stl_path: Path
    regions_json_path: Path
    regions_vtp_path: Path
    metrics_path: Path
    status: str


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


def _write_openfoam_file(path: Path, content: str) -> None:
    atomic_write_text(path, content.strip() + "\n")


def _case_status(case_dir: Path) -> str | None:
    status_path = case_dir / "quality" / "status.json"
    if not status_path.exists():
        return None
    try:
        loaded = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    status = loaded.get("status")
    return status if isinstance(status, str) else None


def _surface_export_payload(config: CfdConfig) -> dict[str, Any]:
    return config.surface_export.model_dump(mode="json")


def _simulation_id(params: AeroParams, config: CfdConfig) -> str:
    return params.simulation_id(
        mesh_config=config.mesh.model_dump(),
        surface_export_config=_surface_export_payload(config),
        solver_config=config.solver.model_dump(),
        quality_config=config.quality.model_dump(),
        openfoam_version="OpenFOAM Foundation v13",
    )


def _normalise_case_paths(value: object, *, case_dir: Path) -> object:
    if isinstance(value, dict):
        return {key: _normalise_case_paths(item, case_dir=case_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_case_paths(item, case_dir=case_dir) for item in value]
    if isinstance(value, str):
        try:
            return str(Path(value).relative_to(case_dir))
        except ValueError:
            return value
    return value


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _surface_source(
    *,
    params: AeroParams,
    geometry: GeometryArtifacts,
    config: CfdConfig,
    cfd_surface_dir: Path,
    surface_export_id: str,
) -> SurfaceSource:
    if config.surface_export.method == "cadquery_current":
        return SurfaceSource(
            stl_path=geometry.stl_path,
            regions_json_path=geometry.regions_json_path,
            regions_vtp_path=geometry.regions_vtp_path,
            metrics_path=geometry.validation_path,
            status="EXPORTED",
        )
    if config.surface_export.method == "gmsh_occ_g0_no_healing":
        result = write_gmsh_g0_surface_export(
            params=params,
            out_dir=cfd_surface_dir / "gmsh_occ_g0_no_healing",
            gmsh_path=config.surface_export.gmsh_path,
            mesh_size_min_m=config.surface_export.mesh_size_min_m,
            mesh_size_max_m=config.surface_export.mesh_size_max_m,
            mesh_algorithm=config.surface_export.mesh_algorithm,
            mesh_optimize=config.surface_export.mesh_optimize,
        )
        if (
            result.stl_path is None
            or result.regions_json_path is None
            or result.regions_vtp_path is None
        ):
            message = f"surface export failed for {surface_export_id}"
            raise RuntimeError(message)
        return SurfaceSource(
            stl_path=result.stl_path,
            regions_json_path=result.regions_json_path,
            regions_vtp_path=result.regions_vtp_path,
            metrics_path=result.metrics_path,
            status=result.status,
        )
    message = f"unsupported surface export method: {config.surface_export.method}"
    raise ValueError(message)


def _write_openfoam_patch_surface(
    *,
    case_dir: Path,
    cfd_surface_dir: Path,
    cfd_stl_path: Path,
    cfd_regions_json_path: Path,
    config: CfdConfig,
) -> tuple[Path, dict[str, Any] | None]:
    if not is_multi_patch_mode(config.surface_export.openfoam_patch_mode):
        return cfd_stl_path, None
    mesh = _load_trimesh(cfd_stl_path)
    regions = json.loads(cfd_regions_json_path.read_text(encoding="utf-8"))
    patch_artifacts = write_openfoam_patch_surface(
        mesh=mesh,
        regions=regions,
        out_dir=cfd_surface_dir,
        transition_band_width_m=config.surface_export.transition_band_width_m,
        feature_angle_deg=config.mesh.feature_angle_deg,
        patch_mode=config.surface_export.openfoam_patch_mode,
    )
    return patch_artifacts.obj_path, {
        "obj_path": str(patch_artifacts.obj_path.relative_to(case_dir)),
        "metrics_path": str(patch_artifacts.metrics_path.relative_to(case_dir)),
        "vtp_path": str(patch_artifacts.vtp_path.relative_to(case_dir)),
        "patch_names": list(patch_artifacts.patch_names),
    }


def _write_surface_metrics(
    *,
    case_dir: Path,
    metrics_source: Path,
    cfd_metrics_path: Path,
    cfd_stl_path: Path,
    cfd_openfoam_surface_path: Path,
    cfd_regions_json_path: Path,
    cfd_regions_vtp_path: Path,
    patch_surface_payload: dict[str, Any] | None,
) -> None:
    try:
        metrics_payload = json.loads(metrics_source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        metrics_payload = {"source_metrics_path": str(metrics_source)}
    normalised_metrics = _normalise_case_paths(metrics_payload, case_dir=case_dir)
    if not isinstance(normalised_metrics, dict):
        atomic_write_json(cfd_metrics_path, {"surface_export_metrics": normalised_metrics})
        return

    metrics = cast("dict[str, Any]", normalised_metrics)
    metrics["case_surface_paths"] = {
        "stl_path": str(cfd_stl_path.relative_to(case_dir)),
        "openfoam_surface_path": str(cfd_openfoam_surface_path.relative_to(case_dir)),
        "regions_json_path": str(cfd_regions_json_path.relative_to(case_dir)),
        "regions_vtp_path": str(cfd_regions_vtp_path.relative_to(case_dir)),
    }
    if patch_surface_payload is not None:
        metrics["openfoam_patch_surface"] = patch_surface_payload
    atomic_write_json(cfd_metrics_path, metrics)


def _prepare_cfd_surface(
    *,
    params: AeroParams,
    case_dir: Path,
    tri_surface_dir: Path,
    geometry: GeometryArtifacts,
    config: CfdConfig,
) -> dict[str, Any]:
    cfd_surface_dir = case_dir / "cfd_surface"
    cfd_surface_dir.mkdir(parents=True, exist_ok=True)
    payload = _surface_export_payload(config)
    surface_export_id = stable_id(
        "surface_export",
        {
            "geometry_id": params.geometry_id(),
            "surface_export": payload,
        },
    )

    source = _surface_source(
        params=params,
        geometry=geometry,
        config=config,
        cfd_surface_dir=cfd_surface_dir,
        surface_export_id=surface_export_id,
    )

    cfd_stl_path = cfd_surface_dir / "article.stl"
    cfd_regions_json_path = cfd_surface_dir / "surface_regions.json"
    cfd_regions_vtp_path = cfd_surface_dir / "surface_regions.vtp"
    cfd_metrics_path = cfd_surface_dir / "surface_export_metrics.json"
    shutil.copy2(source.stl_path, cfd_stl_path)
    shutil.copy2(source.regions_json_path, cfd_regions_json_path)
    shutil.copy2(source.regions_vtp_path, cfd_regions_vtp_path)
    cfd_openfoam_surface_path, patch_surface_payload = _write_openfoam_patch_surface(
        case_dir=case_dir,
        cfd_surface_dir=cfd_surface_dir,
        cfd_stl_path=cfd_stl_path,
        cfd_regions_json_path=cfd_regions_json_path,
        config=config,
    )
    _write_surface_metrics(
        case_dir=case_dir,
        metrics_source=source.metrics_path,
        cfd_metrics_path=cfd_metrics_path,
        cfd_stl_path=cfd_stl_path,
        cfd_openfoam_surface_path=cfd_openfoam_surface_path,
        cfd_regions_json_path=cfd_regions_json_path,
        cfd_regions_vtp_path=cfd_regions_vtp_path,
        patch_surface_payload=patch_surface_payload,
    )
    shutil.copy2(cfd_stl_path, tri_surface_dir / "article.stl")
    if cfd_openfoam_surface_path != cfd_stl_path:
        shutil.copy2(cfd_openfoam_surface_path, tri_surface_dir / cfd_openfoam_surface_path.name)
    shutil.copy2(cfd_regions_json_path, tri_surface_dir / "article_surface_regions.json")
    shutil.copy2(cfd_regions_vtp_path, tri_surface_dir / "article_surface_regions.vtp")

    return {
        "surface_export_id": surface_export_id,
        "surface_export": payload,
        "status": source.status,
        "stl_path": str(cfd_stl_path.relative_to(case_dir)),
        "openfoam_surface_path": str(cfd_openfoam_surface_path.relative_to(case_dir)),
        "openfoam_surface_file": cfd_openfoam_surface_path.name,
        "regions_json_path": str(cfd_regions_json_path.relative_to(case_dir)),
        "regions_vtp_path": str(cfd_regions_vtp_path.relative_to(case_dir)),
        "metrics_path": str(cfd_metrics_path.relative_to(case_dir)),
        "source_step_path": str(geometry.step_path.relative_to(case_dir)),
        "stl_sha256": sha256_file(cfd_stl_path),
        "openfoam_surface_sha256": sha256_file(cfd_openfoam_surface_path),
        "regions_json_sha256": sha256_file(cfd_regions_json_path),
        "regions_vtp_sha256": sha256_file(cfd_regions_vtp_path),
        "openfoam_patch_surface": patch_surface_payload,
    }


def _make_case_dirs(case_dir: Path) -> tuple[Path, Path]:
    openfoam_dir = case_dir / "openfoam"
    tri_surface_dir = openfoam_dir / "constant" / "triSurface"
    for path in [
        openfoam_dir / "0",
        openfoam_dir / "constant",
        openfoam_dir / "system",
        tri_surface_dir,
        case_dir / "logs",
        case_dir / "quality",
        case_dir / "outputs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return openfoam_dir, tri_surface_dir


def _write_openfoam_case_files(
    *,
    params: AeroParams,
    config: CfdConfig,
    case_dir: Path,
    openfoam_dir: Path,
    surface_file: str,
) -> None:
    _write_openfoam_file(openfoam_dir / "system" / "blockMeshDict", foam.block_mesh_dict(config))
    _write_openfoam_file(
        openfoam_dir / "system" / "snappyHexMeshDict",
        foam.snappy_hex_mesh_dict(config),
    )
    _write_openfoam_file(
        openfoam_dir / "system" / "surfaceFeaturesDict",
        foam.surface_features_dict(config),
    )
    _write_openfoam_file(openfoam_dir / "system" / "controlDict", foam.control_dict(config))
    _write_openfoam_file(openfoam_dir / "system" / "fvSchemes", foam.fv_schemes())
    _write_openfoam_file(openfoam_dir / "system" / "fvSolution", foam.fv_solution())
    _write_openfoam_file(openfoam_dir / "system" / "meshQualityDict", foam.mesh_quality_dict())
    _write_openfoam_file(
        openfoam_dir / "constant" / "physicalProperties",
        foam.physical_properties(),
    )
    _write_openfoam_file(
        openfoam_dir / "constant" / "momentumTransport",
        foam.momentum_transport(),
    )
    _write_openfoam_file(openfoam_dir / "0" / "U", foam.field_u(params, config))
    _write_openfoam_file(openfoam_dir / "0" / "p", foam.field_p(config))
    _write_openfoam_file(openfoam_dir / "0" / "k", foam.field_k(config))
    _write_openfoam_file(openfoam_dir / "0" / "omega", foam.field_omega(config))
    _write_openfoam_file(openfoam_dir / "0" / "nut", foam.field_nut(config))

    run_script_path = case_dir / "run_openfoam.sh"
    atomic_write_text(run_script_path, _script(surface_file, config))
    run_script_path.chmod(0o755)
    atomic_write_json(case_dir / "quality" / "status.json", {"status": "BUILT_NOT_RUN"})
    atomic_write_text(case_dir / "outputs" / ".gitkeep", "")
    atomic_write_text(case_dir / "logs" / ".gitkeep", "")


def _write_case_manifest(
    *,
    params: AeroParams,
    config: CfdConfig,
    case_dir: Path,
    cfd_surface: dict[str, Any],
    geometry: GeometryArtifacts,
    simulation_id: str,
) -> Path:
    file_hashes = {
        str(path.relative_to(case_dir)): sha256_file(path)
        for path in sorted(case_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, Any] = {
        "case_id": params.case_id(),
        "geometry_id": params.geometry_id(),
        "state_id": params.state_id(),
        "simulation_id": simulation_id,
        "surface_export_id": cfd_surface["surface_export_id"],
        "git_sha": _git_sha(),
        "params": params.model_dump(),
        "cfd_config": config.model_dump(mode="json"),
        "cfd_surface": cfd_surface,
        "geometry_validation": geometry.validation.model_dump(mode="json"),
        "file_hashes": file_hashes,
        "status": "BUILT_NOT_RUN",
    }
    manifest_path = case_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _publish_case_dir(
    *,
    case_dir: Path,
    final_case_dir: Path,
    cases_dir: Path,
    simulation_id: str,
) -> None:
    if not final_case_dir.exists():
        case_dir.rename(final_case_dir)
        return

    previous_dir = Path(tempfile.mkdtemp(prefix=f".{simulation_id}.previous.", dir=cases_dir))
    previous_dir.rmdir()
    final_case_dir.rename(previous_dir)
    try:
        case_dir.rename(final_case_dir)
    except OSError:
        previous_dir.rename(final_case_dir)
        raise
    shutil.rmtree(previous_dir, ignore_errors=True)


def _script(surface_file: str, config: CfdConfig) -> str:
    mesh_quality_check = "checkMesh -meshQuality"
    mesh_quality_block = f"""set +e
{mesh_quality_check} > ../logs/checkMesh_meshQuality.log 2>&1
mesh_quality_status=$?
set -e
cp ../logs/checkMesh_meshQuality.log ../logs/checkMesh_fatal.log
printf "%s\\n" "${{mesh_quality_status}}" > ../quality/checkMesh_meshQuality.returncode"""
    mesh_quality_failed_count = (
        "mesh_quality_failed_lines=$(grep -Eic 'Failed [0-9]+ mesh checks?' "
        "../logs/checkMesh_meshQuality.log || true)\n"
        "mesh_quality_failed_checks=$(awk '/Failed [0-9]+ mesh check/ {print $2}' "
        "../logs/checkMesh_meshQuality.log | tail -n 1)\n"
        "mesh_quality_ok_lines=$(grep -Eic 'Mesh OK' ../logs/checkMesh_meshQuality.log || true)"
    )
    mesh_quality_log_gate = "\n".join(
        [
            mesh_quality_failed_count,
            'mesh_quality_failed_checks="${mesh_quality_failed_checks:-0}"',
            'printf "%s\\n" "${mesh_quality_failed_lines}" '
            "> ../quality/checkMesh_meshQuality.failed_lines",
            (
                'printf "%s\\n" "${mesh_quality_failed_checks}" '
                "> ../quality/checkMesh_meshQuality.failed_checks"
            ),
            'printf "%s\\n" "${mesh_quality_ok_lines}" '
            "> ../quality/checkMesh_meshQuality.mesh_ok_lines",
            *(
                [
                    'if [[ "${mesh_quality_status}" -ne 0 ]]; then',
                    ('  echo "fatal checkMesh -meshQuality returned ${mesh_quality_status}" >&2'),
                    "  exit 3",
                    "fi",
                    'if [[ "${mesh_quality_failed_lines}" -gt 1 ]]; then',
                    (
                        '  echo "fatal checkMesh -meshQuality log has ambiguous '
                        '${mesh_quality_failed_lines} failed-check summaries" >&2'
                    ),
                    "  exit 3",
                    "fi",
                    'if [[ "${mesh_quality_failed_checks}" -gt 0 ]]; then',
                    (
                        '  echo "fatal checkMesh -meshQuality reported Failed '
                        '${mesh_quality_failed_checks} mesh checks" >&2'
                    ),
                    "  exit 3",
                    "fi",
                    (
                        'if [[ "${mesh_quality_failed_lines}" -eq 0 && '
                        '"${mesh_quality_ok_lines}" -eq 0 ]]; then'
                    ),
                    (
                        '  echo "fatal checkMesh -meshQuality log lacks Mesh OK '
                        'or Failed N mesh checks summary" >&2'
                    ),
                    "  exit 3",
                    "fi",
                ]
                if config.quality.mesh_quality_fatal
                else []
            ),
        ],
    )
    extended_check = (
        "checkMesh -allGeometry -allTopology -writeSurfaces -writeSets "
        "-surfaceFormat vtk -setFormat vtk"
    )
    if config.quality.extended_diagnostics_required:
        extended_check_block = (
            f"{extended_check} > ../logs/checkMesh.log 2>&1\n"
            'printf "%s\\n" "0" > ../quality/checkMesh_extended.returncode\n'
            'printf "%s\\n" "0" > ../quality/checkMesh_allGeometry.returncode'
            if config.quality.extended_diagnostics_fatal
            else f"""set +e
{extended_check} > ../logs/checkMesh.log 2>&1
extended_check_status=$?
set -e
printf "%s\\n" "${{extended_check_status}}" > ../quality/checkMesh_extended.returncode
printf "%s\\n" "${{extended_check_status}}" > ../quality/checkMesh_allGeometry.returncode"""
        )
    else:
        extended_check_block = (
            'printf "%s\\n" "SKIPPED" > ../quality/checkMesh_extended.returncode\n'
            'printf "%s\\n" "SKIPPED" > ../quality/checkMesh_allGeometry.returncode'
        )
    return f"""#!/usr/bin/env bash
set -eo pipefail

set +eu
if [[ ! -r /opt/openfoam13/etc/bashrc ]]; then
  echo "OpenFOAM v13 environment file not found: /opt/openfoam13/etc/bashrc" >&2
  exit 2
fi
source /opt/openfoam13/etc/bashrc
source_status=$?
if [[ "${{source_status}}" -ne 0 ]]; then
  echo "failed to source OpenFOAM v13 environment" >&2
  exit 2
fi
set -euo pipefail
cd "$(dirname "$0")/openfoam"

mkdir -p ../logs ../quality ../outputs

blockMesh > ../logs/blockMesh.log 2>&1
surfaceCheck constant/triSurface/{surface_file} > ../logs/surfaceCheck.log 2>&1
surfaceFeatures > ../logs/surfaceFeatures.log 2>&1
snappyHexMesh -overwrite > ../logs/snappyHexMesh.log 2>&1
{mesh_quality_block}
{mesh_quality_log_gate}
{extended_check_block}
foamRun -solver incompressibleFluid > ../logs/solver.log 2>&1
postProcess -func yPlus -latestTime > ../logs/yPlus.log 2>&1 || true
postProcess -func wallShearStress -latestTime > ../logs/wallShearStress.log 2>&1 || true
foamToVTK -latestTime > ../logs/foamToVTK.log 2>&1 || true
"""


def build_cfd_case(
    params: AeroParams,
    *,
    cases_dir: Path = Path("cases"),
    config: CfdConfig | None = None,
    overwrite_completed: bool = False,
) -> CfdCaseArtifacts:
    """Create the brief-defined case layout and OpenFOAM dictionaries."""

    config = config or CfdConfig()
    simulation_id = _simulation_id(params, config)
    cases_dir.mkdir(parents=True, exist_ok=True)
    final_case_dir = cases_dir / simulation_id
    lock_dir = cases_dir / f".{simulation_id}.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        message = f"CFD simulation {simulation_id} is already being built"
        raise FileExistsError(message) from exc

    staging_root = Path(tempfile.mkdtemp(prefix=f".{simulation_id}.staging.", dir=cases_dir))
    try:
        existing_status = _case_status(final_case_dir) if final_case_dir.exists() else None
        if (
            final_case_dir.exists()
            and existing_status != "BUILT_NOT_RUN"
            and not overwrite_completed
        ):
            message = (
                f"refusing to overwrite existing CFD simulation {simulation_id} "
                f"with status {existing_status or 'unknown'}"
            )
            raise FileExistsError(message)

        case_dir = staging_root / params.case_id()
        openfoam_dir, tri_surface_dir = _make_case_dirs(case_dir)

        geometry = generate_geometry(params, staging_root)
        cfd_surface = _prepare_cfd_surface(
            params=params,
            case_dir=case_dir,
            tri_surface_dir=tri_surface_dir,
            geometry=geometry,
            config=config,
        )
        _write_openfoam_case_files(
            params=params,
            config=config,
            case_dir=case_dir,
            openfoam_dir=openfoam_dir,
            surface_file=cfd_surface["openfoam_surface_file"],
        )
        _write_case_manifest(
            params=params,
            config=config,
            case_dir=case_dir,
            cfd_surface=cfd_surface,
            geometry=geometry,
            simulation_id=simulation_id,
        )
        _publish_case_dir(
            case_dir=case_dir,
            final_case_dir=final_case_dir,
            cases_dir=cases_dir,
            simulation_id=simulation_id,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        with suppress(FileNotFoundError):
            lock_dir.rmdir()
    final_openfoam_dir = final_case_dir / "openfoam"

    return CfdCaseArtifacts(
        case_id=params.case_id(),
        simulation_id=simulation_id,
        case_dir=final_case_dir,
        openfoam_dir=final_openfoam_dir,
        stl_path=final_case_dir / "geometry" / "article.stl",
        manifest_path=final_case_dir / "manifest.json",
        run_script_path=final_case_dir / "run_openfoam.sh",
    )


def manifest_summary(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"manifest is not a JSON object: {path}"
        raise TypeError(message)
    return loaded

"""AeroMap command-line interface."""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Annotated, Literal

import pyvista as pv
import trimesh
import typer
import yaml
from pydantic import ValidationError

from aeromap.benchmarks.aeromap import (
    AeroMapConfig,
    build_airfrans_geometry_dataset,
    build_airfrans_scalar_dataset,
    extract_airfrans_archive,
    write_active_learning_replay,
    write_aeromap_plan,
    write_airfrans_feasibility,
    write_airfrans_v02_audit,
    write_decision_replay_v02,
    write_decision_replay_v03,
    write_fixture_dataset,
    write_model_baselines_v02,
)
from aeromap.benchmarks.aeromap3d import (
    build_drivaerml_scalar_bridge_dataset,
    write_aeromap3d_metadata_triage,
    write_geometry_readiness_sample,
)
from aeromap.benchmarks.aeromap_cost import (
    write_cost_aware_decision_replay_v05,
    write_cost_aware_report,
    write_cost_proxy_audit,
    write_surface_field_feasibility_precheck,
)
from aeromap.benchmarks.airfrans_field import run_airfrans_surface_pressure_baseline
from aeromap.benchmarks.core_live_loop import (
    DEFAULT_DATASET_PATH as LIVE_CORE_DEFAULT_DATASET_PATH,
)
from aeromap.benchmarks.core_live_loop import (
    DEFAULT_INITIAL_CASES as LIVE_CORE_DEFAULT_INITIAL_CASES,
)
from aeromap.benchmarks.core_live_loop import (
    DEFAULT_OUTPUT_DIR as LIVE_CORE_DEFAULT_OUTPUT_DIR,
)
from aeromap.benchmarks.core_live_loop import (
    DEFAULT_REPORT_PATH as LIVE_CORE_DEFAULT_REPORT_PATH,
)
from aeromap.benchmarks.core_live_loop import (
    write_live_core_acquisition_loop,
)
from aeromap.cfd.alternative_mesher import (
    AlternativeVolumeMesherConfig,
    build_alternative_mesh_inputs,
)
from aeromap.cfd.case_builder import build_cfd_case
from aeromap.cfd.diagnostics import diagnose_mesh
from aeromap.cfd.patch_surface import article_patch_names
from aeromap.cfd.postprocess import postprocess_case
from aeromap.cfd.quality import (
    mesh_layer_coverage_from_vtk,
    parse_snappy_layer_log,
    parse_snappy_layer_retention_log,
)
from aeromap.cfd.reference_lane import run_reference_lane
from aeromap.cfd.region_mapping import RegionMappingError, map_surface_regions_to_vtp
from aeromap.cfd.runner import run_case
from aeromap.cfd.schema import CfdConfig
from aeromap.cfd.spatial_loads import (
    write_case_spatial_loads_report,
    write_urans_spatial_load_history,
)
from aeromap.cfd.steady_restart import (
    analyze_steady_restart_branch,
    compare_steady_restart_wall_series,
    prepare_steady_restart_branches,
    write_steady_restart_wall_series_report,
)
from aeromap.cfd.topology_report import write_topology_report
from aeromap.cfd.urans import (
    prepare_urans_audit,
    write_urans_checkpoint_report,
    write_urans_force_history_report,
    write_urans_retained_field_prune_manifest,
)
from aeromap.cfd.urans_analysis import write_urans_checkpoint_decision_report
from aeromap.cfd.urans_parallel import write_urans_parallel_benchmark_plan
from aeromap.cfd.venturi_core import (
    VenturiCoreConfig,
    build_venturi_core_case,
    write_venturi_core_case_metrics,
    write_venturi_core_design_report,
    write_venturi_core_grid_validation,
    write_venturi_core_wallshear_sign_audit,
)
from aeromap.data.converter import convert_case_to_sample
from aeromap.data.loader import TrainingEligibilityError, load_sample
from aeromap.geometry.diagnostics import diagnose_surface
from aeromap.geometry.evidence import build_geometry_evidence
from aeromap.geometry.generator import generate_geometry
from aeromap.geometry.surface_candidates import generate_surface_candidates
from aeromap.geometry.validate import validate_stl
from aeromap.io import atomic_write_json, atomic_write_text
from aeromap.parameters import AeroParams, corner_params, sobol_params
from aeromap.runtime.device import resolve_device

app = typer.Typer(no_args_is_help=True)
geometry_app = typer.Typer(no_args_is_help=True)
cfd_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
active_app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)

MIN_GEOMETRY_SMOKE_VALID_FRACTION = 0.95

app.add_typer(geometry_app, name="geometry", hidden=True)
app.add_typer(cfd_app, name="cfd", hidden=True)
app.add_typer(data_app, name="data", hidden=True)
app.add_typer(model_app, name="model", hidden=True)
app.add_typer(active_app, name="active", hidden=True)
app.add_typer(benchmark_app, name="benchmark")


def _params_from_options(
    geometry_family: Literal["advanced_challenge", "stable_reference"],
    ride_height_mm: float,
    pitch_deg: float,
    yaw_deg: float,
    throat_offset_mm: float,
    diffuser_angle_deg: float,
    edge_radius_mm: float,
) -> AeroParams:
    try:
        return AeroParams(
            geometry_family=geometry_family,
            ride_height_mm=ride_height_mm,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            throat_offset_mm=throat_offset_mm,
            diffuser_angle_deg=diffuser_angle_deg,
            edge_radius_mm=edge_radius_mm,
        )
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def doctor() -> None:
    """Report local resource availability without weakening gate requirements."""

    generic_device = resolve_device("auto")
    cuda_status: str
    try:
        cuda_status = resolve_device("auto", require_cuda=True).resolved
    except RuntimeError as exc:
        cuda_status = f"blocked: {exc}"

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uv": shutil.which("uv") or "",
        "docker": shutil.which("docker") or "",
        "foamRun": shutil.which("foamRun") or "",
        "nvidia-smi": shutil.which("nvidia-smi") or "",
        "generic_torch_device": generic_device.__dict__,
        "optional_cuda_device": cuda_status,
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@geometry_app.command("generate")
def geometry_generate(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("artifacts/geometry"),
    geometry_family: Annotated[
        Literal["advanced_challenge", "stable_reference"],
        typer.Option("--geometry-family"),
    ] = "advanced_challenge",
    ride_height_mm: float = 40.0,
    pitch_deg: float = 0.4,
    yaw_deg: float = 0.0,
    throat_offset_mm: float = 35.0,
    diffuser_angle_deg: float = 1.25,
    edge_radius_mm: float = 12.0,
) -> None:
    """Generate one transformed article geometry."""

    params = _params_from_options(
        geometry_family,
        ride_height_mm,
        pitch_deg,
        yaw_deg,
        throat_offset_mm,
        diffuser_angle_deg,
        edge_radius_mm,
    )
    artifacts = generate_geometry(params, out)
    typer.echo(artifacts.model_dump_json(indent=2))
    if not artifacts.validation.valid:
        raise typer.Exit(2)


@geometry_app.command("validate")
def geometry_validate(stl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate an STL with the current geometry gates."""

    validation = validate_stl(stl)
    typer.echo(validation.model_dump_json(indent=2))
    if not validation.valid:
        raise typer.Exit(2)


@geometry_app.command("diagnose-surface")
def geometry_diagnose_surface(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/gate2/attempts",
    ),
    parent_attempt_id: Annotated[str | None, typer.Option("--parent-attempt-id")] = None,
    openfoam_image_digest: Annotated[str, typer.Option("--openfoam-image-digest")] = "unknown",
) -> None:
    """Diagnose CAD/BRep and STL surface quality without changing the geometry."""

    artifacts = diagnose_surface(
        params=AeroParams.canonical(),
        attempts_dir=out,
        parent_attempt_id=parent_attempt_id,
        openfoam_image_digest=openfoam_image_digest,
    )
    typer.echo(
        json.dumps(
            {
                "attempt_id": artifacts.attempt_id,
                "attempt_dir": str(artifacts.attempt_dir),
                "attempt_manifest_path": str(artifacts.attempt_manifest_path),
                "metrics_path": str(artifacts.metrics_path),
                "cad_faces_vtp_path": str(artifacts.cad_faces_vtp_path),
                "stl_triangles_vtp_path": str(artifacts.stl_triangles_vtp_path),
                "bad_triangles_csv_path": str(artifacts.bad_triangles_csv_path),
            },
            indent=2,
            sort_keys=True,
        ),
    )


@geometry_app.command("surface-candidates")
def geometry_surface_candidates(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/gate2/surface_candidates",
    ),
    *,
    include_gmsh: Annotated[bool, typer.Option("--include-gmsh/--skip-gmsh")] = True,
    gmsh_path: Annotated[Path | None, typer.Option("--gmsh-path", dir_okay=False)] = None,
    include_cadquery_matrix: Annotated[
        bool,
        typer.Option("--include-cadquery-matrix/--skip-cadquery-matrix"),
    ] = True,
) -> None:
    """Generate bounded surface-export candidates without changing CAD."""

    matrix = generate_surface_candidates(
        params=AeroParams.canonical(),
        out_dir=out,
        include_gmsh=include_gmsh,
        gmsh_path=gmsh_path,
        include_cadquery_fixed_matrix=include_cadquery_matrix,
    )
    typer.echo(
        json.dumps(
            {
                "manifest_path": str(matrix.manifest_path),
                "out_dir": str(matrix.out_dir),
                "candidates": [
                    {
                        "candidate_id": item.candidate_id,
                        "candidate_dir": str(item.candidate_dir),
                        "status": item.status,
                        "stl_path": str(item.stl_path) if item.stl_path is not None else None,
                        "metrics_path": str(item.metrics_path),
                        "regions_json_path": (
                            str(item.regions_json_path)
                            if getattr(item, "regions_json_path", None) is not None
                            else None
                        ),
                        "regions_vtp_path": (
                            str(item.regions_vtp_path)
                            if getattr(item, "regions_vtp_path", None) is not None
                            else None
                        ),
                    }
                    for item in matrix.candidates
                ],
            },
            indent=2,
            sort_keys=True,
        ),
    )


@geometry_app.command("preview")
def geometry_preview(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("artifacts/geometry"),
) -> None:
    """Generate the canonical article and print preview paths."""

    artifacts = generate_geometry(AeroParams.canonical(), out)
    typer.echo(str(artifacts.preview_html_path))


@geometry_app.command("smoke")
def geometry_smoke(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("artifacts/geometry_smoke"),
    samples: Annotated[int, typer.Option("--samples", min=1)] = 100,
    seed: int = 1729,
) -> None:
    """Generate the canonical article and deterministic Sobol variants."""

    out.mkdir(parents=True, exist_ok=True)
    generated = [generate_geometry(AeroParams.canonical(), out)]
    generated.extend(generate_geometry(params, out) for params in corner_params().values())
    generated.extend(generate_geometry(params, out) for params in sobol_params(samples, seed=seed))

    valid_count = sum(item.validation.valid for item in generated)
    valid_fraction = valid_count / len(generated)
    summary = {
        "samples_requested": samples,
        "corner_cases_generated": len(corner_params()),
        "geometries_generated": len(generated),
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
        "canonical_case_id": generated[0].case_id,
        "canonical_stl": str(generated[0].stl_path),
        "canonical_preview_html": str(generated[0].preview_html_path),
        "invalid": [
            {"case_id": item.case_id, "reasons": item.validation.reasons}
            for item in generated
            if not item.validation.valid
        ],
    }
    atomic_write_json(out / "geometry_smoke_summary.json", summary)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if valid_fraction < MIN_GEOMETRY_SMOKE_VALID_FRACTION:
        raise typer.Exit(2)


@geometry_app.command("evidence")
def geometry_evidence(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/geometry/evidence",
    ),
    samples: Annotated[int, typer.Option("--samples", min=1)] = 100,
    seed: int = 1729,
) -> None:
    """Build compact committed geometry evidence."""

    summary = build_geometry_evidence(out, samples=samples, seed=seed)
    typer.echo(
        json.dumps(
            {
                "canonical_case_id": summary["canonical_case_id"],
                "evidence_dir": str(out),
                "geometries_generated": summary["geometries_generated"],
                "invalid_count": len(summary["invalid"]),
                "summary_path": str(out / "geometry_smoke_summary.json"),
                "valid_count": summary["valid_count"],
                "valid_fraction": summary["valid_fraction"],
            },
            indent=2,
            sort_keys=True,
        ),
    )
    if summary["invalid"]:
        raise typer.Exit(2)


def _blocked(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(2)


def _requested_device(device: str) -> str:
    return os.environ.get("AEROMAP_DEVICE", device).lower()


def _require_cuda_workload(device: str) -> None:
    request = _requested_device(device)
    if request != "cuda":
        return
    try:
        spec = resolve_device("cuda", require_cuda=True)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if spec.resolved != "cuda":
        typer.echo("CUDA workload requested but resolver did not return cuda", err=True)
        raise typer.Exit(2)


def _load_cfd_config(config_path: Path | None) -> CfdConfig:
    if config_path is None:
        return CfdConfig()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        typer.echo(f"CFD config YAML is invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(loaded, dict):
        typer.echo(f"CFD config must be a YAML mapping: {config_path}", err=True)
        raise typer.Exit(2)
    try:
        return CfdConfig.model_validate(loaded)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def _load_venturi_core_config(config_path: Path | None) -> VenturiCoreConfig:
    if config_path is None:
        return VenturiCoreConfig()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        typer.echo(f"Venturi Core config YAML is invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(loaded, dict):
        typer.echo(f"Venturi Core config must be a YAML mapping: {config_path}", err=True)
        raise typer.Exit(2)
    try:
        return VenturiCoreConfig.model_validate(loaded)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


def _load_alt_mesh_config(config_path: Path | None) -> AlternativeVolumeMesherConfig:
    if config_path is None:
        return AlternativeVolumeMesherConfig()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        typer.echo(f"Alternative mesher YAML is invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(loaded, dict):
        typer.echo(f"Alternative mesher config must be a YAML mapping: {config_path}", err=True)
        raise typer.Exit(2)
    try:
        return AlternativeVolumeMesherConfig.model_validate(loaded)
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@cfd_app.command("build")
def cfd_build(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("cases"),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False),
    ] = None,
    geometry_family: Annotated[
        Literal["advanced_challenge", "stable_reference"],
        typer.Option("--geometry-family"),
    ] = "advanced_challenge",
    ride_height_mm: float = 40.0,
    pitch_deg: float = 0.4,
    yaw_deg: float = 0.0,
    throat_offset_mm: float = 35.0,
    diffuser_angle_deg: float = 1.25,
    edge_radius_mm: float = 12.0,
) -> None:
    params = _params_from_options(
        geometry_family,
        ride_height_mm,
        pitch_deg,
        yaw_deg,
        throat_offset_mm,
        diffuser_angle_deg,
        edge_radius_mm,
    )
    artifacts = build_cfd_case(params, cases_dir=out, config=_load_cfd_config(config_path))
    typer.echo(artifacts.model_dump_json(indent=2))


@cfd_app.command("build-venturi-core")
def cfd_build_venturi_core(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/venturi_core/cases",
    ),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False),
    ] = None,
    *,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Build a structured Venturi Core / Venturi Lab case without snappyHexMesh."""

    try:
        artifacts = build_venturi_core_case(
            _load_venturi_core_config(config_path),
            cases_dir=out,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "case_id": artifacts.case_id,
                "case_dir": str(artifacts.case_dir),
                "manifest_path": str(artifacts.manifest_path),
                "profile_path": str(artifacts.profile_path),
                "run_mesh_script_path": str(artifacts.run_mesh_script_path),
                "run_solver_script_path": str(artifacts.run_solver_script_path),
                "status": "BUILT_NOT_RUN",
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("venturi-core-report")
def cfd_venturi_core_report(
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/reports/venturi_core_venturi_lab.md",
    ),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False),
    ] = Path("configs/cfd/venturi_core_canonical_medium.yaml"),
) -> None:
    """Write the Venturi Core design and claim-boundary report."""

    report_path = write_venturi_core_design_report(
        config=_load_venturi_core_config(config_path),
        out=out,
    )
    typer.echo(json.dumps({"report_path": str(report_path)}, indent=2, sort_keys=True))


@cfd_app.command("postprocess-venturi-core")
def cfd_postprocess_venturi_core(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
) -> None:
    """Extract compact pressure/load/cliff metrics from a completed Core case."""

    try:
        metrics_path = write_venturi_core_case_metrics(case_dir, out=out)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({"metrics_path": str(metrics_path)}, indent=2, sort_keys=True))


@cfd_app.command("validate-venturi-core-grids")
def cfd_validate_venturi_core_grids(
    coarse_case: Annotated[Path, typer.Option("--coarse", exists=True, file_okay=False)],
    medium_case: Annotated[Path, typer.Option("--medium", exists=True, file_okay=False)],
    fine_case: Annotated[Path, typer.Option("--fine", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/cfd/venturi_core/canonical_three_grid_validation.json",
    ),
) -> None:
    """Validate the canonical Core anchor against a coarse/medium/fine grid family."""

    try:
        validation_path = write_venturi_core_grid_validation(
            coarse_case=coarse_case,
            medium_case=medium_case,
            fine_case=fine_case,
            out=out,
        )
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({"validation_path": str(validation_path)}, indent=2, sort_keys=True))


@cfd_app.command("audit-venturi-core-wallshear")
def cfd_audit_venturi_core_wallshear(
    coarse_case: Annotated[Path, typer.Option("--coarse", exists=True, file_okay=False)],
    medium_case: Annotated[Path, typer.Option("--medium", exists=True, file_okay=False)],
    fine_case: Annotated[Path, typer.Option("--fine", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/cfd/venturi_core/wallshear_sign_convention_audit.json",
    ),
) -> None:
    """Audit Core wallShearStress sign convention against near-wall velocity."""

    try:
        audit_path = write_venturi_core_wallshear_sign_audit(
            coarse_case=coarse_case,
            medium_case=medium_case,
            fine_case=fine_case,
            out=out,
        )
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({"audit_path": str(audit_path)}, indent=2, sort_keys=True))


@cfd_app.command("alt-mesh-inputs")
def cfd_alt_mesh_inputs(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/campaign/alt_mesh_medium",
    ),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Write Gmsh/OpenCASCADE campaign fallback inputs without running a heavy mesh."""

    artifacts = build_alternative_mesh_inputs(
        out_dir=out,
        config=_load_alt_mesh_config(config_path),
    )
    typer.echo(artifacts.model_dump_json(indent=2))


@cfd_app.command("run")
def cfd_run(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    try:
        return_code = run_case(case_dir, dry_run=dry_run)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if return_code != 0:
        raise typer.Exit(return_code)


@cfd_app.command("validate")
def cfd_validate() -> None:
    _blocked("CFD validation requires a real OpenFOAM case artifact.")


@cfd_app.command("postprocess")
def cfd_postprocess(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Parse a completed OpenFOAM smoke case into CFD quality/output artifacts."""

    artifacts = postprocess_case(case_dir)
    typer.echo(json.dumps({key: str(value) for key, value in artifacts.__dict__.items()}, indent=2))


@cfd_app.command("prepare-urans-audit")
def cfd_prepare_urans_audit(
    source_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/campaign/urans_audit_plans",
    ),
    audit_purpose: Annotated[str, typer.Option("--audit-purpose")] = "general_reconnaissance",
    steady_time: Annotated[str | None, typer.Option("--steady-time")] = None,
    initial_delta_t_s: Annotated[float, typer.Option("--initial-delta-t-s", min=0.0)] = 1.0e-5,
    max_delta_t_s: Annotated[float, typer.Option("--max-delta-t-s", min=0.0)] = 2.5e-5,
    end_time_s: Annotated[float, typer.Option("--end-time-s", min=0.0)] = 0.02,
    write_interval_s: Annotated[float, typer.Option("--write-interval-s", min=0.0)] = 5.0e-4,
    purge_write: Annotated[int, typer.Option("--purge-write", min=0)] = 0,
    write_compression: Annotated[  # noqa: FBT002 - Typer boolean flag default.
        bool,
        typer.Option("--write-compression/--no-write-compression"),
    ] = False,
    max_co: Annotated[float, typer.Option("--max-co", min=0.0)] = 1.0,
) -> None:
    """Prepare, but do not run, a bounded URANS audit from a steady case."""

    try:
        artifacts = prepare_urans_audit(
            source_case=source_case,
            out_dir=out,
            audit_purpose=audit_purpose,
            steady_time=steady_time,
            initial_delta_t_s=initial_delta_t_s,
            max_delta_t_s=max_delta_t_s,
            end_time_s=end_time_s,
            write_interval_s=write_interval_s,
            purge_write=purge_write,
            write_compression=write_compression,
            max_co=max_co,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "audit_id": artifacts.audit_id,
                "audit_dir": str(artifacts.audit_dir),
                "manifest_path": str(artifacts.manifest_path),
                "control_dict_path": str(artifacts.control_dict_path),
                "fv_schemes_path": str(artifacts.fv_schemes_path),
                "fv_solution_path": str(artifacts.fv_solution_path),
                "run_script_path": str(artifacts.run_script_path),
                "status": "PREPARED_NOT_RUN",
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("merge-urans-force-history")
def cfd_merge_urans_force_history(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    max_time_s: Annotated[float | None, typer.Option("--max-time-s", min=0.0)] = None,
) -> None:
    """Merge restart-overlapping URANS forces with provenance."""

    try:
        artifacts = write_urans_force_history_report(
            work_case,
            out_json=out,
            max_time_s=max_time_s,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({"report_path": str(artifacts.report_path)}, indent=2, sort_keys=True))


@cfd_app.command("summarize-urans-checkpoint")
def cfd_summarize_urans_checkpoint(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/cfd/urans/medium_urans_resume_checkpoint.json",
    ),
    planned_end_time_s: Annotated[float, typer.Option("--planned-end-time-s", min=0.0)] = 0.12,
    source_case: Annotated[Path | None, typer.Option("--source-case", file_okay=False)] = None,
    audit_id: Annotated[str | None, typer.Option("--audit-id")] = None,
    audit_purpose: Annotated[str, typer.Option("--audit-purpose")] = "medium_reconnaissance",
) -> None:
    """Summarize retained URANS restart fields without asserting acceptance."""

    try:
        artifacts = write_urans_checkpoint_report(
            work_case,
            out_json=out,
            planned_end_time_s=planned_end_time_s,
            source_case=source_case,
            audit_id=audit_id,
            audit_purpose=audit_purpose,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "report_path": str(artifacts.report_path),
                "latest_complete_time_s": artifacts.latest_complete_time_s,
                "accepted": False,
                "training_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("analyze-urans-checkpoint")
def cfd_analyze_urans_checkpoint(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/cfd/urans/urans_checkpoint_decision.json",
    ),
    force_history: Annotated[Path | None, typer.Option("--force-history", dir_okay=False)] = None,
    spatial_history: Annotated[
        Path | None,
        typer.Option("--spatial-history", dir_okay=False),
    ] = None,
    checkpoint_report: Annotated[
        Path | None,
        typer.Option("--checkpoint-report", dir_okay=False),
    ] = None,
    prune_manifest: Annotated[Path | None, typer.Option("--prune-manifest", dir_okay=False)] = None,
    analysis_start_s: Annotated[float, typer.Option("--analysis-start-s", min=0.0)] = 0.024,
    window_split_s: Annotated[float, typer.Option("--window-split-s", min=0.0)] = 0.044,
    analysis_end_s: Annotated[float, typer.Option("--analysis-end-s", min=0.0)] = 0.064,
) -> None:
    """Analyze a partial URANS checkpoint before any further continuation."""

    try:
        artifacts = write_urans_checkpoint_decision_report(
            work_case=work_case,
            out_json=out,
            force_history_path=force_history,
            spatial_history_path=spatial_history,
            checkpoint_report_path=checkpoint_report,
            prune_manifest_path=prune_manifest,
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
            window_split_s=window_split_s,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "report_path": str(artifacts.report_path),
                "classification": artifacts.classification,
                "accepted": False,
                "training_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("prepare-urans-parallel-benchmark")
def cfd_prepare_urans_parallel_benchmark(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/campaign/urans_parallel_benchmarks/gate2g_0p064",
    ),
    checkpoint_report: Annotated[
        Path,
        typer.Option("--checkpoint-report", exists=True, dir_okay=False),
    ] = Path("artifacts/cfd/urans/medium_urans_resume_checkpoint.json"),
    ranks: Annotated[
        list[int] | None,
        typer.Option("--rank", help="MPI rank count to prepare; repeatable."),
    ] = None,
    continuation_s: Annotated[float, typer.Option("--continuation-s", min=0.0)] = 0.002,
) -> None:
    """Prepare disposable short OpenFOAM MPI benchmark continuations without running them."""

    try:
        artifacts = write_urans_parallel_benchmark_plan(
            work_case=work_case,
            out_dir=out,
            checkpoint_report_path=checkpoint_report,
            ranks=tuple(ranks or [1, 4, 8, 16]),
            continuation_s=continuation_s,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "manifest_path": str(artifacts.manifest_path),
                "run_scripts": [str(path) for path in artifacts.run_scripts],
                "status": "PREPARED_NOT_RUN",
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("prune-urans-retained-fields")
def cfd_prune_urans_retained_fields(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    keep_latest: Annotated[int, typer.Option("--keep-latest", min=1)] = 8,
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    apply: Annotated[  # noqa: FBT002 - Typer boolean flag default.
        bool,
        typer.Option("--apply/--dry-run", help="Actually delete candidate field directories."),
    ] = False,
) -> None:
    """Plan or apply pruning of old retained URANS field directories."""

    try:
        artifacts = write_urans_retained_field_prune_manifest(
            work_case,
            keep_latest=keep_latest,
            out_json=out,
            dry_run=not apply,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "manifest_path": str(artifacts.manifest_path),
                "dry_run": artifacts.dry_run,
                "candidate_count": artifacts.candidate_count,
                "candidate_bytes": artifacts.candidate_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _params_from_case(case_dir: Path) -> AeroParams:
    manifest = case_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return AeroParams.model_validate(payload["params"])
    params_json = case_dir / "params.json"
    if params_json.exists():
        return AeroParams.model_validate_json(params_json.read_text(encoding="utf-8"))
    msg = f"could not find AeroCliff params in {case_dir}"
    raise FileNotFoundError(msg)


def _params_for_urans_work_case(work_case: Path, source_case: Path | None) -> AeroParams:
    if source_case is not None:
        return _params_from_case(source_case)
    manifest = work_case / "urans_audit_manifest.json"
    if not manifest.exists():
        msg = f"source case is required when the URANS work case lacks {manifest.name}: {work_case}"
        raise FileNotFoundError(msg)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_path = payload.get("source", {}).get("case_dir")
    if not source_path:
        msg = f"URANS audit manifest lacks source.case_dir: {manifest}"
        raise ValueError(msg)
    return _params_from_case(Path(str(source_path)))


@cfd_app.command("extract-urans-spatial-load-history")
def cfd_extract_urans_spatial_load_history(
    work_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    source_case: Annotated[Path | None, typer.Option("--source-case", file_okay=False)] = None,
    time_dirs: Annotated[
        list[str] | None,
        typer.Option("--time-dir", help="Specific OpenFOAM time directory to extract."),
    ] = None,
    patches: Annotated[
        list[str] | None,
        typer.Option("--patch", help="Article wall patch to include; repeatable."),
    ] = None,
    streamwise_bins: Annotated[int, typer.Option("--streamwise-bins", min=1)] = 16,
) -> None:
    """Extract compact left/right and streamwise URANS wall-load diagnostics."""

    try:
        params = _params_for_urans_work_case(work_case, source_case)
        report = write_urans_spatial_load_history(
            work_case=work_case,
            params=params,
            out_json=out,
            time_dirs=time_dirs,
            patches=patches,
            streamwise_bins=streamwise_bins,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "report_path": str(
                    out or work_case / "quality" / "urans_spatial_load_history.json"
                ),
                "row_count": report["row_count"],
                "accepted": False,
                "training_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("prepare-steady-restarts")
def cfd_prepare_steady_restarts(
    source_case: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/campaign/steady_restart_plans",
    ),
    steady_time: Annotated[str | None, typer.Option("--steady-time")] = None,
    iterations: Annotated[int, typer.Option("--iterations", min=1)] = 120,
    write_interval: Annotated[int, typer.Option("--write-interval", min=1)] = 5,
    momentum_relaxation: Annotated[
        float,
        typer.Option("--momentum-relaxation", min=0.0, max=1.0),
    ] = 0.70,
) -> None:
    """Prepare steady restart branches without running them."""

    try:
        artifacts = prepare_steady_restart_branches(
            source_case=source_case,
            out_dir=out,
            steady_time=steady_time,
            iterations=iterations,
            write_interval=write_interval,
            momentum_relaxation=momentum_relaxation,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "plan_id": artifacts.plan_id,
                "plan_dir": str(artifacts.plan_dir),
                "status": "PREPARED_NOT_RUN",
                "branches": {
                    branch.branch_name: {
                        "manifest_path": str(branch.manifest_path),
                        "run_script_path": str(branch.run_script_path),
                    }
                    for branch in artifacts.branches
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("analyze-steady-restart")
def cfd_analyze_steady_restart(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
) -> None:
    """Summarize one completed steady restart branch."""

    try:
        report = analyze_steady_restart_branch(case_dir, out_json=out)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    cdf = report["steady_diagnostics"]["coefficients"]["c_df"]["final_window"]
    typer.echo(
        json.dumps(
            {
                "status": report["status"],
                "case_dir": str(case_dir),
                "branch_name": report["branch_name"],
                "completed_iterations": report["time_window"][
                    "completed_force_coefficient_iterations"
                ],
                "c_df_final_window": cdf,
                "training_eligible": report["training_eligible"],
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("steady-restart-wall-series")
def cfd_steady_restart_wall_series(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    times: Annotated[
        str | None,
        typer.Option(
            "--times",
            help="Comma-separated OpenFOAM time names. Defaults to discovered wall VTK times.",
        ),
    ] = None,
    streamwise_bins: Annotated[int, typer.Option("--streamwise-bins", min=1, max=20)] = 16,
) -> None:
    """Map and integrate a restart wall-field time series."""

    time_tuple = tuple(item.strip() for item in times.split(",") if item.strip()) if times else None
    try:
        report = write_steady_restart_wall_series_report(
            case_dir=case_dir,
            out_json=out,
            times=time_tuple,
            streamwise_bins=streamwise_bins,
        )
    except (FileNotFoundError, KeyError, RegionMappingError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "status": "OK",
                "case_dir": str(case_dir),
                "branch_name": report["branch_name"],
                "sample_count": len(report["samples"]),
                "mapping_min_area_coverage": report["mapping_summary"]["min_area_coverage"],
                "left_right_phase": report["phase_summary"]["left_right_c_df_phase"],
                "training_eligible": report["training_eligible"],
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("compare-steady-restart-wall-series")
def cfd_compare_steady_restart_wall_series(
    reports: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[
        Path,
        typer.Option("--out", dir_okay=False),
    ] = Path("artifacts/cfd/urans/wall_phase_comparison.json"),
) -> None:
    """Compare restart wall-series diagnostics across bounded steady branches."""

    try:
        comparison = compare_steady_restart_wall_series(
            reports=tuple(reports),
            out_json=out,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "status": "OK",
                "classification": comparison["classification"],
                "reports": comparison["reports"],
                "training_eligible": comparison["training_eligible"],
                "out": str(out),
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("spatial-loads")
def cfd_spatial_loads(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    latest_time: Annotated[str | None, typer.Option("--latest-time")] = None,
    streamwise_bins: Annotated[int, typer.Option("--streamwise-bins", min=1, max=20)] = 16,
) -> None:
    """Integrate offline regional and streamwise wall loads from mapped wall fields."""

    try:
        report = write_case_spatial_loads_report(
            case_dir=case_dir,
            out_json=out,
            latest_time=latest_time,
            streamwise_bins=streamwise_bins,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "status": "OK",
                "case_dir": str(case_dir),
                "source_wall_vtp": report["source_wall_vtp"],
                "cell_count": report["cell_count"],
                "streamwise_bins": report["streamwise_bins"]["bin_count"],
                "total_coefficients": report["total"]["coefficients"],
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("layer-report")
def cfd_layer_report(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
) -> None:
    """Parse snappyHexMesh layer-attempt evidence into a compact JSON report."""

    log_path = case_dir / "logs" / "snappyHexMesh.log"
    if not log_path.exists():
        typer.echo(f"missing snappyHexMesh log: {log_path}", err=True)
        raise typer.Exit(2)
    report = parse_snappy_layer_log(log_path)
    out_path = out or case_dir / "quality" / "layer_attempt.json"
    atomic_write_json(out_path, report)
    typer.echo(json.dumps({"layer_attempt": str(out_path), **report}, indent=2, sort_keys=True))


@cfd_app.command("layer-retention-report")
def cfd_layer_retention_report(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    layer_coverage: Annotated[
        Path | None,
        typer.Option("--layer-coverage", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Parse per-iteration snappy layer retention into compact JSON evidence."""

    log_path = case_dir / "logs" / "snappyHexMesh.log"
    if not log_path.exists():
        typer.echo(f"missing snappyHexMesh log: {log_path}", err=True)
        raise typer.Exit(2)
    report = parse_snappy_layer_retention_log(log_path)
    coverage_path = layer_coverage or case_dir / "quality" / "layer_coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        report["final_area_coverage"] = {
            "source": str(coverage_path),
            "critical_area_coverage": coverage.get("critical_area_coverage", {}),
            "critical_area_coverage_ok": coverage.get("critical_area_coverage_ok"),
        }
    out_path = out or case_dir / "quality" / "layer_retention.json"
    atomic_write_json(out_path, report)
    typer.echo(json.dumps({"layer_retention": str(out_path), **report}, indent=2, sort_keys=True))


@cfd_app.command("layer-coverage")
def cfd_layer_coverage(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
    time_name: Annotated[str, typer.Option("--time-name")] = "0",
) -> None:
    """Report area-weighted layer coverage from foamToVTK boundary patch outputs."""

    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"missing manifest: {manifest_path}", err=True)
        raise typer.Exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = CfdConfig.model_validate(manifest["cfd_config"])
    report = mesh_layer_coverage_from_vtk(
        case_dir=case_dir,
        patch_names=article_patch_names(
            patch_mode=config.surface_export.openfoam_patch_mode,
        ),
        time_name=time_name,
    )
    out_path = out or case_dir / "quality" / "layer_coverage.json"
    atomic_write_json(out_path, report)
    typer.echo(json.dumps({"layer_coverage": str(out_path), **report}, indent=2, sort_keys=True))


@cfd_app.command("map-regions")
def cfd_map_regions(
    source_stl: Annotated[Path, typer.Option("--source-stl", exists=True, dir_okay=False)],
    source_regions: Annotated[
        Path,
        typer.Option("--source-regions", exists=True, dir_okay=False),
    ],
    target_vtp: Annotated[Path, typer.Option("--target-vtp", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "outputs/article_wall_regions.vtp",
    ),
    report: Annotated[Path, typer.Option("--report", dir_okay=False)] = Path(
        "quality/region_mapping.json",
    ),
    max_distance_m: Annotated[float, typer.Option("--max-distance-m", min=0.0)] = 0.005,
) -> None:
    """Map source STL region IDs onto an exported post-mesh vehicle-wall VTP."""

    source_mesh = trimesh.load_mesh(source_stl, process=True)
    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh = trimesh.util.concatenate(tuple(source_mesh.geometry.values()))
    regions = json.loads(source_regions.read_text(encoding="utf-8"))
    target_surface = pv.read(target_vtp)
    try:
        result = map_surface_regions_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=target_surface,
            output_vtp_path=out,
            report_path=report,
            max_distance_m=max_distance_m,
        )
    except RegionMappingError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))


@cfd_app.command("diagnose-mesh")
def cfd_diagnose_mesh(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/gate2/attempts",
    ),
    parent_attempt_id: Annotated[str | None, typer.Option("--parent-attempt-id")] = None,
    openfoam_image_digest: Annotated[str, typer.Option("--openfoam-image-digest")] = "unknown",
    near_feature_distance_m: Annotated[
        float,
        typer.Option("--near-feature-distance-m", min=0.0),
    ] = 0.01,
    target_cell_width_m: Annotated[
        float,
        typer.Option("--target-cell-width-m", min=0.0),
    ] = 0.012,
) -> None:
    """Map checkMesh problem sets back to STL triangles, regions, and CAD faces."""

    try:
        artifacts = diagnose_mesh(
            case_dir=case_dir,
            attempts_dir=out,
            parent_attempt_id=parent_attempt_id,
            openfoam_image_digest=openfoam_image_digest,
            near_feature_distance_m=near_feature_distance_m,
            target_cell_width_m=target_cell_width_m,
        )
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "attempt_id": artifacts.attempt_id,
                "attempt_dir": str(artifacts.attempt_dir),
                "attempt_manifest_path": str(artifacts.attempt_manifest_path),
                "metrics_path": str(artifacts.metrics_path),
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("reference-lane")
def cfd_reference_lane(
    case_name: Annotated[
        str,
        typer.Option("--case", help="Official OpenFOAM v13 tutorial case name."),
    ] = "drivaerFastback",
    mode: Annotated[
        str,
        typer.Option("--mode", help="inspect, block-mesh, mesh, or solve."),
    ] = "inspect",
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/gate2/reference_lane",
    ),
    image: Annotated[str, typer.Option("--image")] = "aeromap/openfoam13:dev",
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run a non-headline official OpenFOAM v13 reference case lane."""

    reference_out = out / case_name
    try:
        result = run_reference_lane(
            case_name=case_name,
            mode=mode,
            out_dir=reference_out,
            image=image,
            dry_run=dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    typer.echo(
        json.dumps(
            {
                "case_name": result.case_name,
                "mode": result.mode,
                "out_dir": str(result.out_dir),
                "summary_path": str(result.summary_path) if result.summary_path else None,
                "return_code": result.return_code,
                "dry_run": dry_run,
                "non_headline": True,
            },
            indent=2,
            sort_keys=True,
        ),
    )
    if result.return_code is not None and result.return_code != 0:
        raise typer.Exit(result.return_code)


@cfd_app.command("topology-report")
def cfd_topology_report(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    surface_diagnostics: Annotated[
        Path,
        typer.Option("--surface-diagnostics", exists=True, dir_okay=False),
    ],
    mesh_diagnostics: Annotated[
        Path,
        typer.Option("--mesh-diagnostics", exists=True, dir_okay=False),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path(
        "artifacts/gate2/topology_report",
    ),
) -> None:
    """Summarise surface topology evidence before Gmsh or CAD changes."""

    artifacts = write_topology_report(
        case_dir=case_dir,
        surface_diagnostics_path=surface_diagnostics,
        mesh_diagnostics_path=mesh_diagnostics,
        out_dir=out,
    )
    typer.echo(
        json.dumps(
            {
                "report_json_path": str(artifacts.report_json_path),
                "report_markdown_path": str(artifacts.report_markdown_path),
            },
            indent=2,
            sort_keys=True,
        ),
    )


@cfd_app.command("smoke")
def cfd_smoke(
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("cases"),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False),
    ] = None,
    case_name: Annotated[str, typer.Option("--case-name")] = "canonical",
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    if case_name == "canonical":
        params = AeroParams.canonical()
    else:
        corners = corner_params()
        if case_name not in corners:
            typer.echo(
                f"Unknown case name {case_name!r}; expected canonical or one of {sorted(corners)}",
                err=True,
            )
            raise typer.Exit(2)
        params = corners[case_name]
    artifacts = build_cfd_case(params, cases_dir=out, config=_load_cfd_config(config_path))
    typer.echo(artifacts.model_dump_json(indent=2))
    try:
        return_code = run_case(artifacts.case_dir, dry_run=dry_run)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if return_code != 0:
        raise typer.Exit(return_code)


@data_app.command("register")
def data_register(
    sample_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    registry: Annotated[Path, typer.Option("--registry", dir_okay=False)] = Path(
        "data/registry.jsonl",
    ),
    *,
    allow_non_campaign: Annotated[bool, typer.Option("--allow-non-campaign")] = False,
) -> None:
    """Append a converted sample manifest to a compact JSONL registry."""

    try:
        sample = load_sample(sample_dir, allow_non_campaign=allow_non_campaign)
    except (TrainingEligibilityError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    registry.parent.mkdir(parents=True, exist_ok=True)
    existing_entries = []
    try:
        if registry.exists():
            for line in registry.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                existing = json.loads(line)
                if existing.get("sample_id") != sample.manifest.sample_id:
                    existing_entries.append(json.dumps(existing, sort_keys=True))
    except json.JSONDecodeError as exc:
        typer.echo(f"Registry JSONL is invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    existing_entries.append(sample.manifest.model_dump_json())
    atomic_write_text(registry, "\n".join(existing_entries) + "\n")
    typer.echo(
        json.dumps(
            {
                "registry": str(registry),
                "sample_id": sample.manifest.sample_id,
                "case_class": sample.manifest.case_class,
                "training_eligible": sample.manifest.training_eligible,
            },
            indent=2,
            sort_keys=True,
        ),
    )


@data_app.command("convert")
def data_convert(
    case_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("data/interim"),
) -> None:
    """Convert one post-processed OpenFOAM case into an immutable sample."""

    try:
        artifacts = convert_case_to_sample(case_dir, out)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(artifacts.model_dump_json(indent=2))


@data_app.command("validate")
def data_validate(
    sample_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    *,
    allow_non_campaign: Annotated[bool, typer.Option("--allow-non-campaign")] = False,
) -> None:
    """Validate and load one converted sample."""

    try:
        sample = load_sample(sample_dir, allow_non_campaign=allow_non_campaign)
    except (TrainingEligibilityError, ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "sample_id": sample.manifest.sample_id,
                "case_class": sample.manifest.case_class,
                "training_eligible": sample.manifest.training_eligible,
                "arrays": sorted(sample.arrays),
                "counts": sample.manifest.counts,
            },
            indent=2,
            sort_keys=True,
        ),
    )


@model_app.command("base-predict")
def model_base_predict() -> None:
    _blocked("Base prediction is not part of the public AeroMap package.")


@model_app.command("train")
def model_train(device: Annotated[str, typer.Option("--device")] = "auto") -> None:
    _require_cuda_workload(device)
    _blocked("Model training beyond tiny interface tests starts at model-integration.")


@model_app.command("evaluate")
def model_evaluate() -> None:
    _blocked("Model evaluation requires model artifacts.")


@active_app.command("propose")
def active_propose() -> None:
    _blocked(
        "Active acquisition is acquisition-integration work after uncertainty interfaces exist."
    )


@active_app.command("run-round")
def active_run_round() -> None:
    _blocked("Real active rounds are solver-loop integration work after the solver adapter exists.")


@active_app.command("resume")
def active_resume() -> None:
    _blocked("Use the offline replay commands listed in README.md for this release.")


@benchmark_app.command("run", hidden=True)
def benchmark_run(device: Annotated[str, typer.Option("--device")] = "auto") -> None:
    _require_cuda_workload(device)
    _blocked("Use the explicit AeroMap replay commands listed in README.md.")


@benchmark_app.command("aeromap-plan", hidden=True)
def benchmark_aeromap_plan(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap_plan.json",
    ),
) -> None:
    """Write the local AeroMap Mission Control plan."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        plan_path = write_aeromap_plan(benchmark_config, out)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"AeroMap config is invalid: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(plan_path),
        "benchmark_class": payload["benchmark_class"],
        "headline": payload["headline"],
        "preferred_dataset": payload["open_dataset"]["preferred"],
        "cloud": payload["cost_policy"]["cloud"],
        "custom_cfd_solves": payload["cost_policy"]["custom_cfd_solves"],
        "methods": payload["budget_protocol"]["acquisition_methods"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-airfrans-feasibility", hidden=True)
def benchmark_aeromap_airfrans_feasibility(
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/airfrans_feasibility.json",
    ),
    data_root: Annotated[Path | None, typer.Option("--data-root", file_okay=False)] = None,
) -> None:
    """Inspect AirfRANS package/data readiness without downloading the dataset."""

    report_path = write_airfrans_feasibility(out, data_root=data_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "package_installed": payload["package"]["installed"],
        "package_version": payload["package"]["version"],
        "archive_gib": payload["dataset_archive"]["content_length_gib"],
        "download_attempted": payload["dataset_archive"]["download_attempted"],
        "real_airfrans_benchmark_ready": payload["mvp_decision"]["real_airfrans_benchmark_ready"],
        "fixture_replay_allowed": payload["mvp_decision"]["fixture_replay_allowed"],
        "target_status": payload["target_contract"]["target_status"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-fixture", hidden=True)
def benchmark_aeromap_fixture(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/aeromap_fixture_dataset.json",
    ),
) -> None:
    """Write a compact deterministic AirfRANS-contract fixture dataset."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        manifest_path = write_fixture_dataset(benchmark_config, out)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"AeroMap fixture generation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(manifest_path),
        "npz_path": payload["npz_path"],
        "case_count": payload["case_count"],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
        "aerocliff_result": payload["claim_boundary"]["aerocliff_result"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-airfrans-extract", hidden=True)
def benchmark_aeromap_airfrans_extract(
    archive: Annotated[
        Path,
        typer.Option("--archive", exists=True, dir_okay=False, readable=True),
    ] = Path("artifacts/benchmark/airfrans/Dataset.zip"),
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path(
        "artifacts/benchmark/airfrans/processed",
    ),
) -> None:
    """Extract a previously downloaded AirfRANS processed archive."""

    try:
        manifest = extract_airfrans_archive(archive, root)
    except (OSError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"AirfRANS extraction failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        json.dumps(
            {
                "archive": str(archive),
                "root": str(root),
                "manifest": str(manifest),
            },
            indent=2,
            sort_keys=True,
        ),
    )


@benchmark_app.command("aeromap-airfrans-scalars", hidden=True)
def benchmark_aeromap_airfrans_scalars(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, readable=True),
    ] = Path("artifacts/benchmark/airfrans/processed"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/airfrans_scalar_dataset.json",
    ),
    task: Annotated[str, typer.Option("--task")] = "full",
    split: Annotated[str, typer.Option("--split")] = "both",
    max_cases: Annotated[int | None, typer.Option("--max-cases")] = None,
) -> None:
    """Build the real AirfRANS scalar dataset from documented force coefficients."""

    try:
        manifest = build_airfrans_scalar_dataset(
            root,
            out,
            task=task,
            split=split,
            max_cases=max_cases,
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"AirfRANS scalar dataset build failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    summary = {
        "path": str(manifest),
        "classification": payload["classification"],
        "completed_case_count": payload["completed_case_count"],
        "failed_case_count": payload["failed_case_count"],
        "npz_path": payload["npz_path"],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
        "aerocliff_result": payload["claim_boundary"]["aerocliff_result"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-airfrans-geometry", hidden=True)
def benchmark_aeromap_airfrans_geometry(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, readable=True),
    ] = Path("artifacts/benchmark/airfrans/processed"),
    scalar_npz: Annotated[
        Path,
        typer.Option("--scalar-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("artifacts/benchmark/aeromap/airfrans_scalar_dataset.npz"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap/airfrans_geometry_scalar_dataset.json",
    ),
    feature_contract_out: Annotated[Path | None, typer.Option("--feature-contract-out")] = Path(
        "docs/evidence/aeromap/airfrans_feature_contract.json",
    ),
) -> None:
    """Append deterministic geometry descriptors to the real AirfRANS scalar dataset."""

    try:
        manifest = build_airfrans_geometry_dataset(
            root,
            scalar_npz,
            out,
            feature_contract_out=feature_contract_out,
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"AirfRANS geometry dataset build failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    summary = {
        "path": str(manifest),
        "classification": payload["classification"],
        "case_count": payload["case_count"],
        "feature_count": payload["feature_count"],
        "geometry_feature_count": payload["geometry_feature_count"],
        "unique_geometry_group_count": payload["unique_geometry_group_count"],
        "npz_path": payload["npz_path"],
        "feature_contract_out": str(feature_contract_out) if feature_contract_out else None,
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-replay", hidden=True)
def benchmark_aeromap_replay(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("artifacts/benchmark/aeromap/aeromap_fixture_dataset.npz"),
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/aeromap_fixture_replay.json",
    ),
    svg_out: Annotated[Path | None, typer.Option("--svg-out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/aeromap_fixture_learning_curve.svg",
    ),
) -> None:
    """Run the local AeroMap active-learning replay on a prepared dataset."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_active_learning_replay(
            dataset_npz,
            benchmark_config,
            out,
            svg_out=svg_out,
        )
    except (OSError, ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(f"AeroMap replay failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "best_method_by_final_rmse_cd": payload["best_method_by_final_rmse_cd"],
        "engineering_utility_vs_random": payload["engineering_utility_vs_random"],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
        "svg_out": str(svg_out) if svg_out is not None else None,
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-decision-replay-v02", hidden=True)
def benchmark_aeromap_decision_replay_v02(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz"),
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control_v03.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/airfrans_decision_replay_v02.json",
    ),
    svg_dir: Annotated[Path | None, typer.Option("--svg-dir", file_okay=False)] = Path(
        "docs/evidence/aeromap",
    ),
) -> None:
    """Run the v0.2 decision-quality AeroMap replay."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_decision_replay_v02(
            dataset_npz,
            benchmark_config,
            out,
            svg_dir=svg_dir,
        )
    except (OSError, ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(f"AeroMap decision replay failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "split_modes": payload["split_modes"],
        "method_winners": payload["method_winners"],
        "engineering_decision_utility_v1_assessment": payload[
            "engineering_decision_utility_v1_assessment"
        ],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
        "svg_dir": str(svg_dir) if svg_dir is not None else None,
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-decision-replay-v03")
def benchmark_aeromap_decision_replay_v03(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz"),
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control_v03.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap/airfrans_decision_replay_v03.json",
    ),
    svg_dir: Annotated[Path | None, typer.Option("--svg-dir", file_okay=False)] = Path(
        "docs/evidence/aeromap",
    ),
) -> None:
    """Run the v0.3 decision-quality AeroMap replay with regret-aware acquisition."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_decision_replay_v03(
            dataset_npz,
            benchmark_config,
            out,
            svg_dir=svg_dir,
        )
    except (OSError, ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(f"AeroMap v0.3 decision replay failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "split_modes": payload["split_modes"],
        "method_winners": payload["method_winners"],
        "headline_readiness": payload["headline_readiness"],
        "engineering_decision_utility_assessment": payload[
            "engineering_decision_utility_assessment"
        ],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
        "svg_dir": str(svg_dir) if svg_dir is not None else None,
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("airfrans-field-baseline")
def benchmark_airfrans_field_baseline(
    root: Annotated[
        Path,
        typer.Option("--root", exists=True, file_okay=False, readable=True),
    ] = Path("artifacts/benchmark/airfrans/processed"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/field/airfrans_surface_pressure_baseline_v0_1.json",
    ),
    visual_out: Annotated[Path, typer.Option("--visual-out", dir_okay=False)] = Path(
        "docs/assets/aeromap/airfrans_surface_pressure_field_examples.png",
    ),
    summary_plot_out: Annotated[Path, typer.Option("--summary-plot-out", dir_okay=False)] = Path(
        "docs/assets/aeromap/airfrans_surface_pressure_baseline_metrics.png",
    ),
    train_cases: Annotated[int, typer.Option("--train-cases", min=2)] = 80,
    val_cases: Annotated[int, typer.Option("--val-cases", min=1)] = 16,
    test_cases: Annotated[int, typer.Option("--test-cases", min=1)] = 32,
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 80,
    batch_size: Annotated[int, typer.Option("--batch-size", min=16)] = 8192,
    hidden_width: Annotated[int, typer.Option("--hidden-width", min=8)] = 64,
    seed: Annotated[int, typer.Option("--seed")] = 20260630,
) -> None:
    """Train a compact AirfRANS surface-pressure field baseline."""

    try:
        report_path = run_airfrans_surface_pressure_baseline(
            root=root,
            out=out,
            visual_out=visual_out,
            summary_plot_out=summary_plot_out,
            train_cases=train_cases,
            val_cases=val_cases,
            test_cases=test_cases,
            epochs=epochs,
            batch_size=batch_size,
            hidden_width=hidden_width,
            seed=seed,
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"AirfRANS field baseline failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "train_cases": payload["split"]["train_cases"],
        "test_cases": payload["split"]["test_cases"],
        "best_method_by_rmse": payload["metrics"]["best_method_by_rmse"],
        "pointwise_mlp_rmse": payload["metrics"]["by_method"]["pointwise_mlp"]["rmse"],
        "nearest_case_rmse": payload["metrics"]["by_method"]["nearest_case"]["rmse"],
        "visual_panel": payload["artifacts"]["visual_panel"],
        "summary_plot": payload["artifacts"]["summary_plot"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-3d-triage")
def benchmark_aeromap_3d_triage(
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap3d/metadata_triage.json",
    ),
) -> None:
    """Inspect compact 3D AeroMap bridge candidates without large downloads."""

    try:
        report_path = write_aeromap3d_metadata_triage(out)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"AeroMap 3D metadata triage failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "selected_dataset": payload["selected_dataset"],
        "selection_reason": payload["selection_reason"],
        "ranked_by_positive_result_likelihood": payload["ranked_by_positive_result_likelihood"],
        "download_policy": payload["download_policy"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-3d-drivaerml-scalars")
def benchmark_aeromap_3d_drivaerml_scalars(
    cache_dir: Annotated[Path, typer.Option("--cache-dir", file_okay=False)] = Path(
        "artifacts/benchmark/aeromap3d/drivaerml",
    ),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.json",
    ),
) -> None:
    """Build the compact DrivAerML 3D scalar bridge dataset from root CSV metadata."""

    try:
        report_path = build_drivaerml_scalar_bridge_dataset(cache_dir, out)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"AeroMap 3D DrivAerML scalar bridge failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "source_dataset": payload["source_dataset"],
        "case_count": payload["case_count"],
        "feature_count": payload["feature_count"],
        "target_names": payload["target_names"],
        "npz_path": payload["npz_path"],
        "claim_boundary": payload["claim_boundary"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-3d-geometry-sample", hidden=True)
def benchmark_aeromap_3d_geometry_sample(
    stl: Annotated[
        list[Path] | None,
        typer.Option("--stl", exists=True, dir_okay=False, readable=True),
    ] = None,
    aerocliff_stl: Annotated[
        Path | None,
        typer.Option("--aeromap-stl", dir_okay=False, readable=True),
    ] = Path("artifacts/geometry/evidence/canonical/article.stl"),
    sample_points: Annotated[int, typer.Option("--sample-points", min=64, max=4096)] = 512,
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap3d/drivaerml_geometry_readiness_sample.json",
    ),
) -> None:
    """Compute compact descriptors for three to five real 3D aero STLs."""

    stl_paths = stl or [
        Path("artifacts/benchmark/drivaerml_cache/run_2/drivaer_2.stl"),
        Path("artifacts/benchmark/drivaerml_cache/run_10/drivaer_10.stl"),
        Path("artifacts/benchmark/drivaerml_cache/run_102/drivaer_102.stl"),
    ]
    try:
        report_path = write_geometry_readiness_sample(
            stl_paths,
            out,
            aerocliff_stl=(
                aerocliff_stl if aerocliff_stl is not None and aerocliff_stl.exists() else None
            ),
            sample_points_per_geometry=sample_points,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"AeroMap 3D geometry readiness sample failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "source_dataset": payload["source_dataset"],
        "stl_count": payload["stl_count"],
        "aerocliff_comparison_available": payload["aerocliff_comparison"] is not None,
        "points_npz_path": payload["points_npz_path"],
        "claim_boundary": payload["claim_boundary"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-v02-audit", hidden=True)
def benchmark_aeromap_v02_audit(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz"),
    feature_contract: Annotated[
        Path,
        typer.Option("--feature-contract", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_feature_contract.json"),
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control_v03.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/airfrans_feature_audit.json",
    ),
    decision_report: Annotated[
        Path | None,
        typer.Option("--decision-report", dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_decision_replay_v03.json"),
) -> None:
    """Audit AeroMap AirfRANS feature, split and claim contracts."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_airfrans_v02_audit(
            dataset_npz,
            feature_contract,
            benchmark_config,
            out,
            decision_report=decision_report,
        )
    except (OSError, ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(f"AeroMap v0.2/v0.3 audit failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "feature_count": payload["dataset"]["feature_count"],
        "passes_no_target_leakage_check": payload["feature_contract"][
            "passes_no_target_leakage_check"
        ],
        "split_audit": payload["split_audit"],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-cost-proxy-audit-v05")
def benchmark_aeromap_cost_proxy_audit_v05(
    airfrans_dataset_npz: Annotated[
        Path,
        typer.Option("--airfrans-dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz"),
    drivaerml_dataset_npz: Annotated[
        Path,
        typer.Option("--drivaerml-dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap3d/drivaerml_scalar_bridge_dataset.npz"),
    airfrans_processed_root: Annotated[
        Path,
        typer.Option("--airfrans-processed-root", file_okay=False),
    ] = Path("artifacts/benchmark/airfrans/processed"),
    drivaerml_cache_root: Annotated[
        Path,
        typer.Option("--drivaerml-cache-root", file_okay=False),
    ] = Path("artifacts/benchmark/drivaerml_cache"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap/cost_proxy_audit_v0_5.json",
    ),
) -> None:
    """Audit cost proxies before cost-aware AeroMap replay."""

    try:
        report_path = write_cost_proxy_audit(
            airfrans_dataset_npz=airfrans_dataset_npz,
            drivaerml_dataset_npz=drivaerml_dataset_npz,
            airfrans_processed_root=airfrans_processed_root,
            drivaerml_cache_root=drivaerml_cache_root,
            out=out,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"AeroMap cost proxy audit failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "cost_aware_replay_authorised": payload["selection_decision"][
            "cost_aware_replay_authorised"
        ],
        "airfrans_proxy": payload["datasets"]["airfrans"]["proxy_kind"],
        "drivaerml_proxy": payload["datasets"]["drivaerml"]["proxy_kind"],
        "new_data_downloaded": payload["claim_boundary"]["new_data_downloaded"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-cost-aware-replay-v05")
def benchmark_aeromap_cost_aware_replay_v05(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    dataset_name: Annotated[str, typer.Option("--dataset-name")],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)],
    airfrans_processed_root: Annotated[
        Path,
        typer.Option("--airfrans-processed-root", file_okay=False),
    ] = Path("artifacts/benchmark/airfrans/processed"),
) -> None:
    """Run v0.5 cost-proxy-aware AeroMap replay."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_cost_aware_decision_replay_v05(
            dataset_npz=dataset_npz,
            config=benchmark_config,
            dataset_name=dataset_name,
            out=out,
            airfrans_processed_root=airfrans_processed_root,
        )
    except (OSError, RuntimeError, ValidationError, yaml.YAMLError, ValueError) as exc:
        typer.echo(f"AeroMap v0.5 cost-aware replay failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "dataset_classification": payload["dataset_classification"],
        "cost_proxy_kind": payload["cost_proxy"]["kind"],
        "method_winners": payload["method_winners"],
        "cost_metric_winners": payload["cost_metric_winners"],
        "live_cfd_savings": payload["claim_boundary"]["live_cfd_savings"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("live-core-loop")
def benchmark_live_core_loop(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ] = LIVE_CORE_DEFAULT_DATASET_PATH,
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)] = (
        LIVE_CORE_DEFAULT_OUTPUT_DIR
    ),
    report_path: Annotated[Path, typer.Option("--report", dir_okay=False)] = (
        LIVE_CORE_DEFAULT_REPORT_PATH
    ),
    acquisition_policy: Annotated[
        Literal["random", "diversity", "engineering_utility", "cost_aware_utility"],
        typer.Option("--acquisition-policy"),
    ] = "engineering_utility",
    mode: Annotated[
        Literal["replay-live", "real-live"],
        typer.Option("--mode"),
    ] = "replay-live",
    max_iterations: Annotated[int, typer.Option("--max-iterations", min=1, max=12)] = 4,
    initial_case: Annotated[
        list[str] | None,
        typer.Option("--initial-case"),
    ] = None,
    candidate_case: Annotated[
        list[str] | None,
        typer.Option("--candidate-case"),
    ] = None,
    *,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
    random_seed: Annotated[int, typer.Option("--random-seed")] = 20260630,
) -> None:
    """Run a local live/replay acquisition loop on the Venturi Core map."""

    try:
        manifest_path = write_live_core_acquisition_loop(
            dataset_path=dataset_path,
            output_dir=output_dir,
            report_path=report_path,
            acquisition_policy=acquisition_policy,
            max_iterations=max_iterations,
            mode=mode,
            initial_cases=tuple(initial_case or LIVE_CORE_DEFAULT_INITIAL_CASES),
            candidate_cases=tuple(candidate_case or ()),
            dry_run=dry_run,
            overwrite=overwrite,
            random_seed=random_seed,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        typer.echo(f"AeroMap live Core loop failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(manifest_path),
        "classification": payload["classification"],
        "mode_requested": payload["mode_requested"],
        "mode_executed": payload["mode_executed"],
        "primary_policy": payload["primary_policy"],
        "completed_iterations": payload["primary_loop"]["completed_iterations"],
        "best_method_by_curve_error_area": payload["policy_comparison"][
            "best_method_by_curve_error_area"
        ],
        "report": payload["artifacts"]["report"],
        "learning_curve_svg": payload["artifacts"]["learning_curve_svg"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-3d-surface-field-precheck-v05")
def benchmark_aeromap_3d_surface_field_precheck_v05(
    drivaerml_cache_root: Annotated[
        Path,
        typer.Option("--drivaerml-cache-root", file_okay=False),
    ] = Path("artifacts/benchmark/drivaerml_cache"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/evidence/aeromap3d/surface_field_readiness_v0_5.json",
    ),
    visual_dir: Annotated[Path, typer.Option("--visual-dir", file_okay=False)] = Path(
        "docs/evidence/aeromap3d",
    ),
    max_cases: Annotated[int, typer.Option("--max-cases", min=0, max=2)] = 1,
) -> None:
    """Inspect already-cached DrivAerML boundary fields without downloading data."""

    try:
        report_path = write_surface_field_feasibility_precheck(
            drivaerml_cache_root=drivaerml_cache_root,
            out=out,
            visual_dir=visual_dir,
            max_cases=max_cases,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"AeroMap 3D surface-field precheck failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "cached_boundary_vtp_count": payload["cached_boundary_vtp_count"],
        "inspected_case_count": payload["inspected_case_count"],
        "new_downloads": payload["claim_boundary"]["new_downloads"],
        "field_prediction": payload["claim_boundary"]["field_prediction"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-cost-aware-report-v05")
def benchmark_aeromap_cost_aware_report_v05(
    airfrans_report: Annotated[
        Path,
        typer.Option("--airfrans-report", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_cost_aware_replay_v0_5.json"),
    drivaerml_report: Annotated[
        Path,
        typer.Option("--drivaerml-report", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap3d/drivaerml_cost_aware_replay_v0_5.json"),
    surface_report: Annotated[
        Path,
        typer.Option("--surface-report", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap3d/surface_field_readiness_v0_5.json"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "docs/reports/aeromap_cost_aware_v0_5_report.md",
    ),
) -> None:
    """Write a compact v0.5 cost-aware and field-readiness report."""

    try:
        report_path = write_cost_aware_report(
            airfrans_report=airfrans_report,
            drivaerml_report=drivaerml_report,
            surface_report=surface_report,
            out=out,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        typer.echo(f"AeroMap v0.5 report generation failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({"path": str(report_path)}, indent=2, sort_keys=True))


@benchmark_app.command("aeromap-model-baselines-v02", hidden=True)
def benchmark_aeromap_model_baselines_v02(
    dataset_npz: Annotated[
        Path,
        typer.Option("--dataset-npz", exists=True, dir_okay=False, readable=True),
    ] = Path("docs/evidence/aeromap/airfrans_geometry_scalar_dataset.npz"),
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/benchmark/aeromap_mission_control_v03.yaml"),
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path(
        "artifacts/benchmark/aeromap/airfrans_model_baselines_v02.json",
    ),
) -> None:
    """Run local v0.2 scalar model baselines for the AeroMap benchmark."""

    try:
        raw_config = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        benchmark_config = AeroMapConfig.model_validate(raw_config)
        report_path = write_model_baselines_v02(dataset_npz, benchmark_config, out)
    except (OSError, ValidationError, yaml.YAMLError, ValueError, RuntimeError) as exc:
        typer.echo(f"AeroMap model baselines failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(report_path),
        "classification": payload["classification"],
        "split_modes": payload["split_modes"],
        "open_cfd_result": payload["claim_boundary"]["open_cfd_result"],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@benchmark_app.command("report", hidden=True)
def benchmark_report() -> None:
    _blocked("Use the committed reports in docs/reports for this release.")


@app.command(hidden=True)
def serve() -> None:
    _blocked("Minimum inference API is implemented after stable model/data interfaces.")


@app.command()
def demo() -> None:
    typer.echo("Open docs/demo/aeromap_mission_control.html")

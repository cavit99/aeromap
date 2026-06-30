from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeromap.cfd.case_builder import build_cfd_case
from aeromap.cfd.dictionaries import (
    block_mesh_dict,
    control_dict,
    field_u,
    fv_schemes,
    mesh_quality_dict,
    momentum_transport,
    snappy_hex_mesh_dict,
    surface_features_dict,
)
from aeromap.cfd.schema import (
    CfdConfig,
    MeshConfig,
    PatchLayerConfig,
    QualityConfig,
    RefinementBox,
    SpanRefinement,
    SurfaceExportConfig,
)
from aeromap.constants import REF
from aeromap.io import sha256_file
from aeromap.parameters import AeroParams


def _simulation_id(params: AeroParams, config: CfdConfig) -> str:
    return params.simulation_id(
        mesh_config=config.mesh.model_dump(),
        surface_export_config=config.surface_export.model_dump(mode="json"),
        solver_config=config.solver.model_dump(),
        quality_config=config.quality.model_dump(),
        openfoam_version="OpenFOAM Foundation v13",
    )


def test_openfoam_dictionaries_contain_v13_solver_contract() -> None:
    params = AeroParams.canonical()
    config = CfdConfig()
    rendered_u = field_u(params)
    rendered_snappy = snappy_hex_mesh_dict(config)
    rendered_control = control_dict(config)
    assert "incompressibleFluid" in rendered_control
    assert 'libs            ("libforces.so");' in rendered_control
    assert 'libs            ("libfieldFunctionObjects.so");' in rendered_control
    assert "type            forceCoeffs;" in rendered_control
    assert "liftDir         (0 0 -1);" in rendered_control
    assert f"magUInf         {REF.u_inf_m_s:g};" in rendered_control
    assert f"Aref            {REF.a_ref_m2:g};" in rendered_control
    assert "inletFlowRate" in rendered_control
    assert "outletFlowRate" in rendered_control
    assert "operation       sum;" in rendered_control
    assert "inlet" in block_mesh_dict(config)
    assert 'file "article.stl";' in rendered_snappy
    assert "type triSurface;" in rendered_snappy
    assert 'file "article.eMesh";' in rendered_snappy
    assert "explicitFeatureSnap true;" in rendered_snappy
    assert "nFeatureSnapIter 10;" in rendered_snappy
    assert 'surfaces ("article.stl");' in surface_features_dict()
    assert "model           kOmegaSST;" in momentum_transport()
    assert "wallDist" in fv_schemes()
    assert "method          meshWave;" in fv_schemes()
    assert "movingWallVelocity" in rendered_u
    assert f"uniform ({REF.u_inf_m_s:g} 0 0)" in rendered_u
    assert "meshQualityDict.cfg" in mesh_quality_dict()


def test_snappy_dictionary_can_disable_layers_and_explicit_features() -> None:
    config = CfdConfig(
        mesh=MeshConfig(
            add_layers=False,
            implicit_feature_snap=True,
            explicit_feature_snap=False,
            snap_solve_iterations=100,
            surface_level=(0, 0),
        ),
    )

    rendered_snappy = snappy_hex_mesh_dict(config)

    assert "addLayers       false;" in rendered_snappy
    assert 'file "article.eMesh";' not in rendered_snappy
    assert "nSolveIter 100;" in rendered_snappy
    assert "implicitFeatureSnap true;" in rendered_snappy
    assert "explicitFeatureSnap false;" in rendered_snappy
    assert "level (0 0);" in rendered_snappy


def test_snappy_dictionary_renders_refinement_boxes() -> None:
    config = CfdConfig(
        mesh=MeshConfig(
            n_cells_between_levels=3,
            n_surface_layers=1,
            layer_relative_sizes=False,
            final_layer_thickness=0.15,
            min_layer_thickness=0.02,
            max_thickness_to_medial_ratio=0.5,
            layer_feature_angle_deg=100,
            layer_slip_feature_angle_deg=30,
            layer_n_relaxed_iter=20,
            layer_n_medial_axis_iter=10,
            layer_additional_reporting=True,
            layer_n_grow=1,
            layer_n_buffer_cells_no_extrude=2,
            refinement_boxes=(
                RefinementBox(
                    name="underfloor_tunnels",
                    bounds_min=(-0.1, -0.55, 0.0),
                    bounds_max=(2.1, 0.55, 0.16),
                    level=4,
                ),
            ),
        ),
    )

    rendered_snappy = snappy_hex_mesh_dict(config)

    assert "nCellsBetweenLevels 3;" in rendered_snappy
    assert "article { nSurfaceLayers 1; }" in rendered_snappy
    assert "relativeSizes false;" in rendered_snappy
    assert "finalLayerThickness 0.15;" in rendered_snappy
    assert "minThickness 0.02;" in rendered_snappy
    assert "maxThicknessToMedialRatio 0.5;" in rendered_snappy
    assert "featureAngle 100;" in rendered_snappy
    assert "slipFeatureAngle 30;" in rendered_snappy
    assert "nRelaxedIter 20;" in rendered_snappy
    assert "nMedialAxisIter 10;" in rendered_snappy
    assert "additionalReporting true;" in rendered_snappy
    assert "minMedialAxisAngle 90;" in rendered_snappy
    assert "nGrow 1;" in rendered_snappy
    assert "nBufferCellsNoExtrude 2;" in rendered_snappy
    assert "underfloor_tunnels" in rendered_snappy
    assert "type box;" in rendered_snappy
    assert "min (-0.1 -0.55 0);" in rendered_snappy
    assert "max (2.1 0.55 0.16);" in rendered_snappy
    assert "mode inside;" in rendered_snappy
    assert "level 4;" in rendered_snappy


def test_snappy_dictionary_renders_gate2b_patch_selective_controls() -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(
            method="gmsh_occ_g0_no_healing",
            openfoam_patch_mode="gate2b_core_transition",
        ),
        mesh=MeshConfig(
            layer_relative_sizes=False,
            first_layer_thickness=0.0006,
            min_layer_thickness=0.0002,
            patch_layers=(
                PatchLayerConfig(patch="tunnel_roofs_core", n_surface_layers=3),
                PatchLayerConfig(patch="diffuser_core", n_surface_layers=3),
                PatchLayerConfig(patch="underfloor_core", n_surface_layers=3),
                PatchLayerConfig(patch="upper_body", n_surface_layers=1),
                PatchLayerConfig(patch="layer_transition_band", n_surface_layers=0),
            ),
            span_refinements=(SpanRefinement(surface="article", level=4, cells_across_span=12),),
        ),
    )

    rendered_snappy = snappy_hex_mesh_dict(config)
    rendered_features = surface_features_dict(config)
    rendered_control = control_dict(config)
    rendered_u = field_u(AeroParams.canonical(), config)

    assert 'file "article.obj";' in rendered_snappy
    assert "tunnel_roofs_core" in rendered_snappy
    assert "diffuser_core { nSurfaceLayers 3; }" in rendered_snappy
    assert "upper_body { nSurfaceLayers 1; }" in rendered_snappy
    assert "layer_transition_band { nSurfaceLayers 0; }" in rendered_snappy
    assert "relativeSizes false;" in rendered_snappy
    assert "firstLayerThickness 0.0006;" in rendered_snappy
    assert "finalLayerThickness" not in rendered_snappy
    assert "mode insideSpan;" in rendered_snappy
    assert "cellsAcrossSpan 12;" in rendered_snappy
    assert 'surfaces ("article.obj");' in rendered_features
    assert "pointCloseness yes;" in rendered_features
    assert "patches         (tunnel_roofs_core diffuser_core underfloor_core" in rendered_control
    assert "tunnel_roofs_core      { type noSlip; }" in rendered_u


def test_gate2b_default_layers_target_real_openfoam_patches() -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="gate2b_core_transition"),
        mesh=MeshConfig(n_surface_layers=3),
    )

    rendered_snappy = snappy_hex_mesh_dict(config)

    assert "article { nSurfaceLayers" not in rendered_snappy
    assert "tunnel_roofs_core { nSurfaceLayers 3; }" in rendered_snappy
    assert "diffuser_core { nSurfaceLayers 3; }" in rendered_snappy
    assert "underfloor_core { nSurfaceLayers 3; }" in rendered_snappy
    assert "upper_body { nSurfaceLayers 1; }" in rendered_snappy
    assert "floor_edges { nSurfaceLayers 0; }" in rendered_snappy
    assert "keel { nSurfaceLayers 0; }" in rendered_snappy
    assert "layer_transition_band { nSurfaceLayers 0; }" in rendered_snappy


def test_critical_underfloor_dictionary_targets_combined_patch() -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(
            method="gmsh_occ_g0_no_healing",
            openfoam_patch_mode="critical_underfloor",
        ),
        mesh=MeshConfig(
            n_surface_layers=1,
            layer_relative_sizes=False,
            first_layer_thickness=0.0005,
            min_layer_thickness=0.00016,
        ),
    )

    rendered_snappy = snappy_hex_mesh_dict(config)
    rendered_control = control_dict(config)
    rendered_u = field_u(AeroParams.canonical(), config)

    assert 'file "article.obj";' in rendered_snappy
    assert "critical_underfloor { nSurfaceLayers 1; }" in rendered_snappy
    assert "diffuser_core" not in rendered_snappy
    assert "tunnel_roofs_core" not in rendered_snappy
    assert "underfloor_core" not in rendered_snappy
    assert "upper_body { nSurfaceLayers 1; }" in rendered_snappy
    assert "floor_edges { nSurfaceLayers 0; }" in rendered_snappy
    assert "patches         (critical_underfloor upper_body floor_edges" in rendered_control
    assert "critical_underfloor" in rendered_u
    assert "type noSlip;" in rendered_u


def test_cfd_case_builder_writes_expected_layout(tmp_path: Path) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)
    case_dir = artifacts.case_dir

    assert case_dir.name == artifacts.simulation_id
    assert (case_dir / "params.json").exists()
    assert (case_dir / "manifest.json").exists()
    assert (case_dir / "geometry" / "article.stl").exists()
    assert (case_dir / "cfd_surface" / "article.stl").exists()
    assert (case_dir / "cfd_surface" / "surface_regions.json").exists()
    assert (case_dir / "openfoam" / "0" / "U").exists()
    assert (case_dir / "openfoam" / "constant" / "triSurface" / "article.stl").exists()
    assert (
        case_dir / "openfoam" / "constant" / "triSurface" / "article_surface_regions.json"
    ).exists()
    assert (
        case_dir / "openfoam" / "constant" / "triSurface" / "article_surface_regions.vtp"
    ).exists()
    assert (case_dir / "openfoam" / "system" / "snappyHexMeshDict").exists()
    assert (case_dir / "openfoam" / "system" / "meshQualityDict").exists()
    assert (case_dir / "openfoam" / "system" / "surfaceFeaturesDict").exists()
    assert (case_dir / "run_openfoam.sh").exists()
    script = (case_dir / "run_openfoam.sh").read_text(encoding="utf-8")
    assert "checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log 2>&1" in script
    assert "checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log\n" not in script


def test_cfd_case_builder_critical_underfloor_keeps_single_part_stl(
    tmp_path: Path,
) -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(
            method="cadquery_current",
            openfoam_patch_mode="critical_underfloor",
            transition_band_width_m=1e-9,
        ),
        mesh=MeshConfig(feature_angle_deg=180.0),
    )

    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path, config=config)
    case_dir = artifacts.case_dir
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    script = (case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert (case_dir / "cfd_surface" / "article.obj").exists()
    assert (case_dir / "openfoam" / "constant" / "triSurface" / "article.obj").exists()
    assert (
        manifest["cfd_surface"]["stl_sha256"] == manifest["file_hashes"]["cfd_surface/article.stl"]
    )
    assert (
        manifest["file_hashes"]["geometry/article.stl"]
        == manifest["file_hashes"]["cfd_surface/article.stl"]
    )
    assert manifest["cfd_surface"]["openfoam_patch_surface"]["patch_names"] == [
        "critical_underfloor",
        "upper_body",
        "floor_edges",
        "keel",
        "layer_transition_band",
    ]
    assert "mesh_quality_status=$?" in script
    assert "cp ../logs/checkMesh_meshQuality.log ../logs/checkMesh_fatal.log" in script
    assert (
        'printf "%s\\n" "${mesh_quality_status}" > ../quality/checkMesh_meshQuality.returncode'
    ) in script
    assert (
        "mesh_quality_failed_lines=$(grep -Eic 'Failed [0-9]+ mesh checks?' "
        "../logs/checkMesh_meshQuality.log || true)"
    ) in script
    assert 'printf "%s\\n" "${mesh_quality_failed_lines}"' in script
    assert 'printf "%s\\n" "${mesh_quality_failed_checks}"' in script
    assert 'printf "%s\\n" "${mesh_quality_ok_lines}"' in script
    assert 'if [[ "${mesh_quality_status}" -ne 0 ]]; then' in script
    assert 'if [[ "${mesh_quality_failed_checks}" -gt 0 ]]; then' in script
    assert "foamRun -solver incompressibleFluid" in script
    assert "checkMesh -allGeometry -allTopology -writeSurfaces -writeSets" in script
    assert "extended_check_status=$?" in script
    assert (
        'printf "%s\\n" "${extended_check_status}" > ../quality/checkMesh_extended.returncode'
        in script
    )


def test_cfd_case_rebuild_removes_stale_outputs_and_hashes_final_files(tmp_path: Path) -> None:
    first = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)
    (first.case_dir / "logs" / "stale.log").write_text("old log", encoding="utf-8")
    (first.case_dir / "outputs" / "stale.vtu").write_text("old output", encoding="utf-8")

    rebuilt = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    assert not (rebuilt.case_dir / "logs" / "stale.log").exists()
    assert not (rebuilt.case_dir / "outputs" / "stale.vtu").exists()
    assert json.loads((rebuilt.case_dir / "quality" / "status.json").read_text()) == {
        "status": "BUILT_NOT_RUN",
    }

    manifest = json.loads(rebuilt.manifest_path.read_text(encoding="utf-8"))
    assert manifest["geometry_id"].startswith("geometry_")
    assert manifest["state_id"].startswith("state_")
    assert manifest["simulation_id"].startswith("simulation_")
    assert "manifest.json" not in manifest["file_hashes"]
    assert "openfoam/system/meshQualityDict" in manifest["file_hashes"]
    assert "quality/status.json" in manifest["file_hashes"]
    assert manifest["surface_export_id"].startswith("surface_export_")
    assert manifest["cfd_surface"]["surface_export"]["method"] == "cadquery_current"
    assert manifest["cfd_surface"]["stl_path"] == "cfd_surface/article.stl"
    for relative_path, digest in manifest["file_hashes"].items():
        assert sha256_file(rebuilt.case_dir / relative_path) == digest


def test_cfd_case_rebuild_refuses_completed_simulation(tmp_path: Path) -> None:
    first = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)
    completed_output = first.case_dir / "outputs" / "accepted.vtp"
    completed_output.write_text("accepted result", encoding="utf-8")
    (first.case_dir / "quality" / "status.json").write_text('{"status":"DONE"}', encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    assert completed_output.read_text(encoding="utf-8") == "accepted result"


def test_cfd_case_builder_does_not_delete_legacy_deterministic_staging(
    tmp_path: Path,
) -> None:
    params = AeroParams.canonical()
    config = CfdConfig()
    simulation_id = _simulation_id(params, config)
    legacy_staging = tmp_path / f".{simulation_id}.staging"
    legacy_staging.mkdir()
    sentinel = legacy_staging / "sentinel.txt"
    sentinel.write_text("owned by another worker\n", encoding="utf-8")

    build_cfd_case(params, cases_dir=tmp_path, config=config)

    assert sentinel.read_text(encoding="utf-8") == "owned by another worker\n"


def test_cfd_case_builder_refuses_concurrent_simulation_lock(tmp_path: Path) -> None:
    params = AeroParams.canonical()
    config = CfdConfig()
    simulation_id = _simulation_id(params, config)
    (tmp_path / f".{simulation_id}.lock").mkdir()

    with pytest.raises(FileExistsError, match="already being built"):
        build_cfd_case(params, cases_dir=tmp_path, config=config)


def test_smoke_quality_mode_is_nonfatal_for_full_geometry_check(tmp_path: Path) -> None:
    config = CfdConfig(
        quality=QualityConfig(
            case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
            extended_diagnostics_fatal=False,
        ),
    )

    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path, config=config)
    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert "extended_check_status=$?" in script
    assert manifest["cfd_config"]["quality"]["case_class"] == "NON_CAMPAIGN_ENGINEERING_SMOKE"


def test_configured_mesh_quality_success_path_records_return_code_and_metrics(
    tmp_path: Path,
) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert (
        "set +e\n"
        "checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log 2>&1\n"
        "mesh_quality_status=$?\n"
        "set -e"
    ) in script
    assert 'printf "%s\\n" "${mesh_quality_status}"' in script
    assert "mesh_quality_ok_lines=$(grep -Eic 'Mesh OK'" in script
    assert "foamRun -solver incompressibleFluid" in script


def test_configured_mesh_quality_failure_is_fatal_by_return_code_or_log(
    tmp_path: Path,
) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert 'if [[ "${mesh_quality_status}" -ne 0 ]]; then' in script
    assert 'echo "fatal checkMesh -meshQuality returned ${mesh_quality_status}" >&2' in script
    assert 'if [[ "${mesh_quality_failed_checks}" -gt 0 ]]; then' in script
    assert (
        'echo "fatal checkMesh -meshQuality reported Failed '
        '${mesh_quality_failed_checks} mesh checks" >&2'
    ) in script


def test_extended_diagnostics_findings_remain_advisory_for_campaign(
    tmp_path: Path,
) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")
    extended_index = script.index("checkMesh -allGeometry -allTopology")
    solver_index = script.index("foamRun -solver incompressibleFluid")
    extended_block = script[extended_index:solver_index]

    assert "extended_check_status=$?" in extended_block
    assert 'printf "%s\\n" "${extended_check_status}"' in extended_block
    assert 'if [[ "${extended_check_status}"' not in extended_block


def test_configured_mesh_quality_missing_or_ambiguous_metrics_fail_closed(
    tmp_path: Path,
) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert (
        "mesh_quality_failed_lines=$(grep -Eic 'Failed [0-9]+ mesh checks?' "
        "../logs/checkMesh_meshQuality.log || true)"
    ) in script
    assert 'if [[ "${mesh_quality_failed_lines}" -gt 1 ]]; then' in script
    assert (
        'echo "fatal checkMesh -meshQuality log has ambiguous '
        '${mesh_quality_failed_lines} failed-check summaries" >&2'
    ) in script
    assert (
        'if [[ "${mesh_quality_failed_lines}" -eq 0 && "${mesh_quality_ok_lines}" -eq 0 ]]; then'
    ) in script
    assert (
        'echo "fatal checkMesh -meshQuality log lacks Mesh OK or Failed N mesh checks summary" >&2'
        in script
    )


def test_nonfatal_mesh_quality_mode_records_failed_check_count_without_abort(
    tmp_path: Path,
) -> None:
    artifacts = build_cfd_case(
        AeroParams.canonical(),
        cases_dir=tmp_path,
        config=CfdConfig(quality=QualityConfig(mesh_quality_fatal=False)),
    )

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert "mesh_quality_status=$?" in script
    assert 'printf "%s\\n" "${mesh_quality_failed_checks}"' in script
    assert 'if [[ "${mesh_quality_status}" -ne 0 ]]; then' not in script
    assert 'if [[ "${mesh_quality_failed_checks}" -gt 0 ]]; then' not in script


def test_campaign_quality_mode_keeps_extended_diagnostics_advisory(tmp_path: Path) -> None:
    artifacts = build_cfd_case(AeroParams.canonical(), cases_dir=tmp_path)

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert "extended_check_status=$?" in script
    assert "checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log 2>&1" in script
    assert "checkMesh -meshQuality > ../logs/checkMesh_meshQuality.log\n" not in script
    assert (
        "checkMesh -allGeometry -allTopology -writeSurfaces -writeSets "
        "-surfaceFormat vtk -setFormat vtk > ../logs/checkMesh.log 2>&1"
    ) in script


def test_extended_diagnostics_can_be_configured_fatal(tmp_path: Path) -> None:
    artifacts = build_cfd_case(
        AeroParams.canonical(),
        cases_dir=tmp_path,
        config=CfdConfig(quality=QualityConfig(extended_diagnostics_fatal=True)),
    )

    script = (artifacts.case_dir / "run_openfoam.sh").read_text(encoding="utf-8")

    assert "extended_check_status=$?" not in script
    assert 'printf "%s\\n" "0" > ../quality/checkMesh_extended.returncode' in script


def test_simulation_id_includes_quality_mode() -> None:
    params = AeroParams.canonical()
    strict = params.simulation_id(
        mesh_config={},
        solver_config={},
        quality_config={"case_class": "CAMPAIGN_REFERENCE_CFD"},
        openfoam_version="OpenFOAM Foundation v13",
    )
    smoke = params.simulation_id(
        mesh_config={},
        solver_config={},
        quality_config={"case_class": "NON_CAMPAIGN_ENGINEERING_SMOKE"},
        openfoam_version="OpenFOAM Foundation v13",
    )

    assert strict != smoke

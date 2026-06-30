from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from typer.testing import CliRunner

from aeromap.cli import app
from aeromap.data.loader import TrainingEligibilityError
from aeromap.data.schema import DataSampleManifest

if TYPE_CHECKING:
    import pytest

    from aeromap.cfd.schema import CfdConfig


def test_doctor_cli_reports_resources() -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "physicsnemo_domino_cuda" in result.stdout


def test_cfd_gate_fails_clearly() -> None:
    result = CliRunner().invoke(app, ["cfd", "validate"])
    assert result.exit_code == 2
    assert "real OpenFOAM case artifact" in result.stderr


def test_cfd_build_cli_accepts_yaml_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "cfd.yaml"
    config_path.write_text(
        """
profile: cli_config_test
surface_export:
  method: gmsh_occ_g0_no_healing
mesh:
  add_layers: false
  implicit_feature_snap: true
  explicit_feature_snap: false
  snap_solve_iterations: 100
  n_cells_between_levels: 3
  refinement_boxes:
    - name: underfloor_tunnels
      bounds_min: [-0.1, -0.55, 0.0]
      bounds_max: [2.1, 0.55, 0.16]
      level: 4
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build_cfd_case(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured.update(kwargs)

        def model_dump_json(*, indent: int) -> str:
            return json.dumps({}, indent=indent)

        return SimpleNamespace(model_dump_json=model_dump_json)

    monkeypatch.setattr("aeromap.cli.build_cfd_case", fake_build_cfd_case)

    result = CliRunner().invoke(
        app,
        ["cfd", "build", "--out", str(tmp_path / "cases"), "--config", str(config_path)],
    )

    assert result.exit_code == 0
    config = cast("CfdConfig", captured["config"])
    assert config.profile == "cli_config_test"
    assert config.surface_export.method == "gmsh_occ_g0_no_healing"
    assert config.mesh.add_layers is False
    assert config.mesh.implicit_feature_snap is True
    assert config.mesh.explicit_feature_snap is False
    assert config.mesh.snap_solve_iterations == 100
    assert config.mesh.n_cells_between_levels == 3
    assert config.mesh.refinement_boxes[0].name == "underfloor_tunnels"


def test_cfd_build_cli_rejects_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("profile: [", encoding="utf-8")

    result = CliRunner().invoke(app, ["cfd", "build", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "CFD config YAML is invalid" in result.stderr


def test_cfd_build_venturi_core_cli_writes_case(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["cfd", "build-venturi-core", "--out", str(tmp_path / "cases")],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "BUILT_NOT_RUN"
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["run_mesh_script_path"]).name == "run_core_mesh.sh"


def test_cfd_venturi_core_report_cli_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "core.md"

    result = CliRunner().invoke(app, ["cfd", "venturi-core-report", "--out", str(report)])

    assert result.exit_code == 0
    assert report.exists()
    assert (tmp_path / "core.json").exists()
    assert "Venturi Lab" in report.read_text(encoding="utf-8")


def test_geometry_generate_cli_rejects_out_of_range_parameters() -> None:
    result = CliRunner().invoke(app, ["geometry", "generate", "--ride-height-mm", "10"])

    assert result.exit_code == 2
    assert "ride_height_mm" in result.stderr


def test_cuda_model_train_request_fails_before_placeholder_workload() -> None:
    result = CliRunner().invoke(app, ["model", "train", "--device", "cuda"])
    assert result.exit_code == 2
    assert "Linux NVIDIA CUDA host" in result.stderr


def test_domino_export_cli_outputs_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDominoGeometryAdapter:
        def __init__(self, *, target_surface_points: int) -> None:
            self.target_surface_points = target_surface_points

        def export(self, _params: object, output_dir: Path) -> SimpleNamespace:
            return SimpleNamespace(
                stl_path=output_dir / "domino_article_highres.stl",
                manifest_path=output_dir / "domino_geometry_manifest.json",
                surface_point_count=301_000,
                meets_target_surface_points=True,
                subdivision_steps=1,
                nim_input_contract_valid=True,
            )

    monkeypatch.setattr("aeromap.cli.DominoGeometryAdapter", FakeDominoGeometryAdapter)

    result = CliRunner().invoke(
        app,
        [
            "geometry",
            "domino-export",
            "--out",
            str(tmp_path),
            "--target-surface-points",
            "300000",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "manifest_path": str(tmp_path / "domino_geometry_manifest.json"),
        "meets_target_surface_points": True,
        "nim_input_contract_valid": True,
        "stl_path": str(tmp_path / "domino_article_highres.stl"),
        "subdivision_steps": 1,
        "surface_point_count": 301_000,
    }


def test_domino_export_cli_fails_when_export_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDominoGeometryAdapter:
        def __init__(self, *, target_surface_points: int) -> None:
            self.target_surface_points = target_surface_points

        def export(self, _params: object, output_dir: Path) -> SimpleNamespace:
            return SimpleNamespace(
                stl_path=output_dir / "domino_article_highres.stl",
                manifest_path=output_dir / "domino_geometry_manifest.json",
                surface_point_count=299_999,
                meets_target_surface_points=False,
                subdivision_steps=0,
                nim_input_contract_valid=False,
            )

    monkeypatch.setattr("aeromap.cli.DominoGeometryAdapter", FakeDominoGeometryAdapter)

    result = CliRunner().invoke(
        app,
        [
            "geometry",
            "domino-export",
            "--out",
            str(tmp_path),
            "--target-surface-points",
            "300000",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["nim_input_contract_valid"] is False
    assert payload["meets_target_surface_points"] is False


def test_benchmark_drivaerml_plan_cli_writes_external_plan(tmp_path: Path) -> None:
    out = tmp_path / "drivaerml_plan.json"

    result = CliRunner().invoke(app, ["benchmark", "drivaerml-plan", "--out", str(out)])

    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["path"] == str(out)
    assert summary["ec2_usage"] == "not_required_for_plan_generation"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["benchmark_class"] == "EXTERNAL_DRIVAERML_POOL_BENCHMARK"
    assert payload["split"]["counts"]["total"] == 144
    assert payload["training_eligibility"]["external_benchmark"] is True
    assert payload["training_eligibility"]["aerocliff_custom_model"] is False
    assert payload["claim_eligibility"]["aerocliff_accuracy"] is False
    assert payload["download_policy"]["status"] == "PLAN_ONLY_NO_DOWNLOAD"


def test_benchmark_drivaerml_assets_cli_writes_dry_run_manifest(tmp_path: Path) -> None:
    plan = tmp_path / "drivaerml_plan.json"
    assets = tmp_path / "assets.json"
    cache = tmp_path / "cache"
    plan_result = CliRunner().invoke(app, ["benchmark", "drivaerml-plan", "--out", str(plan)])
    assert plan_result.exit_code == 0

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "drivaerml-assets",
            "--plan",
            str(plan),
            "--out",
            str(assets),
            "--cache-dir",
            str(cache),
            "--split",
            "initial_labelled",
            "--max-cases",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["selected_case_count"] == 2
    assert summary["dry_run"] is True
    assert summary["data_ready"] is False
    assert summary["validation_ok"] is True
    manifest = json.loads(assets.read_text(encoding="utf-8"))
    assert manifest["training_eligibility"]["external_benchmark"] is True
    assert manifest["training_eligibility"]["aerocliff_custom_model"] is False
    assert manifest["no_volume_vtu_confirmed"] is True
    assert {asset["status"] for asset in manifest["assets"]} == {"DRY_RUN_PLANNED"}


def test_benchmark_drivaerml_samples_cli_rejects_dry_run_manifest(tmp_path: Path) -> None:
    plan = tmp_path / "drivaerml_plan.json"
    assets = tmp_path / "assets.json"
    samples = tmp_path / "samples.json"
    plan_result = CliRunner().invoke(app, ["benchmark", "drivaerml-plan", "--out", str(plan)])
    assert plan_result.exit_code == 0
    asset_result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "drivaerml-assets",
            "--plan",
            str(plan),
            "--out",
            str(assets),
            "--max-cases",
            "1",
            "--dry-run",
        ],
    )
    assert asset_result.exit_code == 0

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "drivaerml-samples",
            "--asset-manifest",
            str(assets),
            "--out",
            str(samples),
        ],
    )

    assert result.exit_code == 2
    assert "not data-ready" in result.stderr


def test_benchmark_drivaerml_cuda_stubs_fail_without_data(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "missing_assets.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "drivaerml-cache-frozen-predictor",
            "--asset-manifest",
            str(missing_manifest),
        ],
    )

    assert result.exit_code == 2
    assert "asset manifest not found" in result.stderr


def test_surface_diagnostics_cli_outputs_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_diagnose_surface(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            attempt_id="attempt_surface",
            attempt_dir=tmp_path,
            attempt_manifest_path=tmp_path / "attempt_manifest.json",
            metrics_path=tmp_path / "surface_diagnostics.json",
            cad_faces_vtp_path=tmp_path / "cad_faces_by_id.vtp",
            stl_triangles_vtp_path=tmp_path / "stl_triangle_diagnostics.vtp",
            bad_triangles_csv_path=tmp_path / "bad_stl_triangles.csv",
        )

    monkeypatch.setattr("aeromap.cli.diagnose_surface", fake_diagnose_surface)

    result = CliRunner().invoke(app, ["geometry", "diagnose-surface", "--out", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["attempt_id"] == "attempt_surface"
    assert payload["metrics_path"] == str(tmp_path / "surface_diagnostics.json")


def test_mesh_diagnostics_cli_outputs_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_diagnose_mesh(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            attempt_id="attempt_mesh",
            attempt_dir=tmp_path,
            attempt_manifest_path=tmp_path / "attempt_manifest.json",
            metrics_path=tmp_path / "mesh_diagnostics.json",
        )

    monkeypatch.setattr("aeromap.cli.diagnose_mesh", fake_diagnose_mesh)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    result = CliRunner().invoke(
        app, ["cfd", "diagnose-mesh", str(case_dir), "--out", str(tmp_path)]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["attempt_id"] == "attempt_mesh"
    assert payload["metrics_path"] == str(tmp_path / "mesh_diagnostics.json")


def test_cfd_postprocess_cli_outputs_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_postprocess_case(_case_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            mesh_json=tmp_path / "mesh.json",
            convergence_json=tmp_path / "convergence.json",
            layers_json=tmp_path / "layers.json",
            yplus_json=tmp_path / "yplus.json",
            force_integration_json=tmp_path / "force_integration.json",
            status_json=tmp_path / "status.json",
            scalars_parquet=tmp_path / "scalars.parquet",
            volume_vtu=tmp_path / "volume.vtu",
            wall_vtp=tmp_path / "wall.vtp",
            mapped_wall_vtp=tmp_path / "wall_regions.vtp",
        )

    monkeypatch.setattr("aeromap.cli.postprocess_case", fake_postprocess_case)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    result = CliRunner().invoke(app, ["cfd", "postprocess", str(case_dir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status_json"] == str(tmp_path / "status.json")
    assert payload["volume_vtu"] == str(tmp_path / "volume.vtu")


def test_data_convert_cli_outputs_sample_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_convert_case_to_sample(_case_dir: Path, _out: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model_dump_json=lambda *, indent: json.dumps(
                {
                    "sample_id": "sample_demo",
                    "sample_dir": str(tmp_path / "sample_demo"),
                    "manifest_path": str(tmp_path / "sample_demo" / "manifest.json"),
                    "arrays_path": str(tmp_path / "sample_demo" / "arrays.npz"),
                },
                indent=indent,
            ),
        )

    monkeypatch.setattr("aeromap.cli.convert_case_to_sample", fake_convert_case_to_sample)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    result = CliRunner().invoke(app, ["data", "convert", str(case_dir), "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["sample_id"] == "sample_demo"


def test_data_validate_cli_rejects_non_campaign_without_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_sample(*_args: object, **_kwargs: object) -> object:
        raise TrainingEligibilityError("not training eligible")

    monkeypatch.setattr("aeromap.cli.load_sample", fake_load_sample)
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()

    result = CliRunner().invoke(app, ["data", "validate", str(sample_dir)])

    assert result.exit_code == 2
    assert "not training eligible" in result.stderr


def test_data_register_cli_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = DataSampleManifest.model_validate(
        {
            "sample_id": "sample_demo",
            "geometry_id": "geometry_demo",
            "state_id": "state_demo",
            "simulation_id": "simulation_demo",
            "attempt_id": "case_demo",
            "case_class": "CAMPAIGN_REFERENCE_CFD",
            "training_eligible": True,
            "source_case_dir": "case",
            "arrays_path": "arrays.npz",
            "arrays_sha256": "abc",
            "reference": {},
            "counts": {},
            "array_names": [],
            "vtk_workflow": {
                "surface_adapter": "surface",
                "volume_adapter": "volume",
                "surface_path": "wall.vtp",
                "volume_path": "volume.vtu",
                "semantics": {"surface": "wall", "volume": "cellID"},
            },
            "volume_provenance": {
                "source_openfoam_cell_count": 1,
                "exported_vtu_cell_count": 1,
                "cellid_count": 1,
                "cellid_unique_source_count": 1,
                "cellid_missing_source_count": 0,
                "cellid_min": 0,
                "cellid_max": 0,
                "cellid_maps_all_exported_cells": True,
                "cellid_covers_all_source_cells": True,
                "duplicated_source_cell_count": 0,
                "duplicated_exported_child_cell_count": 0,
                "foam_to_vtk_decomposition": {
                    "source_polyhedra_decomposed": 0,
                    "child_tetrahedra": 0,
                    "child_pyramids": 0,
                    "exported_child_cells": 0,
                    "net_exported_cell_increase": 0,
                },
                "duplicated_child_field_validation": {},
                "source_reduction_semantics": "aggregate through cellID",
            },
            "field_validation": {
                "checks": {
                    "surface_cp": {
                        "equation": "rho * p / q_inf",
                        "dimensional_array": "surface_pressure_kinematic",
                        "nondimensional_array": "surface_cp",
                        "max_abs_error": 0.0,
                        "tolerance": 1e-12,
                        "passed": True,
                    },
                },
            },
            "loads": {},
            "quality": {},
        },
    )

    def fake_load_sample(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(manifest=manifest)

    monkeypatch.setattr("aeromap.cli.load_sample", fake_load_sample)
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    registry = tmp_path / "registry.jsonl"

    first = CliRunner().invoke(
        app,
        ["data", "register", str(sample_dir), "--registry", str(registry)],
    )
    second = CliRunner().invoke(
        app,
        ["data", "register", str(sample_dir), "--registry", str(registry)],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert len(registry.read_text(encoding="utf-8").splitlines()) == 1


def test_surface_candidates_cli_outputs_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_generate_surface_candidates(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            out_dir=tmp_path,
            manifest_path=tmp_path / "surface_candidate_matrix.json",
            candidates=(
                SimpleNamespace(
                    candidate_id="surface_candidate_demo",
                    candidate_dir=tmp_path / "candidate",
                    status="EXPORTED",
                    stl_path=tmp_path / "candidate" / "article.stl",
                    metrics_path=tmp_path / "candidate" / "surface_candidate_metrics.json",
                ),
            ),
        )

    monkeypatch.setattr(
        "aeromap.cli.generate_surface_candidates",
        fake_generate_surface_candidates,
    )

    result = CliRunner().invoke(
        app,
        ["geometry", "surface-candidates", "--out", str(tmp_path), "--skip-gmsh"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["manifest_path"] == str(tmp_path / "surface_candidate_matrix.json")
    assert payload["candidates"][0]["candidate_id"] == "surface_candidate_demo"


def test_reference_lane_cli_dry_run_outputs_non_headline_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_reference_lane(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            case_name="drivaerFastback",
            mode="inspect",
            out_dir=tmp_path / "drivaerFastback",
            summary_path=None,
            return_code=None,
        )

    monkeypatch.setattr("aeromap.cli.run_reference_lane", fake_run_reference_lane)

    result = CliRunner().invoke(
        app,
        [
            "cfd",
            "reference-lane",
            "--case",
            "drivaerFastback",
            "--mode",
            "inspect",
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["non_headline"] is True
    assert payload["dry_run"] is True


def test_topology_report_cli_outputs_report_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    surface_path = tmp_path / "surface.json"
    mesh_path = tmp_path / "mesh.json"
    surface_path.write_text("{}\n", encoding="utf-8")
    mesh_path.write_text("{}\n", encoding="utf-8")

    def fake_write_topology_report(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            report_json_path=tmp_path / "report" / "topology_report.json",
            report_markdown_path=tmp_path / "report" / "topology_report.md",
        )

    monkeypatch.setattr("aeromap.cli.write_topology_report", fake_write_topology_report)

    result = CliRunner().invoke(
        app,
        [
            "cfd",
            "topology-report",
            str(case_dir),
            "--surface-diagnostics",
            str(surface_path),
            "--mesh-diagnostics",
            str(mesh_path),
            "--out",
            str(tmp_path / "report"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["report_json_path"] == str(tmp_path / "report" / "topology_report.json")
    assert payload["report_markdown_path"] == str(tmp_path / "report" / "topology_report.md")

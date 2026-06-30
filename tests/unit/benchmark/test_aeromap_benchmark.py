from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from typer.testing import CliRunner

from aeromap.benchmarks.aeromap import (
    AEROMAP_CLASS,
    AeroMapConfig,
    airfrans_case_features_from_name,
    build_aeromap_plan,
    build_airfrans_scalar_dataset,
    load_dataset_npz,
    write_active_learning_replay,
    write_airfrans_feasibility,
    write_airfrans_v02_audit,
    write_decision_replay_v02,
    write_decision_replay_v03,
    write_fixture_dataset,
)
from aeromap.benchmarks.aeromap3d import (
    AEROMAP3D_CLASSIFICATION,
    build_drivaerml_scalar_bridge_dataset_from_paths,
    write_geometry_readiness_sample,
)
from aeromap.benchmarks.airfrans_field import run_airfrans_surface_pressure_baseline
from aeromap.cli import app


def _small_config() -> AeroMapConfig:
    return AeroMapConfig(
        fixture_case_count=72,
        initial_labels=12,
        acquisition_batch=8,
        max_labels=36,
        test_fraction=0.25,
        ensemble_members=3,
    )


def test_aeromap_plan_sets_claim_boundaries() -> None:
    plan = build_aeromap_plan(_small_config())

    assert plan["benchmark_class"] == AEROMAP_CLASS
    assert plan["headline"] == "AeroMap Mission Control"
    assert plan["lanes"]["lane_a"]["name"] == "AEROMAP_BUDGET_BENCHMARK"
    assert plan["lanes"]["lane_b"]["name"] == "AEROCLIFF_CORE_EXTENSION"
    assert plan["cost_policy"] == {
        "cloud": "forbidden_for_this_goal",
        "ec2": "forbidden_for_this_goal",
        "nim": "forbidden_for_this_goal",
        "custom_cfd_solves": "forbidden_for_this_goal",
    }
    assert plan["claim_boundaries"]["not_aerocliff_cfd"] is True
    assert plan["claim_boundaries"]["aerocliff_accuracy_claim"] is False
    assert "stop if AirfRANS scalar targets are ambiguous" in plan["stop_conditions"]


def test_fixture_dataset_round_trips_and_preserves_non_claim_boundary(tmp_path: Path) -> None:
    manifest_path = write_fixture_dataset(_small_config(), tmp_path / "fixture.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_dataset_npz(Path(manifest["npz_path"]))

    assert manifest["classification"] == "AEROMAP_AIRFRANS_CONTRACT_FIXTURE"
    assert manifest["claim_boundary"] == {
        "aerocliff_result": False,
        "open_cfd_result": False,
        "purpose": "exercise Mission Control active-learning code path locally",
    }
    assert dataset.features.shape == (72, 6)
    assert dataset.targets.shape == (72, 2)
    assert dataset.target_names == ["integrated_cd", "integrated_cl"]


def test_active_learning_replay_writes_expected_methods(tmp_path: Path) -> None:
    config = _small_config()
    manifest_path = write_fixture_dataset(config, tmp_path / "fixture.json")
    dataset_npz = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["npz_path"])
    report_path = write_active_learning_replay(
        dataset_npz,
        config,
        tmp_path / "replay.json",
        svg_out=tmp_path / "curve.svg",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AEROMAP_CONTRACT_FIXTURE_ACTIVE_REPLAY"
    assert report["claim_boundary"]["open_cfd_result"] is False
    assert report["claim_boundary"]["aerocliff_result"] is False
    assert set(report["final_metrics_by_method"]) == set(config.acquisition_methods)
    assert "engineering_utility" in report["learning_curve_area_rmse_cd"]
    assert (tmp_path / "curve.svg").read_text(encoding="utf-8").startswith("<svg")
    for metrics in report["final_metrics_by_method"].values():
        assert metrics["rmse_cd"] >= 0.0
        assert metrics["rmse_cl"] >= 0.0
        assert -1.0 <= metrics["spearman_efficiency"] <= 1.0
        assert 0.0 <= metrics["top_k_efficiency_overlap"] <= 1.0


def test_decision_replay_v02_reports_split_modes_and_regret(tmp_path: Path) -> None:
    config = AeroMapConfig(
        fixture_case_count=96,
        initial_labels=16,
        acquisition_batch=8,
        max_labels=32,
        test_fraction=0.25,
        ensemble_members=3,
        replay_seeds=(101, 102),
        acquisition_methods=(
            "random",
            "diversity",
            "uncertainty_plus_diversity",
            "engineering_decision_utility_v1",
        ),
    )
    base_manifest = write_fixture_dataset(config, tmp_path / "fixture.json")
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_dataset = load_dataset_npz(Path(base_payload["npz_path"]))
    geom_a = base_dataset.features[:, [0, 1]]
    geom_b = base_dataset.features[:, [4, 5]]
    enriched_features = np.column_stack([base_dataset.features, geom_a, geom_b])
    npz_path = tmp_path / "geometry_fixture.npz"
    np.savez_compressed(
        npz_path,
        features=enriched_features,
        targets=base_dataset.targets,
        case_ids=np.array(base_dataset.case_ids),
        feature_names=np.array(
            [
                *base_dataset.feature_names,
                "geom_thickness_cos_0",
                "geom_camber_cos_0",
                "geom_max_thickness",
                "geom_max_abs_camber",
            ],
        ),
        target_names=np.array(base_dataset.target_names),
        classification=np.array("AIRFRANS_REAL_GEOMETRY_SCALAR_DATASET"),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.array([f"group_{idx}" for idx in range(96)]),
    )

    report_path = write_decision_replay_v02(
        npz_path,
        config,
        tmp_path / "decision.json",
        svg_dir=tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AIRFRANS_REAL_SCALAR_DECISION_REPLAY_V02"
    assert report["claim_boundary"]["open_cfd_result"] is True
    assert set(report["split_modes"]) == {"map_completion", "geometry_heldout"}
    assert report["split_reports"]["map_completion"]["seed_count"] == 2
    assert "best_design_regret" in report["method_winners"]["map_completion"]
    assert report["engineering_decision_utility_v1_assessment"]["geometry_heldout"]["available"]
    assert (tmp_path / "airfrans_decision_map_completion_topk.svg").exists()


def test_decision_replay_v03_reports_statistics_and_regret_aware_method(
    tmp_path: Path,
) -> None:
    config = AeroMapConfig(
        fixture_case_count=96,
        initial_labels=16,
        acquisition_batch=8,
        max_labels=32,
        test_fraction=0.25,
        ensemble_members=3,
        replay_seeds=(101, 102, 103),
        acquisition_methods=(
            "random",
            "diversity",
            "uncertainty_plus_diversity",
            "engineering_decision_utility_v1",
            "engineering_decision_utility_v2_regret_aware",
        ),
    )
    base_manifest = write_fixture_dataset(config, tmp_path / "fixture.json")
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_dataset = load_dataset_npz(Path(base_payload["npz_path"]))
    enriched_features = np.column_stack(
        [
            base_dataset.features,
            base_dataset.features[:, [0, 1]],
            base_dataset.features[:, [4, 5]],
        ],
    )
    npz_path = tmp_path / "geometry_fixture.npz"
    np.savez_compressed(
        npz_path,
        features=enriched_features,
        targets=base_dataset.targets,
        case_ids=np.array(base_dataset.case_ids),
        feature_names=np.array(
            [
                *base_dataset.feature_names,
                "geom_thickness_cos_0",
                "geom_camber_cos_0",
                "geom_max_thickness",
                "geom_max_abs_camber",
            ],
        ),
        target_names=np.array(base_dataset.target_names),
        classification=np.array("AIRFRANS_REAL_GEOMETRY_SCALAR_DATASET"),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.array([f"group_{idx}" for idx in range(96)]),
    )

    report_path = write_decision_replay_v03(
        npz_path,
        config,
        tmp_path / "decision_v03.json",
        svg_dir=tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AIRFRANS_AEROMAP_V0_3_DECISION_REPLAY"
    assert set(report["split_modes"]) == {"map_completion", "geometry_heldout"}
    assert (
        "engineering_decision_utility_v2_regret_aware"
        in report["method_winners"]["geometry_heldout"]["best_design_regret"]
        or "engineering_decision_utility_v2_regret_aware"
        in report["split_reports"]["geometry_heldout"]["final_metrics_by_method"]
    )
    assert report["statistics"]["geometry_heldout"]["budget_statistics"]
    assert report["statistics"]["geometry_heldout"]["paired_differences"]
    assert report["statistics"]["geometry_heldout"]["learning_curve_area"]
    assert "release_headline_ready" in report["headline_readiness"]
    assert (tmp_path / "airfrans_v03_map_completion_regret.svg").exists()


def _write_airfrans_surface_fixture(root: Path, case_id: str, *, pressure_offset: float) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    x = np.linspace(0.0, 1.0, 9)
    y = 0.04 * np.sin(np.pi * x) + 0.001 * pressure_offset
    points = np.column_stack([x, y, np.full_like(x, 0.5)])
    lines = []
    for idx in range(points.shape[0] - 1):
        lines.extend([2, idx, idx + 1])
    mesh = pv.PolyData(points, lines=np.asarray(lines))
    cell_count = points.shape[0] - 1
    centers_x = 0.5 * (x[:-1] + x[1:])
    pressure = pressure_offset + 5.0 * centers_x + 0.5 * centers_x**2
    mesh.cell_data["p"] = pressure.astype(np.float32)
    mesh.cell_data["Normals"] = np.tile(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (cell_count, 1)
    )
    mesh.cell_data["Length"] = np.full(cell_count, 1.0 / cell_count, dtype=np.float64)
    mesh.save(case_dir / f"{case_id}_aerofoil.vtp")


def _write_airfrans_surface_fixture_dataset(tmp_path: Path) -> Path:
    processed_root = tmp_path / "processed"
    dataset_root = processed_root / "Dataset"
    train = [
        "airFoil2D_SST_40.0_0.0_0.5_3.0_0.0_8.0",
        "airFoil2D_SST_42.0_2.0_0.6_3.5_1.0_9.0",
        "airFoil2D_SST_44.0_4.0_0.7_4.0_0.0_10.0",
        "airFoil2D_SST_46.0_6.0_0.8_4.5_1.0_11.0",
    ]
    test = [
        "airFoil2D_SST_48.0_8.0_0.9_5.0_0.0_12.0",
        "airFoil2D_SST_50.0_10.0_1.0_5.5_1.0_13.0",
    ]
    for idx, case_id in enumerate([*train, *test]):
        _write_airfrans_surface_fixture(dataset_root, case_id, pressure_offset=float(idx))
    (dataset_root / "manifest.json").write_text(
        json.dumps({"full_train": train, "full_test": test}),
        encoding="utf-8",
    )
    return processed_root


def test_airfrans_surface_pressure_baseline_runs_on_fixture(tmp_path: Path) -> None:
    processed_root = _write_airfrans_surface_fixture_dataset(tmp_path)
    report_path = run_airfrans_surface_pressure_baseline(
        root=processed_root,
        out=tmp_path / "field.json",
        visual_out=tmp_path / "field.png",
        summary_plot_out=tmp_path / "summary.png",
        train_cases=3,
        val_cases=1,
        test_cases=2,
        epochs=1,
        batch_size=16,
        hidden_width=8,
        seed=123,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AEROMAP_AIRFRANS_SURFACE_PRESSURE_FIELD_BASELINE_V0_1"
    assert report["split"]["train_cases"] == 3
    assert report["split"]["test_cases"] == 2
    assert set(report["metrics"]["by_method"]) == {"train_mean", "nearest_case", "pointwise_mlp"}
    assert report["claim_boundary"]["field_level_baseline"] is True
    assert (tmp_path / "field.png").exists()
    assert (tmp_path / "summary.png").exists()


def test_airfrans_surface_pressure_baseline_cli_runs_on_fixture(tmp_path: Path) -> None:
    processed_root = _write_airfrans_surface_fixture_dataset(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "airfrans-field-baseline",
            "--root",
            str(processed_root),
            "--out",
            str(tmp_path / "field_cli.json"),
            "--visual-out",
            str(tmp_path / "field_cli.png"),
            "--summary-plot-out",
            str(tmp_path / "summary_cli.png"),
            "--train-cases",
            "3",
            "--val-cases",
            "1",
            "--test-cases",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--hidden-width",
            "8",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["classification"] == "AEROMAP_AIRFRANS_SURFACE_PRESSURE_FIELD_BASELINE_V0_1"
    assert payload["train_cases"] == 3
    assert payload["test_cases"] == 2
    assert payload["best_method_by_rmse"] in {"train_mean", "nearest_case", "pointwise_mlp"}
    assert Path(payload["path"]).exists()
    assert (tmp_path / "field_cli.png").exists()
    assert (tmp_path / "summary_cli.png").exists()


def test_aeromap3d_scalar_bridge_dataset_uses_compact_drivaerml_csvs(
    tmp_path: Path,
) -> None:
    force_constref = tmp_path / "force_mom_constref_all.csv"
    force_varref = tmp_path / "force_mom_all.csv"
    geo = tmp_path / "geo_parameters_all.csv"
    force_rows = ["run, cd, cl, clf, clr, cs"]
    geo_rows = [
        "Run, Vehicle_Length,Vehicle_Width,Vehicle_Height,Rear_Diffusor_Angle,Vehicle_Ride_Height"
    ]
    for idx in range(1, 56):
        force_rows.append(f"{idx},{0.20 + idx * 0.001:.6f},{0.02 + idx * 0.0005:.6f},0,0,0")
        geo_rows.append(f"{idx},{4.5 + idx * 0.001:.6f},1.8,1.4,{3.0 + idx * 0.01:.6f},0.12")
    force_constref.write_text("\n".join(force_rows), encoding="utf-8")
    force_varref.write_text("\n".join(force_rows), encoding="utf-8")
    geo.write_text("\n".join(geo_rows), encoding="utf-8")

    report_path = build_drivaerml_scalar_bridge_dataset_from_paths(
        force_constref_path=force_constref,
        force_varref_path=force_varref,
        geo_path=geo,
        out=tmp_path / "bridge.json",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = load_dataset_npz(Path(report["npz_path"]))

    assert report["classification"] == AEROMAP3D_CLASSIFICATION
    assert report["case_count"] == 55
    assert report["claim_boundary"]["compact_3d_scalar_bridge"] is True
    assert report["claim_boundary"]["f1_geometry"] is False
    assert dataset.features.shape == (55, 5)
    assert dataset.targets.shape == (55, 2)
    assert dataset.classification == AEROMAP3D_CLASSIFICATION
    assert dataset.open_cfd_result is True


def test_aeromap3d_geometry_sample_reads_tiny_stl(tmp_path: Path) -> None:
    stl = tmp_path / "tiny.stl"
    stl.write_text(
        """solid tiny
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid tiny
""",
        encoding="utf-8",
    )

    report_path = write_geometry_readiness_sample(
        [stl],
        tmp_path / "geometry.json",
        sample_points_per_geometry=64,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AEROMAP_3D_GEOMETRY_READINESS_SAMPLE"
    assert report["stl_count"] == 1
    assert report["stl_summaries"][0]["triangle_count"] == 1
    assert report["claim_boundary"]["accuracy_result"] is False
    assert Path(report["points_npz_path"]).exists()


def test_decision_replay_v03_classifies_3d_scalar_bridge(tmp_path: Path) -> None:
    config = AeroMapConfig(
        preferred_dataset="DrivAerML",
        fixture_case_count=96,
        initial_labels=16,
        acquisition_batch=8,
        max_labels=32,
        test_fraction=0.25,
        ensemble_members=3,
        replay_seeds=(201, 202),
        acquisition_methods=(
            "random",
            "diversity",
            "uncertainty_plus_diversity",
            "engineering_decision_utility_v2_regret_aware",
        ),
    )
    base_manifest = write_fixture_dataset(config, tmp_path / "fixture.json")
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_dataset = load_dataset_npz(Path(base_payload["npz_path"]))
    npz_path = tmp_path / "aeromap3d_fixture.npz"
    np.savez_compressed(
        npz_path,
        features=base_dataset.features,
        targets=base_dataset.targets,
        case_ids=np.array([f"drivaerml_run_{idx}" for idx in range(96)]),
        feature_names=np.array([f"geom_{name}" for name in base_dataset.feature_names]),
        target_names=np.array(base_dataset.target_names),
        classification=np.array(AEROMAP3D_CLASSIFICATION),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.array([f"drivaerml_geometry_{idx}" for idx in range(96)]),
    )

    report_path = write_decision_replay_v03(
        npz_path,
        config,
        tmp_path / "decision_3d.json",
        svg_dir=tmp_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AEROMAP_3D_OPEN_CFD_SCALAR_BRIDGE"
    assert report["dataset_classification"] == AEROMAP3D_CLASSIFICATION
    assert (
        report["claim_boundary"]["active_learning_claim"]
        == "compact_3d_open_cfd_scalar_bridge_offline_replay"
    )
    assert (tmp_path / "aeromap3d_v03_map_completion_topk.svg").exists()


def test_airfrans_v02_audit_reports_feature_and_split_contracts(tmp_path: Path) -> None:
    config = AeroMapConfig(
        fixture_case_count=64,
        initial_labels=12,
        acquisition_batch=8,
        max_labels=28,
        test_fraction=0.25,
        ensemble_members=3,
        acquisition_methods=("random", "engineering_decision_utility_v2_regret_aware"),
    )
    base_manifest = write_fixture_dataset(config, tmp_path / "fixture.json")
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_dataset = load_dataset_npz(Path(base_payload["npz_path"]))
    enriched_features = np.column_stack([base_dataset.features, base_dataset.features[:, [0, 1]]])
    feature_names = [
        *base_dataset.feature_names,
        "geom_thickness_cos_0",
        "geom_camber_cos_0",
    ]
    npz_path = tmp_path / "geometry_fixture.npz"
    np.savez_compressed(
        npz_path,
        features=enriched_features,
        targets=base_dataset.targets,
        case_ids=np.array(base_dataset.case_ids),
        feature_names=np.array(feature_names),
        target_names=np.array(base_dataset.target_names),
        classification=np.array("AIRFRANS_REAL_GEOMETRY_SCALAR_DATASET"),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.array([f"group_{idx}" for idx in range(64)]),
    )
    contract = {
        "features": [
            {
                "name": name,
                "category": "geometry_descriptor" if name.startswith("geom_") else "fixture",
                "target_leakage": False,
            }
            for name in feature_names
        ],
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    audit_path = write_airfrans_v02_audit(
        npz_path,
        contract_path,
        config,
        tmp_path / "audit.json",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit["classification"] == "AIRFRANS_AEROMAP_V0_3_AUDIT"
    assert audit["feature_contract"]["passes_no_target_leakage_check"] is True
    assert audit["split_audit"]["geometry_heldout"]["same_geometry_leakage_detected"] is False
    assert audit["claim_boundary"]["aerocliff_result"] is False


def test_airfrans_feasibility_never_downloads_dataset(tmp_path: Path) -> None:
    report_path = write_airfrans_feasibility(
        tmp_path / "airfrans.json",
        data_root=tmp_path / "data",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["classification"] == "AIRFRANS_FEASIBILITY"
    assert report["dataset_archive"]["download_attempted"] is False
    assert report["mvp_decision"]["fixture_replay_allowed"] is True
    assert report["target_contract"]["fail_closed"] is True


def test_airfrans_case_feature_parser_uses_velocity_aoa_and_shape() -> None:
    features, names = airfrans_case_features_from_name("airFoil2D_SST_30_4_0_0_1_2_extra")

    assert names[:5] == [
        "inlet_velocity_m_s",
        "aoa_deg",
        "sin_aoa",
        "cos_aoa",
        "log10_reynolds",
    ]
    assert features[0] == 30.0
    assert features[1] == 4.0
    assert len(features) == len(names)


def test_airfrans_scalar_dataset_uses_force_coefficient_contract(tmp_path: Path) -> None:
    root = tmp_path / "airfrans"
    root.mkdir()
    manifest = {
        "full_train": [
            "airFoil2D_SST_30_4_0_0_1_2_extra",
            "airFoil2D_SST_40_6_0_0_1_5_extra",
        ],
        "full_test": ["airFoil2D_SST_35_2_0_0_1_0_extra"],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeSimulation:
        def __init__(self, _root: Path, name: str) -> None:
            self.name = name

        def force_coefficient(
            self, *, reference: bool
        ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            assert reference is True
            velocity = float(self.name.split("_")[2])
            aoa = float(self.name.split("_")[3])
            return (velocity / 1000.0, 0.0, 0.0), (aoa / 10.0, 0.0, 0.0)

    def factory(fake_root: Path, name: str) -> FakeSimulation:
        assert fake_root == root
        return FakeSimulation(fake_root, name)

    out = build_airfrans_scalar_dataset(root, tmp_path / "scalars.json", simulation_factory=factory)
    payload = json.loads(out.read_text(encoding="utf-8"))
    dataset = load_dataset_npz(Path(payload["npz_path"]))

    assert payload["classification"] == "AIRFRANS_REAL_SCALAR_DATASET"
    assert payload["completed_case_count"] == 3
    assert payload["failed_case_count"] == 0
    assert payload["claim_boundary"]["open_cfd_result"] is True
    assert payload["claim_boundary"]["aerocliff_result"] is False
    assert dataset.targets.shape == (3, 2)
    assert sorted(dataset.targets[:, 0].round(3).tolist()) == [0.03, 0.035, 0.04]


def test_aeromap_cli_plan_fixture_and_replay(tmp_path: Path) -> None:
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(
        """fixture_case_count: 72
initial_labels: 12
acquisition_batch: 8
max_labels: 36
test_fraction: 0.25
ensemble_members: 3
""",
        encoding="utf-8",
    )

    plan = tmp_path / "plan.json"
    result = runner.invoke(
        app,
        ["benchmark", "aeromap-plan", "--config", str(config), "--out", str(plan)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(plan.read_text(encoding="utf-8"))["benchmark_class"] == AEROMAP_CLASS

    fixture = tmp_path / "fixture.json"
    result = runner.invoke(
        app,
        ["benchmark", "aeromap-fixture", "--config", str(config), "--out", str(fixture)],
    )
    assert result.exit_code == 0, result.output
    fixture_payload = json.loads(fixture.read_text(encoding="utf-8"))

    replay = tmp_path / "replay.json"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "aeromap-replay",
            "--config",
            str(config),
            "--dataset-npz",
            fixture_payload["npz_path"],
            "--out",
            str(replay),
            "--svg-out",
            str(tmp_path / "curve.svg"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(replay.read_text(encoding="utf-8"))["benchmark_class"] == AEROMAP_CLASS

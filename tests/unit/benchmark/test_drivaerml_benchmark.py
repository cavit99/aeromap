from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import yaml
from pydantic import ValidationError

from aeromap.benchmarks.drivaerml import (
    DrivAerMLBenchmarkConfig,
    build_drivaerml_benchmark_plan,
    load_drivaerml_fixture_sample,
    validate_drivaerml_asset_manifest,
    write_drivaerml_asset_manifest,
    write_drivaerml_benchmark_plan,
    write_drivaerml_cuda_bundle,
    write_drivaerml_heldout_pilot_split,
    write_drivaerml_sample_manifest,
    write_drivaerml_sampling_manifest,
)


def test_drivaerml_benchmark_plan_is_external_and_cost_guarded() -> None:
    plan = build_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig())

    assert plan["benchmark_class"] == "EXTERNAL_DRIVAERML_POOL_BENCHMARK"
    assert plan["aerocliff_geometry"] is False
    assert plan["aerocliff_campaign_cfd"] is False
    assert plan["live_solver_loop"] is False
    assert plan["offline_pool_replay"] is True
    assert plan["training_eligibility"] == {
        "external_benchmark": True,
        "aerocliff_custom_model": False,
    }
    assert plan["claim_eligibility"] == {
        "external_label_efficiency": True,
        "aerocliff_accuracy": False,
        "aerocliff_solver_loop": False,
        "ground_effect_cliff": False,
    }
    assert "AeroCliff campaign CFD" in plan["prohibited_claims"]
    assert "AeroCliff Venturi-underfloor cliff discovery" in plan["prohibited_claims"]
    assert plan["download_policy"]["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert plan["assets"]["volume_vtu"] is False
    assert plan["cost_policy"]["ec2_usage"] == "not_required_for_plan_generation"
    assert "idle spot instances" in plan["cost_policy"]["avoid"]


def test_drivaerml_benchmark_split_counts_and_no_overlap() -> None:
    config = DrivAerMLBenchmarkConfig()
    plan = build_drivaerml_benchmark_plan(config)
    split = plan["split"]

    assert split["counts"] == {
        "initial_labelled": 16,
        "acquisition_pool": 64,
        "calibration": 32,
        "heldout_test": 32,
        "total": 144,
    }
    all_ids = [case_id for case_ids in split["case_ids"].values() for case_id in case_ids]
    assert len(all_ids) == 144
    assert len(set(all_ids)) == 144
    assert plan["active_learning"]["acquisition_rounds"] == 4
    assert plan["active_learning"]["sobol_reserved_for_aerocliff_parametric_campaign"] is True


def test_drivaerml_benchmark_config_rejects_volume_vtu_mvp() -> None:
    with pytest.raises(ValidationError, match="volume_vtu is disabled"):
        DrivAerMLBenchmarkConfig.model_validate({"assets": {"volume_vtu": True}})


def test_drivaerml_benchmark_config_file_matches_expected_mvp() -> None:
    payload = yaml.safe_load(
        Path("configs/benchmark/drivaerml_mvp.yaml").read_text(encoding="utf-8"),
    )
    config = DrivAerMLBenchmarkConfig.model_validate(payload)
    plan = build_drivaerml_benchmark_plan(config)

    assert plan["split"]["counts"]["total"] == 144
    assert plan["target_fields"] == [
        "surface_cp",
        "integrated_cd",
        "integrated_cl",
    ]
    assert plan["target_contract"]["integrated_cl"]["sign_convention"] == "positive_lift"
    assert plan["target_contract"]["surface_cp"]["association"] == "cell"
    assert plan["target_contract"]["surface_cp"]["weights"] == "cell_area"
    assert plan["excluded_targets"] == [
        "integrated_cm_pitch",
        "volume_fields",
        "wall_shear",
        "separation_fraction",
        "ground_effect_cliff",
    ]
    first_assets = plan["cases"]["initial_labelled"][0]["assets"]
    assert "geo_parameters_" in first_assets["geometry_parameters"]
    assert "force_mom_constref_" in first_assets["fixed_reference_forces"]


def test_write_drivaerml_benchmark_plan_round_trips_json(tmp_path: Path) -> None:
    out = tmp_path / "plan.json"

    result = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), out)
    loaded = json.loads(result.read_text(encoding="utf-8"))

    assert result == out
    assert loaded["schema_version"].startswith("aerocliff_external_drivaerml")
    assert loaded["cases"]["initial_labelled"][0]["hash_status"] == "PENDING_DOWNLOAD"


def test_drivaerml_asset_manifest_dry_run_is_plan_evidence_not_data(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=tmp_path / "cache",
        splits=("initial_labelled",),
        max_cases=2,
        dry_run=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["benchmark_class"] == "EXTERNAL_DRIVAERML_POOL_BENCHMARK"
    assert manifest["selected_case_count"] == 2
    assert manifest["download_policy"]["dry_run"] is True
    assert manifest["storage"]["downloaded_bytes"] == 0
    assert manifest["storage"]["local_download_budget_bytes"] == 0
    assert manifest["no_volume_vtu_confirmed"] is True
    assert manifest["data_ready"] is False
    assert {asset["status"] for asset in manifest["assets"]} == {"DRY_RUN_PLANNED"}

    validation = validate_drivaerml_asset_manifest(manifest_path)
    assert validation["ok"] is True
    assert validation["dry_run_plan_evidence"] is True
    assert validation["data_ready"] is False


def test_drivaerml_asset_manifest_hashes_existing_files_without_network(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    case = plan["cases"]["initial_labelled"][0]
    cache_dir = tmp_path / "cache"
    stl_path = cache_dir / case["case_id"] / f"drivaer_{case['run_number']}.stl"
    stl_path.parent.mkdir(parents=True)
    stl_path.write_text("solid tiny\nendsolid tiny\n", encoding="utf-8")

    manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=cache_dir,
        splits=(),
        case_ids=(case["case_id"],),
        dry_run=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stl_assets = [asset for asset in manifest["assets"] if asset["asset_name"] == "stl"]

    assert len(stl_assets) == 1
    assert stl_assets[0]["status"] == "PRESENT_HASHED"
    assert stl_assets[0]["byte_count"] == stl_path.stat().st_size
    assert isinstance(stl_assets[0]["sha256"], str)
    assert validate_drivaerml_asset_manifest(manifest_path)["ok"] is True


def test_drivaerml_asset_manifest_rejects_volume_vtu_plan(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["assets"]["volume_vtu"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="volume_vtu"):
        write_drivaerml_asset_manifest(
            plan_path,
            tmp_path / "assets.json",
            cache_dir=tmp_path / "cache",
            dry_run=True,
        )


def test_drivaerml_asset_manifest_case_id_only_reports_real_split(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    case = plan["cases"]["calibration"][0]

    manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=tmp_path / "cache",
        splits=(),
        case_ids=(case["case_id"],),
        dry_run=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["selected_case_ids"] == [case["case_id"]]
    assert manifest["selected_splits"] == ["calibration"]


def test_drivaerml_asset_manifest_rejects_untrusted_plan_entries(tmp_path: Path) -> None:
    plan = build_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig())
    plan["cases"]["initial_labelled"][0]["case_id"] = "../escape"
    malicious_plan = tmp_path / "malicious_case.json"
    malicious_plan.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid DrivAerML case_id"):
        write_drivaerml_asset_manifest(
            malicious_plan,
            tmp_path / "assets.json",
            cache_dir=tmp_path / "cache",
            dry_run=True,
        )

    plan = build_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig())
    plan["cases"]["initial_labelled"][0]["assets"]["stl"] = "https://example.com/drivaer_4.stl"
    untrusted_url_plan = tmp_path / "untrusted_url.json"
    untrusted_url_plan.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="host is not trusted"):
        write_drivaerml_asset_manifest(
            untrusted_url_plan,
            tmp_path / "assets.json",
            cache_dir=tmp_path / "cache",
            dry_run=True,
        )


def test_load_drivaerml_fixture_sample_preserves_external_targets(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    faces = np.array([3, 0, 1, 2])
    mesh = pv.PolyData(points, faces)
    mesh.cell_data["CpMeanTrim"] = np.array([0.1])
    mesh.save(tmp_path / "surface.vtp")
    (tmp_path / "loads.csv").write_text(
        "case_id,integrated_cd,integrated_cl\nrun_1,0.31,0.72\n",
        encoding="utf-8",
    )
    (tmp_path / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "run_1",
                "geometry_group": "run_1",
                "split": "initial_labelled",
                "benchmark_class": "EXTERNAL_DRIVAERML_POOL_BENCHMARK",
                "source": "DrivAerML",
                "license": "CC BY-SA 4.0",
                "citation_key": "ashton2024drivaer",
                "surface_vtp": "surface.vtp",
                "loads_csv": "loads.csv",
            },
        ),
        encoding="utf-8",
    )

    sample = load_drivaerml_fixture_sample(tmp_path)

    assert sample["case_id"] == "run_1"
    assert sample["geometry_group"] == "run_1"
    assert sample["split"] == "initial_labelled"
    assert sample["source"] == "DrivAerML"
    assert sample["license"] == "CC BY-SA 4.0"
    assert sample["surface_targets"]["surface_cp"] == [0.1]
    assert sample["surface_target_contract"]["surface_cp"]["association"] == "cell"
    assert sample["surface_target_contract"]["surface_cp"]["weights"] == "cell_area"
    assert sample["integrated_targets"] == {
        "integrated_cd": 0.31,
        "integrated_cl": 0.72,
    }
    assert sample["training_eligibility"]["external_benchmark"] is True
    assert sample["training_eligibility"]["aerocliff_custom_model"] is False


def test_load_drivaerml_fixture_sample_rejects_point_only_cp(tmp_path: Path) -> None:
    mesh = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([3, 0, 1, 2]),
    )
    mesh.point_data["Cp"] = np.array([0.1, -0.2, -0.3])
    mesh.save(tmp_path / "surface.vtp")
    (tmp_path / "loads.csv").write_text(
        "case_id,integrated_cd,integrated_cl\nrun_1,0.31,0.72\n",
        encoding="utf-8",
    )
    (tmp_path / "case_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "run_1",
                "geometry_group": "run_1",
                "split": "initial_labelled",
                "benchmark_class": "EXTERNAL_DRIVAERML_POOL_BENCHMARK",
                "source": "DrivAerML",
                "license": "CC BY-SA 4.0",
                "citation_key": "ashton2024drivaer",
                "surface_vtp": "surface.vtp",
                "loads_csv": "loads.csv",
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cell-associated"):
        load_drivaerml_fixture_sample(tmp_path)


def test_write_drivaerml_sample_manifest_from_local_assets(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    case = plan["cases"]["initial_labelled"][0]
    cache_dir = tmp_path / "cache"

    stl_path = cache_dir / case["case_id"] / f"drivaer_{case['run_number']}.stl"
    stl_path.parent.mkdir(parents=True)
    stl_path.write_text("solid tiny\nendsolid tiny\n", encoding="utf-8")
    geometry_path = cache_dir / case["case_id"] / f"geo_parameters_{case['run_number']}.csv"
    geometry_path.write_text("run,front_wheelbase\n4,1.1\n", encoding="utf-8")
    forces_path = cache_dir / case["case_id"] / f"force_mom_constref_{case['run_number']}.csv"
    forces_path.parent.mkdir(parents=True, exist_ok=True)
    forces_path.write_text("run,cd,cl,clf,clr\n4,0.29,0.63,0.31,0.32\n", encoding="utf-8")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    mesh = pv.PolyData(points, np.array([3, 0, 1, 2]))
    mesh.cell_data["CpMeanTrim"] = np.array([-0.2])
    mesh.save(cache_dir / case["case_id"] / f"boundary_{case['run_number']}.vtp")

    asset_manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=cache_dir,
        splits=(),
        case_ids=(case["case_id"],),
        dry_run=False,
    )
    sample_manifest_path = write_drivaerml_sample_manifest(
        asset_manifest_path,
        tmp_path / "samples.json",
    )
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))

    assert sample_manifest["benchmark_class"] == "EXTERNAL_DRIVAERML_POOL_BENCHMARK"
    assert sample_manifest["selected_case_count"] == 1
    assert sample_manifest["training_eligibility"]["external_benchmark"] is True
    assert sample_manifest["training_eligibility"]["aerocliff_custom_model"] is False
    assert sample_manifest["no_volume_vtu_confirmed"] is True
    sample = sample_manifest["cases"][0]
    assert sample["case_id"] == case["case_id"]
    assert sample["surface_targets"]["surface_cp"]["source_field"] == "CpMeanTrim"
    assert sample["surface_targets"]["surface_cp"]["association"] == "cell"
    assert sample["surface_targets"]["surface_cp"]["query_positions"] == "cell_centres"
    assert sample["surface_targets"]["surface_cp"]["weights"] == "cell_area"
    assert sample["surface_targets"]["surface_cp"]["finite_count"] == 1
    assert sample["integrated_targets"] == {
        "integrated_cd": 0.29,
        "integrated_cl": 0.63,
    }
    assert sample["scalar_diagnostics"] == {
        "integrated_cl_front": 0.31,
        "integrated_cl_rear": 0.32,
    }


def test_write_drivaerml_cuda_bundle_verifies_cell_cp_contract(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    case = plan["cases"]["initial_labelled"][0]
    cache_dir = tmp_path / "cache"

    stl_path = cache_dir / case["case_id"] / f"drivaer_{case['run_number']}.stl"
    stl_path.parent.mkdir(parents=True)
    stl_path.write_text("solid tiny\nendsolid tiny\n", encoding="utf-8")
    geometry_path = cache_dir / case["case_id"] / f"geo_parameters_{case['run_number']}.csv"
    geometry_path.write_text("run,front_wheelbase\n4,1.1\n", encoding="utf-8")
    forces_path = cache_dir / case["case_id"] / f"force_mom_constref_{case['run_number']}.csv"
    forces_path.parent.mkdir(parents=True, exist_ok=True)
    forces_path.write_text("run,cd,cl,clf,clr\n4,0.29,0.63,0.31,0.32\n", encoding="utf-8")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    mesh = pv.PolyData(points, np.array([3, 0, 1, 2]))
    mesh.cell_data["CpMeanTrim"] = np.array([-0.2])
    mesh.save(cache_dir / case["case_id"] / f"boundary_{case['run_number']}.vtp")

    asset_manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=cache_dir,
        splits=(),
        case_ids=(case["case_id"],),
        dry_run=False,
    )
    sample_manifest_path = write_drivaerml_sample_manifest(
        asset_manifest_path,
        tmp_path / "samples.json",
    )
    bundle_path = write_drivaerml_cuda_bundle(
        asset_manifest_path,
        sample_manifest_path,
        tmp_path / "cuda_bundle.json",
        dataset_revision="abc123",
        physicsnemo_commit="def456",
        physicsnemo_cfd_commit="789abc",
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert bundle["local_preparation_complete"] is True
    assert bundle["cuda_launch_ready"] is True
    assert bundle["cuda_launch_blockers"] == []
    assert bundle["training_eligibility"]["aerocliff_custom_model"] is False
    assert bundle["claim_eligibility"]["external_label_efficiency"] is True
    assert bundle["surface_checks"][0]["cell_centres_generated"] is True
    assert bundle["surface_checks"][0]["cell_areas_generated"] is True
    assert bundle["target_contract"]["surface_cp"]["weights"] == "cell_area"
    assert bundle["normalisation_reference_contract"]["surface_target"]["source_field"] == (
        "CpMeanTrim"
    )
    assert (
        bundle["runtime_captures"]["resolved_nim_image_digest"]["status"] == "captured_at_runtime"
    )
    assert (
        bundle["post_training_outputs"]["trained_corrector_checkpoint_digest"]["status"]
        == "captured_after_training"
    )
    assert bundle["cost_and_observability_policy"]["starting_instance_family"] == "g6e"


def test_write_drivaerml_sampling_manifest_records_deterministic_cells(tmp_path: Path) -> None:
    plan_path = write_drivaerml_benchmark_plan(DrivAerMLBenchmarkConfig(), tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    case = plan["cases"]["initial_labelled"][0]
    cache_dir = tmp_path / "cache"

    stl_path = cache_dir / case["case_id"] / f"drivaer_{case['run_number']}.stl"
    stl_path.parent.mkdir(parents=True)
    stl_path.write_text("solid tiny\nendsolid tiny\n", encoding="utf-8")
    geometry_path = cache_dir / case["case_id"] / f"geo_parameters_{case['run_number']}.csv"
    geometry_path.write_text("run,front_wheelbase\n4,1.1\n", encoding="utf-8")
    forces_path = cache_dir / case["case_id"] / f"force_mom_constref_{case['run_number']}.csv"
    forces_path.write_text("run,cd,cl,clf,clr\n4,0.29,0.63,0.31,0.32\n", encoding="utf-8")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
    )
    mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 3, 4, 5]))
    mesh.cell_data["CpMeanTrim"] = np.array([-0.2, 0.1])
    mesh.save(cache_dir / case["case_id"] / f"boundary_{case['run_number']}.vtp")

    asset_manifest_path = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "assets.json",
        cache_dir=cache_dir,
        splits=(),
        case_ids=(case["case_id"],),
        dry_run=False,
    )
    sample_manifest_path = write_drivaerml_sample_manifest(
        asset_manifest_path,
        tmp_path / "samples.json",
    )

    first = write_drivaerml_sampling_manifest(
        sample_manifest_path,
        tmp_path / "sampling_a.json",
        seed=7,
        sample_count=2,
    )
    second = write_drivaerml_sampling_manifest(
        sample_manifest_path,
        tmp_path / "sampling_b.json",
        seed=7,
        sample_count=2,
    )

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    first_case = first_payload["cases"][0]

    assert (
        first_payload["normalisation_reference_contract_sha256"]
        == second_payload["normalisation_reference_contract_sha256"]
    )
    assert first_case["sampled_cell_ids"] == second_payload["cases"][0]["sampled_cell_ids"]
    assert first_case["sample_count"] == 2
    assert len(first_case["sampling_probabilities"]) == 2
    assert len(first_case["sampled_cell_areas"]) == 2
    expected_areas = np.array([0.5, 2.0])
    expected_probabilities = expected_areas / expected_areas.sum()
    assert first_case["sampled_cell_ids"] == [0, 1]
    assert first_case["sampled_cell_areas"] == pytest.approx(expected_areas.tolist())
    assert first_case["sampling_probabilities"] == pytest.approx(
        expected_probabilities.tolist(),
    )
    assert first_payload["evaluation"]["surface"] == "full_surface"


def test_write_drivaerml_heldout_pilot_split_uses_geometry_maximin(
    tmp_path: Path,
) -> None:
    config = DrivAerMLBenchmarkConfig.model_validate(
        {
            "assets": {
                "stl": False,
                "boundary_vtp": False,
                "geometry_parameters": True,
                "fixed_reference_forces": False,
                "volume_vtu": False,
            },
        },
    )
    plan_path = write_drivaerml_benchmark_plan(config, tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected_records = [
        *plan["cases"]["initial_labelled"],
        *plan["cases"]["acquisition_pool"],
    ][:30]
    cache_dir = tmp_path / "cache"
    for record in selected_records:
        run_number = int(record["run_number"])
        geometry_path = cache_dir / record["case_id"] / f"geo_parameters_{run_number}.csv"
        geometry_path.parent.mkdir(parents=True)
        geometry_path.write_text(
            "run,length,width,roof_height,base_pressure_hint\n"
            f"{run_number},{1.0 + run_number * 0.01},{0.5 + (run_number % 7) * 0.02},"
            f"{0.2 + (run_number % 5) * 0.03},{1000 + run_number}\n",
            encoding="utf-8",
        )
    asset_manifest = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "geometry_assets.json",
        cache_dir=cache_dir,
        splits=("initial_labelled", "acquisition_pool"),
        max_cases=30,
        dry_run=False,
    )

    split_path = write_drivaerml_heldout_pilot_split(
        asset_manifest,
        tmp_path / "heldout_split.json",
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))

    train = split["case_ids"]["train"]
    heldout = split["case_ids"]["heldout_test"]
    selected_order = [record["case_id"] for record in selected_records[:24]]

    assert split["experiment_id"] == "DRIVAERML_24_CASE_HELDOUT_PILOT"
    assert split["counts"]["train"] == 16
    assert split["counts"]["heldout_test"] == 8
    assert len(train) == 16
    assert len(heldout) == 8
    assert not set(train) & set(heldout)
    assert split["selection_method"]["file_order_used_for_split"] is False
    assert split["selection_method"]["selection_standardisation_scope"] == (
        "available_geometry_cases_for_selection_only"
    )
    assert split["selection_method"]["model_feature_standardisation_scope"] == (
        "training_split_only"
    )
    assert heldout != selected_order[16:24]
    train_lengths = []
    train_widths = []
    for case_id in train:
        run_number = int(case_id.removeprefix("run_"))
        train_lengths.append(1.0 + run_number * 0.01)
        train_widths.append(0.5 + (run_number % 7) * 0.02)
    assert split["geometry_parameter_standardisation_scope"] == "training_split_only"
    assert split["geometry_parameter_standardisation"]["mean"]["length"] == pytest.approx(
        float(np.mean(train_lengths)),
    )
    assert split["geometry_parameter_standardisation"]["mean"]["width"] == pytest.approx(
        float(np.mean(train_widths)),
    )
    assert split["geometry_parameter_selection_standardisation"]["mean"]["length"] != pytest.approx(
        float(np.mean(train_lengths)),
    )
    assert split["label_access_policy"] == {
        "test_labels_may_be_used_for_preprocessing_statistics": False,
        "test_labels_may_be_used_for_hyperparameter_selection": False,
        "test_labels_may_be_used_for_early_stopping": False,
        "test_labels_may_be_used_for_checkpoint_selection": False,
        "residual_normalisation_source": "training_split_only",
        "epoch_budget": "fixed_predeclared",
    }
    assert split["target_scope"]["primary_target"] == "surface_cp"
    assert "wall_shear" in split["target_scope"]["excluded"]
    assert split["aerocliff_claim_eligible"] is False
    assert split["active_learning_claim_eligible"] is False


def test_write_drivaerml_heldout_pilot_split_requires_geometry_features(
    tmp_path: Path,
) -> None:
    config = DrivAerMLBenchmarkConfig.model_validate(
        {
            "assets": {
                "stl": False,
                "boundary_vtp": False,
                "geometry_parameters": True,
                "fixed_reference_forces": False,
                "volume_vtu": False,
            },
        },
    )
    plan_path = write_drivaerml_benchmark_plan(config, tmp_path / "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records = [
        *plan["cases"]["initial_labelled"],
        *plan["cases"]["acquisition_pool"],
    ][:24]
    cache_dir = tmp_path / "cache"
    for record in records:
        run_number = int(record["run_number"])
        geometry_path = cache_dir / record["case_id"] / f"geo_parameters_{run_number}.csv"
        geometry_path.parent.mkdir(parents=True)
        geometry_path.write_text(f"run,body_style\n{run_number},fastback\n", encoding="utf-8")
    asset_manifest = write_drivaerml_asset_manifest(
        plan_path,
        tmp_path / "geometry_assets.json",
        cache_dir=cache_dir,
        splits=("initial_labelled", "acquisition_pool"),
        max_cases=24,
        dry_run=False,
    )

    with pytest.raises(ValueError, match="No finite non-label geometry parameters"):
        write_drivaerml_heldout_pilot_split(
            asset_manifest,
            tmp_path / "heldout_split.json",
        )

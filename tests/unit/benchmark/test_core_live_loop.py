from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeromap.benchmarks.core_live_loop import write_live_core_acquisition_loop


def _case(ride_height_mm: float, angle_deg: float, index: int) -> dict[str, object]:
    return {
        "case_id": f"core_case_{index}",
        "ride_height_mm": ride_height_mm,
        "diffuser_angle_deg": angle_deg,
        "throat_ratio": 0.7,
        "clean_for_2d_response_replay": True,
        "source": "unit_fixture",
        "target_status": "medium_response_map_observation",
        "C_D": 0.015 + 0.0002 * angle_deg + 0.00001 * ride_height_mm,
        "suction_downforce": 1.0 + 0.012 * (80.0 - ride_height_mm) - 0.018 * angle_deg,
        "pressure_recovery": 1.8 + 0.01 * (ride_height_mm - 50.0) - 0.04 * angle_deg,
        "diagnostic_corrected_f_sep": 0.0,
        "diagnostic_near_wall_reverse_fraction": 0.0,
        "diagnostic_diffuser_y_plus_mean": 60.0,
        "diagnostic_diffuser_y_plus_max": 90.0,
    }


def _write_dataset(path: Path) -> None:
    cases = [
        _case(50.0, 3.0, 1),
        _case(50.0, 5.0, 2),
        _case(60.0, 5.0, 3),
        _case(60.0, 7.0, 4),
        _case(70.0, 3.0, 5),
        _case(70.0, 7.0, 6),
    ]
    payload = {
        "schema_version": "unit",
        "classification": "VENTURI_CORE_2D_PRESSURE_LOAD_RESPONSE_DATASET_V0",
        "accepted": True,
        "training_eligible": False,
        "cases": cases,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_live_core_acquisition_loop_reveals_committed_core_cases(tmp_path: Path) -> None:
    dataset_path = tmp_path / "core_dataset.json"
    _write_dataset(dataset_path)

    manifest_path = write_live_core_acquisition_loop(
        dataset_path=dataset_path,
        output_dir=tmp_path / "out",
        report_path=tmp_path / "report.md",
        max_iterations=2,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["classification"] == "AEROMAP_LIVE_CORE_ACQUISITION_LOOP_V0_1"
    assert payload["mode_executed"] == "replay-live"
    assert payload["openfoam_runs"] == []
    assert payload["allowed_targets"] == ["C_D", "suction_downforce", "pressure_recovery"]
    assert payload["claim_boundary"]["wall_shear_label"] is False
    assert payload["claim_boundary"]["industrial_live_cfd_savings"] is False
    assert payload["primary_loop"]["completed_iterations"] == 2
    assert len(payload["primary_loop"]["selections"]) == 2
    assert payload["policy_comparison"]["random_baseline_seed_count"] == 10
    assert (tmp_path / "out" / "live_core_loop_learning_curve.svg").exists()
    assert (tmp_path / "report.md").exists()


def test_write_live_core_acquisition_loop_rejects_unknown_initial_case(tmp_path: Path) -> None:
    dataset_path = tmp_path / "core_dataset.json"
    _write_dataset(dataset_path)

    with pytest.raises(ValueError, match="unknown initial Core cases"):
        write_live_core_acquisition_loop(
            dataset_path=dataset_path,
            output_dir=tmp_path / "out",
            report_path=tmp_path / "report.md",
            initial_cases=("99mm/9deg",),
        )

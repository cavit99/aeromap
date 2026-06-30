from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aeromap.cfd.schema import (
    CfdConfig,
    MeshConfig,
    PatchLayerConfig,
    SurfaceExportConfig,
)
from aeromap.cfd.urans import (
    EXPECTED_TRANSIENT_GZIP_FIELDS,
    prepare_urans_audit,
    write_urans_checkpoint_report,
    write_urans_force_history_report,
    write_urans_retained_field_prune_manifest,
)
from aeromap.cfd.urans_analysis import write_urans_checkpoint_decision_report
from aeromap.cfd.urans_parallel import write_urans_parallel_benchmark_plan
from aeromap.constants import REF


def _write(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_gzip(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def _source_case(base_path: Path, *, name: str = "source_case") -> Path:
    case_dir = base_path / name
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="critical_underfloor"),
        mesh=MeshConfig(
            patch_layers=(
                PatchLayerConfig(patch="critical_underfloor", n_surface_layers=5),
                PatchLayerConfig(patch="upper_body", n_surface_layers=1),
            ),
        ),
    )
    _write(
        case_dir / "manifest.json",
        json.dumps(
            {
                "case_id": "case_test",
                "simulation_id": "simulation_test",
                "geometry_id": "geometry_test",
                "state_id": "state_test",
                "cfd_config": config.model_dump(mode="json"),
            },
        ),
    )
    _write(
        case_dir / "quality" / "status.json",
        json.dumps(
            {
                "status": "PROVISIONAL_LIMIT_CYCLE_CANDIDATE",
                "accepted": False,
                "training_eligible": False,
            },
        ),
    )
    _write(
        case_dir / "quality" / "steady_diagnostics.json",
        json.dumps(
            {
                "coefficients": {
                    "c_df": {
                        "spectrum": {
                            "dominant_period_iterations": 16.6666666667,
                        },
                    },
                },
            },
        ),
    )
    for field in ("U", "p", "k", "omega", "nut"):
        _write(case_dir / "openfoam" / "250" / field)
    _write(case_dir / "openfoam" / "constant" / "polyMesh" / "boundary")
    return case_dir


def test_prepare_urans_audit_writes_bounded_transient_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo root"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root, name="source case")

    artifacts = prepare_urans_audit(
        source_case=case_dir,
        out_dir=repo_root / "urans plans",
        audit_purpose="medium_reconnaissance",
        steady_time="250",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    control_dict = artifacts.control_dict_path.read_text(encoding="utf-8")
    fv_schemes = artifacts.fv_schemes_path.read_text(encoding="utf-8")
    fv_solution = artifacts.fv_solution_path.read_text(encoding="utf-8")
    run_script = artifacts.run_script_path.read_text(encoding="utf-8")

    assert manifest["status"] == "PREPARED_NOT_RUN"
    assert manifest["audit_purpose"] == "medium_reconnaissance"
    assert manifest["accepted"] is False
    assert manifest["training_eligible"] is False
    assert manifest["source"]["status"] == "PROVISIONAL_LIMIT_CYCLE_CANDIDATE"
    assert manifest["steady_iteration_cycle"]["c_df_dominant_period_iterations"] == pytest.approx(
        16.6666666667,
    )
    assert manifest["classification_rules"]["never_use_phase_snapshot_as_label"] is True
    assert manifest["classification_rules"]["audit_purpose"] == "medium_reconnaissance"
    assert "adjustTimeStep  yes;" in control_dict
    assert "maxCo           1;" in control_dict
    assert "purgeWrite      0;" in control_dict
    assert "writeCompression off;" in control_dict
    assert "fieldAverage" in control_dict
    assert "patches         (critical_underfloor upper_body floor_edges" in control_dict
    assert "default         Euler;" in fv_schemes
    assert "PIMPLE" in fv_solution
    assert "nOuterCorrectors 3;" in fv_solution
    assert "residualControl" not in fv_solution
    assert "pFinal" in fv_solution
    assert '"(U|k|omega)Final"' in fv_solution
    assert "foamRun -solver incompressibleFluid" in run_script
    assert 'LOCK_DIR="$WORK_CASE.lock"' in run_script
    assert 'if ! mkdir "$LOCK_DIR"' in run_script
    assert "cleanup_lock()" in run_script
    assert 'rm -f "$LOCK_DIR/host_pid" "$LOCK_DIR/started_at_utc"' in run_script
    assert "trap cleanup_lock EXIT" in run_script
    assert "copy source openfoam/250 fields to generated openfoam/0" in json.dumps(manifest)
    assert "SOURCE_CASE_REL='source case'" in run_script
    assert "AUDIT_DIR_REL='urans plans/" in run_script
    assert "AEROMAP_URANS_WORK_CASE" not in run_script
    assert os.access(artifacts.run_script_path, os.X_OK)
    bash = shutil.which("bash") or "/bin/bash"
    syntax = subprocess.run(  # noqa: S603
        [bash, "-n", str(artifacts.run_script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_prepare_urans_audit_can_bound_retained_field_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)

    artifacts = prepare_urans_audit(
        source_case=case_dir,
        out_dir=repo_root / "urans_plans",
        write_interval_s=0.005,
        purge_write=8,
        write_compression=True,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    control_dict = artifacts.control_dict_path.read_text(encoding="utf-8")

    assert "writeInterval   0.005;" in control_dict
    assert "purgeWrite      8;" in control_dict
    assert "writeCompression on;" in control_dict
    assert manifest["time_setup"]["purge_write"] == 8
    assert manifest["time_setup"]["write_compression"] is True
    assert "force and mass-flow" in manifest["time_setup"]["output_budget_note"]


def test_prepare_urans_audit_refuses_missing_restart_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)
    (case_dir / "openfoam" / "250" / "omega").unlink()

    with pytest.raises(FileNotFoundError, match="required URANS restart fields"):
        prepare_urans_audit(source_case=case_dir, out_dir=repo_root / "urans_plans")


def test_prepare_urans_audit_rejects_unknown_purpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)

    with pytest.raises(ValueError, match="audit_purpose"):
        prepare_urans_audit(
            source_case=case_dir,
            out_dir=repo_root / "urans_plans",
            audit_purpose="label_grade_now",
        )


def test_prepare_urans_audit_rejects_out_of_repo_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    external_case = _source_case(tmp_path / "external")

    with pytest.raises(ValueError, match="source_case must be inside project root"):
        prepare_urans_audit(source_case=external_case, out_dir=repo_root / "urans_plans")


def test_prepare_urans_audit_shell_quotes_metacharacter_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root, name="source $(touch injected)")

    artifacts = prepare_urans_audit(
        source_case=case_dir,
        out_dir=repo_root / "urans $(touch injected)",
        steady_time="250",
    )

    run_script = artifacts.run_script_path.read_text(encoding="utf-8")
    assert "SOURCE_CASE_REL='source $(touch injected)'" in run_script
    assert "AUDIT_DIR_REL='urans $(touch injected)/" in run_script
    assert 'CONTAINER_WORK_DIR_ESCAPED="$(printf' in run_script


def test_prepare_urans_audit_rejects_non_numeric_steady_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)

    with pytest.raises(ValueError, match="numeric OpenFOAM time"):
        prepare_urans_audit(
            source_case=case_dir,
            out_dir=repo_root / "urans_plans",
            steady_time="250$(touch injected)",
        )


def test_urans_force_history_merges_restart_overlaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = repo_root / "urans_run"
    _write(
        case_dir / "openfoam" / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
        """# Time Cm Cd Cl Cl(f) Cl(r)
0.000 0.1 0.2 0.3 0.4 0.5
0.001 1.1 1.2 1.3 1.4 1.5
0.002 2.1 2.2 2.3 2.4 2.5
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forceCoeffs" / "0.002" / "forceCoeffs.dat",
        """# restart replaces 0.002 and adds 0.003
0.002 20.1 20.2 20.3 20.4 20.5
0.003 30.1 30.2 30.3 30.4 30.5
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forces" / "0" / "forces.dat",
        """# Time forces moments
0.000 ((1 0 -1) (0.1 0 -0.1)) ((0 1 0) (0 0.1 0))
0.001 ((2 0 -2) (0.2 0 -0.2)) ((0 2 0) (0 0.2 0))
0.002 ((3 0 -3) (0.3 0 -0.3)) ((0 3 0) (0 0.3 0))
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forces" / "0.002" / "forces.dat",
        """0.002 ((30 0 -30) (3 0 -3)) ((0 30 0) (0 3 0))
0.003 ((40 0 -40) (4 0 -4)) ((0 40 0) (0 4 0))
""",
    )

    artifacts = write_urans_force_history_report(case_dir)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    coeff = report["force_coefficients"]
    assert coeff["summary"]["row_count"] == 4
    assert coeff["summary"]["duplicate_time_count"] == 1
    assert [row["time"] for row in coeff["rows"]] == [0.0, 0.001, 0.002, 0.003]
    assert coeff["rows"][2]["c_df"] == pytest.approx(20.3)
    assert coeff["rows"][2]["source_start_time_s"] == pytest.approx(0.002)
    assert coeff["summary"]["duplicates"][0]["replaced_source_start_time_s"] == 0.0
    assert coeff["summary"]["duplicates"][0]["kept_source_start_time_s"] == pytest.approx(0.002)
    assert coeff["summary"]["max_time_filter_s"] is None
    assert coeff["summary"]["rows_excluded_by_max_time"] == 0
    assert report["forces"]["summary"]["duplicate_time_count"] == 1
    assert report["time_alignment"]["times_match_exactly"] is True
    assert report["accepted"] is False
    assert report["training_eligible"] is False

    capped = write_urans_force_history_report(
        case_dir,
        out_json=case_dir / "quality" / "transient_force_history_capped.json",
        max_time_s=0.002,
    )
    capped_report = json.loads(capped.report_path.read_text(encoding="utf-8"))
    capped_coeff = capped_report["force_coefficients"]
    assert [row["time"] for row in capped_coeff["rows"]] == [0.0, 0.001, 0.002]
    assert capped_coeff["rows"][2]["c_df"] == pytest.approx(20.3)
    assert capped_coeff["summary"]["max_time_filter_s"] == pytest.approx(0.002)
    assert capped_coeff["summary"]["rows_excluded_by_max_time"] == 1
    assert capped_report["max_time_filter_s"] == pytest.approx(0.002)


def test_urans_force_history_requires_force_outputs(tmp_path: Path) -> None:
    case_dir = tmp_path / "urans_run"

    with pytest.raises(FileNotFoundError, match="forceCoeffs history"):
        write_urans_force_history_report(case_dir)


def test_urans_checkpoint_report_summarizes_retained_fields_and_forces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    source_case = repo_root / "source_case"
    source_case.mkdir()
    case_dir = repo_root / "urans_run"

    for name in EXPECTED_TRANSIENT_GZIP_FIELDS:
        _write_gzip(case_dir / "openfoam" / "0.001" / name)
        _write_gzip(case_dir / "openfoam" / "0.002" / name)
    _write(case_dir / "openfoam" / "0" / "U")
    _write(
        case_dir / "logs" / "foamRun_urans_recon.log",
        """sigFpe : Floating point exception trapping - not supported on this platform
Courant Number mean: 0.01 max: 0.42
Time = 0.001s
ExecutionTime = 10 s  ClockTime = 10 s
Courant Number mean: 0.01 max: 0.43
Time = 0.002s
ExecutionTime = 20 s  ClockTime = 20 s
End
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
        """# Time Cm Cd Cl Cl(f) Cl(r)
0.000 0.1 0.2 0.3 0.4 0.5
0.001 1.1 1.2 1.3 1.4 1.5
0.002 2.1 2.2 2.3 2.4 2.5
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forces" / "0" / "forces.dat",
        """# Time forces moments
0.000 ((1 0 -1) (0.1 0 -0.1)) ((0 1 0) (0 0.1 0))
0.001 ((2 0 -2) (0.2 0 -0.2)) ((0 2 0) (0 0.2 0))
0.002 ((3 0 -3) (0.3 0 -0.3)) ((0 3 0) (0 0.3 0))
""",
    )

    artifacts = write_urans_checkpoint_report(
        case_dir,
        out_json=repo_root / "checkpoint.json",
        planned_end_time_s=0.12,
        source_case=source_case,
        audit_id="urans_audit_test",
        audit_purpose="medium_reconnaissance",
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    assert artifacts.latest_complete_time_s == pytest.approx(0.002)
    assert report["accepted"] is False
    assert report["training_eligible"] is False
    assert report["audit_id"] == "urans_audit_test"
    assert report["source_case"] == "source_case"
    assert report["latest_complete_written_time_s"] == pytest.approx(0.002)
    assert report["gzip_validation"]["all_expected_gzip_fields_readable"] is True
    assert report["logs"][0]["ends_cleanly"] is True
    assert report["overall_max_courant_observed"] == pytest.approx(0.43)
    assert report["transient_force_history"]["coefficient_row_count"] == 3
    assert report["transient_force_history"]["force_row_count"] == 3
    assert report["transient_force_history"]["time_alignment"]["times_match_exactly"] is True
    assert report["transient_force_history"]["force_coefficients_summary"][
        "max_time_filter_s"
    ] == pytest.approx(0.002)
    assert report["claims_not_established"]


def test_urans_retained_field_prune_manifest_keeps_initial_and_latest_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = repo_root / "urans_run"
    for time_name in ("0", "0.001", "0.002", "0.003", "0.004"):
        _write(case_dir / "openfoam" / time_name / "U", f"{time_name}\n")

    preview = write_urans_retained_field_prune_manifest(
        case_dir,
        keep_latest=2,
    )
    preview_report = json.loads(preview.manifest_path.read_text(encoding="utf-8"))
    assert preview.dry_run is True
    assert preview.candidate_count == 2
    assert preview_report["kept_time_directories"] == ["0", "0.003", "0.004"]
    assert preview_report["candidate_time_directories"] == ["0.001", "0.002"]
    assert (case_dir / "openfoam" / "0.001").exists()

    applied = write_urans_retained_field_prune_manifest(
        case_dir,
        keep_latest=2,
        dry_run=False,
    )
    applied_report = json.loads(applied.manifest_path.read_text(encoding="utf-8"))
    assert applied.dry_run is False
    assert applied_report["applied"] is True
    assert (case_dir / "openfoam" / "0").exists()
    assert not (case_dir / "openfoam" / "0.001").exists()
    assert not (case_dir / "openfoam" / "0.002").exists()
    assert (case_dir / "openfoam" / "0.003").exists()
    assert (case_dir / "openfoam" / "0.004").exists()


def test_urans_retained_field_prune_refuses_active_lock(tmp_path: Path) -> None:
    case_dir = tmp_path / "urans_run"
    _write(case_dir / "openfoam" / "0.001" / "U")
    (tmp_path / "urans_run.lock").mkdir()

    with pytest.raises(RuntimeError, match="work-case lock"):
        write_urans_retained_field_prune_manifest(case_dir, keep_latest=1)


def _write_checkpoint_decision_inputs(case_dir: Path) -> None:
    q_area = REF.q_inf_pa * REF.a_ref_m2
    q_area_length = q_area * REF.l_ref_m
    coefficient_rows = []
    force_rows = []
    for index in range(161):
        time = 0.024 + index * 0.00025
        progress = index / 160
        c_d = 0.025
        c_df = 0.05 - 0.001 * progress
        c_m_pitch = 0.0092 - 0.0002 * progress
        coefficient_rows.append(
            {
                "time": time,
                "c_d": c_d,
                "c_df": c_df,
                "c_df_front": c_df * 0.65,
                "c_df_rear": c_df * 0.35,
                "c_m_pitch": c_m_pitch,
            },
        )
        total_n = [c_d * q_area, 0.0, -c_df * q_area]
        total_moment_nm = [0.0, c_m_pitch * q_area_length, 0.0]
        pressure_n = [0.8 * total_n[0], 0.0, 0.98 * total_n[2]]
        viscous_n = [0.2 * total_n[0], 0.0, 0.02 * total_n[2]]
        pressure_moment_nm = [0.0, 0.97 * total_moment_nm[1], 0.0]
        viscous_moment_nm = [0.0, 0.03 * total_moment_nm[1], 0.0]
        force_rows.append(
            {
                "time": time,
                "pressure_n": pressure_n,
                "viscous_n": viscous_n,
                "total_n": total_n,
                "pressure_moment_nm": pressure_moment_nm,
                "viscous_moment_nm": viscous_moment_nm,
                "total_moment_nm": total_moment_nm,
            },
        )
    _write(
        case_dir / "quality" / "transient_force_history.json",
        json.dumps(
            {
                "force_coefficients": {"rows": coefficient_rows},
                "forces": {"rows": force_rows},
                "accepted": False,
                "training_eligible": False,
            },
        ),
    )
    spatial_rows = []
    for time in (0.024, 0.044, 0.064):
        region = {
            "coefficients": {"c_df": 0.02 + time * 0.01},
            "area_m2": 0.1,
            "cell_count": 2,
            "pressure_n": [0.0, 0.0, -1.0],
            "viscous_n": [0.0, 0.0, -0.01],
            "total_n": [0.0, 0.0, -1.01],
            "pressure_moment_nm": [0.0, 0.1, 0.0],
            "viscous_moment_nm": [0.0, 0.01, 0.0],
            "total_moment_nm": [0.0, 0.11, 0.0],
            "x_cp_m": 1.0,
        }
        spatial_rows.append(
            {
                "time_s": time,
                "critical_underfloor": region,
                "throat_band": region,
                "diffuser_ramp": region,
                "diffuser_exit_band": region,
                "left_tunnel_y_negative": region,
                "right_tunnel_y_positive": region,
                "total": region,
                "streamwise_bins": {
                    "critical_underfloor": [
                        {
                            "x_min_m": 0.0,
                            "x_max_m": 1.0,
                            "coefficients": {"c_df": 0.01 + time * 0.01},
                        },
                        {
                            "x_min_m": 1.0,
                            "x_max_m": 2.0,
                            "coefficients": {"c_df": 0.02 + time * 0.02},
                        },
                    ],
                },
            },
        )
    _write(
        case_dir / "quality" / "urans_spatial_load_history.json",
        json.dumps({"rows": spatial_rows, "accepted": False, "training_eligible": False}),
    )
    _write(
        case_dir / "quality" / "retained_field_prune_manifest.json",
        json.dumps({"kept_time_directories": ["0", "0.064"]}),
    )
    _write_gzip(case_dir / "openfoam" / "0.064" / "U.gz")
    _write_gzip(case_dir / "openfoam" / "0.064" / "p.gz")
    _write(case_dir / "logs" / "foamRun_urans_recon_resume_0p064.log", "End\n")


def test_urans_checkpoint_decision_freezes_hashes_and_rejects_short_transient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = repo_root / "urans_run"
    _write_checkpoint_decision_inputs(case_dir)
    checkpoint_report = repo_root / "checkpoint.json"
    _write(
        checkpoint_report,
        json.dumps({"latest_complete_written_time_dir": "0.064"}),
    )

    artifacts = write_urans_checkpoint_decision_report(
        work_case=case_dir,
        out_json=repo_root / "decision.json",
        checkpoint_report_path=checkpoint_report,
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    assert artifacts.classification == "STILL_TRANSIENT"
    assert report["case_class"] == "PARTIAL_MEDIUM_RECONNAISSANCE_RETAINED_RESTART_CHECKPOINT"
    assert report["accepted"] is False
    assert report["accepted_for_time_step_sensitivity"] is False
    assert report["training_eligible"] is False
    assert report["eligible_targets"]["integrated_drag"] is False
    assert report["eligible_targets"]["integrated_downforce"] is False
    assert report["eligible_targets"]["pitch_moment"] is False
    assert report["article_flow_throughs_at_latest_time"] == pytest.approx(1.28)
    assert report["force_history"]["row_count_in_analysis_window"] == 161
    assert report["force_history"]["metrics"]["c_df"]["late_vs_early_relative_mean_change"] < -0.005
    assert report["spatial_history"]["left_right"]["c_df_correlation"] == pytest.approx(1.0)
    field_hash_paths = {
        entry["path"] for entry in report["frozen_artifacts"]["latest_complete_field_files"]
    }
    assert "urans_run/openfoam/0.064/U.gz" in field_hash_paths
    assert "urans_run/openfoam/0.064/p.gz" in field_hash_paths


def test_urans_checkpoint_decision_marks_broadband_candidate_sensitivity_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = repo_root / "urans_run"
    q_area = REF.q_inf_pa * REF.a_ref_m2
    q_area_length = q_area * REF.l_ref_m
    coefficient_rows = []
    force_rows = []
    for index in range(401):
        time = 0.1 + index * 0.00025
        c_d = 0.025
        c_df = 0.05
        c_m_pitch = 0.0092
        coefficient_rows.append(
            {
                "time": time,
                "c_d": c_d,
                "c_df": c_df,
                "c_df_front": c_df * 0.65,
                "c_df_rear": c_df * 0.35,
                "c_m_pitch": c_m_pitch,
            },
        )
        total_n = [c_d * q_area, 0.0, -c_df * q_area]
        total_moment_nm = [0.0, c_m_pitch * q_area_length, 0.0]
        force_rows.append(
            {
                "time": time,
                "pressure_n": [0.8 * total_n[0], 0.0, 0.98 * total_n[2]],
                "viscous_n": [0.2 * total_n[0], 0.0, 0.02 * total_n[2]],
                "total_n": total_n,
                "pressure_moment_nm": [0.0, 0.97 * total_moment_nm[1], 0.0],
                "viscous_moment_nm": [0.0, 0.03 * total_moment_nm[1], 0.0],
                "total_moment_nm": total_moment_nm,
            },
        )
    _write(
        case_dir / "quality" / "transient_force_history.json",
        json.dumps(
            {
                "force_coefficients": {"rows": coefficient_rows},
                "forces": {"rows": force_rows},
                "accepted": False,
                "training_eligible": False,
            },
        ),
    )
    checkpoint_report = repo_root / "checkpoint.json"
    _write(checkpoint_report, json.dumps({"latest_complete_written_time_dir": "0.2"}))

    artifacts = write_urans_checkpoint_decision_report(
        work_case=case_dir,
        out_json=repo_root / "decision.json",
        checkpoint_report_path=checkpoint_report,
        analysis_start_s=0.1,
        window_split_s=0.15,
        analysis_end_s=0.2,
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    assert artifacts.classification == "STATIONARY_BROADBAND_CANDIDATE"
    assert report["case_class"] == "MEDIUM_MESH_URANS_MEAN_CANDIDATE"
    assert report["accepted_for_time_step_sensitivity"] is True
    assert report["accepted"] is False
    assert report["training_eligible"] is False
    assert report["eligible_targets"] == {
        "surface_pressure": False,
        "integrated_drag": False,
        "integrated_downforce": False,
        "integrated_lateral_force": False,
        "pitch_moment": False,
        "volume_mean_fields": False,
        "wall_shear": False,
        "separation_metrics": False,
        "cliff_boundary": False,
        "unsteady_statistics": False,
    }


def test_urans_checkpoint_decision_requires_ordered_windows(tmp_path: Path) -> None:
    case_dir = tmp_path / "urans_run"
    _write_checkpoint_decision_inputs(case_dir)

    with pytest.raises(ValueError, match="analysis_start_s < window_split_s"):
        write_urans_checkpoint_decision_report(
            work_case=case_dir,
            out_json=tmp_path / "decision.json",
            analysis_start_s=0.05,
            window_split_s=0.04,
            analysis_end_s=0.064,
        )


def test_urans_parallel_benchmark_plan_prepares_disposable_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = repo_root / "artifacts" / "campaign" / "urans_audit_runs" / "urans_run"
    _write(case_dir / "openfoam" / "system" / "controlDict")
    _write(case_dir / "logs" / "foamRun_urans_recon.log", "foamRun -solver incompressibleFluid\n")
    checkpoint = repo_root / "checkpoint.json"
    _write(checkpoint, json.dumps({"latest_complete_written_time_s": 0.064}))

    artifacts = write_urans_parallel_benchmark_plan(
        work_case=case_dir,
        out_dir=repo_root / "artifacts" / "campaign" / "urans_parallel_benchmarks" / "plan",
        checkpoint_report_path=checkpoint,
        ranks=(1, 4),
        continuation_s=0.002,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "PREPARED_NOT_RUN"
    assert manifest["accepted"] is False
    assert manifest["training_eligible"] is False
    assert manifest["start_time_s"] == pytest.approx(0.064)
    assert manifest["end_time_s"] == pytest.approx(0.066)
    assert manifest["rank_counts"] == [1, 4]
    assert manifest["current_run_serial_evidence"]["processor_directories_present"] is False
    assert manifest["current_run_serial_evidence"]["existing_logs_use_parallel_flag"] is False
    rank4_script = artifacts.run_scripts[1].read_text(encoding="utf-8")
    assert "mpirun --allow-run-as-root -np 4 foamRun -solver incompressibleFluid -parallel" in (
        rank4_script
    )
    assert "/usr/bin/time" not in rank4_script
    assert "TIMEFORMAT='real %3R user %3U sys %3S'" in rank4_script
    assert 'cp "$REPO_ROOT/$DECOMPOSE_REL"' in rank4_script
    decompose_text = (artifacts.manifest_path.parent / "decomposeParDict_rank4").read_text(
        encoding="utf-8",
    )
    assert "hierarchicalCoeffs" in decompose_text


def test_urans_parallel_benchmark_plan_rejects_invalid_ranks(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    checkpoint = tmp_path / "checkpoint.json"
    _write(case_dir / "openfoam" / "system" / "controlDict")
    _write(checkpoint, json.dumps({"latest_complete_written_time_s": 0.064}))

    with pytest.raises(ValueError, match="positive MPI rank"):
        write_urans_parallel_benchmark_plan(
            work_case=case_dir,
            out_dir=tmp_path / "plan",
            checkpoint_report_path=checkpoint,
            ranks=(0,),
        )

from __future__ import annotations

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
from aeromap.cfd.steady_restart import (
    analyze_steady_restart_branch,
    compare_steady_restart_wall_series,
    prepare_steady_restart_branches,
)


def _write(path: Path, text: str = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    for field in ("U", "p", "k", "omega", "nut"):
        _write(case_dir / "openfoam" / "250" / field)
    _write(case_dir / "openfoam" / "constant" / "polyMesh" / "boundary")
    _write(case_dir / "openfoam" / "system" / "fvSchemes", "ddtSchemes{}\n")
    _write(
        case_dir / "openfoam" / "system" / "fvSolution",
        """
solvers {}
SIMPLE
{
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
        U       0.9;
        ".*"    0.9;
    }
}
""",
    )
    return case_dir


def test_prepare_steady_restart_branches_writes_two_bounded_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo root"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root, name="source case")

    artifacts = prepare_steady_restart_branches(
        source_case=case_dir,
        out_dir=repo_root / "steady plans",
        steady_time="250",
    )

    assert {branch.branch_name for branch in artifacts.branches} == {
        "unchanged",
        "momentum_relaxation_0p70",
    }
    unchanged = next(branch for branch in artifacts.branches if branch.branch_name == "unchanged")
    perturbed = next(
        branch for branch in artifacts.branches if branch.branch_name == "momentum_relaxation_0p70"
    )
    unchanged_manifest = json.loads(unchanged.manifest_path.read_text(encoding="utf-8"))
    perturbed_manifest = json.loads(perturbed.manifest_path.read_text(encoding="utf-8"))
    control_dict = unchanged.control_dict_path.read_text(encoding="utf-8")
    unchanged_solution = unchanged.fv_solution_path.read_text(encoding="utf-8")
    perturbed_solution = perturbed.fv_solution_path.read_text(encoding="utf-8")
    run_script = unchanged.run_script_path.read_text(encoding="utf-8")

    assert unchanged_manifest["status"] == "PREPARED_NOT_RUN"
    assert unchanged_manifest["accepted"] is False
    assert unchanged_manifest["training_eligible"] is False
    assert unchanged_manifest["source"]["status"] == "PROVISIONAL_LIMIT_CYCLE_CANDIDATE"
    assert unchanged_manifest["run_setup"]["intended_simple_iterations"] == 120
    assert unchanged_manifest["run_setup"]["source_start_time"] == 250.0
    assert unchanged_manifest["run_setup"]["end_time"] == 370.0
    assert unchanged_manifest["branch_perturbation"]["momentum_relaxation"] is None
    assert perturbed_manifest["branch_perturbation"]["momentum_relaxation"] == pytest.approx(0.70)
    assert "startFrom       latestTime;" in control_dict
    assert "startTime       250;" in control_dict
    assert "endTime         370;" in control_dict
    assert "writeInterval   5;" in control_dict
    assert "writeInterval   1;" in control_dict
    assert "U       0.9;" in unchanged_solution
    assert "U       1e-3;" in perturbed_solution
    assert "U       0.7;" in perturbed_solution
    assert "p       0.9;" in perturbed_solution
    assert "pcorr   0.9;" in perturbed_solution
    assert "k       0.9;" in perturbed_solution
    assert "omega   0.9;" in perturbed_solution
    assert '".*"    0.9;' not in perturbed_solution
    assert '"source case"' not in run_script
    assert "SOURCE_CASE_REL='source case'" in run_script
    assert "foamRun -solver incompressibleFluid" in run_script
    assert "AEROMAP_STEADY_RESTART_OVERWRITE" in run_script
    assert os.access(unchanged.run_script_path, os.X_OK)
    bash = shutil.which("bash") or "/bin/bash"
    syntax = subprocess.run(  # noqa: S603
        [bash, "-n", str(unchanged.run_script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_prepare_steady_restart_refuses_missing_restart_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)
    (case_dir / "openfoam" / "250" / "omega").unlink()

    with pytest.raises(FileNotFoundError, match="required steady restart fields"):
        prepare_steady_restart_branches(source_case=case_dir, out_dir=repo_root / "steady_plans")


def test_prepare_steady_restart_rejects_out_of_repo_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    external_case = _source_case(tmp_path / "external")

    with pytest.raises(ValueError, match="source_case must be inside project root"):
        prepare_steady_restart_branches(
            source_case=external_case, out_dir=repo_root / "steady_plans"
        )


def test_prepare_steady_restart_rejects_non_numeric_steady_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    case_dir = _source_case(repo_root)

    with pytest.raises(ValueError, match="numeric OpenFOAM time"):
        prepare_steady_restart_branches(
            source_case=case_dir,
            out_dir=repo_root / "steady_plans",
            steady_time="250$(touch injected)",
        )


def test_analyze_steady_restart_branch_filters_to_restart_window(tmp_path: Path) -> None:
    case_dir = tmp_path / "restart_case"
    _write(
        case_dir / "steady_restart_manifest.json",
        json.dumps(
            {
                "plan_id": "steady_restart_test",
                "branch_name": "unchanged",
                "run_setup": {
                    "source_start_time": 250.0,
                    "end_time": 253.0,
                    "intended_simple_iterations": 3,
                },
            },
        ),
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forceCoeffs" / "250" / "forceCoeffs.dat",
        """# Time Cm Cd Cl Cl(f) Cl(r)
250 0.0 9.0 9.0 0.0 0.0
251 0.1 0.020 0.030 0.015 0.015
252 0.2 0.022 0.032 0.016 0.016
253 0.3 0.024 0.034 0.017 0.017
""",
    )
    _write(
        case_dir / "openfoam" / "postProcessing" / "forces" / "250" / "forces.dat",
        """# Time forces
251 1 2 3 4 5 6 0.1 0.2 0.3 0.4 0.5 0.6
252 2 3 4 5 6 7 0.2 0.3 0.4 0.5 0.6 0.7
253 3 4 5 6 7 8 0.3 0.4 0.5 0.6 0.7 0.8
""",
    )
    _write(
        case_dir
        / "openfoam"
        / "postProcessing"
        / "inletFlowRate"
        / "250"
        / "surfaceFieldValue.dat",
        "251 -1\n252 -1\n253 -1\n",
    )
    _write(
        case_dir
        / "openfoam"
        / "postProcessing"
        / "outletFlowRate"
        / "250"
        / "surfaceFieldValue.dat",
        "251 1\n252 1\n253 1\n",
    )
    for time_name in ("250", "251", "252", "253"):
        (case_dir / "openfoam" / time_name).mkdir(parents=True)
    _write(
        case_dir / "logs" / "foamRun_steady_restart_unchanged.log",
        """Time = 251s
smoothSolver:  Solving for Ux, Initial residual = 1e-3, Final residual = 1e-5, No Iterations 2
time step continuity errors : sum local = 1e-4, global = 1e-6, cumulative = 1e-6
""",
    )

    report = analyze_steady_restart_branch(case_dir)

    assert report["status"] == "COMPLETE"
    assert report["accepted"] is False
    assert report["training_eligible"] is False
    assert report["time_window"]["completed_force_coefficient_iterations"] == 3
    assert report["time_window"]["first_force_coefficient_time"] == 251.0
    assert report["time_window"]["last_force_coefficient_time"] == 253.0
    assert report["steady_diagnostics"]["sample_count"] == 3
    assert report["mass_flow"]["latest_inlet_flow_m3_s"] == -1.0
    assert report["mass_flow"]["latest_outlet_flow_m3_s"] == 1.0
    assert (case_dir / "quality" / "steady_restart_diagnostics.json").exists()


def _wall_series_report(
    path: Path,
    *,
    branch_name: str,
    total_c_df_mean: float,
    total_c_df_peak_to_peak: float,
    coherent: bool,
) -> None:
    correlation = 0.92 if coherent else 0.2
    report = {
        "schema_version": "steady_restart_wall_series_v0.1.0",
        "case_dir": f"fixture_cases/{branch_name}",
        "branch_name": branch_name,
        "mapping_summary": {
            "min_area_coverage": 1.0,
            "min_cross_check_area_coverage": 0.99,
        },
        "samples": [{}, {}, {}, {}],
        "phase_summary": {
            "total_c_d": {"mean": 0.02, "peak_to_peak": 0.001},
            "total_c_df": {
                "mean": total_c_df_mean,
                "peak_to_peak": total_c_df_peak_to_peak,
            },
            "total_c_m_pitch": {"mean": 0.008, "peak_to_peak": 0.001},
            "left_right_c_df_phase": {
                "zero_lag_correlation": correlation,
                "best_lag_correlation": correlation,
            },
            "diffuser_total_c_df_phase": {
                "zero_lag_correlation": 0.1,
                "best_lag_correlation": 0.1,
            },
        },
    }
    _write(path, json.dumps(report))


def test_compare_steady_restart_wall_series_marks_coherent_candidate(tmp_path: Path) -> None:
    first = tmp_path / "unchanged.json"
    second = tmp_path / "relaxed.json"
    _wall_series_report(
        first,
        branch_name="unchanged",
        total_c_df_mean=0.048,
        total_c_df_peak_to_peak=0.004,
        coherent=True,
    )
    _wall_series_report(
        second,
        branch_name="momentum_relaxation_0p70",
        total_c_df_mean=0.0482,
        total_c_df_peak_to_peak=0.0042,
        coherent=True,
    )

    comparison = compare_steady_restart_wall_series(
        reports=(first, second),
        out_json=tmp_path / "comparison.json",
    )

    assert comparison["accepted"] is False
    assert comparison["training_eligible"] is False
    assert comparison["classification"] == "PHYSICAL_UNSTEADINESS_CANDIDATE"


def test_compare_steady_restart_wall_series_marks_numerics_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "unchanged.json"
    second = tmp_path / "relaxed.json"
    _wall_series_report(
        first,
        branch_name="unchanged",
        total_c_df_mean=0.048,
        total_c_df_peak_to_peak=0.004,
        coherent=True,
    )
    _wall_series_report(
        second,
        branch_name="momentum_relaxation_0p70",
        total_c_df_mean=0.060,
        total_c_df_peak_to_peak=0.004,
        coherent=True,
    )

    comparison = compare_steady_restart_wall_series(
        reports=(first, second),
        out_json=tmp_path / "comparison.json",
    )

    assert comparison["classification"] == "NUMERICS_SENSITIVE"

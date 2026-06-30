from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from aeromap.cfd.patch_surface import article_patch_names
from aeromap.cfd.postprocess import (
    _acceptance_blockers,
    _boundary_scalar_values,
    _case_status_name,
    _convert_vtk,
    _force_rows,
    _independent_force_integration,
    _integrate_exported_wall_forces,
    _layer_and_wall_metrics,
    _problem_sets_report,
    _residual_report,
    _spectrum_summary,
    _steady_diagnostics,
    _target_eligibility,
    _wall_condition_report,
    postprocess_case,
)
from aeromap.cfd.quality import contains_negative_volume_failure, parse_check_mesh_log
from aeromap.cfd.schema import CfdConfig, QualityConfig, SurfaceExportConfig
from aeromap.constants import REF


def test_independent_force_integration_uses_polygon_area_vectors(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([4, 0, 1, 2, 3])
    poly = pv.PolyData(points, faces)
    poly.cell_data["p"] = np.array([2.0], dtype=np.float64)
    poly.cell_data["wallShearStress"] = np.array([[-0.5, 0.0, 0.25]], dtype=np.float64)
    wall_vtp = tmp_path / "wall.vtp"
    poly.save(wall_vtp)

    expected_pressure = REF.rho_kg_m3 * np.array([0.0, 0.0, 2.0], dtype=np.float64)
    expected_viscous = -REF.rho_kg_m3 * np.array([-0.5, 0.0, 0.25], dtype=np.float64)
    expected_total = expected_pressure + expected_viscous

    result = _independent_force_integration(
        wall_vtp,
        {
            "pressure_n": expected_pressure.tolist(),
            "viscous_n": expected_viscous.tolist(),
            "total_n": expected_total.tolist(),
        },
    )

    assert result["independent_pressure_n"] == pytest.approx(expected_pressure)
    assert result["independent_viscous_n"] == pytest.approx(expected_viscous)
    assert result["independent_total_n"] == pytest.approx(expected_total)
    assert result["openfoam_pressure_n"] == pytest.approx(expected_pressure)
    assert result["openfoam_viscous_n"] == pytest.approx(expected_viscous)
    assert result["within_1pct"] is True


def test_exported_wall_force_integration_has_analytic_uniform_pressure_case(
    tmp_path: Path,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([4, 0, 1, 2, 3])
    poly = pv.PolyData(points, faces)
    poly.cell_data["p"] = np.array([4.0], dtype=np.float64)
    poly.cell_data["wallShearStress"] = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    wall_vtp = tmp_path / "uniform_pressure.vtp"
    poly.save(wall_vtp)

    result = _integrate_exported_wall_forces(wall_vtp)

    assert result["independent_pressure_n"] == pytest.approx(
        REF.rho_kg_m3 * np.array([0.0, 0.0, 24.0], dtype=np.float64),
    )
    assert result["independent_viscous_n"] == pytest.approx([0.0, 0.0, 0.0])
    assert result["independent_total_n"] == pytest.approx(result["independent_pressure_n"])


def _write_scalar_patch_field(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value_lines = "\n".join(f"{value:g}" for value in values)
    path.write_text(
        f"""
FoamFile
{{
    format ascii;
    class volScalarField;
    object {path.name};
}}
dimensions [];
internalField uniform 0;
boundaryField
{{
    article
    {{
        type fixedValue;
        value nonuniform List<scalar>
{len(values)}
(
{value_lines}
)
;
    }}
}}
""",
        encoding="utf-8",
    )


def _write_multi_patch_scalar_field(path: Path, patch_values: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for patch, values in patch_values.items():
        value_lines = "\n".join(f"{value:g}" for value in values)
        blocks.append(
            f"""
    {patch}
    {{
        type fixedValue;
        value nonuniform List<scalar>
{len(values)}
(
{value_lines}
)
;
    }}""",
        )
    path.write_text(
        f"""
FoamFile
{{
    format ascii;
    class volScalarField;
    object {path.name};
}}
dimensions [];
internalField uniform 0;
boundaryField
{{
{"".join(blocks)}
}}
""",
        encoding="utf-8",
    )


def _write_patch_type_field(path: Path, patch_types: dict[str, str]) -> None:
    blocks = "\n".join(
        f"""
    {patch}
    {{
        type {patch_type};
        value uniform 0;
    }}"""
        for patch, patch_type in patch_types.items()
    )
    _write_case_file(
        path,
        f"""
        FoamFile
        {{
            format ascii;
            class volScalarField;
            object {path.name};
        }}
        dimensions [];
        internalField uniform 0;
        boundaryField
        {{
        {blocks}
        }}
        """,
    )


def _write_case_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")


def _write_tiny_volume_vtk(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA], dtype=np.uint8)
    pv.UnstructuredGrid(cells, celltypes, points).save(path)


def _write_tiny_patch_vtk(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    patch = pv.PolyData(points, np.array([3, 0, 1, 2]))
    patch.cell_data["p"] = np.array([1.0], dtype=np.float64)
    patch.cell_data["wallShearStress"] = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    patch.save(path)


def test_boundary_scalar_values_reads_openfoam_patch_lists(tmp_path: Path) -> None:
    field = tmp_path / "nSurfaceLayers"
    _write_scalar_patch_field(field, [0.0, 1.0, 3.0])

    values = _boundary_scalar_values(field, "article")

    assert values.tolist() == [0.0, 1.0, 3.0]


def test_convert_vtk_skips_absent_zero_face_article_patch(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    latest_time = "12"
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="critical_underfloor"),
    )
    vtk_dir = case_dir / "openfoam" / "VTK"
    _write_tiny_volume_vtk(vtk_dir / f"openfoam_{latest_time}.vtk")
    for patch in ("critical_underfloor", "upper_body", "floor_edges", "layer_transition_band"):
        _write_tiny_patch_vtk(vtk_dir / patch / f"{patch}_{latest_time}.vtk")
    _write_case_file(
        case_dir / "openfoam" / "constant" / "polyMesh" / "boundary",
        """
        4
        (
            critical_underfloor
            {
                type wall;
                nFaces 1;
                startFace 0;
            }
            upper_body
            {
                type wall;
                nFaces 1;
                startFace 1;
            }
            floor_edges
            {
                type wall;
                nFaces 1;
                startFace 2;
            }
            layer_transition_band
            {
                type wall;
                nFaces 1;
                startFace 3;
            }
        )
        """,
    )

    volume_vtu, wall_vtp = _convert_vtk(case_dir, latest_time, config)

    report = json.loads(
        (case_dir / "quality" / f"wall_patch_export_{latest_time}.json").read_text(
            encoding="utf-8",
        ),
    )
    assert volume_vtu.exists()
    assert wall_vtp.exists()
    assert "keel" not in report["present_patches"]
    assert report["omitted_patches"]["keel"]["reason"] == "absent_or_zero_face_openfoam_patch"


def test_convert_vtk_fails_for_missing_positive_face_article_patch(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    latest_time = "12"
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="critical_underfloor"),
    )
    vtk_dir = case_dir / "openfoam" / "VTK"
    _write_tiny_volume_vtk(vtk_dir / f"openfoam_{latest_time}.vtk")
    for patch in ("critical_underfloor", "upper_body", "floor_edges", "layer_transition_band"):
        _write_tiny_patch_vtk(vtk_dir / patch / f"{patch}_{latest_time}.vtk")
    _write_case_file(
        case_dir / "openfoam" / "constant" / "polyMesh" / "boundary",
        """
        5
        (
            critical_underfloor
            {
                type wall;
                nFaces 1;
                startFace 0;
            }
            upper_body
            {
                type wall;
                nFaces 1;
                startFace 1;
            }
            floor_edges
            {
                type wall;
                nFaces 1;
                startFace 2;
            }
            keel
            {
                type wall;
                nFaces 1;
                startFace 3;
            }
            layer_transition_band
            {
                type wall;
                nFaces 1;
                startFace 4;
            }
        )
        """,
    )

    with pytest.raises(FileNotFoundError, match=r"keel_12\.vtk"):
        _convert_vtk(case_dir, latest_time, config)


def test_force_rows_accepts_pressure_and_viscous_components(tmp_path: Path) -> None:
    path = tmp_path / "forces.dat"
    path.write_text("92 (1 2 3) (4 5 6)\n", encoding="utf-8")

    rows = _force_rows(path)

    assert rows == [
        {
            "time": 92.0,
            "pressure_n": [1.0, 2.0, 3.0],
            "viscous_n": [4.0, 5.0, 6.0],
            "total_n": [5.0, 7.0, 9.0],
        },
    ]


def test_force_rows_accepts_moment_components(tmp_path: Path) -> None:
    path = tmp_path / "forces.dat"
    path.write_text("92 (1 2 3) (4 5 6) (7 8 9) (10 11 12)\n", encoding="utf-8")

    row = _force_rows(path)[0]

    assert row["force_row_layout"] == "pressure_viscous_force_and_moment"
    assert row["pressure_moment_nm"] == [7.0, 8.0, 9.0]
    assert row["viscous_moment_nm"] == [10.0, 11.0, 12.0]
    assert row["total_moment_nm"] == [17.0, 19.0, 21.0]


def test_force_rows_accepts_openfoam_porous_force_moment_layout(tmp_path: Path) -> None:
    path = tmp_path / "forces.dat"
    path.write_text(
        "92 ((1 2 3) (4 5 6) (7 8 9)) ((10 11 12) (13 14 15) (16 17 18))\n",
        encoding="utf-8",
    )

    row = _force_rows(path)[0]

    assert row["force_row_layout"] == "pressure_viscous_porous_force_and_moment"
    assert row["pressure_n"] == [1.0, 2.0, 3.0]
    assert row["viscous_n"] == [4.0, 5.0, 6.0]
    assert row["porous_force_n"] == [7.0, 8.0, 9.0]
    assert row["total_n"] == [12.0, 15.0, 18.0]
    assert row["pressure_moment_nm"] == [10.0, 11.0, 12.0]
    assert row["viscous_moment_nm"] == [13.0, 14.0, 15.0]
    assert row["porous_moment_nm"] == [16.0, 17.0, 18.0]
    assert row["total_moment_nm"] == [39.0, 42.0, 45.0]


def test_spectrum_summary_reports_iteration_period() -> None:
    values = np.sin(2.0 * np.pi * np.arange(40, dtype=np.float64) / 10.0)

    summary = _spectrum_summary(values)

    assert summary["status"] == "OK"
    assert summary["dominant_period_iterations"] == pytest.approx(10.0)
    assert "not physical time" in summary["note"]


def test_residual_report_parses_solver_log(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case_file(
        case_dir / "logs" / "solver.log",
        "Time = 7s\n"
        "smoothSolver:  Solving for Ux, Initial residual = 0.1, "
        "Final residual = 0.01, No Iterations 3\n"
        "smoothSolver:  Solving for Uy, Initial residual = 0.2, "
        "Final residual = 0.02, No Iterations 4\n"
        "GAMG:  Solving for p, Initial residual = 0.3, "
        "Final residual = 0.03, No Iterations 2\n"
        "time step continuity errors : sum local = 0.4, global = -0.05, cumulative = 0.06\n"
        "ExecutionTime = 12.5 s  ClockTime = 13 s\n",
    )

    report = _residual_report(case_dir)

    assert report["status"] == "OK"
    assert report["residual_sample_count"] == 3
    assert report["per_field"]["p"]["final"]["full"]["mean"] == pytest.approx(0.03)
    assert report["grouped_fields"]["U"]["initial"]["full"]["max"] == pytest.approx(0.2)
    assert report["continuity"]["global"]["mean"] == pytest.approx(-0.05)
    assert report["execution"]["last"]["clock_time_s"] == pytest.approx(13.0)


def test_wall_condition_report_confirms_expected_wall_functions(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    patch_types = {
        "ground": "nutkWallFunction",
        "article": "nutkWallFunction",
    }
    _write_patch_type_field(case_dir / "openfoam" / "0" / "nut", patch_types)
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "k",
        {"ground": "kqRWallFunction", "article": "kqRWallFunction"},
    )
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "omega",
        {"ground": "omegaWallFunction", "article": "omegaWallFunction"},
    )

    report = _wall_condition_report(case_dir, CfdConfig())

    assert report["status"] == "OK"
    assert report["fields"]["nut"]["patch_types"]["article"] == "nutkWallFunction"
    assert report["fields"]["k"]["patch_types"]["ground"] == "kqRWallFunction"
    assert report["fields"]["omega"]["patch_types"]["article"] == "omegaWallFunction"


def test_wall_condition_report_reads_one_line_openfoam_patch_entries(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case_file(
        case_dir / "openfoam" / "0" / "nut",
        """
        boundaryField
        {
            ground      { type nutkWallFunction; value uniform 0; }
            article     { type nutkWallFunction; value uniform 0; }
        }
        """,
    )
    _write_case_file(
        case_dir / "openfoam" / "0" / "k",
        """
        boundaryField
        {
            ground      { type kqRWallFunction; value uniform 0.24; }
            article     { type kqRWallFunction; value uniform 0.24; }
        }
        """,
    )
    _write_case_file(
        case_dir / "openfoam" / "0" / "omega",
        """
        boundaryField
        {
            ground      { type omegaWallFunction; value uniform 6.3; }
            article     { type omegaWallFunction; value uniform 6.3; }
        }
        """,
    )

    report = _wall_condition_report(case_dir, CfdConfig())

    assert report["status"] == "OK"
    assert report["fields"]["nut"]["patch_types"]["ground"] == "nutkWallFunction"


def test_wall_condition_report_flags_mismatched_patch_type(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "nut",
        {"ground": "nutkWallFunction", "article": "fixedValue"},
    )
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "k",
        {"ground": "kqRWallFunction", "article": "kqRWallFunction"},
    )
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "omega",
        {"ground": "omegaWallFunction", "article": "omegaWallFunction"},
    )

    report = _wall_condition_report(case_dir, CfdConfig())

    assert report["status"] == "FAILED"
    assert report["mismatches"][0]["field"] == "nut"
    assert report["mismatches"][0]["patch"] == "article"


def test_wall_condition_report_records_missing_patch_once(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "nut",
        {"ground": "nutkWallFunction"},
    )
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "k",
        {"ground": "kqRWallFunction", "article": "kqRWallFunction"},
    )
    _write_patch_type_field(
        case_dir / "openfoam" / "0" / "omega",
        {"ground": "omegaWallFunction", "article": "omegaWallFunction"},
    )

    report = _wall_condition_report(case_dir, CfdConfig())
    nut_article_mismatches = [
        mismatch
        for mismatch in report["mismatches"]
        if mismatch["field"] == "nut" and mismatch["patch"] == "article"
    ]

    assert report["status"] == "FAILED"
    assert len(nut_article_mismatches) == 1
    assert nut_article_mismatches[0]["observed"] == "MISSING"
    assert "reason" in nut_article_mismatches[0]


def test_problem_sets_report_maps_warped_faces_to_nearest_wall_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    problem_root = case_dir / "openfoam" / "postProcessing" / "checkMesh" / "constant"
    problem_root.mkdir(parents=True)
    warped = pv.PolyData(
        np.asarray(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ],
            dtype=np.float64,
        ),
        np.asarray([3, 0, 1, 2]),
    )
    warped.save(problem_root / "warpedFaces.vtk")

    wall = pv.PolyData(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([3, 0, 1, 2]),
    )
    wall.cell_data["surface_region"] = np.asarray(["diffuser"])
    wall.cell_data["p"] = np.asarray([2.0], dtype=np.float64)
    wall.cell_data["yPlus"] = np.asarray([42.0], dtype=np.float64)
    wall.cell_data["wallShearStress"] = np.asarray([[3.0, 4.0, 0.0]], dtype=np.float64)
    wall_vtp = tmp_path / "wall.vtp"
    wall.save(wall_vtp)

    report = _problem_sets_report(case_dir, wall_vtp)

    warped_report = report["sets"]["warpedFaces"]
    context = warped_report["nearest_wall_field_context"]
    assert warped_report["cell_count"] == 1
    assert context["status"] == "OK"
    assert context["nearest_wall_region_counts"] == {"diffuser": 1}
    assert context["nearest_wall_yplus"]["mean"] == pytest.approx(42.0)
    assert context["nearest_wall_shear_magnitude"]["mean"] == pytest.approx(5.0)


def test_steady_diagnostics_keeps_iteration_cycle_as_diagnostic_only() -> None:
    coeff_rows = [
        {
            "time": float(index),
            "c_m_pitch": 0.1 * np.sin(index),
            "c_d": 0.2,
            "c_df": 0.3 + 0.01 * np.sin(index),
            "c_df_front": 0.2,
            "c_df_rear": 0.1,
        }
        for index in range(20)
    ]
    force_rows = [
        {
            "time": float(index),
            "pressure_n": [1.0, 0.0, -2.0],
            "viscous_n": [0.5, 0.0, -0.5],
            "total_n": [1.5, 0.0, -2.5],
            "pressure_moment_nm": [0.0, 0.5, 0.0],
            "viscous_moment_nm": [0.0, 0.25, 0.0],
            "total_moment_nm": [0.0, 0.75, 0.0],
        }
        for index in range(20)
    ]

    diagnostics = _steady_diagnostics(coeff_rows=coeff_rows, force_rows=force_rows)

    assert diagnostics["final_window_count"] == 4
    assert "not physical time" in diagnostics["steady_iteration_note"]
    assert diagnostics["forces_final_window_n"]["total"]["status"] == "OK"
    assert diagnostics["moments_final_window_nm"]["total"]["status"] == "OK"
    assert diagnostics["streamwise_center_of_pressure"]["final_window"]["count"] == 4.0


def test_acceptance_blockers_accept_clean_smoke_checks() -> None:
    config = CfdConfig.model_validate(
        {
            "quality": {
                "case_class": "NON_CAMPAIGN_ENGINEERING_SMOKE",
                "mesh_quality_fatal": True,
                "extended_diagnostics_required": True,
                "extended_diagnostics_fatal": False,
            },
        },
    )

    blockers = _acceptance_blockers(
        mesh_report={
            "mesh_quality_returncode": 0,
            "mesh_quality_check": {"mesh_ok": True, "contains_negative_volume": False},
            "contains_negative_volume": False,
            "extended_diagnostics_returncode": 1,
        },
        convergence={"force_stable": True, "mass_balance_ok": True},
        force_integration={"within_1pct": True},
        mapping={"status": "OK"},
        config=config,
    )

    assert blockers == []


def test_acceptance_blockers_reject_failed_smoke_gates() -> None:
    config = CfdConfig.model_validate(
        {
            "quality": {
                "case_class": "NON_CAMPAIGN_ENGINEERING_SMOKE",
                "mesh_quality_fatal": True,
                "extended_diagnostics_required": True,
            },
        },
    )

    blockers = _acceptance_blockers(
        mesh_report={
            "mesh_quality_returncode": 2,
            "mesh_quality_check": {
                "mesh_ok": False,
                "failed_mesh_checks": 1,
                "contains_negative_volume": True,
            },
            "contains_negative_volume": True,
            "extended_diagnostics_returncode": None,
        },
        convergence={"force_stable": False, "mass_balance_ok": False},
        force_integration={"within_1pct": False},
        mapping={"status": "FAILED"},
        config=config,
    )

    assert "fatal mesh-quality check returned 2" in blockers
    assert "fatal mesh-quality check reported failed mesh checks" in blockers
    assert "fatal mesh-quality check did not report Mesh OK" in blockers
    assert "fatal mesh-quality check reported negative-volume cells" in blockers
    assert "extended mesh diagnostics reported negative-volume cells" in blockers
    assert "extended mesh diagnostic return code missing" in blockers
    assert "force coefficients are not stable in the final window" in blockers
    assert "mass-flow imbalance exceeds the acceptance limit" in blockers
    assert "surface region mapping failed" in blockers
    assert "independent force integration exceeds 1%" in blockers


def test_campaign_force_instability_is_named_limit_cycle_candidate() -> None:
    status = _case_status_name(
        config=CfdConfig(),
        convergence={"force_stable": False},
        blockers=["force coefficients are not stable in the final window"],
    )

    assert status == "PROVISIONAL_LIMIT_CYCLE_CANDIDATE"


def test_target_eligibility_keeps_provisional_cases_ineligible() -> None:
    eligibility = _target_eligibility(
        accepted=False,
        training_eligible=False,
        config=CfdConfig(),
        convergence={"force_stable": False, "mass_balance_ok": True},
        force_integration={"within_1pct": True},
        mapping={"status": "OK"},
        layers={"status": "OK"},
    )

    assert eligibility["surface_pressure"] is False
    assert eligibility["integrated_drag"] is False
    assert eligibility["integrated_downforce"] is False
    assert eligibility["integrated_lateral_force"] is False
    assert eligibility["pitch_moment"] is False
    assert eligibility["volume_mean_fields"] is False
    assert eligibility["wall_shear"] is False
    assert eligibility["separation_metrics"] is False
    assert eligibility["cliff_boundary"] is False


def test_target_eligibility_separates_pressure_loads_from_wall_shear() -> None:
    eligibility = _target_eligibility(
        accepted=True,
        training_eligible=True,
        config=CfdConfig(),
        convergence={"force_stable": True, "mass_balance_ok": True},
        force_integration={"within_1pct": True},
        mapping={"status": "OK"},
        layers={"status": "OK"},
    )

    assert eligibility["surface_pressure"] is True
    assert eligibility["integrated_drag"] is True
    assert eligibility["integrated_downforce"] is True
    assert eligibility["integrated_lateral_force"] is True
    assert eligibility["pitch_moment"] is True
    assert eligibility["volume_mean_fields"] is True
    assert eligibility["wall_shear"] is False
    assert eligibility["separation_metrics"] is False
    assert eligibility["cliff_boundary"] is False


def test_checkmesh_log_failure_is_fatal_even_with_zero_return_code(tmp_path: Path) -> None:
    log_path = tmp_path / "checkMesh_meshQuality.log"
    log_path.write_text(
        """
Mesh stats
    cells:            424416
    Max skewness = 3.337961 OK.
Failed 1 mesh checks.
""",
        encoding="utf-8",
    )
    parsed = parse_check_mesh_log(log_path)
    config = CfdConfig.model_validate(
        {
            "quality": {
                "case_class": "NON_CAMPAIGN_ENGINEERING_SMOKE",
                "mesh_quality_fatal": True,
                "extended_diagnostics_required": True,
            },
        },
    )

    blockers = _acceptance_blockers(
        mesh_report={
            "mesh_quality_returncode": 0,
            "mesh_quality_check": parsed,
            "contains_negative_volume": False,
            "extended_diagnostics_returncode": 0,
        },
        convergence={"force_stable": True, "mass_balance_ok": True},
        force_integration={"within_1pct": True},
        mapping={"status": "OK"},
        config=config,
    )

    assert parsed["failed_mesh_checks"] == 1
    assert "fatal mesh-quality check reported failed mesh checks" in blockers
    assert "fatal mesh-quality check did not report Mesh OK" in blockers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("***Cells with negative volume found, number of cells: 3", True),
        ("Detected 2 illegal faces (concave, zero area or negative cell pyramid volume)", True),
        ("No negative volume cells detected.", False),
        ("***Cells with negative volume found, number of cells: 0", False),
        ("Detected 0 illegal faces (concave, zero area or negative cell pyramid volume)", False),
        ("negative volume diagnostics are discussed in this report", False),
    ],
)
def test_negative_volume_detection_requires_positive_count(text: str, expected: object) -> None:
    assert contains_negative_volume_failure(text) is expected


def test_layer_and_wall_metrics_reports_region_coverage(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    field_dir = case_dir / "openfoam" / "0"
    _write_scalar_patch_field(field_dir / "nSurfaceLayers", [0.0, 2.0, 1.0])
    _write_scalar_patch_field(field_dir / "thickness", [0.0, 0.002, 0.001])
    _write_scalar_patch_field(field_dir / "thicknessFraction", [0.0, 0.5, 0.25])

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5, 3, 6, 7, 8])
    wall = pv.PolyData(points, faces)
    wall.cell_data["surface_region"] = np.array(["diffuser", "diffuser", "keel"])
    wall.cell_data["local_face_area_m2"] = np.array([1.0, 1.0, 2.0])
    wall.cell_data["yPlus"] = np.array([100.0, 200.0, 50.0])
    wall_vtp = tmp_path / "wall_regions.vtp"
    wall.save(wall_vtp)

    metrics = _layer_and_wall_metrics(case_dir, wall_vtp)

    assert metrics["faces_with_layers"] == 2
    assert metrics["area_fraction_with_layers"] == pytest.approx(0.75)
    assert metrics["per_region"]["diffuser"]["face_fraction_with_layers"] == pytest.approx(0.5)
    assert metrics["per_region"]["keel"]["area_fraction_with_layers"] == pytest.approx(1.0)
    assert metrics["per_region"]["keel"]["yplus"]["mean"] == pytest.approx(50.0)


def test_layer_and_wall_metrics_reads_gate2b_patch_fields(tmp_path: Path) -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="gate2b_core_transition"),
    )
    patches = article_patch_names(patch_mode="gate2b_core_transition")
    case_dir = tmp_path / "case"
    field_dir = case_dir / "openfoam" / "0"
    patch_values = {patch: [0.0] for patch in patches}
    patch_values["diffuser_core"] = [2.0]
    _write_multi_patch_scalar_field(field_dir / "nSurfaceLayers", patch_values)
    _write_multi_patch_scalar_field(field_dir / "thickness", patch_values)
    _write_multi_patch_scalar_field(field_dir / "thicknessFraction", patch_values)

    points = []
    faces = []
    regions = []
    areas = []
    yplus = []
    for index, patch in enumerate(patches):
        offset = float(index * 2)
        points.extend(
            [
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, 0.0],
            ],
        )
        base = index * 3
        faces.extend([3, base, base + 1, base + 2])
        regions.append("diffuser" if patch == "diffuser_core" else "upper_body")
        areas.append(1.0)
        yplus.append(100.0)
    wall = pv.PolyData(np.asarray(points, dtype=np.float64), np.asarray(faces))
    wall.cell_data["surface_region"] = np.asarray(regions, dtype=object)
    wall.cell_data["local_face_area_m2"] = np.asarray(areas, dtype=np.float64)
    wall.cell_data["yPlus"] = np.asarray(yplus, dtype=np.float64)
    wall_vtp = tmp_path / "wall_regions.vtp"
    wall.save(wall_vtp)

    metrics = _layer_and_wall_metrics(case_dir, wall_vtp, config)

    assert metrics["status"] == "OK"
    assert metrics["faces_with_layers"] == 1
    assert metrics["per_region"]["diffuser"]["mean_layers"] == pytest.approx(2.0)


def test_layer_and_wall_metrics_expands_uniform_patch_fields_by_patch_id(
    tmp_path: Path,
) -> None:
    config = CfdConfig(
        surface_export=SurfaceExportConfig(openfoam_patch_mode="critical_underfloor"),
    )
    patches = article_patch_names(patch_mode="critical_underfloor")
    case_dir = tmp_path / "case"
    field_dir = case_dir / "openfoam" / "0"
    patch_values = {patch: [0.0] for patch in patches}
    patch_values["critical_underfloor"] = [5.0, 0.0]
    _write_multi_patch_scalar_field(field_dir / "nSurfaceLayers", patch_values)
    _write_multi_patch_scalar_field(field_dir / "thickness", patch_values)
    _write_multi_patch_scalar_field(field_dir / "thicknessFraction", patch_values)

    points = []
    faces = []
    regions = []
    areas = []
    yplus = []
    patch_ids = []
    patch_cell_counts = {
        "critical_underfloor": 2,
        "upper_body": 2,
        "floor_edges": 1,
        "keel": 1,
        "layer_transition_band": 1,
    }
    patch_id_start = 6
    cell_index = 0
    for patch_index, patch in enumerate(patches):
        for local_index in range(patch_cell_counts[patch]):
            offset = float(cell_index * 2)
            points.extend(
                [
                    [offset, 0.0, 0.0],
                    [offset + 1.0, 0.0, 0.0],
                    [offset, 1.0, 0.0],
                ],
            )
            base = cell_index * 3
            faces.extend([3, base, base + 1, base + 2])
            regions.append("diffuser" if patch == "critical_underfloor" else patch)
            areas.append(1.0)
            yplus.append(50.0 + local_index)
            patch_ids.append(patch_id_start + patch_index)
            cell_index += 1
    wall = pv.PolyData(np.asarray(points, dtype=np.float64), np.asarray(faces))
    wall.cell_data["surface_region"] = np.asarray(regions, dtype=object)
    wall.cell_data["local_face_area_m2"] = np.asarray(areas, dtype=np.float64)
    wall.cell_data["yPlus"] = np.asarray(yplus, dtype=np.float64)
    wall.cell_data["patchID"] = np.asarray(patch_ids, dtype=np.float64)
    wall_vtp = tmp_path / "wall_regions.vtp"
    wall.save(wall_vtp)

    metrics = _layer_and_wall_metrics(case_dir, wall_vtp, config)

    assert metrics["status"] == "OK"
    assert metrics["face_count"] == 7
    assert metrics["faces_with_layers"] == 1
    assert metrics["per_region"]["diffuser"]["face_count"] == 2
    assert metrics["per_region"]["diffuser"]["faces_with_layers"] == 1
    assert metrics["per_region"]["upper_body"]["faces_with_layers"] == 0


def test_layer_and_wall_metrics_skips_missing_layer_fields_for_no_layer_smoke(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    wall = pv.PolyData(points, np.array([3, 0, 1, 2]))
    wall.cell_data["surface_region"] = np.array(["diffuser"])
    wall.cell_data["local_face_area_m2"] = np.array([1.0])
    wall.cell_data["yPlus"] = np.array([100.0])
    wall_vtp = tmp_path / "wall_regions.vtp"
    wall.save(wall_vtp)

    metrics = _layer_and_wall_metrics(case_dir, wall_vtp)

    assert metrics["status"] == "SKIPPED"
    assert "nSurfaceLayers" in metrics["missing_fields"]


def test_postprocess_records_skipped_optional_artifacts(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    config = CfdConfig(
        quality=QualityConfig(
            case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
            extended_diagnostics_required=False,
        ),
    )
    _write_case_file(
        case_dir / "manifest.json",
        json.dumps({"cfd_config": config.model_dump(mode="json")}),
    )
    _write_case_file(
        case_dir / "openfoam" / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
        """
        1 0 0.20 0.30 0 0
        2 0 0.20 0.30 0 0
        """,
    )
    _write_case_file(
        case_dir / "openfoam" / "postProcessing" / "forces" / "0" / "forces.dat",
        "2 (1 2 3) (4 5 6)\n",
    )
    _write_case_file(
        case_dir / "openfoam" / "postProcessing" / "inletFlowRate" / "0" / "surfaceFieldValue.dat",
        "2 10\n",
    )
    _write_case_file(
        case_dir / "openfoam" / "postProcessing" / "outletFlowRate" / "0" / "surfaceFieldValue.dat",
        "2 -10\n",
    )
    _write_case_file(case_dir / "quality" / "checkMesh_meshQuality.returncode", "0\n")
    _write_case_file(
        case_dir / "logs" / "checkMesh_meshQuality.log",
        """
        Mesh OK.
        cells: 1
        """,
    )
    _write_case_file(case_dir / "quality" / "checkMesh_extended.returncode", "SKIPPED\n")

    artifacts = postprocess_case(case_dir)

    mesh = json.loads(artifacts.mesh_json.read_text(encoding="utf-8"))
    yplus = json.loads(artifacts.yplus_json.read_text(encoding="utf-8"))
    force = json.loads(artifacts.force_integration_json.read_text(encoding="utf-8"))
    layers = json.loads(artifacts.layers_json.read_text(encoding="utf-8"))
    status = json.loads(artifacts.status_json.read_text(encoding="utf-8"))

    assert mesh["extended_diagnostics_status"] == "SKIPPED"
    assert mesh["foamToVTK"]["status"] == "SKIPPED"
    assert yplus["status"] == "SKIPPED"
    assert force["status"] == "SKIPPED"
    assert layers["status"] == "SKIPPED"
    assert "surface region mapping skipped" in status["blockers"]
    assert "independent force integration skipped" in status["blockers"]
    assert "yPlus post-processing skipped" in status["blockers"]
    assert "foamToVTK export skipped" in status["blockers"]

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from aeromap.cfd.quality import (
    mesh_layer_coverage_from_vtk,
    parse_snappy_layer_log,
    parse_snappy_layer_retention_log,
)


def test_parse_snappy_layer_log_reports_extrusion_collapse(tmp_path: Path) -> None:
    log = tmp_path / "snappyHexMesh.log"
    log_lines = [
        "Handling feature edges ...",
        "Set displacement to zero for points on 88 feature edges",
        "displacementMedialAxis : Reducing layer thickness at 96 nodes "
        "where thickness to medial axis distance is large",
        "displacementMedialAxis : Number of isolated points extrusion stopped : 61",
        "Detected 2 illegal faces (concave, zero area or negative cell pyramid volume)",
        "patch   faces    layers avg thickness[m]",
        "-----   -----    ------ ---------------",
        "article 696      3      0.02      0.0402",
        "Extruding 480 out of 696 faces (68.96552%). Removed extrusion at 2 faces.",
        "Added 918 out of 2088 cells (43.96552%).",
        "displacementMedialAxis : Reducing layer thickness at 0 nodes "
        "where thickness to medial axis distance is large",
        "displacementMedialAxis : Number of isolated points extrusion stopped : 0",
        "Detected 0 illegal faces (concave, zero area or negative cell pyramid volume)",
        "Extruding 60 out of 696 faces (8.62069%). Removed extrusion at 0 faces.",
        "Added 60 out of 2088 cells (2.873563%).",
        "",
        "patch   faces    layers   overall thickness",
        "                          [m]       [%]",
        "-----   -----    ------   ---       ---",
        "article 696      0.0862   0.00197   3.34",
        "",
        "Layer mesh : cells:503900  faces:1532392  points:524820",
    ]
    log.write_text("\n".join(log_lines), encoding="utf-8")

    report = parse_snappy_layer_log(log)

    assert report["feature_edges_zeroed"] == 88
    assert report["iteration_count"] == 2
    assert report["first_extrusion"]["faces"] == 480
    assert report["final_extrusion"]["faces"] == 60
    assert report["final_added_cells"]["cells"] == 60
    assert report["final_article_layers"]["average_layers"] == pytest.approx(0.0862)
    assert report["medial_axis_reduction_summary"] == {"first": 96, "last": 0, "max": 96}
    assert report["isolated_points_summary"] == {"first": 61, "last": 0, "max": 61}
    assert report["contains_layer_mesh"] is True


def test_parse_snappy_layer_retention_log_reports_iteration_deltas(tmp_path: Path) -> None:
    log = tmp_path / "snappyHexMesh.log"
    log.write_text(
        (
            "Layer addition iteration 0\n"
            "displacementMedialAxis : Reducing layer thickness at 0 nodes "
            "where thickness to medial axis distance is large\n"
            "displacementMedialAxis : Number of isolated points extrusion stopped : 10\n"
            "Checking mesh with layer ...\n"
            "    non-orthogonality > 65  degrees                        : 2\n"
            "    faces with interpolation weights (0..1)  < 0.05        : 50\n"
            "    faces on cells with determinant < 0.001                : 3\n"
            "Detected 55 illegal faces (concave, zero area or negative cell pyramid volume)\n"
            "Extruding 900 out of 1000 faces (90%). Removed extrusion at 20 faces.\n"
            "Added 920 out of 1000 cells (92%).\n"
            "Layer addition iteration 1\n"
            "displacementMedialAxis : Reducing layer thickness at 7 nodes "
            "where thickness to medial axis distance is large\n"
            "displacementMedialAxis : Number of isolated points extrusion stopped : 4\n"
            "Checking mesh with layer ...\n"
            "    non-orthogonality > 65  degrees                        : 0\n"
            "    faces with interpolation weights (0..1)  < 0.05        : 12\n"
            "    faces on cells with determinant < 0.001                : 0\n"
            "Detected 12 illegal faces (concave, zero area or negative cell pyramid volume)\n"
            "Extruding 700 out of 1000 faces (70%). Removed extrusion at 8 faces.\n"
            "Added 712 out of 1000 cells (71.2%)."
        ),
        encoding="utf-8",
    )

    report = parse_snappy_layer_retention_log(log)

    assert report["iteration_count"] == 2
    assert report["summary"]["first_extruded_faces"] == 900
    assert report["summary"]["final_extruded_faces"] == 700
    assert report["summary"]["retained_fraction_of_first_extrusion"] == pytest.approx(7 / 9)
    assert report["summary"]["total_extruded_faces_lost"] == 200
    first, second = report["iterations"]
    assert first["removed_faces_reported"] == 20
    assert first["quality_counts"]["low_interpolation_weight_faces"] == 50
    assert first["quality_counts"]["low_determinant_faces"] == 3
    assert first["extruded_faces_delta_from_previous"] is None
    assert second["medial_axis_reduction_nodes"] == 7
    assert second["isolated_points_stopped"] == 4
    assert second["extruded_faces_delta_from_previous"] == -200


def test_mesh_layer_coverage_uses_patch_area_weights(tmp_path: Path) -> None:
    patch_dir = tmp_path / "case" / "openfoam" / "VTK" / "diffuser_core"
    patch_dir.mkdir(parents=True)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])
    patch = pv.PolyData(points, faces)
    patch.cell_data["nSurfaceLayers"] = np.array([0.0, 1.0])
    patch.save(patch_dir / "diffuser_core_0.vtk")

    report = mesh_layer_coverage_from_vtk(
        case_dir=tmp_path / "case",
        patch_names=("diffuser_core",),
        critical_patches=("diffuser_core",),
        min_area_coverage=0.75,
    )

    diffuser = report["patches"]["diffuser_core"]
    assert diffuser["face_fraction_with_layers"] == pytest.approx(0.5)
    assert diffuser["area_fraction_with_layers"] == pytest.approx(0.8)
    assert report["critical_area_coverage_ok"] is True


def test_mesh_layer_coverage_ignores_missing_noncritical_patch_for_critical_gate(
    tmp_path: Path,
) -> None:
    patch_dir = tmp_path / "case" / "openfoam" / "VTK" / "critical_underfloor"
    patch_dir.mkdir(parents=True)
    patch = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array([3, 0, 1, 2]),
    )
    patch.cell_data["nSurfaceLayers"] = np.array([3.0])
    patch.save(patch_dir / "critical_underfloor_0.vtk")

    report = mesh_layer_coverage_from_vtk(
        case_dir=tmp_path / "case",
        patch_names=("critical_underfloor", "keel"),
        critical_patches=("critical_underfloor",),
        min_area_coverage=0.8,
    )

    assert report["missing_patches"] == ["keel"]
    assert report["missing_critical_patches"] == []
    assert report["critical_area_coverage"]["critical_underfloor"] == pytest.approx(1.0)
    assert report["critical_area_coverage_ok"] is True

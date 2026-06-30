from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aeromap.cfd.diagnostics import (
    _mesh_input_hashes,
    _min_distance_to_segments,
    _read_openfoam_internal_vectors,
    _read_openfoam_label_list,
)
from aeromap.cfd.topology_report import (
    MESH_SET_COLUMNS,
    _cluster_label,
    _read_csv,
    _reported_problem_count,
    _summarize_mesh_set,
)


def test_min_distance_to_segments_projects_to_finite_segment() -> None:
    points = np.array([[0.5, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    segments = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)

    distances = _min_distance_to_segments(points, segments)

    assert distances[0] == pytest.approx(1.0)
    assert distances[1] == pytest.approx(1.0)


def test_openfoam_label_list_reader_counts_entries(tmp_path: Path) -> None:
    path = tmp_path / "concaveCells"
    path.write_text(
        """
FoamFile {}

3
(
42
7
9
)
""",
        encoding="utf-8",
    )

    assert _read_openfoam_label_list(path) == [42, 7, 9]


def test_openfoam_label_list_reader_handles_compact_repeated_form(tmp_path: Path) -> None:
    path = tmp_path / "cellLevel"
    path.write_text(
        """
FoamFile {}

5{0}
""",
        encoding="utf-8",
    )

    assert _read_openfoam_label_list(path) == [0, 0, 0, 0, 0]


def test_openfoam_internal_vector_reader_ignores_boundary_field(tmp_path: Path) -> None:
    path = tmp_path / "C"
    path.write_text(
        """
FoamFile {}

internalField   nonuniform List<vector>
2
(
(1 2 3)
(4 5 6)
)
;

boundaryField
{
    wall {}
}
""",
        encoding="utf-8",
    )

    vectors = _read_openfoam_internal_vectors(path)

    assert vectors.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_mesh_input_hashes_include_actual_probe_dictionaries(tmp_path: Path) -> None:
    snappy = tmp_path / "openfoam" / "system" / "snappyHexMeshDict"
    check_mesh = tmp_path / "logs" / "checkMesh.log"
    snappy.parent.mkdir(parents=True)
    check_mesh.parent.mkdir(parents=True)
    snappy.write_text("features ();", encoding="utf-8")
    check_mesh.write_text("Failed 1 mesh checks.", encoding="utf-8")

    hashes = _mesh_input_hashes(tmp_path)

    assert hashes["system/snappyHexMeshDict"] is not None
    assert hashes["logs/checkMesh.log"] is not None
    assert hashes["system/blockMeshDict"] is None


def test_topology_report_cluster_label_detects_dominant_cad_face() -> None:
    label = _cluster_label(
        [{"value": "25", "count": 8, "fraction": 8 / 9}],
        [{"value": "diffuser", "count": 4, "fraction": 4 / 9}],
    )

    assert label == "single_cad_face_dominant"


def test_topology_report_mesh_set_summary_includes_bounds_and_locations() -> None:
    rows = [
        {
            "diagnostic_element_index": "0",
            "x_m": "1.0",
            "y_m": "-0.2",
            "z_m": "0.1",
            "nearest_stl_triangle": "10",
            "nearest_cad_face_id": "25",
            "surface_region": "floor_edges",
            "distance_to_surface_m": "0.03",
        },
        {
            "diagnostic_element_index": "1",
            "x_m": "1.2",
            "y_m": "-0.1",
            "z_m": "0.2",
            "nearest_stl_triangle": "11",
            "nearest_cad_face_id": "25",
            "surface_region": "floor_edges",
            "distance_to_surface_m": "0.04",
        },
    ]

    summary = _summarize_mesh_set(
        set_name="underdeterminedCells",
        rows=rows,
        raw_summary={"openfoam_reported_problem_count": 2},
    )

    assert summary["openfoam_reported_problem_count"] == 2
    assert summary["bounds_m"]["min_m"] == [1.0, -0.2, 0.1]
    assert summary["bounds_m"]["max_m"] == [1.2, -0.1, 0.2]
    assert summary["cluster_label"] == "single_cad_face_dominant"
    assert summary["sample_locations"][0]["nearest_cad_face_id"] == 25


def test_topology_report_requires_diagnostic_csv_with_schema(tmp_path: Path) -> None:
    missing = tmp_path / "concaveCells_mapped.csv"

    with pytest.raises(FileNotFoundError, match="required diagnostic CSV is missing"):
        _read_csv(missing, required_columns=MESH_SET_COLUMNS)

    malformed = tmp_path / "concaveCells_mapped.csv"
    malformed.write_text("x_m,y_m,z_m\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        _read_csv(malformed, required_columns=MESH_SET_COLUMNS)


def test_topology_report_detects_nonzero_raw_problem_counts() -> None:
    assert _reported_problem_count({"openfoam_reported_problem_count": 7}) == 7
    assert _reported_problem_count({"diagnostic_vtk_element_count": 3}) == 3
    assert _reported_problem_count({"openfoam_reported_problem_count": 0}) == 0

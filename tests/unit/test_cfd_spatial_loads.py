from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from aeromap.cfd.spatial_loads import (
    integrate_openfoam_boundary_spatial_loads,
    integrate_spatial_loads,
    write_urans_spatial_load_history,
)
from aeromap.constants import REF
from aeromap.parameters import AeroParams


def _two_panel_wall(path: Path) -> None:
    points = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([4, 0, 1, 2, 3, 4, 4, 5, 6, 7], dtype=np.int64)
    wall = pv.PolyData(points, faces)
    wall.cell_data["p"] = np.asarray([2.0, 4.0], dtype=np.float64)
    wall.cell_data["wallShearStress"] = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    wall.cell_data["surface_region"] = np.asarray(["underfloor", "underfloor"], dtype=object)
    wall.cell_data["local_face_area_m2"] = np.asarray([1.0, 1.0], dtype=np.float64)
    wall.cell_data["body_local_centroid_m"] = np.asarray(
        [[0.5, -0.5, 0.0], [1.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    wall.save(path)


def _write_openfoam_boundary_fixture(root: Path) -> None:
    mesh = root / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    (root / "constant" / "triSurface").mkdir(parents=True)
    points = np.asarray(
        [
            [0.0, -0.3, 0.0],
            [1.0, -0.3, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.3, 0.0],
            [0.0, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([4, 0, 3, 2, 1, 4, 4, 7, 6, 5], dtype=np.int64)
    source = pv.PolyData(points, faces)
    source.save(root / "constant" / "triSurface" / "article_surface_regions.vtp")
    point_lines = "\n".join(f"({x:g} {y:g} {z:g})" for x, y, z in points)
    (mesh / "points").write_text(
        f"""
FoamFile {{ object points; }}
8
(
{point_lines}
)
""".lstrip(),
        encoding="utf-8",
    )
    (mesh / "faces").write_text(
        """
FoamFile { object faces; }
2
(
4(0 3 2 1)
4(4 7 6 5)
)
""".lstrip(),
        encoding="utf-8",
    )
    (mesh / "owner").write_text(
        """
FoamFile { object owner; }
2
(
0
1
)
""".lstrip(),
        encoding="utf-8",
    )
    (mesh / "boundary").write_text(
        """
FoamFile { object boundary; }
1
(
    critical_underfloor
    {
        type wall;
        nFaces 2;
        startFace 0;
    }
)
""".lstrip(),
        encoding="utf-8",
    )


def _write_time_fields(root: Path, time_dir: str, pressure: tuple[float, float]) -> None:
    time_path = root / time_dir
    time_path.mkdir(parents=True)
    time_path.joinpath("p").write_text(
        f"""
FoamFile {{ object p; }}
boundaryField
{{
    critical_underfloor
    {{
        type zeroGradient;
        value nonuniform List<scalar>
        2
        (
        {pressure[0]:g}
        {pressure[1]:g}
        )
        ;
    }}
}}
""".lstrip(),
        encoding="utf-8",
    )
    time_path.joinpath("wallShearStress").write_text(
        """
FoamFile { object wallShearStress; }
boundaryField
{
    critical_underfloor
    {
        type calculated;
        value nonuniform List<vector>
        2
        (
        (1 0 0)
        (2 0 0)
        )
        ;
    }
}
""".lstrip(),
        encoding="utf-8",
    )


def test_spatial_loads_split_left_right_and_streamwise_bins(tmp_path: Path) -> None:
    wall_vtp = tmp_path / "wall.vtp"
    _two_panel_wall(wall_vtp)

    report = integrate_spatial_loads(
        wall_vtp,
        params=AeroParams.canonical(),
        streamwise_bins=2,
    )

    total = report["total"]
    assert total["pressure_n"] == pytest.approx([0.0, 0.0, REF.rho_kg_m3 * 6.0])
    assert total["viscous_n"] == pytest.approx([-REF.rho_kg_m3 * 3.0, 0.0, 0.0])
    assert total["total_n"] == pytest.approx(
        [-REF.rho_kg_m3 * 3.0, 0.0, REF.rho_kg_m3 * 6.0],
    )
    assert total["coefficients"]["c_df"] == pytest.approx(
        -(REF.rho_kg_m3 * 6.0) / (REF.q_inf_pa * REF.a_ref_m2),
    )

    groups = report["named_groups"]["loads"]
    assert groups["left_tunnel_y_negative"]["pressure_n"] == pytest.approx(
        [0.0, 0.0, REF.rho_kg_m3 * 2.0],
    )
    assert groups["right_tunnel_y_positive"]["pressure_n"] == pytest.approx(
        [0.0, 0.0, REF.rho_kg_m3 * 4.0],
    )

    bins = report["streamwise_bins"]["all_article"]
    assert len(bins) == 2
    assert bins[0]["cell_count"] == 1
    assert bins[1]["cell_count"] == 1
    assert bins[0]["pressure_n"] == pytest.approx([0.0, 0.0, REF.rho_kg_m3 * 2.0])
    assert bins[1]["pressure_n"] == pytest.approx([0.0, 0.0, REF.rho_kg_m3 * 4.0])
    assert report["phase_relation"]["status"] == "UNAVAILABLE_SINGLE_SNAPSHOT"


def test_spatial_loads_requires_exported_wall_fields(tmp_path: Path) -> None:
    wall = pv.PolyData(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([3, 0, 1, 2]),
    )
    wall.cell_data["p"] = np.asarray([1.0])
    wall_vtp = tmp_path / "missing.vtp"
    wall.save(wall_vtp)

    with pytest.raises(KeyError, match="required cell fields"):
        integrate_spatial_loads(wall_vtp, params=AeroParams.canonical())


def test_openfoam_boundary_spatial_loads_integrate_raw_fields(tmp_path: Path) -> None:
    openfoam = tmp_path / "openfoam"
    _write_openfoam_boundary_fixture(openfoam)
    _write_time_fields(openfoam, "0.001", (2.0, 4.0))

    report = integrate_openfoam_boundary_spatial_loads(
        openfoam_dir=openfoam,
        time_dir="0.001",
        params=AeroParams.canonical(),
        streamwise_bins=2,
    )

    assert report["cell_count"] == 2
    assert report["regions_present"] == ["tunnel_roofs"]
    assert report["total"]["pressure_n"] == pytest.approx([0.0, 0.0, -REF.rho_kg_m3 * 1.8])
    assert report["total"]["viscous_n"] == pytest.approx([-REF.rho_kg_m3 * 0.9, 0.0, 0.0])
    groups = report["named_groups"]["loads"]
    assert groups["left_tunnel_y_negative"]["pressure_n"] == pytest.approx(
        [0.0, 0.0, -REF.rho_kg_m3 * 0.6],
    )
    assert groups["right_tunnel_y_positive"]["pressure_n"] == pytest.approx(
        [0.0, 0.0, -REF.rho_kg_m3 * 1.2],
    )
    assert report["total"]["coefficients"]["c_df"] == pytest.approx(
        (REF.rho_kg_m3 * 1.8) / (REF.q_inf_pa * REF.a_ref_m2),
    )


def test_urans_spatial_load_history_writes_compact_time_series(tmp_path: Path) -> None:
    work_case = tmp_path / "urans_run"
    openfoam = work_case / "openfoam"
    _write_openfoam_boundary_fixture(openfoam)
    _write_time_fields(openfoam, "0.001", (2.0, 4.0))
    _write_time_fields(openfoam, "0.002", (3.0, 5.0))

    report = write_urans_spatial_load_history(
        work_case=work_case,
        params=AeroParams.canonical(),
        streamwise_bins=2,
    )

    assert report["accepted"] is False
    assert report["training_eligible"] is False
    assert report["row_count"] == 2
    assert report["time_dirs"] == ["0.001", "0.002"]
    assert report["rows"][1]["total"]["pressure_n"] == pytest.approx(
        [0.0, 0.0, -REF.rho_kg_m3 * 2.4],
    )
    assert (work_case / "quality" / "urans_spatial_load_history.json").exists()

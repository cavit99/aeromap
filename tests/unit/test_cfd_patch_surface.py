from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from aeromap.cfd.patch_surface import write_gate2b_patch_surface, write_openfoam_patch_surface


def test_gate2b_patch_surface_writes_obj_groups_and_metrics(tmp_path: Path) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    regions = {
        "face_regions": [
            {"face_index": 0, "region": "diffuser", "region_id": 0},
            {"face_index": 1, "region": "underfloor", "region_id": 2},
        ],
    }

    artifacts = write_gate2b_patch_surface(
        mesh=mesh,
        regions=regions,
        out_dir=tmp_path,
        transition_band_width_m=0.0,
        feature_angle_deg=30.0,
    )

    obj = artifacts.obj_path.read_text(encoding="utf-8")
    assert "g diffuser_core" in obj
    assert "g underfloor_core" in obj
    assert "g layer_transition_band" in obj
    metrics = artifacts.metrics_path.read_text(encoding="utf-8")
    assert "diffuser_core" in metrics
    assert "underfloor_core" in metrics
    vtp = pv.read(artifacts.vtp_path)
    assert "openfoam_patch" in vtp.cell_data


def test_critical_underfloor_patch_surface_combines_analytical_regions(
    tmp_path: Path,
) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array(
            [
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
            ],
            dtype=np.int64,
        ),
        process=False,
    )
    regions = {
        "face_regions": [
            {"face_index": 0, "region": "tunnel_roofs", "region_id": 1},
            {"face_index": 1, "region": "diffuser", "region_id": 0},
            {"face_index": 2, "region": "underfloor", "region_id": 2},
            {"face_index": 3, "region": "upper_body", "region_id": 3},
        ],
    }

    artifacts = write_openfoam_patch_surface(
        mesh=mesh,
        regions=regions,
        out_dir=tmp_path,
        transition_band_width_m=0.0,
        feature_angle_deg=180.0,
        patch_mode="critical_underfloor",
    )

    obj = artifacts.obj_path.read_text(encoding="utf-8")
    assert "g critical_underfloor" in obj
    assert "g diffuser_core" not in obj
    assert artifacts.patch_names == (
        "critical_underfloor",
        "upper_body",
        "floor_edges",
        "keel",
        "layer_transition_band",
    )

    vtp = pv.read(artifacts.vtp_path)
    assert set(vtp.cell_data["openfoam_patch"][:3]) == {"critical_underfloor"}
    assert set(vtp.cell_data["source_surface_region"][:3]) == {
        "tunnel_roofs",
        "diffuser",
        "underfloor",
    }

    metrics = artifacts.metrics_path.read_text(encoding="utf-8")
    assert '"patch_mode": "critical_underfloor"' in metrics
    assert '"connected_watertight_shell_preserved": true' in metrics

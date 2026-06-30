from __future__ import annotations

import numpy as np
import trimesh

from aeromap.geometry.validate import validate_mesh


def test_validate_mesh_returns_invalid_for_empty_mesh() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.empty((0, 3), dtype=np.float64),
        faces=np.empty((0, 3), dtype=np.int64),
        process=False,
    )

    validation = validate_mesh(mesh)

    assert not validation.valid
    assert validation.metrics is None
    assert "empty_or_invalid_vertices" in validation.reasons
    assert "empty_or_invalid_faces" in validation.reasons


def test_validate_mesh_returns_invalid_for_non_finite_vertices() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )

    validation = validate_mesh(mesh)

    assert not validation.valid
    assert validation.metrics is None
    assert validation.reasons == ("non_finite_vertices",)

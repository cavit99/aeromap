from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh

from aeromap.geometry.domino import DominoGeometryAdapter
from aeromap.parameters import AeroParams

if TYPE_CHECKING:
    import pytest


class FakeValidation:
    valid = True

    def model_dump(self, mode: str) -> dict[str, Any]:
        return {"mode": mode, "valid": True}


def test_domino_refinement_moves_mesh_toward_target() -> None:
    mesh = trimesh.creation.box()
    adapter = DominoGeometryAdapter(target_surface_points=len(mesh.vertices) + 1)

    refined, steps = adapter.subdivide_to_target(mesh)

    assert len(refined.vertices) >= adapter.target_surface_points
    assert steps > 0


def test_domino_export_reports_unmet_target_as_not_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("aeromap.geometry.domino.build_article", lambda _params: object())
    monkeypatch.setattr(
        "aeromap.geometry.domino.cq.exporters.export",
        lambda _article, path, **_kwargs: Path(path).write_text(
            "solid empty\nendsolid empty\n",
            encoding="utf-8",
        ),
    )
    monkeypatch.setattr(
        "aeromap.geometry.domino.trimesh.load_mesh",
        lambda *_args, **_kwargs: trimesh.creation.box(),
    )
    monkeypatch.setattr(
        "aeromap.geometry.domino.validate_mesh",
        lambda _mesh: FakeValidation(),
    )

    export = DominoGeometryAdapter(target_surface_points=10**12).export(
        AeroParams.canonical(),
        tmp_path,
    )

    manifest = json.loads(export.manifest_path.read_text(encoding="utf-8"))
    assert not export.nim_input_contract_valid
    assert not export.meets_target_surface_points
    assert manifest["nim_input_contract_valid"] is False
    assert manifest["meets_target_surface_points"] is False
    assert "cad_retessellation" in manifest
    assert "cad_to_nim_transform_4x4" in manifest

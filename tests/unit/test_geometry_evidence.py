from __future__ import annotations

from pathlib import Path

import pytest

from aeromap.geometry.evidence import build_geometry_evidence


def test_geometry_evidence_refuses_non_evidence_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "not_evidence"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="non-evidence directory"):
        build_geometry_evidence(output_dir, samples=1)

    assert (output_dir / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_geometry_evidence_preserves_existing_output_on_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "geometry"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "old.txt").write_text("old evidence", encoding="utf-8")

    def fail_generate(*_args: object, **_kwargs: object) -> object:
        message = "forced generation failure"
        raise RuntimeError(message)

    monkeypatch.setattr("aeromap.geometry.evidence.generate_geometry", fail_generate)

    with pytest.raises(RuntimeError, match="forced generation failure"):
        build_geometry_evidence(output_dir, samples=1)

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "old.txt").read_text(encoding="utf-8") == "old evidence"


def test_geometry_evidence_preserves_unrelated_hidden_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "geometry"
    staging_collision = tmp_path / ".geometry.staging"
    previous_collision = tmp_path / ".geometry.previous"
    staging_collision.mkdir()
    previous_collision.mkdir()
    (staging_collision / "sentinel.txt").write_text("staging", encoding="utf-8")
    (previous_collision / "sentinel.txt").write_text("previous", encoding="utf-8")

    def fail_generate(*_args: object, **_kwargs: object) -> object:
        message = "forced generation failure"
        raise RuntimeError(message)

    monkeypatch.setattr("aeromap.geometry.evidence.generate_geometry", fail_generate)

    with pytest.raises(RuntimeError, match="forced generation failure"):
        build_geometry_evidence(output_dir, samples=1)

    assert (staging_collision / "sentinel.txt").read_text(encoding="utf-8") == "staging"
    assert (previous_collision / "sentinel.txt").read_text(encoding="utf-8") == "previous"

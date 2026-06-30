"""Preview utilities."""

from __future__ import annotations

from pathlib import Path

from aeromap.geometry.generator import generate_geometry
from aeromap.parameters import AeroParams


def generate_canonical_preview(output_dir: Path) -> Path:
    artifacts = generate_geometry(AeroParams.canonical(), output_dir)
    return artifacts.preview_html_path

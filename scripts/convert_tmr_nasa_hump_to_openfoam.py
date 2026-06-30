"""Materialise a local OpenFOAM conversion scaffold for the NASA/TMR hump grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeromap.cfd.nasa_hump import write_conversion_scaffold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "artifacts/methodology/nasa_hump/raw/hump2newtop_noplenumZ103x28.p2dfmt.gz"
DEFAULT_OUT = ROOT / "artifacts/methodology/nasa_hump/openfoam_scaffold"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a local conversion scaffold for the NASA/TMR 103 x 28 no-plenum "
            "PLOT3D grid. This does not run OpenFOAM."
        ),
    )
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.grid.exists():
        msg = (
            f"grid file not found: {args.grid}. Run "
            "`uv run python scripts/prepare_nasa_hump_methodology.py` first."
        )
        raise FileNotFoundError(msg)

    manifest = write_conversion_scaffold(grid_path=args.grid, out_dir=args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

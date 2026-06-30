"""Run a bounded OpenFOAM SST candidate on the NASA/TMR medium hump grid."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_nasa_hump_sst_smoke import run_smoke_case  # noqa: E402

GRID_ZIP_URL = "https://www.nasa.gov/wp-content/uploads/2026/02/nasahump-grids.zip"
GRID_MEMBER_409X109 = (
    "u/piyer/nasa_tmr/gitlab/turbmodels/Nasahump_grids/hump2newtop_noplenumZ409x109.p2dfmt.gz"
)
DEFAULT_GRID = ROOT / "artifacts/methodology/nasa_hump/raw/hump2newtop_noplenumZ409x109.p2dfmt.gz"
DEFAULT_CASE_DIR = ROOT / "artifacts/methodology/nasa_hump/sst_medium_grid_case"


def download_grid(path: Path, *, refresh: bool) -> None:
    if path.exists() and not refresh:
        return
    response = requests.get(GRID_ZIP_URL, timeout=90)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(GRID_MEMBER_409X109))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--end-time", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh-grid", action="store_true")
    args = parser.parse_args()

    download_grid(args.grid, refresh=args.refresh_grid)
    summary = run_smoke_case(
        grid_path=args.grid,
        case_dir=args.case_dir,
        overwrite=args.overwrite,
        end_time=args.end_time,
    )
    summary["classification"] = "OPENFOAM_NASA_HUMP_MEDIUM_GRID_SST_RUN_ATTEMPT"
    summary["grid"] = str(args.grid)
    summary["grid_family"] = "NASA/TMR no-plenum 409 x 109"
    summary["report_command"] = (
        f"uv run python scripts/report_nasa_hump_medium_grid_sst.py --case-dir {args.case_dir}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

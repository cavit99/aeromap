"""Non-headline OpenFOAM v13 official tutorial reference lane."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aeromap.cfd.quality import parse_check_mesh_log
from aeromap.io import atomic_write_json, sha256_file

ReferenceCaseName = Literal["drivaerFastback", "motorBikeSteady"]
ReferenceMode = Literal["inspect", "block-mesh", "mesh", "solve"]

REFERENCE_LANE_SCHEMA_VERSION = "openfoam_v13_reference_lane_v0.1.0"
DEFAULT_OPENFOAM_IMAGE = "aeromap/openfoam13:dev"
REFERENCE_CASE_PATHS: dict[str, str] = {
    "drivaerFastback": "$FOAM_TUTORIALS/incompressibleFluid/drivaerFastback",
    "motorBikeSteady": "$FOAM_TUTORIALS/incompressibleFluid/motorBikeSteady",
}
SURFACE_CHECK_RE = {
    "triangles": re.compile(r"Triangles\s*:\s*(?P<value>\d+)"),
    "vertices": re.compile(r"Vertices\s*:\s*(?P<value>\d+)"),
    "quality_0_to_0p05": re.compile(r"0 \.\. 0\.05\s*:\s*(?P<value>[0-9.eE+-]+)"),
    "min_quality": re.compile(r"min\s+(?P<value>[0-9.eE+-]+)\s+for triangle"),
    "min_edge_m": re.compile(r"Edges:\s*\n\s*min\s+(?P<value>[0-9.eE+-]+)"),
    "unconnected_parts": re.compile(r"Number of unconnected parts\s*:\s*(?P<value>\d+)"),
}


@dataclass(frozen=True)
class ReferenceLaneResult:
    case_name: str
    mode: str
    out_dir: Path
    summary_path: Path | None
    command: list[str]
    script: str
    return_code: int | None


def _project_relative(path: Path) -> str:
    root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        message = f"reference lane output must be inside the project root: {path}"
        raise RuntimeError(message) from exc


def _validate_case_name(case_name: str) -> ReferenceCaseName:
    if case_name not in REFERENCE_CASE_PATHS:
        message = f"unknown OpenFOAM reference case {case_name!r}"
        raise ValueError(message)
    return case_name  # type: ignore[return-value]


def _validate_mode(mode: str) -> ReferenceMode:
    if mode not in {"inspect", "block-mesh", "mesh", "solve"}:
        message = f"unknown reference lane mode {mode!r}"
        raise ValueError(message)
    return mode  # type: ignore[return-value]


def _reference_setup_script(
    *, case_name: ReferenceCaseName, mode: ReferenceMode, out_rel: str
) -> str:
    out_dir_literal = f"/work/{shlex.quote(out_rel)}"
    return f"""set +eu
source /opt/openfoam13/etc/bashrc
source_status=$?
if [ "$source_status" -ne 0 ]; then
  echo "failed to source OpenFOAM v13 environment" >&2
  exit 2
fi
set -euo pipefail

case_name={case_name}
mode={mode}
out_dir={out_dir_literal}
source_case=$FOAM_TUTORIALS/incompressibleFluid/$case_name
case_dir=$out_dir/case
log_dir=$out_dir/logs
quality_dir=$out_dir/quality

rm -rf "$case_dir" "$log_dir" "$quality_dir"
mkdir -p "$out_dir" "$log_dir" "$quality_dir"
cp -R "$source_case" "$case_dir"
cd "$case_dir"

if [ "$case_name" = "drivaerFastback" ]; then
  cp system/controlDict.orig system/controlDict
  cp system/snappyHexMeshDict.orig system/snappyHexMeshDict
  cp system/decomposeParDict.orig system/decomposeParDict
  gunzip -kf constant/geometry/*.gz
elif [ "$case_name" = "motorBikeSteady" ]; then
  cp 0/U.orig 0/U
  cp "$FOAM_TUTORIALS/resources/geometry/motorBike.obj.gz" constant/geometry/
  gunzip -kf constant/geometry/motorBike.obj.gz
fi

for surface in constant/geometry/*.obj constant/geometry/*.stl; do
  if [ -f "$surface" ]; then
    name=$(basename "$surface")
    surface_abs=$(readlink -f "$surface")
    surface_scratch="$quality_dir/surfaceCheck_${{name}}_scratch"
    rm -rf "$surface_scratch"
    mkdir -p "$surface_scratch"
    set +e
    (
      cd "$surface_scratch" && surfaceCheck "$surface_abs"
    ) > "$log_dir/surfaceCheck_${{name}}.log" 2>&1
    status=$?
    set -e
    printf "%s\\n" "$status" > "$quality_dir/surfaceCheck_${{name}}.returncode"
    rm -rf "$surface_scratch"
  fi
done

if [ "$mode" != "inspect" ]; then
  blockMesh > "$log_dir/blockMesh.log" 2>&1

  set +e
  checkMesh > "$log_dir/checkMesh_blockMesh.log" 2>&1
  status=$?
  set -e
  printf "%s\\n" "$status" > "$quality_dir/checkMesh_blockMesh.returncode"

  if [ "$mode" != "block-mesh" ]; then
    if [ -f system/surfaceFeaturesDict ]; then
      surfaceFeatures > "$log_dir/surfaceFeatures.log" 2>&1
    fi

    set +e
    snappyHexMesh -overwrite > "$log_dir/snappyHexMesh.log" 2>&1
    snappy_status=$?
    set -e
    printf "%s\\n" "$snappy_status" > "$quality_dir/snappyHexMesh.returncode"

    if [ "$snappy_status" -eq 0 ]; then
      set +e
      checkMesh -allGeometry -allTopology -writeSurfaces -writeSets \\
        -surfaceFormat vtk -setFormat vtk > "$log_dir/checkMesh.log" 2>&1
      status=$?
      set -e
      printf "%s\\n" "$status" > "$quality_dir/checkMesh.returncode"

      if [ "$mode" = "solve" ]; then
        set +e
        foamRun -solver incompressibleFluid > "$log_dir/foamRun.log" 2>&1
        solver_status=$?
        postProcess -func yPlus -latestTime > "$log_dir/yPlus.log" 2>&1
        foamToVTK -latestTime > "$log_dir/foamToVTK.log" 2>&1
        set -e
        printf "%s\\n" "$solver_status" > "$quality_dir/foamRun.returncode"
      fi
    fi
  fi
fi

true
"""


def reference_lane_command(
    *,
    case_name: str,
    mode: str,
    out_dir: Path,
    image: str = DEFAULT_OPENFOAM_IMAGE,
) -> tuple[list[str], str]:
    """Build the Docker command and script for the official reference lane."""

    validated_case = _validate_case_name(case_name)
    validated_mode = _validate_mode(mode)
    out_rel = _project_relative(out_dir)
    script = _reference_setup_script(case_name=validated_case, mode=validated_mode, out_rel=out_rel)
    command = [
        shutil.which("docker") or "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:/work",
        "-w",
        "/work",
        "--entrypoint",
        "/bin/bash",
        image,
        "-lc",
        script,
    ]
    return command, script


def _return_code(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text else None


def parse_surface_check_log(path: Path) -> dict[str, Any]:
    """Parse the small surfaceCheck subset used by the reference lane."""

    text = path.read_text(encoding="utf-8", errors="replace")
    parsed: dict[str, Any] = {
        "surface_check_ok": "Surface is closed" in text and "no illegal triangles" in text,
        "log_path": str(path),
    }
    for name, pattern in SURFACE_CHECK_RE.items():
        match = pattern.search(text)
        if match is None:
            parsed[name] = None
            continue
        value = match.group("value")
        parsed[name] = float(value) if "." in value or "e" in value.lower() else int(value)
    return parsed


def summarize_reference_lane(*, case_name: str, mode: str, out_dir: Path) -> Path:
    """Write an AeroCliff-style JSON summary for a copied official tutorial case."""

    logs_dir = out_dir / "logs"
    quality_dir = out_dir / "quality"
    case_dir = out_dir / "case"
    surface_logs = sorted(logs_dir.glob("surfaceCheck_*.log"))
    check_mesh_logs = {
        path.name.removesuffix(".log"): parse_check_mesh_log(path)
        for path in sorted(logs_dir.glob("checkMesh*.log"))
    }
    summary: dict[str, Any] = {
        "schema_version": REFERENCE_LANE_SCHEMA_VERSION,
        "case_name": case_name,
        "mode": mode,
        "source": "OpenFOAM Foundation v13 installed tutorials",
        "non_headline": True,
        "purpose": (
            "Reference-lane diagnostics only; outside the Venturi Core benchmark "
            "and public scientific results."
        ),
        "case_dir": str(case_dir),
        "surface_checks": [parse_surface_check_log(path) for path in surface_logs],
        "check_mesh": check_mesh_logs,
        "return_codes": {
            path.name.removesuffix(".returncode"): _return_code(path)
            for path in sorted(quality_dir.glob("*.returncode"))
        },
        "log_hashes": {
            str(path.relative_to(out_dir)): sha256_file(path)
            for path in sorted(logs_dir.glob("*"))
            if path.is_file()
        },
        "dictionary_hashes": {
            str(path.relative_to(case_dir)): sha256_file(path)
            for path in sorted((case_dir / "system").glob("*"))
            if path.is_file()
        },
    }
    summary_path = out_dir / "reference_lane_summary.json"
    atomic_write_json(summary_path, summary)
    return summary_path


def run_reference_lane(
    *,
    case_name: str,
    mode: str,
    out_dir: Path,
    image: str = DEFAULT_OPENFOAM_IMAGE,
    dry_run: bool = False,
) -> ReferenceLaneResult:
    """Run or describe a non-headline official OpenFOAM reference lane."""

    command, script = reference_lane_command(
        case_name=case_name,
        mode=mode,
        out_dir=out_dir,
        image=image,
    )
    if dry_run:
        return ReferenceLaneResult(
            case_name=case_name,
            mode=mode,
            out_dir=out_dir,
            summary_path=None,
            command=command,
            script=script,
            return_code=None,
        )

    completed = subprocess.run(command, check=False)  # noqa: S603
    summary_path = summarize_reference_lane(case_name=case_name, mode=mode, out_dir=out_dir)
    return ReferenceLaneResult(
        case_name=case_name,
        mode=mode,
        out_dir=out_dir,
        summary_path=summary_path,
        command=command,
        script=script,
        return_code=completed.returncode,
    )

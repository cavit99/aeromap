"""Post-process completed OpenFOAM cases into typed CFD artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from numpy.typing import NDArray

from aeromap.cfd.patch_surface import article_patch_names
from aeromap.cfd.quality import parse_check_mesh_log
from aeromap.cfd.region_mapping import RegionMappingError, map_wall_regions_analytically_to_vtp
from aeromap.cfd.schema import CfdConfig
from aeromap.constants import REF
from aeromap.geometry.regions import REGION_NAMES
from aeromap.io import atomic_write_json
from aeromap.parameters import AeroParams

FloatArray = NDArray[np.float64]
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
FORCE_WINDOW_FRACTION = 0.20
FORCE_STABILITY_LIMIT = 0.005
MASS_IMBALANCE_LIMIT = 0.005
INDEPENDENT_FORCE_LIMIT = 0.01
FORCE_COEFF_COLUMN_COUNT = 6
FORCE_COLUMN_COUNT = 7
FORCE_MOMENT_COLUMN_COUNT = 13
FORCE_POROUS_MOMENT_COLUMN_COUNT = 19
FLOW_COLUMN_COUNT = 2
YPLUS_COLUMN_COUNT = 5
MIN_DETREND_SAMPLES = 2
MIN_AUTOCORRELATION_SAMPLES = 3
MIN_SPECTRUM_SAMPLES = 4
MIN_COP_FORCE_N = 1.0e-12
YPLUS_BANDS = (
    ("0_5", 0.0, 5.0),
    ("5_30", 5.0, 30.0),
    ("30_100", 30.0, 100.0),
    ("100_200", 100.0, 200.0),
)
YPLUS_GT_200 = 200.0
OPENFOAM_TIME_RE = re.compile(r"^Time = (?P<time>\S+)")
RESIDUAL_RE = re.compile(
    r"Solving for (?P<field>[A-Za-z0-9_]+), Initial residual = (?P<initial>"
    + FLOAT_RE.pattern
    + r"), Final residual = (?P<final>"
    + FLOAT_RE.pattern
    + r"), No Iterations (?P<iterations>\d+)",
)
CONTINUITY_RE = re.compile(
    r"time step continuity errors : sum local = (?P<local>"
    + FLOAT_RE.pattern
    + r"), global = (?P<global>"
    + FLOAT_RE.pattern
    + r"), cumulative = (?P<cumulative>"
    + FLOAT_RE.pattern
    + r")",
)
EXECUTION_RE = re.compile(
    r"ExecutionTime = (?P<execution>"
    + FLOAT_RE.pattern
    + r") s\s+ClockTime = (?P<clock>"
    + FLOAT_RE.pattern
    + r") s",
)
PATCH_TYPE_RE = re.compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")


@dataclass(frozen=True)
class PostprocessArtifacts:
    mesh_json: Path
    convergence_json: Path
    steady_diagnostics_json: Path
    residuals_json: Path
    wall_conditions_json: Path
    layers_json: Path
    yplus_json: Path
    force_integration_json: Path
    status_json: Path
    scalars_parquet: Path
    volume_vtu: Path
    wall_vtp: Path
    mapped_wall_vtp: Path | None


def _numbers(line: str) -> list[float]:
    return [float(value) for value in FLOAT_RE.findall(line)]


def _data_lines(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and not line.startswith("#"):
            values = _numbers(line)
            if values:
                rows.append(values)
    return rows


def _coefficient_rows(path: Path) -> list[dict[str, float]]:
    return [
        {
            "time": values[0],
            "c_m_pitch": values[1],
            "c_d": values[2],
            "c_df": values[3],
            "c_df_front": values[4],
            "c_df_rear": values[5],
        }
        for values in _data_lines(path)
        if len(values) >= FORCE_COEFF_COLUMN_COUNT
    ]


def _force_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values in _data_lines(path):
        if len(values) >= FORCE_COLUMN_COUNT:
            pressure = np.asarray(values[1:4], dtype=np.float64)
            viscous = np.asarray(values[4:7], dtype=np.float64)
            porous_force = (
                np.asarray(values[7:10], dtype=np.float64)
                if len(values) >= FORCE_POROUS_MOMENT_COLUMN_COUNT
                else np.zeros(3, dtype=np.float64)
            )
            total = pressure + viscous + porous_force
            moments: dict[str, Any] = {}
            if len(values) >= FORCE_POROUS_MOMENT_COLUMN_COUNT:
                pressure_moment = np.asarray(values[10:13], dtype=np.float64)
                viscous_moment = np.asarray(values[13:16], dtype=np.float64)
                porous_moment = np.asarray(values[16:19], dtype=np.float64)
                moments = {
                    "pressure_moment_nm": pressure_moment.tolist(),
                    "viscous_moment_nm": viscous_moment.tolist(),
                    "porous_force_n": porous_force.tolist(),
                    "porous_moment_nm": porous_moment.tolist(),
                    "total_moment_nm": (pressure_moment + viscous_moment + porous_moment).tolist(),
                    "force_row_layout": "pressure_viscous_porous_force_and_moment",
                }
            elif len(values) >= FORCE_MOMENT_COLUMN_COUNT:
                pressure_moment = np.asarray(values[7:10], dtype=np.float64)
                viscous_moment = np.asarray(values[10:13], dtype=np.float64)
                moments = {
                    "pressure_moment_nm": pressure_moment.tolist(),
                    "viscous_moment_nm": viscous_moment.tolist(),
                    "total_moment_nm": (pressure_moment + viscous_moment).tolist(),
                    "force_row_layout": "pressure_viscous_force_and_moment",
                }
            rows.append(
                {
                    "time": values[0],
                    "pressure_n": pressure.tolist(),
                    "viscous_n": viscous.tolist(),
                    "total_n": total.tolist(),
                    **moments,
                },
            )
    return rows


def _flow_rows(path: Path) -> list[dict[str, float]]:
    return [
        {"time": row[0], "flow_m3_s": row[1]}
        for row in _data_lines(path)
        if len(row) >= FLOW_COLUMN_COUNT
    ]


def _yplus_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= YPLUS_COLUMN_COUNT:
            rows.append(
                {
                    "time": float(parts[0]),
                    "patch": parts[1],
                    "min": float(parts[2]),
                    "max": float(parts[3]),
                    "average": float(parts[4]),
                },
            )
    return rows


def _patch_block(text: str, patch: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != patch:
            continue
        for open_index in range(index + 1, len(lines)):
            if lines[open_index].strip() == "{":
                depth = 0
                block: list[str] = []
                for block_line in lines[open_index:]:
                    depth += block_line.count("{")
                    depth -= block_line.count("}")
                    block.append(block_line)
                    if depth == 0:
                        return "\n".join(block)
                break
    msg = f"patch {patch!r} not found in OpenFOAM field"
    raise ValueError(msg)


def _patch_type(path: Path, patch: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    inline_match = re.search(
        rf"^\s*{re.escape(patch)}\s*\{{(?P<body>.*?)\}}\s*$",
        text,
        flags=re.MULTILINE,
    )
    block = inline_match.group("body") if inline_match else _patch_block(text, patch)
    match = PATCH_TYPE_RE.search(block)
    if not match:
        msg = f"patch {patch!r} has no boundary type in {path}"
        raise ValueError(msg)
    return match.group(1)


def _wall_condition_report(case_dir: Path, config: CfdConfig) -> dict[str, Any]:
    patches = ("ground", *article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode))
    expected_types = {
        "nut": "nutkWallFunction",
        "k": "kqRWallFunction",
        "omega": "omegaWallFunction",
    }
    fields: dict[str, Any] = {}
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []
    for field, expected_type in expected_types.items():
        path = case_dir / "openfoam" / "0" / field
        field_report: dict[str, Any] = {
            "source_path": str(path),
            "expected_article_and_ground_type": expected_type,
            "patch_types": {},
        }
        if not path.exists():
            missing.append(str(path))
            field_report["status"] = "MISSING"
            fields[field] = field_report
            continue
        patch_types: dict[str, str] = {}
        for patch in patches:
            try:
                observed_type = _patch_type(path, patch)
            except ValueError as exc:
                observed_type = "MISSING"
                patch_types[patch] = observed_type
                mismatches.append(
                    {
                        "field": field,
                        "patch": patch,
                        "expected": expected_type,
                        "observed": observed_type,
                        "reason": str(exc),
                    },
                )
                continue
            patch_types[patch] = observed_type
            if observed_type != expected_type:
                mismatches.append(
                    {
                        "field": field,
                        "patch": patch,
                        "expected": expected_type,
                        "observed": observed_type,
                    },
                )
        field_report["patch_types"] = patch_types
        field_report["status"] = (
            "OK" if not any(item["field"] == field for item in mismatches) else "MISMATCH"
        )
        fields[field] = field_report

    status = "OK" if not missing and not mismatches else "FAILED"
    return {
        "status": status,
        "route": "high_re_wall_function",
        "note": (
            "AeroCliff records the exact Foundation OpenFOAM wall-function patch types. "
            "This verifies configuration only; achieved y+ and wall-shear reliability "
            "are judged separately from layer/yPlus evidence."
        ),
        "patches_checked": list(patches),
        "expected_types": expected_types,
        "fields": fields,
        "missing_files": missing,
        "mismatches": mismatches,
    }


def _boundary_scalar_values(path: Path, patch: str) -> FloatArray:
    block = _patch_block(path.read_text(encoding="utf-8", errors="replace"), patch)
    uniform_match = re.search(r"value\s+uniform\s+([^;]+);", block)
    if uniform_match:
        return np.asarray([float(uniform_match.group(1))], dtype=np.float64)
    nonuniform_match = re.search(
        r"value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        block,
        flags=re.DOTALL,
    )
    if not nonuniform_match:
        msg = f"patch {patch!r} has no scalar value list in {path}"
        raise ValueError(msg)
    expected_count = int(nonuniform_match.group(1))
    values = np.asarray(_numbers(nonuniform_match.group(2)), dtype=np.float64)
    if len(values) != expected_count:
        msg = f"expected {expected_count} scalar values for patch {patch!r}, got {len(values)}"
        raise ValueError(msg)
    return values


def _boundary_scalar_values_for_patches(path: Path, patches: tuple[str, ...]) -> FloatArray:
    values = [_boundary_scalar_values(path, patch) for patch in patches]
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _boundary_scalar_values_for_wall(
    path: Path,
    patches: tuple[str, ...],
    wall: pv.PolyData,
) -> FloatArray:
    if "patchID" not in wall.cell_data:
        return _boundary_scalar_values_for_patches(path, patches)
    patch_ids = np.asarray(wall.cell_data["patchID"], dtype=np.int64).reshape(-1)
    unique_patch_ids = tuple(sorted({int(value) for value in patch_ids}))
    if len(unique_patch_ids) != len(patches):
        return _boundary_scalar_values_for_patches(path, patches)

    values = np.zeros(wall.n_cells, dtype=np.float64)
    for patch, patch_id in zip(patches, unique_patch_ids, strict=True):
        patch_values = _boundary_scalar_values(path, patch)
        patch_mask = patch_ids == patch_id
        patch_cell_count = int(np.count_nonzero(patch_mask))
        if len(patch_values) == 1:
            values[patch_mask] = patch_values[0]
        elif len(patch_values) == patch_cell_count:
            values[patch_mask] = patch_values
        else:
            msg = (
                f"patch {patch!r} scalar value count does not match VTP patch cells: "
                f"{len(patch_values)} vs {patch_cell_count}"
            )
            raise ValueError(msg)
    return values


def _boundary_patch_face_counts(case_dir: Path, patches: tuple[str, ...]) -> dict[str, int] | None:
    boundary_path = case_dir / "openfoam" / "constant" / "polyMesh" / "boundary"
    if not boundary_path.exists():
        return None
    text = boundary_path.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for patch in patches:
        try:
            block = _patch_block(text, patch)
        except ValueError:
            continue
        match = re.search(r"\bnFaces\s+(\d+)\s*;", block)
        if match:
            counts[patch] = int(match.group(1))
    return counts


def _wall_patch_export_report_path(case_dir: Path, latest_time: str) -> Path:
    return case_dir / "quality" / f"wall_patch_export_{latest_time}.json"


def _wall_patches_for_exported_wall(
    case_dir: Path,
    latest_time: str,
    config: CfdConfig,
) -> tuple[str, ...]:
    report_path = _wall_patch_export_report_path(case_dir, latest_time)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        patches = report.get("present_patches")
        if isinstance(patches, list) and all(isinstance(item, str) for item in patches):
            return tuple(patches)
    return article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)


def _window_stats(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    if not rows:
        return {"mean": float("nan"), "cv": float("nan"), "drift": float("nan")}
    count = max(2, int(len(rows) * FORCE_WINDOW_FRACTION))
    values = np.asarray([row[key] for row in rows[-count:]], dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "cv": abs(std / mean) if mean else float("nan"),
        "drift": abs(float(values[-1] - values[0]) / mean) if mean else float("nan"),
        "first": float(values[0]),
        "last": float(values[-1]),
        "count": float(count),
    }


def _series_stats(values: FloatArray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "drift": float("nan"),
            "first": float("nan"),
            "last": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = float(np.mean(values))
    return {
        "count": float(len(values)),
        "mean": mean,
        "std": float(np.std(values)),
        "cv": abs(float(np.std(values)) / mean) if mean else float("nan"),
        "drift": abs(float(values[-1] - values[0]) / mean) if mean else float("nan"),
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _detrended(values: FloatArray) -> FloatArray:
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    if len(values) < MIN_DETREND_SAMPLES:
        return np.asarray(values - np.mean(values), dtype=np.float64)
    x = np.arange(len(values), dtype=np.float64)
    coefficients = np.asarray(np.polyfit(x, values, deg=1), dtype=np.float64)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    return np.asarray(values - (slope * x + intercept), dtype=np.float64)


def _autocorrelation_summary(values: FloatArray) -> dict[str, Any]:
    if len(values) < MIN_AUTOCORRELATION_SAMPLES:
        return {"status": "SKIPPED", "reason": "fewer than three samples"}
    signal = _detrended(values)
    signal = signal - np.mean(signal)
    denominator = float(np.dot(signal, signal))
    if denominator <= 0.0:
        return {"status": "SKIPPED", "reason": "zero detrended variance"}
    max_lag = min(50, len(values) - 1)
    lags = np.arange(1, max_lag + 1, dtype=np.int64)
    correlations = np.asarray(
        [float(np.dot(signal[:-lag], signal[lag:]) / denominator) for lag in lags],
        dtype=np.float64,
    )
    strongest_index = int(np.argmax(np.abs(correlations)))
    return {
        "status": "OK",
        "max_lag": int(max_lag),
        "strongest_lag": int(lags[strongest_index]),
        "strongest_correlation": float(correlations[strongest_index]),
        "lag_1": float(correlations[0]),
        "lags": lags.tolist(),
        "correlations": correlations.tolist(),
    }


def _spectrum_summary(values: FloatArray) -> dict[str, Any]:
    if len(values) < MIN_SPECTRUM_SAMPLES:
        return {"status": "SKIPPED", "reason": "fewer than four samples"}
    signal = _detrended(values)
    signal = signal - np.mean(signal)
    amplitudes = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0)
    if len(amplitudes) <= 1 or float(np.max(amplitudes[1:])) <= 0.0:
        return {"status": "SKIPPED", "reason": "no non-zero spectral amplitude"}
    index = int(np.argmax(amplitudes[1:]) + 1)
    frequency = float(frequencies[index])
    return {
        "status": "OK",
        "dominant_frequency_cycles_per_iteration": frequency,
        "dominant_period_iterations": float(1.0 / frequency) if frequency else float("inf"),
        "dominant_amplitude": float(amplitudes[index]),
        "note": "Steady SIMPLE iteration number is not physical time.",
    }


def _vector_window_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
    if values.size == 0:
        return {"status": "SKIPPED", "reason": f"{key} unavailable"}
    return {
        "status": "OK",
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "first": values[0].tolist(),
        "last": values[-1].tolist(),
    }


def _streamwise_cop_rows(force_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in force_rows:
        if "total_moment_nm" not in row:
            continue
        force = np.asarray(row["total_n"], dtype=np.float64)
        moment = np.asarray(row["total_moment_nm"], dtype=np.float64)
        fz = float(force[2])
        if abs(fz) < MIN_COP_FORCE_N:
            continue
        rows.append(
            {
                "time": float(row["time"]),
                "x_cp_m": float(REF.l_ref_m / 2.0 - moment[1] / fz),
            },
        )
    return rows


def _steady_diagnostics(
    *,
    coeff_rows: list[dict[str, float]],
    force_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    window_count = max(2, int(len(coeff_rows) * FORCE_WINDOW_FRACTION)) if coeff_rows else 0
    window_coeff_rows = coeff_rows[-window_count:] if window_count else []
    window_force_rows = force_rows[-window_count:] if window_count else []
    coefficient_series: dict[str, Any] = {}
    for key in ("c_d", "c_df", "c_m_pitch", "c_df_front", "c_df_rear"):
        full_values = np.asarray([row[key] for row in coeff_rows], dtype=np.float64)
        window_values = np.asarray([row[key] for row in window_coeff_rows], dtype=np.float64)
        coefficient_series[key] = {
            "full": _series_stats(full_values),
            "final_window": _series_stats(window_values),
            "autocorrelation": _autocorrelation_summary(window_values),
            "spectrum": _spectrum_summary(window_values),
        }

    cop_rows = _streamwise_cop_rows(force_rows)
    cop_values = np.asarray([row["x_cp_m"] for row in cop_rows[-window_count:]], dtype=np.float64)
    return {
        "schema_version": "steady_diagnostics_v0.1.0",
        "steady_iteration_note": (
            "OpenFOAM steady SIMPLE iteration/time index is not physical time; "
            "autocorrelation and spectrum are iteration-sequence diagnostics only."
        ),
        "sample_count": len(coeff_rows),
        "final_window_count": window_count,
        "coefficients": coefficient_series,
        "forces_final_window_n": {
            "pressure": _vector_window_stats(window_force_rows, "pressure_n"),
            "viscous": _vector_window_stats(window_force_rows, "viscous_n"),
            "total": _vector_window_stats(window_force_rows, "total_n"),
        },
        "moments_final_window_nm": {
            "pressure": _vector_window_stats(window_force_rows, "pressure_moment_nm"),
            "viscous": _vector_window_stats(window_force_rows, "viscous_moment_nm"),
            "total": _vector_window_stats(window_force_rows, "total_moment_nm"),
        },
        "streamwise_center_of_pressure": {
            "definition": (
                "x_cp = L_ref/2 - M_y/F_z, assuming the effective force acts at the "
                "reference z plane; use as a trend diagnostic, not a full load-path proof."
            ),
            "final_window": _series_stats(cop_values),
            "rows": cop_rows,
        },
        "unavailable": [
            (
                "left/right tunnel force decomposition requires patch/group-specific "
                "force function objects"
            ),
            "streamwise field bins over time require additional sampled/exported wall fields",
        ],
    }


def _openfoam_time_value(raw: str) -> float:
    return float(raw.strip().removesuffix("s"))


def _parse_solver_log(path: Path) -> dict[str, Any]:
    current_time = float("nan")
    residuals: list[dict[str, Any]] = []
    continuity: list[dict[str, float]] = []
    execution: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if time_match := OPENFOAM_TIME_RE.search(line):
            current_time = _openfoam_time_value(time_match.group("time"))
            continue
        if residual_match := RESIDUAL_RE.search(line):
            residuals.append(
                {
                    "time": current_time,
                    "field": residual_match.group("field"),
                    "initial": float(residual_match.group("initial")),
                    "final": float(residual_match.group("final")),
                    "iterations": int(residual_match.group("iterations")),
                },
            )
            continue
        if continuity_match := CONTINUITY_RE.search(line):
            continuity.append(
                {
                    "time": current_time,
                    "sum_local": float(continuity_match.group("local")),
                    "global": float(continuity_match.group("global")),
                    "cumulative": float(continuity_match.group("cumulative")),
                },
            )
            continue
        if execution_match := EXECUTION_RE.search(line):
            execution.append(
                {
                    "time": current_time,
                    "execution_time_s": float(execution_match.group("execution")),
                    "clock_time_s": float(execution_match.group("clock")),
                },
            )
    return {
        "source_path": str(path),
        "residuals": residuals,
        "continuity": continuity,
        "execution": execution,
    }


def _residual_field_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for field in sorted({str(record["field"]) for record in records}):
        field_records = [record for record in records if record["field"] == field]
        count = max(2, int(len(field_records) * FORCE_WINDOW_FRACTION)) if field_records else 0
        window = field_records[-count:] if count else []
        summaries[field] = {
            "sample_count": len(field_records),
            "final_window_count": count,
            "initial": {
                "full": _series_stats(
                    np.asarray([record["initial"] for record in field_records], dtype=np.float64),
                ),
                "final_window": _series_stats(
                    np.asarray([record["initial"] for record in window], dtype=np.float64),
                ),
            },
            "final": {
                "full": _series_stats(
                    np.asarray([record["final"] for record in field_records], dtype=np.float64),
                ),
                "final_window": _series_stats(
                    np.asarray([record["final"] for record in window], dtype=np.float64),
                ),
            },
            "iterations": {
                "full": _series_stats(
                    np.asarray(
                        [record["iterations"] for record in field_records],
                        dtype=np.float64,
                    ),
                ),
                "final_window": _series_stats(
                    np.asarray([record["iterations"] for record in window], dtype=np.float64),
                ),
            },
        }
    return summaries


def _vector_component_max_summary(
    records: list[dict[str, Any]],
    *,
    prefix: str,
    value_key: str,
) -> dict[str, Any]:
    grouped: dict[float, list[float]] = {}
    for record in records:
        field = str(record["field"])
        if not field.startswith(prefix):
            continue
        grouped.setdefault(float(record["time"]), []).append(float(record[value_key]))
    values = np.asarray([max(items) for _time, items in sorted(grouped.items())], dtype=np.float64)
    count = max(2, int(len(values) * FORCE_WINDOW_FRACTION)) if len(values) else 0
    return {
        "sample_count": len(values),
        "full": _series_stats(values),
        "final_window": _series_stats(
            values[-count:] if count else np.asarray([], dtype=np.float64)
        ),
        "definition": f"maximum {value_key} residual across {prefix} components at each iteration",
    }


def _solver_log_paths(case_dir: Path) -> list[Path]:
    log_dir = case_dir / "logs"
    paths = sorted(log_dir.glob("solver*.log"))
    if not paths:
        paths = sorted(log_dir.glob("foamRun*.log"))
    return paths


def _residual_report(case_dir: Path) -> dict[str, Any]:
    paths = _solver_log_paths(case_dir)
    if not paths:
        return {
            "status": "SKIPPED",
            "reason": "no solver or foamRun log found",
            "source_paths": [],
        }
    parsed_logs = [_parse_solver_log(path) for path in paths]
    residuals = [record for parsed in parsed_logs for record in parsed["residuals"]]
    continuity = [record for parsed in parsed_logs for record in parsed["continuity"]]
    execution = [record for parsed in parsed_logs for record in parsed["execution"]]
    return {
        "status": "OK" if residuals else "SKIPPED",
        "source_paths": [str(path) for path in paths],
        "steady_iteration_note": (
            "Residual rows are indexed by OpenFOAM steady SIMPLE iteration/time label; "
            "they are convergence diagnostics, not physical-time samples."
        ),
        "residual_sample_count": len(residuals),
        "per_field": _residual_field_summary(residuals),
        "grouped_fields": {
            "U": {
                "initial": _vector_component_max_summary(
                    residuals, prefix="U", value_key="initial"
                ),
                "final": _vector_component_max_summary(residuals, prefix="U", value_key="final"),
            },
        },
        "continuity": {
            "sample_count": len(continuity),
            "sum_local": _series_stats(
                np.asarray([record["sum_local"] for record in continuity], dtype=np.float64),
            ),
            "global": _series_stats(
                np.asarray([record["global"] for record in continuity], dtype=np.float64),
            ),
            "cumulative": _series_stats(
                np.asarray([record["cumulative"] for record in continuity], dtype=np.float64),
            ),
        },
        "execution": {
            "sample_count": len(execution),
            "last": execution[-1] if execution else None,
        },
        "logs": parsed_logs,
    }


def _bounds_payload(bounds: Sequence[float]) -> dict[str, list[float]]:
    return {
        "min_m": [float(bounds[0]), float(bounds[2]), float(bounds[4])],
        "max_m": [float(bounds[1]), float(bounds[3]), float(bounds[5])],
    }


def _nearest_wall_field_context(
    problem: pv.DataSet, mapped_wall_vtp: Path | None
) -> dict[str, Any]:
    if mapped_wall_vtp is None or not mapped_wall_vtp.exists() or problem.n_cells == 0:
        return {"status": "SKIPPED", "reason": "mapped wall VTP or problem cells unavailable"}
    wall_loaded = pv.read(mapped_wall_vtp)
    wall = wall_loaded if isinstance(wall_loaded, pv.PolyData) else wall_loaded.extract_surface()
    wall_centers = np.asarray(wall.cell_centers().points, dtype=np.float64)
    problem_centers = np.asarray(problem.cell_centers().points, dtype=np.float64)
    if len(wall_centers) == 0 or len(problem_centers) == 0:
        return {"status": "SKIPPED", "reason": "empty wall or problem-set centers"}
    nearest_indices: list[int] = []
    nearest_distances: list[float] = []
    for center in problem_centers:
        distances = np.linalg.norm(wall_centers - center, axis=1)
        index = int(np.argmin(distances))
        nearest_indices.append(index)
        nearest_distances.append(float(distances[index]))

    index_array = np.asarray(nearest_indices, dtype=np.int64)
    region_values = (
        np.asarray(wall.cell_data["surface_region"], dtype=object)[index_array]
        if "surface_region" in wall.cell_data
        else np.asarray(["unknown"] * len(index_array), dtype=object)
    )
    per_region = {
        str(region): int(np.count_nonzero(region_values == region))
        for region in sorted({str(value) for value in region_values})
    }
    context: dict[str, Any] = {
        "status": "OK",
        "mapped_wall_vtp": str(mapped_wall_vtp),
        "nearest_wall_distance_m": _finite_stats(np.asarray(nearest_distances, dtype=np.float64)),
        "nearest_wall_region_counts": per_region,
        "nearest_wall_indices": nearest_indices,
    }
    if "p" in wall.cell_data:
        context["nearest_wall_kinematic_pressure"] = _finite_stats(
            np.asarray(wall.cell_data["p"], dtype=np.float64).reshape(-1)[index_array],
        )
    if "yPlus" in wall.cell_data:
        context["nearest_wall_yplus"] = _finite_stats(
            np.asarray(wall.cell_data["yPlus"], dtype=np.float64).reshape(-1)[index_array],
        )
    if "wallShearStress" in wall.cell_data:
        shear = np.asarray(wall.cell_data["wallShearStress"], dtype=np.float64)[index_array]
        context["nearest_wall_shear_magnitude"] = _finite_stats(np.linalg.norm(shear, axis=1))
    return context


def _problem_sets_report(case_dir: Path, mapped_wall_vtp: Path | None) -> dict[str, Any]:
    root = case_dir / "openfoam" / "postProcessing" / "checkMesh" / "constant"
    reports: dict[str, Any] = {}
    for name in ("warpedFaces", "concaveFaces", "concaveCells"):
        path = root / f"{name}.vtk"
        if not path.exists():
            reports[name] = {"status": "MISSING", "source_path": str(path)}
            continue
        dataset = pv.read(path)
        item: dict[str, Any] = {
            "status": "OK",
            "source_path": str(path),
            "cell_count": int(dataset.n_cells),
            "point_count": int(dataset.n_points),
            "bounds_m": _bounds_payload(dataset.bounds),
        }
        if name == "warpedFaces":
            item["nearest_wall_field_context"] = _nearest_wall_field_context(
                dataset,
                mapped_wall_vtp,
            )
        reports[name] = item
    return {
        "status": "OK"
        if any(item.get("status") == "OK" for item in reports.values())
        else "MISSING",
        "source_directory": str(root),
        "sets": reports,
    }


def _latest_time_from_coefficients(rows: list[dict[str, float]]) -> str:
    if not rows:
        msg = "force coefficient output contains no rows"
        raise ValueError(msg)
    time = rows[-1]["time"]
    return str(int(time)) if float(time).is_integer() else str(time)


def _return_code(path: Path) -> int | str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _optional_check_mesh_log(
    path: Path,
    *,
    skipped: bool,
    missing_reason: str,
) -> dict[str, Any]:
    if path.exists():
        return {
            **parse_check_mesh_log(path),
            "status": "OK",
            "log_path": str(path),
        }
    return {
        "mesh_ok": False,
        "cells": None,
        "failed_mesh_checks": 0,
        "contains_negative_volume": False,
        "status": "SKIPPED" if skipped else "MISSING",
        "log_path": str(path),
        "log_missing": True,
        "reason": missing_reason,
    }


def _yplus_report(path: Path, latest_time: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "SKIPPED",
            "source_path": str(path),
            "rows": [],
            "final": [],
            "reason": "yPlus postProcess output unavailable",
        }
    rows = _yplus_rows(path)
    return {
        "status": "OK",
        "source_path": str(path),
        "rows": rows,
        "final": [row for row in rows if row["time"] == latest_time],
    }


def _acceptance_blockers(
    *,
    mesh_report: dict[str, Any],
    convergence: dict[str, Any],
    force_integration: dict[str, Any],
    mapping: dict[str, Any],
    config: CfdConfig,
) -> list[str]:
    blockers: list[str] = []
    if config.quality.mesh_quality_fatal:
        fatal_returncode = mesh_report.get("mesh_quality_returncode")
        if fatal_returncode is None:
            blockers.append("fatal mesh-quality return code missing")
        elif fatal_returncode != 0:
            blockers.append(f"fatal mesh-quality check returned {fatal_returncode}")
        fatal_check = mesh_report.get("mesh_quality_check") or mesh_report.get(
            "configured_fatal_check",
        )
        if isinstance(fatal_check, dict):
            if fatal_check.get("failed_mesh_checks", 0) > 0:
                blockers.append("fatal mesh-quality check reported failed mesh checks")
            if not fatal_check.get("mesh_ok", False):
                blockers.append("fatal mesh-quality check did not report Mesh OK")
            if fatal_check.get("contains_negative_volume", False):
                blockers.append("fatal mesh-quality check reported negative-volume cells")
    if mesh_report.get("contains_negative_volume", False):
        blockers.append("extended mesh diagnostics reported negative-volume cells")

    extended_returncode = mesh_report.get("extended_diagnostics_returncode")
    if config.quality.extended_diagnostics_required and extended_returncode is None:
        blockers.append("extended mesh diagnostic return code missing")
    if config.quality.extended_diagnostics_fatal and extended_returncode != 0:
        blockers.append(f"extended mesh diagnostics returned {extended_returncode}")

    if not convergence.get("force_stable", False):
        blockers.append("force coefficients are not stable in the final window")
    if not convergence.get("mass_balance_ok", False):
        blockers.append("mass-flow imbalance exceeds the acceptance limit")
    if mapping.get("status") == "SKIPPED":
        blockers.append("surface region mapping skipped")
    elif mapping.get("status") != "OK":
        blockers.append("surface region mapping failed")
    if force_integration.get("status") == "SKIPPED":
        blockers.append("independent force integration skipped")
    elif force_integration.get("status") == "FAILED":
        blockers.append("independent force integration failed")
    elif not force_integration.get("within_1pct", False):
        blockers.append("independent force integration exceeds 1%")
    if config.quality.case_class != "NON_CAMPAIGN_ENGINEERING_SMOKE":
        blockers.extend(
            [
                "campaign acceptance requires near-wall layer/y-plus evidence",
                "worst-clearance case not run",
            ],
        )
    return blockers


def _convert_vtk(case_dir: Path, latest_time: str, config: CfdConfig) -> tuple[Path, Path]:
    vtk_dir = case_dir / "openfoam" / "VTK"
    volume_vtk = vtk_dir / f"openfoam_{latest_time}.vtk"
    outputs = case_dir / "outputs"
    outputs.mkdir(exist_ok=True)
    volume_vtu = outputs / f"volume_{latest_time}.vtu"
    wall_vtp = outputs / f"article_wall_{latest_time}.vtp"
    configured_wall_patches = article_patch_names(
        patch_mode=config.surface_export.openfoam_patch_mode,
    )
    patch_face_counts = _boundary_patch_face_counts(case_dir, configured_wall_patches)
    wall_patches: list[str] = []
    omitted_patches: dict[str, Any] = {}
    missing_required: list[str] = []
    for patch in configured_wall_patches:
        wall_vtk = vtk_dir / patch / f"{patch}_{latest_time}.vtk"
        if wall_vtk.exists():
            wall_patches.append(patch)
            continue
        if patch_face_counts is not None and patch_face_counts.get(patch, 0) == 0:
            omitted_patches[patch] = {
                "reason": "absent_or_zero_face_openfoam_patch",
                "nFaces": patch_face_counts.get(patch, 0),
            }
            continue
        missing_required.append(str(wall_vtk))
    missing = [str(volume_vtk)] if not volume_vtk.exists() else []
    missing.extend(missing_required)
    if missing:
        message = "foamToVTK export missing required files: " + ", ".join(missing)
        raise FileNotFoundError(message)
    if not wall_patches:
        message = "foamToVTK export produced no article wall patch files"
        raise FileNotFoundError(message)
    pv.read(volume_vtk).save(volume_vtu)
    wall_polys: list[pv.PolyData] = []
    for patch in wall_patches:
        wall_vtk = vtk_dir / patch / f"{patch}_{latest_time}.vtk"
        loaded = pv.read(wall_vtk)
        wall_polys.append(loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface())
    wall = wall_polys[0] if len(wall_polys) == 1 else pv.MultiBlock(wall_polys).combine()
    wall_poly = (
        wall if isinstance(wall, pv.PolyData) else wall.extract_surface(algorithm="dataset_surface")
    )
    wall_poly.save(wall_vtp)
    atomic_write_json(
        _wall_patch_export_report_path(case_dir, latest_time),
        {
            "boundary_face_counts": patch_face_counts,
            "configured_patches": list(configured_wall_patches),
            "latest_time": latest_time,
            "omitted_patches": omitted_patches,
            "present_patches": wall_patches,
            "wall_vtp": str(wall_vtp),
        },
    )
    return volume_vtu, wall_vtp


def _map_regions(
    case_dir: Path, wall_vtp: Path, latest_time: str, config: CfdConfig
) -> tuple[Path | None, dict[str, Any]]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    params = AeroParams.model_validate(manifest["params"])
    source_mesh = trimesh.load_mesh(case_dir / "cfd_surface" / "article.stl", process=True)
    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh = trimesh.util.concatenate(tuple(source_mesh.geometry.values()))
    regions = json.loads((case_dir / "cfd_surface" / "surface_regions.json").read_text())
    output = case_dir / "outputs" / f"article_wall_regions_{latest_time}.vtp"
    report = case_dir / "quality" / "region_mapping.json"
    required_regions = REGION_NAMES
    if params.geometry_family == "stable_reference":
        required_regions = tuple(region for region in REGION_NAMES if region != "keel")
    try:
        result = map_wall_regions_analytically_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=pv.read(wall_vtp),
            params=params,
            output_vtp_path=output,
            report_path=report,
            max_distance_face_scale=config.quality.region_mapping_max_distance_face_scale,
            min_coverage=config.quality.region_mapping_min_coverage,
            required_regions=required_regions,
        )
    except RegionMappingError as exc:
        return output if output.exists() else None, {"status": "FAILED", "reason": str(exc)}
    return output, {"status": "OK", **result.as_dict()}


def _cell_area_vectors(poly: pv.PolyData) -> tuple[FloatArray, FloatArray]:
    faces = np.asarray(poly.faces, dtype=np.int64)
    points = np.asarray(poly.points, dtype=np.float64)
    vectors: list[FloatArray] = []
    cursor = 0
    while cursor < len(faces):
        vertex_count = int(faces[cursor])
        vertex_ids = faces[cursor + 1 : cursor + 1 + vertex_count]
        cursor += vertex_count + 1
        vertices = points[vertex_ids]
        area_vector = np.zeros(3, dtype=np.float64)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            area_vector += np.cross(start, end)
        vectors.append(0.5 * area_vector)
    area_vectors = np.asarray(vectors, dtype=np.float64)
    areas = np.linalg.norm(area_vectors, axis=1)
    return area_vectors, areas


def _integrate_exported_wall_forces(wall_vtp: Path) -> dict[str, Any]:
    dataset = pv.read(wall_vtp)
    if isinstance(dataset, pv.MultiBlock):
        msg = f"expected a wall PolyData export, got MultiBlock: {wall_vtp}"
        raise TypeError(msg)
    poly = dataset if isinstance(dataset, pv.PolyData) else dataset.extract_surface()
    area_vectors, areas = _cell_area_vectors(poly)
    pressure = np.asarray(poly.cell_data["p"], dtype=np.float64).reshape(-1)
    shear = np.asarray(poly.cell_data["wallShearStress"], dtype=np.float64)

    pressure_force = REF.rho_kg_m3 * np.sum(pressure[:, None] * area_vectors, axis=0)
    viscous_force = -REF.rho_kg_m3 * np.sum(shear * areas[:, None], axis=0)
    total_force = pressure_force + viscous_force
    return {
        "independent_pressure_n": pressure_force.tolist(),
        "independent_viscous_n": viscous_force.tolist(),
        "independent_total_n": total_force.tolist(),
        "pressure_note": "OpenFOAM incompressible pressure is kinematic; multiplied by rho.",
        "integration_note": (
            "Pressure uses exact exported polygon area vectors; viscous force uses "
            "polygon areas and kinematic wallShearStress. This step reads only "
            "exported wall fields, face areas, and face normals."
        ),
    }


def _component_relative_error(independent: FloatArray, reference: FloatArray) -> FloatArray:
    denominator = np.maximum(np.abs(reference), 1.0)
    return np.abs(independent - reference) / denominator


def _independent_force_integration(
    wall_vtp: Path, openfoam_force: dict[str, Any]
) -> dict[str, Any]:
    integrated = _integrate_exported_wall_forces(wall_vtp)
    independent_pressure = np.asarray(integrated["independent_pressure_n"], dtype=np.float64)
    independent_viscous = np.asarray(integrated["independent_viscous_n"], dtype=np.float64)
    independent_total = np.asarray(integrated["independent_total_n"], dtype=np.float64)
    reference_pressure = np.asarray(openfoam_force["pressure_n"], dtype=np.float64)
    reference_viscous = np.asarray(openfoam_force["viscous_n"], dtype=np.float64)
    reference = np.asarray(openfoam_force["total_n"], dtype=np.float64)
    pressure_relative_error = _component_relative_error(independent_pressure, reference_pressure)
    viscous_relative_error = _component_relative_error(independent_viscous, reference_viscous)
    relative_error = _component_relative_error(independent_total, reference)
    max_component_error = max(
        float(np.max(pressure_relative_error)),
        float(np.max(viscous_relative_error)),
        float(np.max(relative_error)),
    )
    return {
        **integrated,
        "openfoam_pressure_n": reference_pressure.tolist(),
        "openfoam_viscous_n": reference_viscous.tolist(),
        "openfoam_total_n": reference.tolist(),
        "pressure_relative_error": pressure_relative_error.tolist(),
        "viscous_relative_error": viscous_relative_error.tolist(),
        "relative_error": relative_error.tolist(),
        "max_relative_error": float(np.max(relative_error)),
        "max_component_relative_error": max_component_error,
        "within_1pct": bool(max_component_error <= INDEPENDENT_FORCE_LIMIT),
        "comparison_note": (
            "OpenFOAM force history values are used only as the reference comparison; "
            "they are not consumed by the exported-field integration path."
        ),
    }


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0 or len(values) == 0:
        return float("nan")
    return float(np.sum(values * weights) / total_weight)


def _weighted_fraction(mask: NDArray[np.bool_], weights: FloatArray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.0
    return float(np.sum(weights[mask]) / total_weight)


def _finite_stats(values: FloatArray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "mean": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _yplus_band_fractions(values: FloatArray, weights: FloatArray) -> dict[str, float]:
    fractions = {
        name: _weighted_fraction((values >= lower) & (values < upper), weights)
        for name, lower, upper in YPLUS_BANDS
    }
    fractions["gt_200"] = _weighted_fraction(values >= YPLUS_GT_200, weights)
    return fractions


def _layer_and_wall_metrics(
    case_dir: Path,
    mapped_wall_vtp: Path | None,
    config: CfdConfig | None = None,
    latest_time: str | None = None,
) -> dict[str, Any]:
    if mapped_wall_vtp is None or not mapped_wall_vtp.exists():
        return {"status": "SKIPPED", "reason": "mapped wall VTP unavailable"}
    field_paths = {
        "nSurfaceLayers": case_dir / "openfoam" / "0" / "nSurfaceLayers",
        "thickness": case_dir / "openfoam" / "0" / "thickness",
        "thicknessFraction": case_dir / "openfoam" / "0" / "thicknessFraction",
    }
    missing = [name for name, path in field_paths.items() if not path.exists()]
    if missing:
        return {
            "status": "SKIPPED",
            "reason": (
                "layer fields unavailable; no-layer smoke meshes are permitted only as "
                "non-campaign evidence"
            ),
            "missing_fields": missing,
        }
    config = config or CfdConfig()
    wall_patches = (
        _wall_patches_for_exported_wall(case_dir, latest_time, config)
        if latest_time is not None
        else article_patch_names(patch_mode=config.surface_export.openfoam_patch_mode)
    )
    loaded_wall = pv.read(mapped_wall_vtp)
    if isinstance(loaded_wall, pv.MultiBlock):
        return {
            "status": "SKIPPED",
            "reason": "mapped wall VTP is a MultiBlock dataset, expected a wall PolyData",
            "mapped_wall_vtp": str(mapped_wall_vtp),
        }
    wall = loaded_wall if isinstance(loaded_wall, pv.PolyData) else loaded_wall.extract_surface()
    missing_cell_data = [
        name
        for name in ["local_face_area_m2", "surface_region", "yPlus"]
        if name not in wall.cell_data
    ]
    if missing_cell_data:
        return {
            "status": "SKIPPED",
            "reason": "mapped wall VTP lacks required wall metric fields",
            "missing_cell_data": missing_cell_data,
            "mapped_wall_vtp": str(mapped_wall_vtp),
        }
    layers = _boundary_scalar_values_for_wall(field_paths["nSurfaceLayers"], wall_patches, wall)
    thickness = _boundary_scalar_values_for_wall(field_paths["thickness"], wall_patches, wall)
    thickness_fraction = _boundary_scalar_values_for_wall(
        field_paths["thicknessFraction"],
        wall_patches,
        wall,
    )
    if not (len(layers) == len(thickness) == len(thickness_fraction) == wall.n_cells):
        msg = (
            "article layer field lengths do not match mapped wall cells: "
            f"{len(layers)}, {len(thickness)}, {len(thickness_fraction)}, {wall.n_cells}"
        )
        raise ValueError(msg)

    areas = np.asarray(wall.cell_data["local_face_area_m2"], dtype=np.float64)
    regions = np.asarray(wall.cell_data["surface_region"], dtype=object)
    yplus = np.asarray(wall.cell_data["yPlus"], dtype=np.float64)
    layer_mask = layers > 0.0
    per_region: dict[str, Any] = {}
    for region in sorted({str(value) for value in regions}):
        region_mask = regions == region
        region_layers = layers[region_mask]
        region_thickness = thickness[region_mask]
        region_thickness_fraction = thickness_fraction[region_mask]
        region_yplus = yplus[region_mask]
        region_areas = areas[region_mask]
        region_layer_mask = region_layers > 0.0
        per_region[region] = {
            "face_count": int(np.count_nonzero(region_mask)),
            "area_m2": float(np.sum(region_areas)),
            "faces_with_layers": int(np.count_nonzero(region_layer_mask)),
            "face_fraction_with_layers": float(np.mean(region_layer_mask))
            if len(region_layer_mask)
            else 0.0,
            "area_fraction_with_layers": _weighted_fraction(region_layer_mask, region_areas),
            "mean_layers": _weighted_mean(region_layers, region_areas),
            "max_layers": float(np.max(region_layers)) if len(region_layers) else 0.0,
            "mean_thickness_m": _weighted_mean(region_thickness, region_areas),
            "mean_thickness_fraction": _weighted_mean(region_thickness_fraction, region_areas),
            "yplus": _finite_stats(region_yplus),
            "yplus_area_fraction_bands": _yplus_band_fractions(region_yplus, region_areas),
        }

    return {
        "status": "OK",
        "source_fields": {
            "nSurfaceLayers": str(case_dir / "openfoam" / "0" / "nSurfaceLayers"),
            "thickness": str(case_dir / "openfoam" / "0" / "thickness"),
            "thicknessFraction": str(case_dir / "openfoam" / "0" / "thicknessFraction"),
            "mapped_wall_vtp": str(mapped_wall_vtp),
        },
        "face_count": int(wall.n_cells),
        "surface_area_m2": float(np.sum(areas)),
        "faces_with_layers": int(np.count_nonzero(layer_mask)),
        "face_fraction_with_layers": float(np.mean(layer_mask)) if len(layer_mask) else 0.0,
        "area_fraction_with_layers": _weighted_fraction(layer_mask, areas),
        "mean_layers": _weighted_mean(layers, areas),
        "max_layers": float(np.max(layers)) if len(layers) else 0.0,
        "mean_thickness_m": _weighted_mean(thickness, areas),
        "mean_thickness_fraction": _weighted_mean(thickness_fraction, areas),
        "yplus": _finite_stats(yplus),
        "yplus_area_fraction_bands": _yplus_band_fractions(yplus, areas),
        "per_region": per_region,
    }


def _case_status_name(
    *,
    config: CfdConfig,
    convergence: dict[str, Any],
    blockers: list[str],
) -> str:
    if config.quality.case_class == "NON_CAMPAIGN_ENGINEERING_SMOKE":
        return "NON_CAMPAIGN_ENGINEERING_SMOKE"
    if blockers and not convergence.get("force_stable", False):
        return "PROVISIONAL_LIMIT_CYCLE_CANDIDATE"
    return "PROVISIONAL_CAMPAIGN_SOLVE"


def _target_eligibility(
    *,
    accepted: bool,
    training_eligible: bool,
    config: CfdConfig,
    convergence: dict[str, Any],
    force_integration: dict[str, Any],
    mapping: dict[str, Any],
    layers: dict[str, Any],
) -> dict[str, Any]:
    pressure_volume_ready = (
        accepted
        and training_eligible
        and config.quality.case_class == "CAMPAIGN_REFERENCE_CFD"
        and convergence.get("force_stable", False)
        and convergence.get("mass_balance_ok", False)
        and force_integration.get("within_1pct", False)
        and mapping.get("status") == "OK"
    )
    wall_ready = False
    wall_reason = (
        "regional y-plus/layer evidence is not accepted for wall-shear or separation use"
        if layers.get("status") == "OK"
        else "regional y-plus/layer metrics are unavailable"
    )
    return {
        "surface_pressure": pressure_volume_ready,
        "integrated_drag": pressure_volume_ready,
        "integrated_downforce": pressure_volume_ready,
        "integrated_lateral_force": pressure_volume_ready,
        "pitch_moment": pressure_volume_ready,
        "volume_mean_fields": pressure_volume_ready,
        "wall_shear": wall_ready,
        "separation_metrics": wall_ready,
        "cliff_boundary": False,
        "unsteady_statistics": False,
        "policy": (
            "Pressure, scalar-force, pitch-moment and volume-mean targets require an "
            "accepted steady or repeatable URANS mean with target-specific stability, "
            "mass balance, mapping and independent force checks. Wall-shear, separation "
            "and cliff-boundary targets additionally require accepted regional y-plus, "
            "layer and mesh-sensitivity evidence."
        ),
        "reasons": {
            "pressure_drag_volume": []
            if pressure_volume_ready
            else [
                "pressure/drag mean evidence is not in the selected target registry",
            ],
            "downforce_lateral_pitch": []
            if pressure_volume_ready
            else [
                "downforce, lateral-force and pitch-moment targets require separate "
                "target-specific acceptance",
            ],
            "wall_shear_separation": []
            if wall_ready
            else [
                wall_reason,
            ],
            "cliff_boundary": [
                "cliff-boundary evidence is outside the selected target registry",
            ],
            "unsteady_statistics": [
                "repeatable transient mean/RMS/frequency evidence is outside this registry",
            ],
        },
    }


def _vtk_and_mapping_report(
    case_dir: Path,
    latest_time: str,
    config: CfdConfig,
) -> tuple[Path, Path, Path | None, dict[str, Any], dict[str, Any]]:
    outputs = case_dir / "outputs"
    volume_vtu = outputs / f"volume_{latest_time}.vtu"
    wall_vtp = outputs / f"article_wall_{latest_time}.vtp"
    try:
        volume_vtu, wall_vtp = _convert_vtk(case_dir, latest_time, config)
    except FileNotFoundError as exc:
        if volume_vtu.exists() and wall_vtp.exists():
            vtk_export = {
                "status": "REUSED_EXISTING",
                "reason": str(exc),
                "volume_vtu": str(volume_vtu),
                "wall_vtp": str(wall_vtp),
            }
            mapped_wall_vtp, mapping = _map_regions(case_dir, wall_vtp, latest_time, config)
            return volume_vtu, wall_vtp, mapped_wall_vtp, mapping, vtk_export
        vtk_export = {
            "status": "SKIPPED",
            "reason": str(exc),
            "volume_vtu": str(volume_vtu),
            "wall_vtp": str(wall_vtp),
        }
        mapping = {
            "status": "SKIPPED",
            "reason": "foamToVTK export unavailable",
            "foamToVTK": vtk_export,
        }
        return volume_vtu, wall_vtp, None, mapping, vtk_export

    vtk_export = {
        "status": "OK",
        "volume_vtu": str(volume_vtu),
        "wall_vtp": str(wall_vtp),
    }
    mapped_wall_vtp, mapping = _map_regions(case_dir, wall_vtp, latest_time, config)
    return volume_vtu, wall_vtp, mapped_wall_vtp, mapping, vtk_export


def _force_integration_report(wall_vtp: Path, openfoam_force: dict[str, Any]) -> dict[str, Any]:
    if not wall_vtp.exists():
        return {
            "status": "SKIPPED",
            "within_1pct": False,
            "reason": "wall VTP unavailable; foamToVTK may have failed or been skipped",
            "wall_vtp": str(wall_vtp),
        }
    try:
        return {
            "status": "OK",
            **_independent_force_integration(wall_vtp, openfoam_force),
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "status": "FAILED",
            "within_1pct": False,
            "reason": str(exc),
            "wall_vtp": str(wall_vtp),
        }


def postprocess_case(case_dir: Path) -> PostprocessArtifacts:  # noqa: PLR0915
    """Parse a completed OpenFOAM case and emit CFD post-processing artifacts."""

    quality = case_dir / "quality"
    outputs = case_dir / "outputs"
    quality.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    pp = case_dir / "openfoam" / "postProcessing"
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    config = CfdConfig.model_validate(manifest["cfd_config"])

    coeff_rows = _coefficient_rows(pp / "forceCoeffs" / "0" / "forceCoeffs.dat")
    force_rows = _force_rows(pp / "forces" / "0" / "forces.dat")
    if not force_rows:
        msg = "forces output contains no parseable pressure/viscous force rows"
        raise ValueError(msg)
    inlet_rows = _flow_rows(pp / "inletFlowRate" / "0" / "surfaceFieldValue.dat")
    outlet_rows = _flow_rows(pp / "outletFlowRate" / "0" / "surfaceFieldValue.dat")
    latest_time = _latest_time_from_coefficients(coeff_rows)
    latest_coeff_time = coeff_rows[-1]["time"]
    yplus = _yplus_report(pp / "yPlus" / "0" / "yPlus.dat", latest_coeff_time)
    volume_vtu, wall_vtp, mapped_wall_vtp, mapping, vtk_export = _vtk_and_mapping_report(
        case_dir,
        latest_time,
        config,
    )

    cd_stats = _window_stats(coeff_rows, "c_d")
    cdf_stats = _window_stats(coeff_rows, "c_df")
    flow_imbalance = abs(inlet_rows[-1]["flow_m3_s"] + outlet_rows[-1]["flow_m3_s"]) / max(
        abs(inlet_rows[-1]["flow_m3_s"]),
        abs(outlet_rows[-1]["flow_m3_s"]),
    )
    force_integration = _force_integration_report(wall_vtp, force_rows[-1])
    layers = _layer_and_wall_metrics(case_dir, mapped_wall_vtp, config, latest_time)

    mesh_json = quality / "mesh.json"
    convergence_json = quality / "convergence.json"
    steady_diagnostics_json = quality / "steady_diagnostics.json"
    residuals_json = quality / "residuals.json"
    wall_conditions_json = quality / "wall_conditions.json"
    layers_json = quality / "layers.json"
    yplus_json = quality / "yplus.json"
    force_json = quality / "force_integration.json"
    status_json = quality / "status.json"
    scalars_parquet = outputs / "scalars.parquet"

    extended_returncode = _return_code(case_dir / "quality" / "checkMesh_extended.returncode")
    extended_report = _optional_check_mesh_log(
        case_dir / "logs" / "checkMesh.log",
        skipped=extended_returncode == "SKIPPED"
        or not config.quality.extended_diagnostics_required,
        missing_reason="extended checkMesh diagnostics were skipped or unavailable",
    )
    mesh_report = {
        **extended_report,
        "configured_fatal_check": parse_check_mesh_log(case_dir / "logs" / "checkMesh_fatal.log")
        if (case_dir / "logs" / "checkMesh_fatal.log").exists()
        else None,
        "mesh_quality_check": parse_check_mesh_log(case_dir / "logs" / "checkMesh_meshQuality.log")
        if (case_dir / "logs" / "checkMesh_meshQuality.log").exists()
        else None,
        "mesh_quality_fatal": config.quality.mesh_quality_fatal,
        "mesh_quality_returncode": _return_code(
            case_dir / "quality" / "checkMesh_meshQuality.returncode",
        ),
        "extended_diagnostics_required": config.quality.extended_diagnostics_required,
        "extended_diagnostics_fatal": config.quality.extended_diagnostics_fatal,
        "extended_diagnostics_returncode": extended_returncode,
        "extended_diagnostics_status": extended_report["status"],
        "extended_diagnostics_log_missing": extended_report.get("log_missing", False),
        "case_class": config.quality.case_class,
        "latest_time": latest_time,
        "foamToVTK": vtk_export,
        "problem_sets": _problem_sets_report(case_dir, mapped_wall_vtp),
        "layer_coverage_note": (
            "No-layer or sparse-layer smoke meshes are pipeline evidence only; "
            "campaign CFD requires separate near-wall/y-plus and sensitivity evidence."
        ),
    }
    convergence = {
        "latest_time": coeff_rows[-1]["time"],
        "force_window_fraction": FORCE_WINDOW_FRACTION,
        "c_d": cd_stats,
        "c_df": cdf_stats,
        "force_stable": cd_stats["cv"] <= FORCE_STABILITY_LIMIT
        and cdf_stats["cv"] <= FORCE_STABILITY_LIMIT,
        "inlet_flow_m3_s": inlet_rows[-1]["flow_m3_s"],
        "outlet_flow_m3_s": outlet_rows[-1]["flow_m3_s"],
        "relative_flow_imbalance": flow_imbalance,
        "mass_balance_ok": flow_imbalance <= MASS_IMBALANCE_LIMIT,
    }
    steady_diagnostics = _steady_diagnostics(coeff_rows=coeff_rows, force_rows=force_rows)
    residuals = _residual_report(case_dir)
    wall_conditions = _wall_condition_report(case_dir, config)
    blockers = _acceptance_blockers(
        mesh_report=mesh_report,
        convergence=convergence,
        force_integration=force_integration,
        mapping=mapping,
        config=config,
    )
    if yplus["status"] != "OK":
        blockers.append("yPlus post-processing skipped")
    if residuals["status"] != "OK":
        blockers.append("solver residual history unavailable")
    if wall_conditions["status"] != "OK":
        blockers.append("wall-condition audit failed")
    if vtk_export["status"] not in {"OK", "REUSED_EXISTING"}:
        blockers.append("foamToVTK export skipped")
    accepted = not blockers
    training_eligible = accepted and config.quality.case_class == "CAMPAIGN_REFERENCE_CFD"
    eligible_targets = _target_eligibility(
        accepted=accepted,
        training_eligible=training_eligible,
        config=config,
        convergence=convergence,
        force_integration=force_integration,
        mapping=mapping,
        layers=layers,
    )
    status_name = _case_status_name(
        config=config,
        convergence=convergence,
        blockers=blockers,
    )
    status = {
        "status": status_name,
        "accepted": accepted,
        "accepted_scope": config.quality.case_class if accepted else None,
        "case_class": config.quality.case_class,
        "training_eligible": training_eligible,
        "eligible_targets": eligible_targets,
        "never_use_for": [
            "training labels",
            "cliff identification",
            "headline aerodynamics",
            "accuracy claims",
        ]
        if config.quality.case_class == "NON_CAMPAIGN_ENGINEERING_SMOKE" or not training_eligible
        else [],
        "blockers": blockers,
        "artifacts": {
            "mesh": str(mesh_json),
            "convergence": str(convergence_json),
            "steady_diagnostics": str(steady_diagnostics_json),
            "residuals": str(residuals_json),
            "wall_conditions": str(wall_conditions_json),
            "layers": str(layers_json),
            "yplus": str(yplus_json),
            "force_integration": str(force_json),
            "scalars": str(scalars_parquet),
            "volume_vtu": str(volume_vtu),
            "wall_vtp": str(wall_vtp),
            "mapped_wall_vtp": str(mapped_wall_vtp) if mapped_wall_vtp else None,
        },
    }

    atomic_write_json(mesh_json, mesh_report)
    atomic_write_json(convergence_json, convergence)
    atomic_write_json(steady_diagnostics_json, steady_diagnostics)
    atomic_write_json(residuals_json, residuals)
    atomic_write_json(wall_conditions_json, wall_conditions)
    atomic_write_json(layers_json, layers)
    atomic_write_json(yplus_json, yplus)
    atomic_write_json(force_json, force_integration)
    atomic_write_json(status_json, status)
    pd.DataFrame(coeff_rows).to_parquet(scalars_parquet, index=False)
    return PostprocessArtifacts(
        mesh_json=mesh_json,
        convergence_json=convergence_json,
        steady_diagnostics_json=steady_diagnostics_json,
        residuals_json=residuals_json,
        wall_conditions_json=wall_conditions_json,
        layers_json=layers_json,
        yplus_json=yplus_json,
        force_integration_json=force_json,
        status_json=status_json,
        scalars_parquet=scalars_parquet,
        volume_vtu=volume_vtu,
        wall_vtp=wall_vtp,
        mapped_wall_vtp=mapped_wall_vtp,
    )

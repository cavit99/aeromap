"""Decision diagnostics for partial URANS checkpoints."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aeromap.constants import REF
from aeromap.io import atomic_write_json, sha256_file

SCHEMA_VERSION = "aerocliff_urans_checkpoint_decision_v0.1.0"
PARTIAL_URANS_CLASS = "PARTIAL_MEDIUM_RECONNAISSANCE_RETAINED_RESTART_CHECKPOINT"
MEDIUM_URANS_MEAN_CANDIDATE_CLASS = "MEDIUM_MESH_URANS_MEAN_CANDIDATE"
DEFAULT_ANALYSIS_START_S = 0.024
DEFAULT_ANALYSIS_END_S = 0.064
DEFAULT_WINDOW_SPLIT_S = 0.044
FLOW_THROUGH_TIME_S = REF.l_ref_m / REF.u_inf_m_s
MIN_ROWS_FOR_DECISION = 64
FORCE_KEYS = ("c_d", "c_df", "c_m_pitch")
NEAR_ZERO = 1.0e-15
TINY_VARIANCE = 1.0e-30
MIN_LINEAR_DRIFT_ROWS = 3
MIN_SPATIAL_ROWS = 3
MIN_SPECTRUM_AMPLITUDE_COUNT = 3
MIN_SUPPORTED_SPECTRUM_CYCLES = 3.0
MIN_SUPPORTED_PEAK_PROMINENCE = 2.5
MIN_STATIONARY_FLOW_THROUGHS = 2.0
MATERIAL_WINDOW_CHANGE = 0.005
IN_PHASE_CORRELATION = 0.9


@dataclass(frozen=True)
class UransCheckpointDecisionArtifacts:
    report_path: Path
    classification: str


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(__import__("json").loads(path.read_text(encoding="utf-8")))


def _window_rows(rows: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if start_s <= float(row.get("time", row.get("time_s", math.nan))) <= end_s
    ]


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_relative_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or abs(old) < NEAR_ZERO:
        return None
    return float((new - old) / abs(old))


def _basic_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "rms": None,
            "std": None,
            "coefficient_of_variation": None,
            "minimum": None,
            "maximum": None,
        }
    mean = float(statistics.fmean(values))
    rms = float(math.sqrt(statistics.fmean([value * value for value in values])))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "rms": rms,
        "std": std,
        "coefficient_of_variation": std / abs(mean) if abs(mean) > NEAR_ZERO else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def _linear_drift(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    pairs = [
        (float(row["time"]), float(row[key]))
        for row in rows
        if "time" in row and key in row and math.isfinite(float(row[key]))
    ]
    if len(pairs) < MIN_LINEAR_DRIFT_ROWS:
        return {
            "count": len(pairs),
            "slope_per_s": None,
            "slope_95pct_ci_per_s": None,
            "relative_change_over_window": None,
            "r_squared": None,
        }
    times = np.asarray([item[0] for item in pairs], dtype=np.float64)
    values = np.asarray([item[1] for item in pairs], dtype=np.float64)
    shifted_times = times - float(times[0])
    design = np.vstack([shifted_times, np.ones_like(shifted_times)]).T
    slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = slope * shifted_times + intercept
    residuals = values - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((values - float(np.mean(values))) ** 2))
    variance = ss_res / max(len(values) - 2, 1)
    sxx = float(np.sum((shifted_times - float(np.mean(shifted_times))) ** 2))
    slope_se = math.sqrt(variance / sxx) if sxx > 0 else None
    window_s = float(times[-1] - times[0])
    mean_abs = abs(float(np.mean(values)))
    return {
        "count": len(pairs),
        "slope_per_s": float(slope),
        "slope_95pct_ci_per_s": float(1.96 * slope_se) if slope_se is not None else None,
        "relative_change_over_window": float(slope * window_s / mean_abs)
        if mean_abs > NEAR_ZERO
        else None,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > TINY_VARIANCE else None,
    }


def _running_mean_samples(
    rows: list[dict[str, Any]],
    key: str,
    *,
    count: int = 5,
) -> list[dict[str, Any]]:
    values = [float(row[key]) for row in rows if key in row]
    times = [float(row["time"]) for row in rows if key in row]
    if not values:
        return []
    cumulative: list[float] = []
    total = 0.0
    for value in values:
        total += value
        cumulative.append(total / (len(cumulative) + 1))
    if len(values) <= count:
        indices = list(range(len(values)))
    else:
        indices = sorted({round(i) for i in np.linspace(0, len(values) - 1, count)})
    return [
        {
            "time_s": times[index],
            "running_mean": float(cumulative[index]),
            "sample_count": index + 1,
        }
        for index in indices
    ]


def _block_means(rows: list[dict[str, Any]], key: str, *, blocks: int = 4) -> list[dict[str, Any]]:
    if not rows:
        return []
    start = float(rows[0]["time"])
    end = float(rows[-1]["time"])
    if end <= start:
        return [
            {
                "start_s": start,
                "end_s": end,
                "count": len(rows),
                "mean": _safe_mean([float(row[key]) for row in rows]),
            },
        ]
    output: list[dict[str, Any]] = []
    for block_index in range(blocks):
        block_start = start + (end - start) * block_index / blocks
        block_end = start + (end - start) * (block_index + 1) / blocks
        if block_index == blocks - 1:
            selected = [row for row in rows if block_start <= float(row["time"]) <= block_end]
        else:
            selected = [row for row in rows if block_start <= float(row["time"]) < block_end]
        values = [float(row[key]) for row in selected if key in row]
        output.append(
            {
                "start_s": block_start,
                "end_s": block_end,
                "count": len(values),
                "mean": _safe_mean(values),
            },
        )
    return output


def _autocorrelation(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
    times = np.asarray([float(row["time"]) for row in rows if key in row], dtype=np.float64)
    if len(values) < MIN_ROWS_FOR_DECISION:
        return {
            "sample_count": len(values),
            "median_delta_t_s": None,
            "integral_correlation_time_s": None,
            "first_zero_crossing_lag": None,
            "lags": [],
        }
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    dt = float(np.median(np.diff(times)))
    if denominator <= TINY_VARIANCE or dt <= 0.0:
        return {
            "sample_count": len(values),
            "median_delta_t_s": dt if dt > 0.0 else None,
            "integral_correlation_time_s": None,
            "first_zero_crossing_lag": None,
            "lags": [],
        }
    max_lag = min(len(values) // 2, 800)
    acf: list[float] = []
    for lag in range(max_lag + 1):
        numerator = float(np.dot(centered[: len(values) - lag], centered[lag:]))
        acf.append(numerator / denominator)
    first_zero = next((index for index, value in enumerate(acf[1:], start=1) if value <= 0.0), None)
    positive_tail = acf[1:first_zero] if first_zero is not None else acf[1:]
    integral = dt * (1.0 + 2.0 * float(sum(positive_tail)))
    requested_lags = [1, 10, 50, 100, 200, 400, 800]
    return {
        "sample_count": len(values),
        "median_delta_t_s": dt,
        "integral_correlation_time_s": integral,
        "first_zero_crossing_lag": first_zero,
        "first_zero_crossing_time_s": first_zero * dt if first_zero is not None else None,
        "lags": [
            {"lag": lag, "time_s": lag * dt, "acf": float(acf[lag])}
            for lag in requested_lags
            if lag < len(acf)
        ],
    }


def _spectrum(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
    times = np.asarray([float(row["time"]) for row in rows if key in row], dtype=np.float64)
    if len(values) < MIN_ROWS_FOR_DECISION:
        return {
            "sample_count": len(values),
            "dominant_frequency_hz": None,
            "dominant_mode_supported": False,
            "reason": "too_few_samples",
        }
    dt = float(np.median(np.diff(times)))
    window_s = float(times[-1] - times[0])
    if dt <= 0.0 or window_s <= 0.0:
        return {
            "sample_count": len(values),
            "dominant_frequency_hz": None,
            "dominant_mode_supported": False,
            "reason": "non_positive_time_spacing",
        }
    detrended = values - np.linspace(float(values[0]), float(values[-1]), len(values))
    window = np.hanning(len(detrended))
    amplitudes = np.abs(np.fft.rfft(detrended * window))
    frequencies = np.fft.rfftfreq(len(detrended), dt)
    if len(amplitudes) <= 1:
        dominant_frequency = None
        cycles = None
        peak_prominence = None
    else:
        index = int(np.argmax(amplitudes[1:]) + 1)
        dominant_frequency = float(frequencies[index])
        cycles = dominant_frequency * window_s
        background = (
            float(np.median(amplitudes[1:]))
            if len(amplitudes) >= MIN_SPECTRUM_AMPLITUDE_COUNT
            else 0.0
        )
        peak_prominence = (
            float(amplitudes[index] / background) if background > TINY_VARIANCE else None
        )
    supported = bool(
        dominant_frequency is not None
        and cycles is not None
        and cycles >= MIN_SUPPORTED_SPECTRUM_CYCLES
        and (peak_prominence is None or peak_prominence >= MIN_SUPPORTED_PEAK_PROMINENCE)
    )
    return {
        "sample_count": len(values),
        "median_delta_t_s": dt,
        "window_s": window_s,
        "frequency_resolution_hz": 1.0 / window_s,
        "nyquist_hz": 0.5 / dt,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_period_s": 1.0 / dominant_frequency
        if dominant_frequency is not None and dominant_frequency > 0.0
        else None,
        "dominant_cycles_in_window": cycles,
        "peak_to_median_amplitude": peak_prominence,
        "dominant_mode_supported": supported,
        "limits": [
            "The force history is short; spectral peaks are reconnaissance evidence only.",
            "No physical period is accepted unless repeatability and timestep sensitivity pass.",
        ],
    }


def _force_component_rows(force_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    q_area = REF.q_inf_pa * REF.a_ref_m2
    q_area_length = q_area * REF.l_ref_m
    components: dict[str, list[dict[str, float]]] = {"pressure": [], "viscous": [], "total": []}
    for row in force_rows:
        time = float(row["time"])
        for component, force_key, moment_key in (
            ("pressure", "pressure_n", "pressure_moment_nm"),
            ("viscous", "viscous_n", "viscous_moment_nm"),
            ("total", "total_n", "total_moment_nm"),
        ):
            force = [float(value) for value in row[force_key]]
            moment = [float(value) for value in row[moment_key]]
            components[component].append(
                {
                    "time": time,
                    "c_d": force[0] / q_area,
                    "c_df": -force[2] / q_area,
                    "c_m_pitch": moment[1] / q_area_length,
                },
            )
    return components


def _metric_report(rows: list[dict[str, Any]], key: str, *, split_s: float) -> dict[str, Any]:
    early = _window_rows(rows, float(rows[0]["time"]), split_s)
    late = _window_rows(rows, split_s, float(rows[-1]["time"]))
    early_stats = _basic_stats(early, key)
    late_stats = _basic_stats(late, key)
    return {
        "overall": _basic_stats(rows, key),
        "early_window": early_stats,
        "late_window": late_stats,
        "late_vs_early_relative_mean_change": _safe_relative_change(
            early_stats["mean"],
            late_stats["mean"],
        ),
        "linear_drift": _linear_drift(rows, key),
        "running_mean_samples": _running_mean_samples(rows, key),
        "block_means": _block_means(rows, key),
        "autocorrelation": _autocorrelation(rows, key),
        "spectrum": _spectrum(rows, key),
    }


def _summarise_spatial_history(
    spatial_history: dict[str, Any],
    *,
    analysis_start_s: float,
    analysis_end_s: float,
) -> dict[str, Any]:
    rows = [
        row
        for row in spatial_history.get("rows", [])
        if analysis_start_s <= float(row.get("time_s", math.nan)) <= analysis_end_s
    ]
    if len(rows) < MIN_SPATIAL_ROWS:
        return {
            "status": "INSUFFICIENT_SPATIAL_ROWS",
            "row_count": len(rows),
            "accepted": False,
            "training_eligible": False,
        }

    left = np.asarray(
        [float(row["left_tunnel_y_negative"]["coefficients"]["c_df"]) for row in rows],
        dtype=np.float64,
    )
    right = np.asarray(
        [float(row["right_tunnel_y_positive"]["coefficients"]["c_df"]) for row in rows],
        dtype=np.float64,
    )
    correlation = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else None
    regions: dict[str, Any] = {}
    for name in (
        "critical_underfloor",
        "throat_band",
        "diffuser_ramp",
        "diffuser_exit_band",
        "left_tunnel_y_negative",
        "right_tunnel_y_positive",
        "total",
    ):
        values = [float(row[name]["coefficients"]["c_df"]) for row in rows if name in row]
        regions[name] = {
            "mean_c_df": _safe_mean(values),
            "std_c_df": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "minimum_c_df": min(values) if values else None,
            "maximum_c_df": max(values) if values else None,
        }

    bins = rows[0].get("streamwise_bins", {}).get("critical_underfloor", [])
    bin_count = len(bins)
    streamwise_variability: list[dict[str, Any]] = []
    for index in range(bin_count):
        values = [
            float(row["streamwise_bins"]["critical_underfloor"][index]["coefficients"]["c_df"])
            for row in rows
        ]
        template = rows[0]["streamwise_bins"]["critical_underfloor"][index]
        streamwise_variability.append(
            {
                "index": index,
                "x_min_m": template.get("x_min_m"),
                "x_max_m": template.get("x_max_m"),
                "mean_c_df": _safe_mean(values),
                "std_c_df": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                "range_c_df": max(values) - min(values) if values else None,
            },
        )
    streamwise_variability.sort(
        key=lambda item: abs(float(item["range_c_df"] or 0.0)),
        reverse=True,
    )

    return {
        "status": "DIAGNOSTIC_ONLY",
        "accepted": False,
        "training_eligible": False,
        "row_count": len(rows),
        "time_range_s": [float(rows[0]["time_s"]), float(rows[-1]["time_s"])],
        "left_right": {
            "c_df_correlation": correlation,
            "mean_left_c_df": float(np.mean(left)),
            "mean_right_c_df": float(np.mean(right)),
            "mean_difference_left_minus_right_c_df": float(np.mean(left - right)),
            "rms_difference_c_df": float(math.sqrt(np.mean((left - right) ** 2))),
            "interpretation": (
                "consistent_with_in_phase_symmetric_motion"
                if correlation is not None and correlation > IN_PHASE_CORRELATION
                else "asymmetry_or_short_history_unresolved"
            ),
            "limits": [
                "Spatial snapshots are too sparse and short to establish a physical mode.",
            ],
        },
        "regions": regions,
        "highest_variability_streamwise_bins": streamwise_variability[:5],
    }


def _hash_entry(path: Path) -> dict[str, Any]:
    return {
        "path": _repo_relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _freeze_artifacts(
    *,
    work_case: Path,
    force_history_path: Path,
    spatial_history_path: Path | None,
    checkpoint_report_path: Path | None,
    prune_manifest_path: Path | None,
    latest_time_dir: str | None,
) -> dict[str, Any]:
    files: list[Path] = [force_history_path]
    files.extend(
        optional
        for optional in (spatial_history_path, checkpoint_report_path, prune_manifest_path)
        if optional is not None and optional.exists()
    )
    log_dir = work_case / "logs"
    if log_dir.exists():
        files.extend(sorted(log_dir.glob("foamRun_urans_recon*.log")))
    frozen: dict[str, Any] = {
        "files": [_hash_entry(path) for path in files if path.exists()],
        "latest_complete_field_time_dir": latest_time_dir,
        "latest_complete_field_files": [],
    }
    if latest_time_dir is not None:
        latest_dir = work_case / "openfoam" / latest_time_dir
        if latest_dir.exists():
            frozen["latest_complete_field_files"] = [
                _hash_entry(path) for path in sorted(latest_dir.iterdir()) if path.is_file()
            ]
    return frozen


def _classify_checkpoint(
    *,
    metric_reports: dict[str, Any],
    row_count: int,
    latest_time_s: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if row_count < MIN_ROWS_FOR_DECISION:
        return "NUMERICALLY_UNRESOLVED", ["too few force samples for a checkpoint decision"]
    flow_throughs = latest_time_s / FLOW_THROUGH_TIME_S
    if flow_throughs < MIN_STATIONARY_FLOW_THROUGHS:
        reasons.append(
            f"latest time spans only {flow_throughs:.2f} article flow-through times",
        )
    material_changes: list[str] = []
    for key in FORCE_KEYS:
        rel = metric_reports[key]["late_vs_early_relative_mean_change"]
        if rel is not None and abs(float(rel)) > MATERIAL_WINDOW_CHANGE:
            material_changes.append(f"{key} late-vs-early mean change {rel:.4g}")
    if material_changes:
        reasons.extend(material_changes)
    supported_periodic = any(
        bool(metric_reports[key]["spectrum"]["dominant_mode_supported"]) for key in FORCE_KEYS
    )
    if reasons:
        return "STILL_TRANSIENT", reasons
    if supported_periodic:
        return "STATIONARY_PERIODIC_CANDIDATE", [
            "short-history spectrum has a supported dominant reconnaissance mode",
        ]
    return "STATIONARY_BROADBAND_CANDIDATE", [
        "window means and drift are bounded, but repeatability still requires continuation",
    ]


def _accepted_for_time_step_sensitivity(classification: str) -> bool:
    return classification in {
        "STATIONARY_PERIODIC_CANDIDATE",
        "STATIONARY_BROADBAND_CANDIDATE",
    }


def write_urans_checkpoint_decision_report(
    *,
    work_case: Path,
    out_json: Path,
    force_history_path: Path | None = None,
    spatial_history_path: Path | None = None,
    checkpoint_report_path: Path | None = None,
    prune_manifest_path: Path | None = None,
    analysis_start_s: float = DEFAULT_ANALYSIS_START_S,
    analysis_end_s: float = DEFAULT_ANALYSIS_END_S,
    window_split_s: float = DEFAULT_WINDOW_SPLIT_S,
) -> UransCheckpointDecisionArtifacts:
    """Analyze a partial URANS checkpoint without accepting it as CFD evidence."""

    if not (analysis_start_s < window_split_s < analysis_end_s):
        message = "analysis_start_s < window_split_s < analysis_end_s is required"
        raise ValueError(message)
    work_case = work_case.resolve()
    force_history_path = force_history_path or (
        work_case / "quality" / "transient_force_history.json"
    )
    spatial_history_path = spatial_history_path or (
        work_case / "quality" / "urans_spatial_load_history.json"
    )
    checkpoint_report_path = checkpoint_report_path or Path(
        "artifacts/cfd/urans/medium_urans_resume_checkpoint.json",
    )
    prune_manifest_path = prune_manifest_path or (
        work_case / "quality" / "retained_field_prune_manifest.json"
    )

    if not force_history_path.exists():
        message = f"URANS force-history report not found: {force_history_path}"
        raise FileNotFoundError(message)
    force_history = _load_json(force_history_path)
    coefficient_rows = _window_rows(
        list(force_history["force_coefficients"]["rows"]),
        analysis_start_s,
        analysis_end_s,
    )
    force_rows = _window_rows(
        list(force_history["forces"]["rows"]),
        analysis_start_s,
        analysis_end_s,
    )
    if not coefficient_rows:
        message = "analysis window contains no force-coefficient rows"
        raise ValueError(message)

    metric_reports = {
        key: _metric_report(coefficient_rows, key, split_s=window_split_s) for key in FORCE_KEYS
    }
    component_rows = _force_component_rows(force_rows)
    component_reports = {
        component: {
            key: _metric_report(rows, key, split_s=window_split_s) for key in FORCE_KEYS if rows
        }
        for component, rows in component_rows.items()
    }
    latest_time_s = float(coefficient_rows[-1]["time"])
    classification, reasons = _classify_checkpoint(
        metric_reports=metric_reports,
        row_count=len(coefficient_rows),
        latest_time_s=latest_time_s,
    )
    accepted_for_time_step_sensitivity = _accepted_for_time_step_sensitivity(classification)
    case_class = (
        MEDIUM_URANS_MEAN_CANDIDATE_CLASS
        if accepted_for_time_step_sensitivity
        else PARTIAL_URANS_CLASS
    )

    spatial_report: dict[str, Any] | None = None
    if spatial_history_path.exists():
        spatial_report = _summarise_spatial_history(
            _load_json(spatial_history_path),
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
        )

    latest_time_dir: str | None = None
    if checkpoint_report_path.exists():
        checkpoint_report = _load_json(checkpoint_report_path)
        latest_time_dir = str(checkpoint_report.get("latest_complete_written_time_dir"))
    elif (work_case / "openfoam").exists():
        numeric_dirs = [
            path.name
            for path in (work_case / "openfoam").iterdir()
            if path.is_dir() and path.name.replace(".", "", 1).replace("-", "", 1).isdigit()
        ]
        latest_time_dir = sorted(numeric_dirs, key=float)[-1] if numeric_dirs else None

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "URANS_CHECKPOINT_ANALYSIS_ONLY",
        "case_class": case_class,
        "accepted": False,
        "accepted_for_time_step_sensitivity": accepted_for_time_step_sensitivity,
        "training_eligible": False,
        "eligible_targets": {
            "surface_pressure": False,
            "integrated_drag": False,
            "integrated_downforce": False,
            "integrated_lateral_force": False,
            "pitch_moment": False,
            "volume_mean_fields": False,
            "wall_shear": False,
            "separation_metrics": False,
            "cliff_boundary": False,
            "unsteady_statistics": False,
        },
        "work_case": _repo_relative(work_case),
        "classification": classification,
        "classification_reasons": reasons,
        "analysis_window_s": [analysis_start_s, analysis_end_s],
        "comparison_windows_s": [
            [analysis_start_s, window_split_s],
            [window_split_s, analysis_end_s],
        ],
        "latest_time_s": latest_time_s,
        "article_flow_through_time_s": FLOW_THROUGH_TIME_S,
        "article_flow_throughs_at_latest_time": latest_time_s / FLOW_THROUGH_TIME_S,
        "force_history": {
            "source_report": _repo_relative(force_history_path),
            "row_count_in_analysis_window": len(coefficient_rows),
            "time_range_s": [
                float(coefficient_rows[0]["time"]),
                float(coefficient_rows[-1]["time"]),
            ],
            "metrics": metric_reports,
            "pressure_viscous_total_components": component_reports,
        },
        "spatial_history": spatial_report
        if spatial_report is not None
        else {
            "status": "MISSING",
            "accepted": False,
            "training_eligible": False,
        },
        "frozen_artifacts": _freeze_artifacts(
            work_case=work_case,
            force_history_path=force_history_path,
            spatial_history_path=spatial_history_path,
            checkpoint_report_path=checkpoint_report_path,
            prune_manifest_path=prune_manifest_path,
            latest_time_dir=latest_time_dir,
        ),
        "claims_established": [
            f"The {latest_time_s:.6g} s checkpoint artifacts have deterministic hashes recorded.",
            "The current force history has been analyzed before any further CFD continuation.",
            "The checkpoint remains non-accepted and training-ineligible.",
            "Candidate status permits only timestep-sensitivity follow-up, not training labels.",
        ],
        "claims_not_established": [
            "No stationary or repeatable URANS mean is accepted.",
            "No timestep sensitivity, fine-mesh confirmation or training label is established.",
            "Spectral estimates are reconnaissance diagnostics only on this short history.",
        ],
    }
    atomic_write_json(out_json, report)
    return UransCheckpointDecisionArtifacts(report_path=out_json, classification=classification)

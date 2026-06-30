"""Minimal live/replay acquisition loop for the structured AeroCliff Core map."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from aeromap.io import atomic_write_json, atomic_write_text, sha256_file

SCHEMA_VERSION = "aeromap_live_core_acquisition_loop_v0.1.0"
CLASSIFICATION = "AEROMAP_LIVE_CORE_ACQUISITION_LOOP_V0_1"
DEFAULT_DATASET_PATH = Path("docs/evidence/cfd/aerocliff_core/core_2d_response_map_dataset_v0.json")
DEFAULT_OUTPUT_DIR = Path("docs/evidence/cfd/aerocliff_core/live_core_loop_v0_1")
DEFAULT_REPORT_PATH = Path("docs/reports/aerocliff_core_live_acquisition_loop.md")
TARGET_NAMES = ("C_D", "suction_downforce", "pressure_recovery")
DEFAULT_INITIAL_CASES = ("50mm/3deg", "60mm/5deg", "70mm/7deg")
DEFAULT_POLICIES = (
    "random",
    "diversity",
    "engineering_utility",
    "cost_aware_utility",
)
RANDOM_BASELINE_SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
EPSILON = 1.0e-15
CASE_KEY_PART_COUNT = 2

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
AcquisitionPolicy = Literal["random", "diversity", "engineering_utility", "cost_aware_utility"]
LoopMode = Literal["replay-live", "real-live"]


def write_live_core_acquisition_loop(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    acquisition_policy: AcquisitionPolicy = "engineering_utility",
    max_iterations: int = 4,
    mode: LoopMode = "replay-live",
    initial_cases: tuple[str, ...] = DEFAULT_INITIAL_CASES,
    candidate_cases: tuple[str, ...] = (),
    dry_run: bool = False,
    overwrite: bool = False,
    random_seed: int = 20260630,
) -> Path:
    """Run a bounded Core acquisition loop over the committed pressure/load pool.

    The public MVP defaults to replay-live mode: existing Core evidence is hidden,
    revealed one case at a time, and ingested as if a local CFD run had just
    completed. Real-live mode is intentionally conservative; if the selected
    evidence already exists, it is reused rather than regenerating OpenFOAM cases.
    """

    if max_iterations < 1:
        msg = "max_iterations must be at least 1"
        raise ValueError(msg)
    if acquisition_policy not in DEFAULT_POLICIES:
        msg = f"unknown acquisition policy: {acquisition_policy}"
        raise ValueError(msg)
    if mode not in {"replay-live", "real-live"}:
        msg = f"unknown loop mode: {mode}"
        raise ValueError(msg)

    dataset = _load_dataset(dataset_path)
    cases = _clean_cases(dataset)
    initial_indices = _indices_for_case_keys(cases, initial_cases, label="initial")
    candidate_indices = _candidate_indices(cases, initial_indices, candidate_cases)
    primary_run = _run_policy_loop(
        cases=cases,
        initial_indices=initial_indices,
        candidate_indices=candidate_indices,
        policy=acquisition_policy,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )
    policy_runs = {
        policy: [
            _run_policy_loop(
                cases=cases,
                initial_indices=initial_indices,
                candidate_indices=candidate_indices,
                policy=cast("AcquisitionPolicy", policy),
                max_iterations=max_iterations,
                random_seed=seed,
            )
            for seed in (RANDOM_BASELINE_SEEDS if policy == "random" else (random_seed,))
        ]
        for policy in DEFAULT_POLICIES
    }
    summary_by_method = {
        policy: _summarise_policy_runs(runs) for policy, runs in policy_runs.items()
    }
    selected_run = primary_run
    best_method = min(
        summary_by_method,
        key=lambda policy: summary_by_method[policy]["area_under_normalised_rmse_mean_curve"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "live_core_loop_learning_curve.svg"
    _write_learning_curve_svg(summary_by_method, svg_path)
    manifest_path = output_dir / "live_core_loop_manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "accepted": True,
        "mode_requested": mode,
        "mode_executed": "replay-live",
        "dry_run": dry_run,
        "overwrite_requested": overwrite,
        "live_execution_status": _live_execution_status(mode),
        "openfoam_runs": [],
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "classification": dataset.get("classification"),
            "clean_case_count": len(cases),
        },
        "target_family": "AEROCLIFF_CORE_PRESSURE_LOAD_RESPONSE_MAP",
        "allowed_targets": list(TARGET_NAMES),
        "diagnostic_only": [
            "diagnostic_corrected_f_sep",
            "diagnostic_near_wall_reverse_fraction",
            "diagnostic_diffuser_y_plus_mean",
            "diagnostic_diffuser_y_plus_max",
        ],
        "initial_labelled_cases": [_case_public_summary(cases[idx]) for idx in initial_indices],
        "candidate_pool": [_case_public_summary(cases[idx]) for idx in candidate_indices],
        "primary_policy": acquisition_policy,
        "primary_loop": selected_run,
        "policy_comparison": {
            "summary_by_method": summary_by_method,
            "best_method_by_curve_error_area": best_method,
            "runs": policy_runs,
            "random_baseline_seed_count": len(RANDOM_BASELINE_SEEDS),
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "learning_curve_svg": str(svg_path),
            "report": str(report_path),
        },
        "claim_boundary": _claim_boundary(),
    }
    atomic_write_json(manifest_path, payload)
    _write_report(report_path, payload)
    return manifest_path


def _live_execution_status(mode: LoopMode) -> str:
    if mode == "real-live":
        return (
            "real-live requested, but selected Core cases already had accepted evidence; "
            "OpenFOAM was not rerun"
        )
    return "existing_committed_core_evidence_reused; no OpenFOAM case was regenerated"


def _load_dataset(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Core dataset must be a JSON object: {path}"
        raise TypeError(msg)
    return loaded


def _clean_cases(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = dataset.get("cases")
    if not isinstance(raw_cases, list):
        msg = "Core dataset does not contain a cases list"
        raise TypeError(msg)
    cases = [
        cast("dict[str, Any]", case)
        for case in raw_cases
        if isinstance(case, dict) and case.get("clean_for_2d_response_replay") is True
    ]
    if len(cases) < len(DEFAULT_INITIAL_CASES) + 1:
        msg = "Core dataset has too few clean cases for a live acquisition loop"
        raise ValueError(msg)
    return sorted(
        cases,
        key=lambda row: (
            _case_float(row, "ride_height_mm"),
            _case_float(row, "diffuser_angle_deg"),
        ),
    )


def _indices_for_case_keys(
    cases: list[dict[str, Any]], keys: tuple[str, ...], *, label: str
) -> list[int]:
    index_by_key = {_case_key(case): idx for idx, case in enumerate(cases)}
    indices: list[int] = []
    missing: list[str] = []
    for key in keys:
        normalised = _normalise_case_key(key)
        if normalised not in index_by_key:
            missing.append(key)
        elif index_by_key[normalised] not in indices:
            indices.append(index_by_key[normalised])
    if missing:
        msg = f"unknown {label} Core cases: {missing}"
        raise ValueError(msg)
    if not indices:
        msg = f"at least one {label} Core case is required"
        raise ValueError(msg)
    return sorted(indices)


def _candidate_indices(
    cases: list[dict[str, Any]],
    initial_indices: list[int],
    candidate_cases: tuple[str, ...],
) -> list[int]:
    if candidate_cases:
        candidates = _indices_for_case_keys(cases, candidate_cases, label="candidate")
    else:
        candidates = [idx for idx in range(len(cases)) if idx not in initial_indices]
    initial_set = set(initial_indices)
    filtered = [idx for idx in candidates if idx not in initial_set]
    if not filtered:
        msg = "candidate pool is empty after removing initial labelled cases"
        raise ValueError(msg)
    return filtered


def _run_policy_loop(
    *,
    cases: list[dict[str, Any]],
    initial_indices: list[int],
    candidate_indices: list[int],
    policy: AcquisitionPolicy,
    max_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    features, targets, target_spread = _feature_target_arrays(cases)
    high_gradient_indices = _high_gradient_indices(features, targets, target_spread, top_n=3)
    labelled = sorted(initial_indices)
    pool = [idx for idx in candidate_indices if idx not in labelled]
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = [
        _metrics_record(
            iteration=0,
            cases=cases,
            features=features,
            targets=targets,
            target_spread=target_spread,
            labelled=labelled,
            high_gradient_indices=high_gradient_indices,
            selected=None,
        ),
    ]
    selections: list[dict[str, Any]] = []
    for iteration in range(1, min(max_iterations, len(pool)) + 1):
        scored = _score_pool(
            policy=policy,
            features=features,
            targets=targets,
            target_spread=target_spread,
            labelled=labelled,
            pool=pool,
            rng=rng,
        )
        selected_index = int(scored[0]["index"])
        selected = {
            "iteration": iteration,
            "selected_case": _case_public_summary(cases[selected_index]),
            "action": "ingest_committed_core_evidence",
            "reason": scored[0]["reason"],
            "score": scored[0]["score"],
            "score_components": scored[0]["components"],
            "candidate_rankings": [
                {
                    "case": _case_public_summary(cases[int(item["index"])]),
                    "score": item["score"],
                    "components": item["components"],
                }
                for item in scored[:5]
            ],
        }
        selections.append(selected)
        labelled = sorted([*labelled, selected_index])
        pool = [idx for idx in pool if idx != selected_index]
        records.append(
            _metrics_record(
                iteration=iteration,
                cases=cases,
                features=features,
                targets=targets,
                target_spread=target_spread,
                labelled=labelled,
                high_gradient_indices=high_gradient_indices,
                selected=selected,
            ),
        )
    return {
        "policy": policy,
        "initial_label_count": len(initial_indices),
        "max_iterations": max_iterations,
        "completed_iterations": len(selections),
        "selections": selections,
        "metrics_by_iteration": records,
    }


def _feature_target_arrays(
    cases: list[dict[str, Any]],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    features = np.array(
        [
            [
                _case_float(case, "ride_height_mm"),
                _case_float(case, "diffuser_angle_deg"),
                _derived_area_ratio(case),
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    targets = np.array(
        [[_case_float(case, target) for target in TARGET_NAMES] for case in cases],
        dtype=np.float64,
    )
    spread = np.maximum(targets.max(axis=0) - targets.min(axis=0), EPSILON)
    return features, targets, spread


def _case_float(case: dict[str, Any], key: str) -> float:
    value = case[key]
    if not isinstance(value, int | float):
        msg = f"Core case field {key!r} is not numeric"
        raise TypeError(msg)
    return float(value)


def _derived_area_ratio(case: dict[str, Any]) -> float:
    ride_height_m = _case_float(case, "ride_height_mm") / 1000.0
    throat_ratio = _case_float(case, "throat_ratio")
    angle_rad = math.radians(_case_float(case, "diffuser_angle_deg"))
    diffuser_length_m = 0.72
    throat_height = ride_height_m * throat_ratio
    exit_height = throat_height + math.tan(angle_rad) * diffuser_length_m
    return exit_height / max(throat_height, EPSILON)


def _normalised_features(features: FloatArray) -> FloatArray:
    spread = np.maximum(features.max(axis=0) - features.min(axis=0), EPSILON)
    return cast("FloatArray", (features - features.min(axis=0)) / spread)


def _predict_idw(features: FloatArray, targets: FloatArray, labelled: list[int]) -> FloatArray:
    x_values = _normalised_features(features)
    labelled_x = x_values[labelled]
    labelled_y = targets[labelled]
    predictions = np.zeros_like(targets)
    labelled_set = set(labelled)
    for idx, point in enumerate(x_values):
        if idx in labelled_set:
            predictions[idx] = targets[idx]
            continue
        distances = np.linalg.norm(labelled_x - point, axis=1)
        if float(distances.min()) < EPSILON:
            predictions[idx] = labelled_y[int(np.argmin(distances))]
            continue
        weights = 1.0 / np.maximum(distances, EPSILON) ** 2
        predictions[idx] = np.sum(labelled_y * weights[:, None], axis=0) / float(weights.sum())
    return predictions


def _score_pool(
    *,
    policy: AcquisitionPolicy,
    features: FloatArray,
    targets: FloatArray,
    target_spread: FloatArray,
    labelled: list[int],
    pool: list[int],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    predictions = _predict_idw(features, targets, labelled)
    diversity = _distance_to_labelled(features, labelled, pool)
    random_scores = rng.random(len(pool))
    if policy == "random":
        score = random_scores
        reason = "random replay baseline"
        component_rows = [{"random": float(value)} for value in random_scores]
    elif policy == "diversity":
        score = diversity
        reason = "largest distance from the labelled Core cases"
        component_rows = [{"diversity": float(value)} for value in diversity]
    else:
        gradient = np.array(
            [
                _gradient_proxy(
                    candidate_idx=idx,
                    features=features,
                    predictions=predictions,
                    target_spread=target_spread,
                )
                for idx in pool
            ],
            dtype=np.float64,
        )
        high_suction = _safe_normalise(np.abs(predictions[pool, 1]))
        diversity_norm = _safe_normalise(diversity)
        gradient_norm = _safe_normalise(gradient)
        score = 0.40 * diversity_norm + 0.35 * gradient_norm + 0.25 * high_suction
        reason = (
            "balances design-space coverage, response-gradient proxy and high-suction relevance"
        )
        if policy == "cost_aware_utility":
            reason = (
                "same engineering utility with constant proxy cost for medium structured Core cases"
            )
        component_rows = [
            {
                "diversity": float(diversity_norm[row_idx]),
                "gradient_proxy": float(gradient_norm[row_idx]),
                "high_suction_relevance": float(high_suction[row_idx]),
                "proxy_cost": 1.0,
            }
            for row_idx in range(len(pool))
        ]
    rows = [
        {
            "index": idx,
            "score": float(score[row_idx]),
            "reason": reason,
            "components": component_rows[row_idx],
        }
        for row_idx, idx in enumerate(pool)
    ]
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _distance_to_labelled(features: FloatArray, labelled: list[int], pool: list[int]) -> FloatArray:
    x_values = _normalised_features(features)
    labelled_x = x_values[labelled]
    distances = [float(np.min(np.linalg.norm(labelled_x - x_values[idx], axis=1))) for idx in pool]
    return np.array(distances, dtype=np.float64)


def _safe_normalise(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    spread = float(values.max() - values.min())
    if spread < EPSILON:
        return np.zeros_like(values)
    return cast("FloatArray", (values - values.min()) / spread)


def _gradient_proxy(
    *,
    candidate_idx: int,
    features: FloatArray,
    predictions: FloatArray,
    target_spread: FloatArray,
) -> float:
    x_values = _normalised_features(features)
    distances = np.linalg.norm(x_values - x_values[candidate_idx], axis=1)
    neighbor_indices = np.argsort(distances)[1:4]
    deltas = np.abs(predictions[neighbor_indices] - predictions[candidate_idx]) / target_spread
    return float(np.mean(deltas))


def _high_gradient_indices(
    features: FloatArray, targets: FloatArray, target_spread: FloatArray, *, top_n: int
) -> set[int]:
    x_values = _normalised_features(features)
    gradients: list[tuple[int, float]] = []
    for idx in range(len(targets)):
        distances = np.linalg.norm(x_values - x_values[idx], axis=1)
        neighbor_indices = np.argsort(distances)[1:4]
        deltas = np.abs(targets[neighbor_indices] - targets[idx]) / target_spread
        gradients.append((idx, float(np.mean(deltas))))
    return {idx for idx, _ in sorted(gradients, key=lambda item: item[1], reverse=True)[:top_n]}


def _metrics_record(
    *,
    iteration: int,
    cases: list[dict[str, Any]],
    features: FloatArray,
    targets: FloatArray,
    target_spread: FloatArray,
    labelled: list[int],
    high_gradient_indices: set[int],
    selected: dict[str, Any] | None,
) -> dict[str, Any]:
    predictions = _predict_idw(features, targets, labelled)
    residual = predictions - targets
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    normalised_rmse = rmse / target_spread
    labelled_set = set(labelled)
    return {
        "iteration": iteration,
        "label_count": len(labelled),
        "selected_case": selected["selected_case"] if selected is not None else None,
        "rmse_C_D": float(rmse[0]),
        "rmse_suction_downforce": float(rmse[1]),
        "rmse_pressure_recovery": float(rmse[2]),
        "normalised_rmse_mean": float(np.mean(normalised_rmse)),
        "high_gradient_region_coverage": len(high_gradient_indices & labelled_set)
        / float(max(1, len(high_gradient_indices))),
        "cumulative_proxy_cost": float(len(labelled)),
        "labelled_cases": [_case_public_summary(cases[idx]) for idx in labelled],
    }


def _summarise_policy_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    records_by_budget: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        for record in run["metrics_by_iteration"]:
            records_by_budget.setdefault(int(record["label_count"]), []).append(record)
    budget_curve = []
    for budget, records in sorted(records_by_budget.items()):
        budget_curve.append(
            {
                "label_count": budget,
                "normalised_rmse_mean": float(
                    np.mean([record["normalised_rmse_mean"] for record in records]),
                ),
                "rmse_C_D": float(np.mean([record["rmse_C_D"] for record in records])),
                "rmse_suction_downforce": float(
                    np.mean([record["rmse_suction_downforce"] for record in records]),
                ),
                "rmse_pressure_recovery": float(
                    np.mean([record["rmse_pressure_recovery"] for record in records]),
                ),
                "high_gradient_region_coverage": float(
                    np.mean([record["high_gradient_region_coverage"] for record in records]),
                ),
            },
        )
    x_values = np.array([record["label_count"] for record in budget_curve], dtype=np.float64)
    y_values = np.array(
        [record["normalised_rmse_mean"] for record in budget_curve],
        dtype=np.float64,
    )
    final = budget_curve[-1]
    return {
        "budget_curve": budget_curve,
        "area_under_normalised_rmse_mean_curve": float(np.trapezoid(y_values, x_values)),
        "final": final,
    }


def _case_key(case: dict[str, Any]) -> str:
    return _format_case_key(
        _case_float(case, "ride_height_mm"),
        _case_float(case, "diffuser_angle_deg"),
    )


def _normalise_case_key(key: str) -> str:
    cleaned = key.strip().lower().replace(" ", "")
    cleaned = cleaned.replace("ride_height_", "").replace("diffuser_angle_", "")
    cleaned = cleaned.replace("mm", "").replace("deg", "")
    cleaned = cleaned.replace(",", "/").replace("_", "/").replace("-", "/")
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) != CASE_KEY_PART_COUNT:
        msg = f"Core case key must look like '60mm/5deg': {key!r}"
        raise ValueError(msg)
    return _format_case_key(float(parts[0]), float(parts[1]))


def _format_case_key(ride_height_mm: float, diffuser_angle_deg: float) -> str:
    return f"{ride_height_mm:g}mm/{diffuser_angle_deg:g}deg"


def _case_public_summary(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_key": _case_key(case),
        "case_id": case.get("case_id"),
        "ride_height_mm": _case_float(case, "ride_height_mm"),
        "diffuser_angle_deg": _case_float(case, "diffuser_angle_deg"),
        "source": case.get("source"),
        "target_status": case.get("target_status"),
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "local_live_or_replay_core_loop": True,
        "pressure_load_response_mapping": True,
        "openfoam_result_ingestion": True,
        "wall_shear_label": False,
        "continuous_f_sep_label": False,
        "cliff_boundary_label": False,
        "field_level_surrogate": False,
        "full_3d_aerocliff_accuracy": False,
        "f1_floor_accuracy": False,
        "domino_accuracy": False,
        "industrial_live_cfd_savings": False,
    }


def _write_learning_curve_svg(summary_by_method: dict[str, dict[str, Any]], path: Path) -> None:
    width = 920
    height = 520
    left = 80
    right = 40
    top = 40
    bottom = 70
    colors = {
        "random": "#8b8f97",
        "diversity": "#2574a9",
        "engineering_utility": "#1b8a5a",
        "cost_aware_utility": "#b45f06",
    }
    curves = {method: rows["budget_curve"] for method, rows in summary_by_method.items()}
    x_values = [float(point["label_count"]) for curve in curves.values() for point in curve]
    y_values = [
        float(point["normalised_rmse_mean"]) for curve in curves.values() for point in curve
    ]
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    y_min = 0.0
    y_max = max(y_values) * 1.05

    def x_scale(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, EPSILON) * (width - left - right)

    def y_scale(value: float) -> float:
        return (
            height
            - bottom
            - (value - y_min) / max(y_max - y_min, EPSILON) * (height - top - bottom)
        )

    polylines = []
    legend = []
    for index, (method, curve) in enumerate(curves.items()):
        points = " ".join(
            (
                f"{x_scale(float(point['label_count'])):.1f},"
                f"{y_scale(float(point['normalised_rmse_mean'])):.1f}"
            )
            for point in curve
        )
        color = colors.get(method, "#333333")
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" />'
        )
        legend_y = 68 + index * 24
        legend.append(
            f'<rect x="650" y="{legend_y - 10}" width="16" height="4" fill="{color}" />'
            f'<text x="674" y="{legend_y}" font-family="Arial" font-size="13" '
            f'fill="#222">{method}</text>'
        )
    svg_lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="24" font-family="Arial" font-size="20" '
            'font-weight="700" fill="#111">AeroCliff Core live/replay '
            "acquisition loop</text>"
        ),
        (
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
            f'y2="{height - bottom}" stroke="#333"/>'
        ),
        (f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>'),
        (
            f'<text x="{width / 2 - 70:.1f}" y="{height - 24}" '
            'font-family="Arial" font-size="14" fill="#333">labelled Core cases</text>'
        ),
        (
            f'<text x="18" y="{height / 2:.1f}" font-family="Arial" '
            'font-size="14" fill="#333" '
            f'transform="rotate(-90 18 {height / 2:.1f})">'
            "mean normalised RMSE</text>"
        ),
        (
            f'<text x="{left}" y="{height - bottom + 24}" font-family="Arial" '
            f'font-size="12" fill="#555">{x_min}</text>'
        ),
        (
            f'<text x="{width - right - 12}" y="{height - bottom + 24}" '
            f'font-family="Arial" font-size="12" fill="#555">{x_max}</text>'
        ),
        (
            f'<text x="42" y="{y_scale(y_max):.1f}" font-family="Arial" '
            f'font-size="12" fill="#555">{y_max:.3f}</text>'
        ),
        (
            f'<text x="48" y="{y_scale(0.0):.1f}" font-family="Arial" '
            'font-size="12" fill="#555">0</text>'
        ),
        "".join(polylines),
        "".join(legend),
        (
            f'<text x="{left}" y="{height - 8}" font-family="Arial" '
            'font-size="12" fill="#555">Local replay-live loop over accepted '
            "Core pressure/load evidence. Lower is better.</text>"
        ),
        "</svg>",
    ]
    svg = "\n".join(svg_lines) + "\n"
    atomic_write_text(path, svg)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    primary = payload["primary_loop"]
    summary_by_method = payload["policy_comparison"]["summary_by_method"]
    rows = [
        "| Method | Curve-error area | Final normalised RMSE | Final C_D RMSE | "
        "Final suction RMSE | Final pressure-recovery RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, summary in summary_by_method.items():
        final = summary["final"]
        rows.append(
            "| "
            f"{method} | "
            f"{summary['area_under_normalised_rmse_mean_curve']:.6f} | "
            f"{final['normalised_rmse_mean']:.6f} | "
            f"{final['rmse_C_D']:.6f} | "
            f"{final['rmse_suction_downforce']:.6f} | "
            f"{final['rmse_pressure_recovery']:.6f} |"
        )
    selection_rows = [
        "| Iteration | Selected case | Reason |",
        "|---:|---|---|",
    ]
    for selection in primary["selections"]:
        selected_case = selection["selected_case"]
        selection_rows.append(
            f"| {selection['iteration']} | {selected_case['case_key']} | {selection['reason']} |"
        )
    report = f"""# AeroCliff Core live acquisition loop

## Executive summary

This report records the first minimal live/replay Mission Control loop on the
structured AeroCliff Core pressure/load response map.

The loop starts with three labelled Core cases, fits a lightweight response
surrogate, selects the next case, ingests the committed Core evidence, and
updates the map metrics. It proves the local acquisition workflow without
claiming live industrial CFD savings.

Classification: `{payload["classification"]}`

## Loop

```text
labelled Core cases
        -> response surrogate
        -> acquisition policy
        -> selected Core simulation
        -> committed evidence ingestion or local OpenFOAM run
        -> updated pressure/load map
```

Mode requested: `{payload["mode_requested"]}`
Mode executed: `{payload["mode_executed"]}`
Live execution status: {payload["live_execution_status"]}

## Initial labelled set

{_markdown_case_list(payload["initial_labelled_cases"])}

## Primary selections

Primary policy: `{payload["primary_policy"]}`

{chr(10).join(selection_rows)}

## Metrics

{chr(10).join(rows)}

Best method by curve-error area:
`{payload["policy_comparison"]["best_method_by_curve_error_area"]}`

## Claim boundary

Allowed:

- local Core live/replay acquisition loop;
- pressure/load response mapping;
- OpenFOAM result ingestion when selected cases have to be generated locally.

Not claimed:

- field-level surrogate;
- wall-shear or continuous separation-fraction labels;
- validated cliff boundary;
- full 3D AeroCliff accuracy;
- F1 floor accuracy;
- DoMINO accuracy;
- industrial live CFD savings.

## Artifacts

- Manifest: `{payload["artifacts"]["manifest"]}`
- Learning curve: `{payload["artifacts"]["learning_curve_svg"]}`
"""
    atomic_write_text(path, report)


def _markdown_case_list(cases: list[dict[str, Any]]) -> str:
    return "\n".join(f"- `{case['case_key']}` (`{case['case_id']}`)" for case in cases)

#!/usr/bin/env python3
"""Run the bounded 2D Venturi Core pressure/load response-map replay."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aeromap.cfd.venturi_core import (
    FORCE_CV_LIMIT,
    MASS_IMBALANCE_LIMIT,
    PRESSURE_RECOVERY_GRID_DIFF_LIMIT,
    SUCTION_GRID_DIFF_LIMIT,
    VenturiCoreConfig,
    build_venturi_core_case,
    write_venturi_core_case_metrics,
)
from aeromap.io import sha256_file

SCHEMA_VERSION = "venturi_core_2d_pressure_load_response_replay_v0.1.0"
DATASET_SCHEMA_VERSION = "venturi_core_2d_pressure_load_response_dataset_v0.1.0"
REPORT_PATH = Path("artifacts/reports/venturi_core_2d_response_map_active_replay.md")
DATASET_PATH = Path("docs/evidence/cfd/venturi_core/core_2d_response_map_dataset_v0.json")
REPLAY_PATH = Path("docs/evidence/cfd/venturi_core/core_2d_response_map_active_replay_v0.json")

CORE_EVIDENCE_DIR = Path("docs/evidence/cfd/venturi_core")
SOURCE_1D_DATASET = CORE_EVIDENCE_DIR / "core_response_map_dataset_v0.json"
RIDE_HEIGHTS_MM = (50.0, 60.0, 70.0)
DIFFUSER_ANGLES_DEG = (3.0, 4.0, 5.0, 6.0, 7.0)
TARGET_NAMES = ("C_D", "suction_downforce", "pressure_recovery")
REPLAY_METHODS = (
    "random",
    "diversity_space_filling",
    "uncertainty",
    "engineering_utility",
    "cost_aware_utility",
)
RANDOM_SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
EPSILON = 1.0e-15
MAX_TOTAL_MAP_FAILURES = 3
REPEATED_ROW_FAILURE_COUNT = 2
MIN_CLEAN_CASES_FOR_REPLAY = 12
INITIAL_LABEL_COUNT = 4
ERROR_THRESHOLD = 0.05
CD_FINE_SANITY_DIFF_LIMIT = 0.05

MEDIUM_GRID = {
    "grid": "medium",
    "x_cells_per_segment": [16, 24, 24, 40, 16],
    "span_cells": 8,
    "wall_normal_cells": 36,
    "wall_normal_grading": 1.0,
}

FINE_GRID = {
    "grid": "fine",
    "x_cells_per_segment": [24, 36, 36, 60, 24],
    "span_cells": 12,
    "wall_normal_cells": 54,
    "wall_normal_grading": 1.0,
}

INITIAL_SOLVER = {
    "max_iterations": 250,
    "write_interval": 50,
    "force_window": 50,
    "u_inf_m_s": 40.0,
    "rho_kg_m3": 1.225,
}

LONG_STEADY_SOLVER = {
    "max_iterations": 750,
    "write_interval": 50,
    "force_window": 150,
    "u_inf_m_s": 40.0,
    "rho_kg_m3": 1.225,
}


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"expected object JSON: {path}"
        raise TypeError(msg)
    return loaded


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_path(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _case_config(
    *,
    ride_height_mm: float,
    diffuser_angle_deg: float,
    mesh: dict[str, Any],
    solver: dict[str, Any],
) -> VenturiCoreConfig:
    return VenturiCoreConfig.model_validate(
        {
            "classification": "VENTURI_CORE_VENTURI_LAB",
            "geometry": {
                "ride_height_mm": ride_height_mm,
                "diffuser_angle_deg": diffuser_angle_deg,
                "throat_ratio": 0.7,
            },
            "mesh": mesh,
            "solver": solver,
        },
    )


def _run_solver(case_dir: Path, repo_root: Path) -> tuple[int, str]:
    script_path = f"/work/{_relative_path(case_dir / 'run_core_solver.sh', repo_root)}"
    result = subprocess.run(  # noqa: S603
        ["docker", "compose", "run", "--rm", "cfd", script_path],  # noqa: S607
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return result.returncode, combined


def _force_ok_from_metrics(metrics: dict[str, Any]) -> bool:
    forces = metrics["force_coefficients_final_window"]
    return bool(forces["Cd"]["cv"] < FORCE_CV_LIMIT and forces["Cl"]["cv"] < FORCE_CV_LIMIT)


def _summary_from_metrics(
    *,
    ride_height_mm: float,
    diffuser_angle_deg: float,
    source: str,
    phase: str,
    target_status: str,
    metrics_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    forces = metrics["force_coefficients_final_window"]
    floor = metrics["floor_metrics"]
    fatal_ok = bool(metrics["mesh"]["fatal_mesh_quality"]["mesh_ok"])
    extended_ok = bool(metrics["mesh"]["extended_mesh"]["mesh_ok"])
    force_ok = _force_ok_from_metrics(metrics)
    mass_ok = bool(metrics["mass_balance"]["relative_imbalance"] < MASS_IMBALANCE_LIMIT)
    clean = bool(fatal_ok and extended_ok and force_ok and mass_ok)
    return {
        "ride_height_mm": ride_height_mm,
        "diffuser_angle_deg": diffuser_angle_deg,
        "throat_ratio": 0.7,
        "case_id": str(metrics["case_id"]),
        "source": source,
        "phase": phase,
        "target_status": target_status,
        "clean_for_2d_response_replay": clean,
        "mesh_ok": bool(fatal_ok and extended_ok),
        "fatal_mesh_quality_ok": fatal_ok,
        "extended_mesh_quality_ok": extended_ok,
        "force_stability_ok": force_ok,
        "mass_balance_ok": mass_ok,
        "mass_imbalance": metrics["mass_balance"]["relative_imbalance"],
        "C_D": forces["Cd"]["mean"],
        "C_D_cv": forces["Cd"]["cv"],
        "C_D_relative_drift": forces["Cd"]["relative_drift"],
        "suction_downforce": forces["Cl"]["mean"],
        "suction_downforce_cv": forces["Cl"]["cv"],
        "suction_downforce_relative_drift": forces["Cl"]["relative_drift"],
        "pressure_recovery": floor["pressure_recovery_cp_exit_minus_cp_throat"],
        "throat_pressure": floor["throat_pressure_mean_cp"],
        "diffuser_exit_pressure": floor["diffuser_exit_pressure_mean_cp"],
        "diagnostic_corrected_f_sep": floor["diffuser_f_sep"],
        "diagnostic_near_wall_reverse_fraction": floor[
            "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0"
        ],
        "diagnostic_diffuser_y_plus_mean": floor["regions"]["diffuser"]["y_plus_mean"],
        "diagnostic_diffuser_y_plus_max": floor["regions"]["diffuser"]["y_plus_max"],
        "metrics_path": _relative_path(metrics_path, repo_root),
        "metrics_sha256": sha256_file(metrics_path),
    }


def _reuse_60mm_cases(repo_root: Path) -> list[dict[str, Any]]:
    source = _load_json(repo_root / SOURCE_1D_DATASET)
    rows = []
    for case in source["cases"]:
        if not math.isclose(case["ride_height_mm"], 60.0):
            continue
        angle = float(case.get("diffuser_angle_deg", case.get("angle_deg")))
        if angle not in DIFFUSER_ANGLES_DEG:
            continue
        rows.append(
            {
                **case,
                "diffuser_angle_deg": angle,
                "phase": case.get("phase", "reused_clean_medium"),
                "clean_for_2d_response_replay": bool(
                    case["clean_medium_for_response_replay"],
                ),
                "source": f"reused_from_{SOURCE_1D_DATASET.name}:{case['source']}",
                "target_status": case["target_status"],
            },
        )
    return sorted(rows, key=lambda item: item["diffuser_angle_deg"])


def _run_case(
    *,
    repo_root: Path,
    cases_dir: Path,
    ride_height_mm: float,
    diffuser_angle_deg: float,
    mesh: dict[str, Any],
    solver: dict[str, Any],
    phase: str,
    overwrite: bool,
) -> dict[str, Any]:
    config = _case_config(
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=diffuser_angle_deg,
        mesh=mesh,
        solver=solver,
    )
    artifacts = build_venturi_core_case(config, cases_dir=cases_dir, overwrite=overwrite)
    return_code, output = _run_solver(artifacts.case_dir, repo_root)
    log_path = artifacts.case_dir / "logs" / f"{phase}_solver_invocation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    if return_code != 0:
        return {
            "ride_height_mm": ride_height_mm,
            "diffuser_angle_deg": diffuser_angle_deg,
            "throat_ratio": 0.7,
            "case_id": artifacts.case_id,
            "source": "generated_core_2d_response_map",
            "phase": phase,
            "target_status": "solver_or_presolve_failed",
            "clean_for_2d_response_replay": False,
            "solver_return_code": return_code,
            "solver_invocation_log": _relative_path(log_path, repo_root),
        }
    metrics_path = write_venturi_core_case_metrics(artifacts.case_dir)
    summary = _summary_from_metrics(
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=diffuser_angle_deg,
        source="generated_core_2d_response_map",
        phase=phase,
        target_status="medium_response_map_observation",
        metrics_path=metrics_path,
        repo_root=repo_root,
    )
    summary["solver_return_code"] = return_code
    summary["solver_invocation_log"] = _relative_path(log_path, repo_root)
    return summary


def _run_medium_with_one_continuation(
    *,
    repo_root: Path,
    ride_height_mm: float,
    diffuser_angle_deg: float,
    overwrite: bool,
) -> dict[str, Any]:
    key = f"{ride_height_mm:.0f}mm_{diffuser_angle_deg:.0f}deg"
    cases_dir = repo_root / "artifacts/venturi_core/core_2d_response_map_v0/cases" / key
    initial = _run_case(
        repo_root=repo_root,
        cases_dir=cases_dir / "initial",
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=diffuser_angle_deg,
        mesh=MEDIUM_GRID,
        solver=INITIAL_SOLVER,
        phase="initial_250",
        overwrite=overwrite,
    )
    if initial.get("clean_for_2d_response_replay"):
        return initial
    if initial.get("solver_return_code") != 0 or not initial.get("mesh_ok", False):
        return initial
    return _run_case(
        repo_root=repo_root,
        cases_dir=cases_dir / "long_steady",
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=diffuser_angle_deg,
        mesh=MEDIUM_GRID,
        solver=LONG_STEADY_SOLVER,
        phase="long_steady_750",
        overwrite=overwrite,
    )


def _build_medium_map(
    repo_root: Path, *, overwrite: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _reuse_60mm_cases(repo_root)
    failures: list[dict[str, Any]] = []
    failure_counts_by_ride_height = {50.0: 0, 70.0: 0}
    stop_reason: str | None = None
    for ride_height_mm in (50.0, 70.0):
        for angle in DIFFUSER_ANGLES_DEG:
            result = _run_medium_with_one_continuation(
                repo_root=repo_root,
                ride_height_mm=ride_height_mm,
                diffuser_angle_deg=angle,
                overwrite=overwrite,
            )
            rows.append(result)
            if not result.get("clean_for_2d_response_replay"):
                failures.append(result)
                failure_counts_by_ride_height[ride_height_mm] += 1
            if len(failures) > MAX_TOTAL_MAP_FAILURES:
                stop_reason = "more_than_3_of_15_medium_map_cases_failed"
                break
            if all(
                count >= REPEATED_ROW_FAILURE_COUNT
                for count in failure_counts_by_ride_height.values()
            ):
                stop_reason = "both_50mm_and_70mm_rows_show_repeated_failures"
                break
        if stop_reason is not None:
            break
    return sorted(rows, key=lambda item: (item["ride_height_mm"], item["diffuser_angle_deg"])), {
        "stop_reason": stop_reason,
        "failure_count": len(failures),
        "failure_counts_by_ride_height": failure_counts_by_ride_height,
        "failures": [
            {
                "ride_height_mm": item["ride_height_mm"],
                "diffuser_angle_deg": item["diffuser_angle_deg"],
                "case_id": item.get("case_id"),
                "phase": item.get("phase"),
                "target_status": item.get("target_status"),
                "solver_return_code": item.get("solver_return_code"),
                "mesh_ok": item.get("mesh_ok"),
                "force_stability_ok": item.get("force_stability_ok"),
                "mass_balance_ok": item.get("mass_balance_ok"),
            }
            for item in failures
        ],
    }


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("clean_for_2d_response_replay")]


def _target_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array(
        [
            [
                row["ride_height_mm"],
                row["diffuser_angle_deg"],
                _derived_area_ratio(row["ride_height_mm"], row["diffuser_angle_deg"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    targets = np.array([[row[name] for name in TARGET_NAMES] for row in rows], dtype=np.float64)
    target_spread = np.maximum(targets.max(axis=0) - targets.min(axis=0), EPSILON)
    return features, targets, target_spread


def _derived_area_ratio(ride_height_mm: float, diffuser_angle_deg: float) -> float:
    config = _case_config(
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=diffuser_angle_deg,
        mesh=MEDIUM_GRID,
        solver=INITIAL_SOLVER,
    )
    return config.geometry.diffuser_exit_height_m / config.geometry.throat_height_m


def _normalised_features(features: np.ndarray) -> np.ndarray:
    spread = np.maximum(features.max(axis=0) - features.min(axis=0), EPSILON)
    return (features - features.min(axis=0)) / spread


def _predict_idw(features: np.ndarray, targets: np.ndarray, labelled: list[int]) -> np.ndarray:
    x = _normalised_features(features)
    labelled_x = x[labelled]
    labelled_y = targets[labelled]
    predictions = np.zeros_like(targets)
    for idx, point in enumerate(x):
        if idx in labelled:
            predictions[idx] = targets[idx]
            continue
        distances = np.linalg.norm(labelled_x - point, axis=1)
        if float(distances.min()) < EPSILON:
            predictions[idx] = labelled_y[int(np.argmin(distances))]
            continue
        weights = 1.0 / np.maximum(distances, EPSILON) ** 2
        predictions[idx] = np.sum(labelled_y * weights[:, None], axis=0) / float(weights.sum())
    return predictions


def _distance_to_labelled(features: np.ndarray, labelled: list[int], pool: list[int]) -> np.ndarray:
    x = _normalised_features(features)
    labelled_x = x[labelled]
    return np.array(
        [float(np.min(np.linalg.norm(labelled_x - x[idx], axis=1))) for idx in pool],
        dtype=np.float64,
    )


def _safe_normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    spread = float(values.max() - values.min())
    if spread < EPSILON:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def _gradient_proxy(
    *,
    candidate_idx: int,
    features: np.ndarray,
    predictions: np.ndarray,
    target_spread: np.ndarray,
) -> float:
    x = _normalised_features(features)
    distances = np.linalg.norm(x - x[candidate_idx], axis=1)
    neighbor_indices = np.argsort(distances)[1:4]
    deltas = np.abs(predictions[neighbor_indices] - predictions[candidate_idx]) / target_spread
    return float(np.mean(deltas))


def _select_next(
    *,
    method: str,
    features: np.ndarray,
    targets: np.ndarray,
    labelled: list[int],
    pool: list[int],
    target_spread: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if method == "random":
        return int(rng.choice(np.array(pool, dtype=np.int64)))
    diversity = _distance_to_labelled(features, labelled, pool)
    if method in {"diversity_space_filling", "uncertainty"}:
        scores = diversity
    elif method in {"engineering_utility", "cost_aware_utility"}:
        predictions = _predict_idw(features, targets, labelled)
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
        high_load = _safe_normalise(np.abs(predictions[pool, 1]))
        scores = (
            0.40 * _safe_normalise(diversity) + 0.35 * _safe_normalise(gradient) + 0.25 * high_load
        )
        if method == "cost_aware_utility":
            # All medium Core labels have the same live cost model in this goal.
            scores = scores - 0.0
    else:
        msg = f"unknown acquisition method: {method}"
        raise ValueError(msg)
    return pool[int(np.argmax(scores))]


def _true_high_gradient_cells(
    *,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    targets: np.ndarray,
    target_spread: np.ndarray,
) -> list[dict[str, Any]]:
    x = _normalised_features(features)
    gradients = []
    for idx, row in enumerate(rows):
        distances = np.linalg.norm(x - x[idx], axis=1)
        neighbor_indices = np.argsort(distances)[1:4]
        deltas = np.abs(targets[neighbor_indices] - targets[idx]) / target_spread
        gradients.append(
            {
                "index": idx,
                "ride_height_mm": row["ride_height_mm"],
                "diffuser_angle_deg": row["diffuser_angle_deg"],
                "normalised_gradient_proxy": float(np.mean(deltas)),
            },
        )
    return sorted(gradients, key=lambda item: item["normalised_gradient_proxy"], reverse=True)


def _metrics(
    *,
    rows: list[dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    labelled: list[int],
    target_spread: np.ndarray,
    high_gradient_indices: set[int],
    top_suction_index: int,
) -> dict[str, Any]:
    residual = predictions - targets
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    normalised_rmse = rmse / target_spread
    labelled_set = set(labelled)
    high_gradient_recall = len(high_gradient_indices & labelled_set) / float(
        max(1, len(high_gradient_indices)),
    )
    predicted_top_suction = int(np.argmax(predictions[:, 1]))
    top_suction_identified = bool(predicted_top_suction == top_suction_index)
    return {
        "rmse_C_D": float(rmse[0]),
        "rmse_suction_downforce": float(rmse[1]),
        "rmse_pressure_recovery": float(rmse[2]),
        "normalised_rmse_mean": float(np.mean(normalised_rmse)),
        "high_gradient_recall": high_gradient_recall,
        "top_suction_identified": top_suction_identified,
        "labelled_points": [
            {
                "ride_height_mm": rows[idx]["ride_height_mm"],
                "diffuser_angle_deg": rows[idx]["diffuser_angle_deg"],
            }
            for idx in labelled
        ],
    }


def _initial_labelled_indices(rows: list[dict[str, Any]]) -> list[int]:
    corners = {(50.0, 3.0), (50.0, 7.0), (70.0, 3.0), (70.0, 7.0)}
    indices = [
        idx
        for idx, row in enumerate(rows)
        if (row["ride_height_mm"], row["diffuser_angle_deg"]) in corners
    ]
    if len(indices) != INITIAL_LABEL_COUNT:
        return [0, len(rows) // 3, (2 * len(rows)) // 3, len(rows) - 1]
    return sorted(indices)


def _run_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < MIN_CLEAN_CASES_FOR_REPLAY:
        return {
            "classification": "VENTURI_CORE_2D_RESPONSE_MAP_INCONCLUSIVE",
            "accepted": False,
            "reason": "too few clean medium cases for a meaningful 2D replay",
            "records": [],
            "summary_by_method": {},
        }
    features, targets, target_spread = _target_matrix(rows)
    high_gradient = _true_high_gradient_cells(
        rows=rows,
        features=features,
        targets=targets,
        target_spread=target_spread,
    )
    high_gradient_indices = {int(item["index"]) for item in high_gradient[:3]}
    top_suction_index = int(np.argmax(targets[:, 1]))
    records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for method in REPLAY_METHODS:
        method_records: list[dict[str, Any]] = []
        seeds = RANDOM_SEEDS if method == "random" else (0,)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            labelled = _initial_labelled_indices(rows)
            pool = [idx for idx in range(len(rows)) if idx not in labelled]
            while True:
                predictions = _predict_idw(features, targets, labelled)
                metrics = _metrics(
                    rows=rows,
                    targets=targets,
                    predictions=predictions,
                    labelled=labelled,
                    target_spread=target_spread,
                    high_gradient_indices=high_gradient_indices,
                    top_suction_index=top_suction_index,
                )
                record = {
                    "method": method,
                    "seed": seed,
                    "label_count": len(labelled),
                    **metrics,
                }
                records.append(record)
                method_records.append(record)
                if not pool:
                    break
                selected = _select_next(
                    method=method,
                    features=features,
                    targets=targets,
                    labelled=labelled,
                    pool=pool,
                    target_spread=target_spread,
                    rng=rng,
                )
                labelled = sorted([*labelled, selected])
                pool = [idx for idx in pool if idx != selected]
        summary[method] = _summarise_method(method_records)
    best_method = min(
        summary,
        key=lambda method: summary[method]["area_under_normalised_rmse_mean_curve"],
    )
    return {
        "classification": "VENTURI_CORE_2D_PRESSURE_LOAD_RESPONSE_REPLAY_V0",
        "accepted": True,
        "reason": "offline 2D Core pressure/load response-map replay completed",
        "initial_label_count": INITIAL_LABEL_COUNT,
        "methods": list(REPLAY_METHODS),
        "records": records,
        "summary_by_method": summary,
        "best_method_by_curve_error_area": best_method,
        "true_high_gradient_points": high_gradient[:5],
        "top_suction_point": {
            "ride_height_mm": rows[top_suction_index]["ride_height_mm"],
            "diffuser_angle_deg": rows[top_suction_index]["diffuser_angle_deg"],
            "suction_downforce": rows[top_suction_index]["suction_downforce"],
        },
    }


def _summarise_method(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_budget: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_budget.setdefault(int(record["label_count"]), []).append(record)
    budget_curve = []
    for budget, rows in sorted(by_budget.items()):
        budget_curve.append(
            {
                "label_count": budget,
                "normalised_rmse_mean": float(
                    np.mean([row["normalised_rmse_mean"] for row in rows]),
                ),
                "rmse_C_D": float(np.mean([row["rmse_C_D"] for row in rows])),
                "rmse_suction_downforce": float(
                    np.mean([row["rmse_suction_downforce"] for row in rows]),
                ),
                "rmse_pressure_recovery": float(
                    np.mean([row["rmse_pressure_recovery"] for row in rows]),
                ),
                "high_gradient_recall": float(
                    np.mean([row["high_gradient_recall"] for row in rows])
                ),
                "top_suction_identified_rate": float(
                    np.mean([row["top_suction_identified"] for row in rows]),
                ),
            },
        )
    x = np.array([row["label_count"] for row in budget_curve], dtype=np.float64)
    y = np.array([row["normalised_rmse_mean"] for row in budget_curve], dtype=np.float64)
    labels_to_error_threshold = next(
        (
            row["label_count"]
            for row in budget_curve
            if row["normalised_rmse_mean"] <= ERROR_THRESHOLD
        ),
        None,
    )
    budget_8 = _budget_row_or_last(budget_curve, label_count=8)
    return {
        "budget_curve": budget_curve,
        "area_under_normalised_rmse_mean_curve": float(np.trapezoid(y, x)),
        "labels_to_error_threshold": labels_to_error_threshold,
        "budget_8_or_nearest": budget_8,
        "final": budget_curve[-1],
    }


def _budget_row_or_last(rows: list[dict[str, Any]], *, label_count: int) -> dict[str, Any]:
    for row in rows:
        if int(row["label_count"]) == label_count:
            return row
    return rows[-1]


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), EPSILON)
    return abs(right - left) / denominator


def _select_fine_sanity_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features, targets, target_spread = _target_matrix(rows)
    high_gradient = _true_high_gradient_cells(
        rows=rows,
        features=features,
        targets=targets,
        target_spread=target_spread,
    )
    desired = [(70.0, 3.0), (50.0, 7.0)]
    points: list[dict[str, Any]] = []
    for ride_height_mm, angle in desired:
        match = next(
            (
                row
                for row in rows
                if math.isclose(row["ride_height_mm"], ride_height_mm)
                and math.isclose(row["diffuser_angle_deg"], angle)
            ),
            None,
        )
        if match is not None:
            points.append(match)
    for gradient in high_gradient:
        match = rows[int(gradient["index"])]
        if all(
            not (
                math.isclose(item["ride_height_mm"], match["ride_height_mm"])
                and math.isclose(item["diffuser_angle_deg"], match["diffuser_angle_deg"])
            )
            for item in points
        ):
            points.append(match)
            break
    return points[:3]


def _run_fine_sanity_case(
    *,
    repo_root: Path,
    medium: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    ride_height_mm = medium["ride_height_mm"]
    angle = medium["diffuser_angle_deg"]
    key = f"{ride_height_mm:.0f}mm_{angle:.0f}deg"
    cases_dir = repo_root / "artifacts/venturi_core/core_2d_response_map_v0/fine_checks" / key
    initial = _run_case(
        repo_root=repo_root,
        cases_dir=cases_dir / "initial",
        ride_height_mm=ride_height_mm,
        diffuser_angle_deg=angle,
        mesh=FINE_GRID,
        solver=INITIAL_SOLVER,
        phase="fine_initial_250",
        overwrite=overwrite,
    )
    fine = initial
    if (
        not initial.get("clean_for_2d_response_replay")
        and initial.get("solver_return_code") == 0
        and initial.get("mesh_ok", False)
    ):
        fine = _run_case(
            repo_root=repo_root,
            cases_dir=cases_dir / "long_steady",
            ride_height_mm=ride_height_mm,
            diffuser_angle_deg=angle,
            mesh=FINE_GRID,
            solver=LONG_STEADY_SOLVER,
            phase="fine_long_steady_750",
            overwrite=overwrite,
        )
    if not fine.get("clean_for_2d_response_replay"):
        return {
            "ride_height_mm": ride_height_mm,
            "diffuser_angle_deg": angle,
            "medium_case_id": medium["case_id"],
            "fine_case_id": fine.get("case_id"),
            "passed": False,
            "reason": "fine case failed mesh/mass/force gate",
            "fine": fine,
        }
    cd_diff = _relative_difference(medium["C_D"], fine["C_D"])
    suction_diff = _relative_difference(medium["suction_downforce"], fine["suction_downforce"])
    pressure_diff = _relative_difference(medium["pressure_recovery"], fine["pressure_recovery"])
    passed = bool(
        cd_diff <= CD_FINE_SANITY_DIFF_LIMIT
        and suction_diff <= SUCTION_GRID_DIFF_LIMIT
        and pressure_diff <= PRESSURE_RECOVERY_GRID_DIFF_LIMIT
    )
    return {
        "ride_height_mm": ride_height_mm,
        "diffuser_angle_deg": angle,
        "medium_case_id": medium["case_id"],
        "fine_case_id": fine["case_id"],
        "passed": passed,
        "reason": "medium/fine pressure-load sanity passed"
        if passed
        else "medium/fine difference large",
        "relative_differences": {
            "C_D": cd_diff,
            "suction_downforce": suction_diff,
            "pressure_recovery": pressure_diff,
        },
        "medium": {
            "C_D": medium["C_D"],
            "suction_downforce": medium["suction_downforce"],
            "pressure_recovery": medium["pressure_recovery"],
        },
        "fine": {
            "case_id": fine["case_id"],
            "phase": fine["phase"],
            "C_D": fine["C_D"],
            "suction_downforce": fine["suction_downforce"],
            "pressure_recovery": fine["pressure_recovery"],
            "mesh_ok": fine["mesh_ok"],
            "force_stability_ok": fine["force_stability_ok"],
            "mass_balance_ok": fine["mass_balance_ok"],
        },
    }


def _run_fine_sanity_checks(
    *,
    repo_root: Path,
    rows: list[dict[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    selected = _select_fine_sanity_points(rows)
    checks = [
        _run_fine_sanity_case(repo_root=repo_root, medium=row, overwrite=overwrite)
        for row in selected
    ]
    return {
        "status": "ran",
        "selected_cases": [
            {
                "ride_height_mm": row["ride_height_mm"],
                "diffuser_angle_deg": row["diffuser_angle_deg"],
                "case_id": row["case_id"],
            }
            for row in selected
        ],
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "screening_only_if_failed": not all(check["passed"] for check in checks),
    }


def _release_scope() -> dict[str, bool]:
    return {
        "core_2d_pressure_load_response_mapping": True,
        "wall_shear_label": False,
        "continuous_f_sep_label": False,
        "cliff_boundary_label": False,
        "active_learning_cliff_discovery": False,
        "full_3d_extension_accuracy": False,
        "f1_floor_accuracy": False,
        "external_predictor_accuracy": False,
        "live_cfd_savings": False,
    }


def _write_dataset(
    *,
    repo_root: Path,
    rows: list[dict[str, Any]],
    medium_status: dict[str, Any],
    fine_sanity: dict[str, Any],
) -> dict[str, Any]:
    clean = _clean_rows(rows)
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "classification": "VENTURI_CORE_2D_PRESSURE_LOAD_RESPONSE_DATASET_V0",
        "accepted": len(clean) >= MIN_CLEAN_CASES_FOR_REPLAY
        and medium_status["stop_reason"] is None,
        "training_eligible": False,
        "case_grid": {
            "ride_height_mm": list(RIDE_HEIGHTS_MM),
            "diffuser_angle_deg": list(DIFFUSER_ANGLES_DEG),
            "throat_ratio": 0.7,
            "u_inf_m_s": 40.0,
        },
        "targets": {
            "allowed": list(TARGET_NAMES),
            "diagnostic_only": [
                "diagnostic_corrected_f_sep",
                "diagnostic_near_wall_reverse_fraction",
                "diagnostic_diffuser_y_plus_mean",
                "diagnostic_diffuser_y_plus_max",
            ],
            "out_of_scope": [
                "wall_shear_magnitude",
                "continuous_separation_fraction",
                "cliff_boundary",
            ],
        },
        "medium_map_status": {
            **medium_status,
            "planned_case_count": len(RIDE_HEIGHTS_MM) * len(DIFFUSER_ANGLES_DEG),
            "observed_case_count": len(rows),
            "clean_case_count": len(clean),
        },
        "fine_sanity_status": fine_sanity,
        "cases": rows,
        "release_scope": _release_scope(),
        "source_1d_dataset": {
            "path": str(SOURCE_1D_DATASET),
            "sha256": sha256_file(repo_root / SOURCE_1D_DATASET),
        },
    }
    _write_json(repo_root / DATASET_PATH, dataset)
    return dataset


def _classification_from_results(
    *,
    dataset: dict[str, Any],
    replay: dict[str, Any],
    fine_sanity: dict[str, Any],
) -> tuple[str, bool, bool, str]:
    medium_status = dataset["medium_map_status"]
    if medium_status["stop_reason"] is not None:
        return (
            "VENTURI_CORE_2D_RESPONSE_MAP_INCONCLUSIVE",
            False,
            False,
            f"medium-map stop condition fired: {medium_status['stop_reason']}",
        )
    if medium_status["clean_case_count"] < MIN_CLEAN_CASES_FOR_REPLAY:
        return (
            "VENTURI_CORE_2D_RESPONSE_MAP_INCONCLUSIVE",
            False,
            False,
            "too few clean medium cases for a meaningful replay",
        )
    if not replay.get("accepted", False):
        return (
            "VENTURI_CORE_2D_RESPONSE_MAP_INCONCLUSIVE",
            False,
            False,
            str(replay.get("reason", "replay did not complete")),
        )
    if fine_sanity.get("status") == "ran" and not fine_sanity.get("all_passed", False):
        return (
            "VENTURI_CORE_2D_PRESSURE_LOAD_RESPONSE_REPLAY_SCREENING_V0",
            True,
            True,
            "medium 2D replay completed, but representative fine checks make it screening-only",
        )
    return (
        "VENTURI_CORE_2D_PRESSURE_LOAD_RESPONSE_REPLAY_V0",
        True,
        False,
        (
            "2D Core pressure/load response-map replay completed with "
            "representative fine sanity checks"
        ),
    )


def _build_payload(
    *,
    repo_root: Path,
    dataset: dict[str, Any],
    replay: dict[str, Any],
    fine_sanity: dict[str, Any],
) -> dict[str, Any]:
    classification, accepted, screening_only, reason = _classification_from_results(
        dataset=dataset,
        replay=replay,
        fine_sanity=fine_sanity,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "accepted": accepted,
        "screening_only": screening_only,
        "training_eligible": False,
        "reason": reason,
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": sha256_file(repo_root / DATASET_PATH),
            "classification": dataset["classification"],
            "clean_case_count": dataset["medium_map_status"]["clean_case_count"],
            "observed_case_count": dataset["medium_map_status"]["observed_case_count"],
        },
        "replay": replay,
        "fine_sanity_status": fine_sanity,
        "release_scope": _release_scope(),
    }
    _write_json(repo_root / REPLAY_PATH, payload)
    return payload


def _fmt(value: object, *, precision: int = 6) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{precision}f}"
    return "n/a"


def _markdown_case_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        (
            "| ride height | angle | case | phase | clean | C_D | "
            "suction/downforce | pressure recovery | f_sep diag |"
        ),
        "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            f"{row['ride_height_mm']:.0f} | "
            f"{row['diffuser_angle_deg']:.2f} | "
            f"`{row.get('case_id', 'n/a')}` | "
            f"{row.get('phase', 'n/a')} | "
            f"`{bool(row.get('clean_for_2d_response_replay'))}` | "
            f"{_fmt(row.get('C_D'))} | "
            f"{_fmt(row.get('suction_downforce'))} | "
            f"{_fmt(row.get('pressure_recovery'))} | "
            f"{_fmt(row.get('diagnostic_corrected_f_sep'))} |"
        )
        for row in rows
    )
    return "\n".join(lines)


def _markdown_method_table(summary_by_method: dict[str, Any]) -> str:
    lines = [
        (
            "| method | curve-error area | labels to threshold | budget-8 norm RMSE | "
            "budget-8 high-gradient recall | budget-8 top-suction rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, summary in summary_by_method.items():
        budget = summary["budget_8_or_nearest"]
        labels_to_threshold = summary["labels_to_error_threshold"]
        labels_text = "n/a" if labels_to_threshold is None else str(labels_to_threshold)
        lines.append(
            "| "
            f"{method} | "
            f"{summary['area_under_normalised_rmse_mean_curve']:.6f} | "
            f"{labels_text} | "
            f"{budget['normalised_rmse_mean']:.6f} | "
            f"{budget['high_gradient_recall']:.3f} | "
            f"{budget['top_suction_identified_rate']:.3f} |",
        )
    return "\n".join(lines)


def _markdown_fine_table(fine_sanity: dict[str, Any]) -> str:
    if fine_sanity["status"] != "ran":
        return f"Fine sanity checks were `{fine_sanity['status']}`: {fine_sanity['reason']}."
    lines = [
        (
            "| ride height | angle | passed | reason | C_D diff | suction diff | "
            "pressure recovery diff |"
        ),
        "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for check in fine_sanity["checks"]:
        diffs = check.get("relative_differences", {})
        lines.append(
            "| "
            f"{check['ride_height_mm']:.0f} | "
            f"{check['diffuser_angle_deg']:.2f} | "
            f"`{bool(check['passed'])}` | "
            f"{check['reason']} | "
            f"{_fmt(diffs.get('C_D'), precision=4)} | "
            f"{_fmt(diffs.get('suction_downforce'), precision=4)} | "
            f"{_fmt(diffs.get('pressure_recovery'), precision=4)} |",
        )
    return "\n".join(lines)


def _write_report(repo_root: Path, payload: dict[str, Any], dataset: dict[str, Any]) -> None:
    replay = payload["replay"]
    medium_status = dataset["medium_map_status"]
    lines = [
        "# Venturi Core 2D Pressure/Load Response-Map Active Replay",
        "",
        (
            "This report upgrades the previous one-dimensional Venturi Core replay "
            "into a small 3 x 5 structured Venturi-underfloor pressure/load response "
            "map. It is continuous response mapping, not cliff classification."
        ),
        "",
        "## Decision",
        "",
        f"- classification: `{payload['classification']}`",
        f"- accepted: `{payload['accepted']}`",
        f"- screening only: `{payload['screening_only']}`",
        "- training eligible: `False`",
        f"- reason: {payload['reason']}",
        "",
        "## Scope",
        "",
        "- structured Core pressure/load response map",
        "- local offline replay over C_D, suction/downforce and pressure recovery",
        "- live solver scheduling and richer field targets are follow-on work",
        "",
        "## Medium Response Map",
        "",
        f"- planned cases: `{medium_status['planned_case_count']}`",
        f"- observed cases: `{medium_status['observed_case_count']}`",
        f"- clean cases: `{medium_status['clean_case_count']}`",
        f"- stop reason: `{medium_status['stop_reason']}`",
        "",
        _markdown_case_table(dataset["cases"]),
        "",
        "## Representative Fine Sanity Checks",
        "",
        _markdown_fine_table(payload["fine_sanity_status"]),
        "",
        "## Active-Replay Protocol",
        "",
        "- inputs: ride height, diffuser angle and derived diffuser/throat area ratio",
        "- targets: `C_D`, suction/downforce coefficient and pressure recovery",
        "- initial labels: four corner cases when available",
        "- surrogate: deterministic inverse-distance interpolation over labelled points",
        "- acquisition batch: one case at a time",
        "- random baseline: ten acquisition seeds",
        "- diagnostic f_sep, near-wall reverse fraction and y+ are not targets",
        "",
    ]
    if replay.get("accepted", False):
        lines.extend(
            [
                "## Replay Result",
                "",
                f"- best method by curve-error area: `{replay['best_method_by_curve_error_area']}`",
                (
                    "- top suction point: "
                    f"`{replay['top_suction_point']['ride_height_mm']:.0f} mm / "
                    f"{replay['top_suction_point']['diffuser_angle_deg']:.2f} deg`"
                ),
                "",
                _markdown_method_table(replay["summary_by_method"]),
                "",
                "## High-Gradient Points",
                "",
                "| ride height | angle | gradient proxy |",
                "| ---: | ---: | ---: |",
            ],
        )
        lines.extend(
            (
                f"| {point['ride_height_mm']:.0f} | "
                f"{point['diffuser_angle_deg']:.2f} | "
                f"{point['normalised_gradient_proxy']:.6f} |"
            )
            for point in replay["true_high_gradient_points"]
        )
        lines.append("")
    else:
        lines.extend(["## Replay Result", "", f"- replay did not complete: {replay['reason']}", ""])
    lines.extend(
        [
            "## Release Scope",
            "",
            (
                "- AeroMap can run a bounded offline response-mapping replay "
                "on the structured Core pressure/load surface."
            ),
            (
                "- The replay targets pressure/load scalars; field and separation "
                "targets are later work."
            ),
            "",
            "## Artifacts",
            "",
            f"- Dataset: `{DATASET_PATH}` (`{payload['dataset']['sha256']}`)",
            f"- Replay: `{REPLAY_PATH}` (`{sha256_file(repo_root / REPLAY_PATH)}`)",
            "- Runner: `scripts/run_venturi_core_2d_response_map_replay.py`",
            "",
        ],
    )
    (repo_root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")


def _not_run_fine_status(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "selected_cases": [],
        "checks": [],
        "all_passed": False,
        "screening_only_if_failed": False,
    }


def _run_from_committed_dataset(repo_root: Path) -> None:
    dataset = _load_json(repo_root / DATASET_PATH)
    rows = list(dataset["cases"])
    clean_rows = _clean_rows(rows)
    fine_sanity = dataset.get(
        "fine_sanity_status",
        _not_run_fine_status("committed dataset did not include fine sanity metadata"),
    )
    replay = _run_replay(clean_rows)
    payload = _build_payload(
        repo_root=repo_root,
        dataset=dataset,
        replay=replay,
        fine_sanity=fine_sanity,
    )
    _write_report(repo_root, payload, dataset)
    print(
        json.dumps(
            {
                "mode": "committed_dataset_replay",
                "classification": payload["classification"],
                "accepted": payload["accepted"],
                "screening_only": payload["screening_only"],
                "clean_case_count": dataset["medium_map_status"]["clean_case_count"],
                "best_method": replay.get("best_method_by_curve_error_area"),
                "report": str(REPORT_PATH),
            },
            indent=2,
            sort_keys=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate OpenFOAM cases with Docker instead of replaying committed evidence",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if not args.overwrite and (repo_root / DATASET_PATH).exists():
        _run_from_committed_dataset(repo_root)
        return

    rows, medium_status = _build_medium_map(repo_root, overwrite=args.overwrite)
    clean_rows = _clean_rows(rows)
    if medium_status["stop_reason"] is None and len(clean_rows) == len(RIDE_HEIGHTS_MM) * len(
        DIFFUSER_ANGLES_DEG,
    ):
        fine_sanity = _run_fine_sanity_checks(
            repo_root=repo_root,
            rows=clean_rows,
            overwrite=args.overwrite,
        )
    else:
        fine_sanity = _not_run_fine_status(
            "medium map was incomplete or hit a stop condition",
        )
    replay = _run_replay(clean_rows)
    dataset = _write_dataset(
        repo_root=repo_root,
        rows=rows,
        medium_status=medium_status,
        fine_sanity=fine_sanity,
    )
    payload = _build_payload(
        repo_root=repo_root,
        dataset=dataset,
        replay=replay,
        fine_sanity=fine_sanity,
    )
    _write_report(repo_root, payload, dataset)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "accepted": payload["accepted"],
                "screening_only": payload["screening_only"],
                "clean_case_count": dataset["medium_map_status"]["clean_case_count"],
                "best_method": replay.get("best_method_by_curve_error_area"),
                "report": str(REPORT_PATH),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()

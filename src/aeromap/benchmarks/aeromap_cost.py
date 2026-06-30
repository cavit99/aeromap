"""Cost-proxy AeroMap replay and cached 3D field-readiness utilities."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from aeromap.benchmarks.aeromap import (
    AEROMAP_CLASS,
    DECISION_METRICS,
    LOWER_IS_BETTER_METRICS,
    AeroMapConfig,
    DatasetArrays,
    IndexArray,
    RidgeEnsemble,
    _decision_metric_winners,
    _diversity_score,
    _fit_scaler,
    _learning_curve_area_statistics,
    _mean_records,
    _pareto_mask,
    _report_seed_records,
    _safe_normalize,
    _select_batch,
    _split_replay_statistics,
    _stable_method_seed,
    evaluate_predictions,
    fit_ridge_ensemble,
    load_dataset_npz,
    split_dataset_by_mode,
)
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file

FloatArray = NDArray[np.float64]

AEROMAP_COST_AUDIT_SCHEMA = "aerocliff_aeromap_cost_proxy_audit_v0.5.0"
AEROMAP_COST_REPLAY_SCHEMA = "aerocliff_aeromap_cost_aware_replay_v0.5.0"
AEROMAP_SURFACE_FIELD_READINESS_SCHEMA = "aerocliff_aeromap3d_surface_field_readiness_v0.5.0"
COST_AWARE_METHOD = "engineering_decision_utility_v2_cost_aware"
COST_AWARE_LAMBDA = 0.35
MAX_MESH_COUNT_SAMPLE = 10
MAX_SURFACE_FIELD_CASES = 2
SURFACE_SAMPLE_POINTS = 75_000
EPSILON = 1.0e-12
STL_HEADER_BYTES = 80
STL_TRIANGLE_COUNT_BYTES = 4
STL_COUNT_HEADER_BYTES = STL_HEADER_BYTES + STL_TRIANGLE_COUNT_BYTES
BINARY_STL_TRIANGLE_BYTES = 50


@dataclass(frozen=True)
class CostProxy:
    """One positive scalar cost proxy per dataset case."""

    values: FloatArray
    kind: str
    source: str
    description: str
    available_for_all_cases: bool
    evidence: dict[str, Any]


def _case_file(root: Path, case_id: str, suffix: str) -> Path:
    return root / case_id / f"{case_id}{suffix}"


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _normalise_positive(values: FloatArray) -> FloatArray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.ones_like(values, dtype=np.float64)
    median = float(np.median(finite))
    if math.isclose(median, 0.0, abs_tol=EPSILON):
        median = 1.0
    scaled = values / median
    scaled = np.where(np.isfinite(scaled), scaled, 1.0)
    return cast("FloatArray", np.clip(scaled, 0.1, 10.0).astype(np.float64))


def _geometry_complexity_cost(dataset: DatasetArrays) -> CostProxy:
    geometry_columns = [
        idx
        for idx, name in enumerate(dataset.feature_names)
        if name.startswith(("geom_", "shape_param_"))
    ]
    if not geometry_columns:
        values = np.ones(dataset.features.shape[0], dtype=np.float64)
        evidence = {"geometry_feature_count": 0, "fallback": "constant_cost"}
    else:
        geometry = dataset.features[:, geometry_columns]
        scaled = _fit_scaler(geometry).transform(geometry)
        distance = np.linalg.norm(scaled - np.median(scaled, axis=0), axis=1)
        values = 1.0 + _safe_normalize(distance)
        evidence = {
            "geometry_feature_count": len(geometry_columns),
            "geometry_features": [dataset.feature_names[idx] for idx in geometry_columns],
            "proxy_formula": (
                "1 + minmax_norm(norm(train-independent geometry descriptor distance from median))"
            ),
        }
    return CostProxy(
        values=values.astype(np.float64),
        kind="derived_complexity_proxy",
        source="compact geometry descriptors",
        description=(
            "Derived geometry-complexity proxy. It is useful for cost-aware policy mechanics "
            "but is not observed CFD runtime."
        ),
        available_for_all_cases=True,
        evidence=evidence,
    )


def _airfrans_file_size_cost(dataset: DatasetArrays, processed_root: Path) -> CostProxy | None:
    dataset_root = processed_root / "Dataset"
    sizes: list[float] = []
    missing: list[str] = []
    for case_id in dataset.case_ids:
        size = _file_size(_case_file(dataset_root, case_id, "_internal.vtu"))
        if size is None:
            missing.append(case_id)
            sizes.append(float("nan"))
        else:
            sizes.append(float(size))
    if missing:
        return None
    raw = np.asarray(sizes, dtype=np.float64)
    return CostProxy(
        values=_normalise_positive(raw),
        kind="real_observed_cost_proxy",
        source="local AirfRANS internal.vtu file size",
        description=(
            "Observed local per-case volume-field file size. This is a real case-size proxy, "
            "not a measured solver wall-clock cost."
        ),
        available_for_all_cases=True,
        evidence={
            "case_count": len(dataset.case_ids),
            "bytes_min": int(raw.min()),
            "bytes_median": float(np.median(raw)),
            "bytes_max": int(raw.max()),
            "normalisation": "case_size_bytes / median_case_size_bytes, clipped to [0.1, 10]",
            "missing_case_count": len(missing),
        },
    )


def _binary_stl_triangle_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        header = handle.read(STL_COUNT_HEADER_BYTES)
    if len(header) < STL_COUNT_HEADER_BYTES:
        return None
    count = struct.unpack("<I", header[STL_HEADER_BYTES:STL_COUNT_HEADER_BYTES])[0]
    expected = STL_COUNT_HEADER_BYTES + int(count) * BINARY_STL_TRIANGLE_BYTES
    if expected == path.stat().st_size:
        return int(count)
    return None


def _summarise_sizes(paths: list[Path]) -> dict[str, Any]:
    sizes = np.asarray([path.stat().st_size for path in paths], dtype=np.float64)
    if sizes.size == 0:
        return {"count": 0}
    return {
        "count": int(sizes.size),
        "bytes_min": int(sizes.min()),
        "bytes_median": float(np.median(sizes)),
        "bytes_max": int(sizes.max()),
        "bytes_total": int(sizes.sum()),
    }


def _sample_airfrans_mesh_counts(processed_root: Path) -> list[dict[str, Any]]:
    dataset_root = processed_root / "Dataset"
    paths = sorted(dataset_root.glob("*/*_internal.vtu"))[:MAX_MESH_COUNT_SAMPLE]
    rows: list[dict[str, Any]] = []
    for path in paths:
        mesh = pv.read(path)
        rows.append(
            {
                "case_id": path.parent.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "points": int(mesh.n_points),
                "cells": int(mesh.n_cells),
            },
        )
    return rows


def _dataset_cost_proxy(
    dataset: DatasetArrays,
    *,
    dataset_name: str,
    airfrans_processed_root: Path,
) -> CostProxy:
    if dataset_name == "airfrans":
        file_proxy = _airfrans_file_size_cost(dataset, airfrans_processed_root)
        if file_proxy is not None:
            return file_proxy
    return _geometry_complexity_cost(dataset)


def write_cost_proxy_audit(
    *,
    airfrans_dataset_npz: Path,
    drivaerml_dataset_npz: Path,
    airfrans_processed_root: Path,
    drivaerml_cache_root: Path,
    out: Path,
) -> Path:
    """Audit available cost proxies without downloading new data."""

    airfrans = load_dataset_npz(airfrans_dataset_npz)
    drivaerml = load_dataset_npz(drivaerml_dataset_npz)
    airfrans_proxy = _dataset_cost_proxy(
        airfrans,
        dataset_name="airfrans",
        airfrans_processed_root=airfrans_processed_root,
    )
    drivaerml_proxy = _dataset_cost_proxy(
        drivaerml,
        dataset_name="drivaerml",
        airfrans_processed_root=airfrans_processed_root,
    )
    stls = sorted(drivaerml_cache_root.glob("run_*/drivaer_*.stl"))
    boundaries = sorted(drivaerml_cache_root.glob("run_*/boundary_*.vtp"))
    stl_triangles = [
        {
            "path": str(path),
            "triangles": _binary_stl_triangle_count(path),
            "bytes": path.stat().st_size,
        }
        for path in stls[:24]
    ]
    payload = {
        "schema_version": AEROMAP_COST_AUDIT_SCHEMA,
        "classification": "AEROMAP_COST_PROXY_AUDIT_V0_5",
        "claim_boundary": {
            "live_cfd_savings": False,
            "costs_are_proxy_values": True,
            "cloud_used": False,
            "new_data_downloaded": False,
        },
        "selection_decision": {
            "cost_aware_replay_authorised": airfrans_proxy.available_for_all_cases
            or drivaerml_proxy.available_for_all_cases,
            "reason": (
                "At least one per-case proxy is available for every compact replay case. "
                "AirfRANS uses observed local case file size when available; DrivAerML uses "
                "derived geometry complexity because observed files are cached for only a subset."
            ),
        },
        "datasets": {
            "airfrans": {
                "case_count": len(airfrans.case_ids),
                "proxy_kind": airfrans_proxy.kind,
                "proxy_source": airfrans_proxy.source,
                "proxy_description": airfrans_proxy.description,
                "proxy_evidence": airfrans_proxy.evidence,
                "local_mesh_count_sample": _sample_airfrans_mesh_counts(airfrans_processed_root),
            },
            "drivaerml": {
                "case_count": len(drivaerml.case_ids),
                "proxy_kind": drivaerml_proxy.kind,
                "proxy_source": drivaerml_proxy.source,
                "proxy_description": drivaerml_proxy.description,
                "proxy_evidence": drivaerml_proxy.evidence,
                "cached_stl_summary": _summarise_sizes(stls),
                "cached_boundary_vtp_summary": _summarise_sizes(boundaries),
                "cached_stl_triangle_sample": stl_triangles,
            },
        },
    }
    atomic_write_json(out, payload)
    return out


def _select_cost_aware_batch(
    *,
    ensemble: RidgeEnsemble,
    features: FloatArray,
    costs: FloatArray,
    labelled_indices: IndexArray,
    pool_indices: IndexArray,
    batch_size: int,
    max_labels: int,
) -> IndexArray:
    count = min(batch_size, int(pool_indices.shape[0]))
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    pool_features = features[pool_indices]
    x_scaled = ensemble.x_scaler.transform(features)
    diversity = _diversity_score(x_scaled[pool_indices], x_scaled[labelled_indices])
    uncertainty = ensemble.uncertainty(pool_features)
    prediction = ensemble.predict(pool_features)
    efficiency = prediction[:, 1] / np.maximum(prediction[:, 0], EPSILON)
    labelled_prediction = ensemble.predict(features[labelled_indices])
    labelled_efficiency = labelled_prediction[:, 1] / np.maximum(
        labelled_prediction[:, 0],
        EPSILON,
    )
    current_best = float(labelled_efficiency.max())
    expected_improvement = np.maximum(efficiency - current_best, 0.0)
    pareto_relevance = _pareto_mask(prediction[:, 0], prediction[:, 1]).astype(np.float64)
    ood_penalty = np.maximum(_safe_normalize(diversity) - 0.85, 0.0) / 0.15
    progress = min(1.0, float(labelled_indices.shape[0]) / float(max_labels))
    exploration_score = (
        0.45 * _safe_normalize(diversity)
        + 0.35 * _safe_normalize(uncertainty)
        + 0.20 * pareto_relevance
    )
    exploitation_score = (
        0.20 * _safe_normalize(uncertainty)
        + 0.10 * _safe_normalize(diversity)
        + 0.20 * pareto_relevance
        + 0.30 * _safe_normalize(efficiency)
        + 0.20 * _safe_normalize(expected_improvement)
    )
    decision_score = (1.0 - progress) * exploration_score + progress * exploitation_score
    decision_score = decision_score - 0.05 * np.clip(ood_penalty, 0.0, 1.0)
    cost_penalty = _safe_normalize(costs[pool_indices])
    score = decision_score - COST_AWARE_LAMBDA * cost_penalty
    selected_local = np.argsort(score)[-count:]
    return cast("IndexArray", np.sort(pool_indices[selected_local]))


def _select_batch_with_cost(
    *,
    method: str,
    ensemble: RidgeEnsemble,
    features: FloatArray,
    costs: FloatArray,
    labelled_indices: IndexArray,
    pool_indices: IndexArray,
    batch_size: int,
    max_labels: int,
    rng: np.random.Generator,
) -> IndexArray:
    if method == COST_AWARE_METHOD:
        return _select_cost_aware_batch(
            ensemble=ensemble,
            features=features,
            costs=costs,
            labelled_indices=labelled_indices,
            pool_indices=pool_indices,
            batch_size=batch_size,
            max_labels=max_labels,
        )
    return _select_batch(
        method=method,
        ensemble=ensemble,
        features=features,
        labelled_indices=labelled_indices,
        pool_indices=pool_indices,
        batch_size=batch_size,
        max_labels=max_labels,
        rng=rng,
    )


def _add_cost_metrics(metrics: dict[str, float], cumulative_cost: float) -> dict[str, float]:
    output = dict(metrics)
    output["cumulative_cost_proxy"] = float(cumulative_cost)
    denominator = max(cumulative_cost, EPSILON)
    output["rmse_cd_per_cost_proxy"] = metrics["rmse_cd"] / denominator
    output["rmse_cl_per_cost_proxy"] = metrics["rmse_cl"] / denominator
    output["top_k_overlap_per_cost_proxy"] = metrics["top_k_efficiency_overlap"] / denominator
    output["pareto_recall_per_cost_proxy"] = metrics["pareto_recall"] / denominator
    output["regret_per_cost_proxy"] = metrics["best_design_regret"] / denominator
    return output


def _run_cost_aware_replay(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    costs: FloatArray,
    *,
    split_mode: str,
) -> dict[str, Any]:
    split = split_dataset_by_mode(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.seed,
        split_mode=split_mode,
    )
    if split.train_pool_indices.shape[0] < config.max_labels:
        msg = "train/pool split is smaller than max_labels"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    final_by_method: dict[str, dict[str, float]] = {}
    for method in config.acquisition_methods:
        rng = np.random.default_rng(config.seed + _stable_method_seed(method))
        labelled = cast(
            "IndexArray",
            np.sort(
                rng.choice(
                    split.train_pool_indices,
                    size=int(config.initial_labels),
                    replace=False,
                ),
            ),
        )
        pool = cast("IndexArray", np.setdiff1d(split.train_pool_indices, labelled))
        round_idx = 0
        while True:
            ensemble = fit_ridge_ensemble(
                dataset.features[labelled],
                dataset.targets[labelled],
                members=int(config.ensemble_members),
                seed=config.seed + round_idx + len(method),
            )
            predictions = ensemble.predict(dataset.features[split.test_indices])
            metrics = evaluate_predictions(dataset.targets[split.test_indices], predictions)
            cumulative_cost = float(costs[labelled].sum())
            cost_metrics = _add_cost_metrics(metrics, cumulative_cost)
            record = {
                "method": method,
                "round": round_idx,
                "label_count": int(labelled.shape[0]),
                "cumulative_cost_proxy": cumulative_cost,
                "mean_label_cost_proxy": float(costs[labelled].mean()),
                **cost_metrics,
            }
            records.append(record)
            final_by_method[method] = cost_metrics
            if int(labelled.shape[0]) >= int(config.max_labels) or pool.shape[0] == 0:
                break
            new_indices = _select_batch_with_cost(
                method=method,
                ensemble=ensemble,
                features=dataset.features,
                costs=costs,
                labelled_indices=labelled,
                pool_indices=pool,
                batch_size=int(config.acquisition_batch),
                max_labels=int(config.max_labels),
                rng=rng,
            )
            labelled = cast("IndexArray", np.sort(np.concatenate([labelled, new_indices])))
            pool = cast("IndexArray", np.setdiff1d(pool, new_indices, assume_unique=True))
            round_idx += 1

    return {
        "split": {
            "mode": split_mode,
            "train_pool_count": int(split.train_pool_indices.shape[0]),
            "test_count": int(split.test_indices.shape[0]),
        },
        "records": records,
        "final_metrics_by_method": final_by_method,
        "cost_aware_method_available": COST_AWARE_METHOD in config.acquisition_methods,
    }


def _aggregate_final_metrics(
    seed_reports: list[dict[str, Any]],
    *,
    reducer: Callable[[FloatArray], np.float64],
) -> dict[str, dict[str, float]]:
    methods = sorted(seed_reports[0]["final_metrics_by_method"])
    result: dict[str, dict[str, float]] = {}
    for method in methods:
        metric_names = sorted(seed_reports[0]["final_metrics_by_method"][method])
        result[method] = {}
        for metric_name in metric_names:
            values = np.asarray(
                [
                    float(report["final_metrics_by_method"][method][metric_name])
                    for report in seed_reports
                ],
                dtype=np.float64,
            )
            result[method][metric_name] = float(reducer(values))
    return result


def _run_cost_aware_replay_suite(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    costs: FloatArray,
    *,
    split_mode: str,
) -> dict[str, Any]:
    replay_seeds = list(config.replay_seeds or (config.seed,))
    seed_reports = [
        _run_cost_aware_replay(
            dataset,
            config.model_copy(update={"seed": int(seed)}),
            costs,
            split_mode=split_mode,
        )
        for seed in replay_seeds
    ]
    seed_records: list[dict[str, Any]] = []
    for seed, report in zip(replay_seeds, seed_reports, strict=True):
        seed_records.extend({"seed": int(seed), **record} for record in report["records"])
    if len(seed_reports) == 1:
        payload = seed_reports[0]
        payload["seed_count"] = 1
        payload["replay_seeds"] = replay_seeds
        payload["seed_records"] = seed_records
        return payload
    first = seed_reports[0]
    return {
        "split": first["split"],
        "records": _mean_records(seed_records),
        "seed_records": seed_records,
        "seed_count": len(replay_seeds),
        "replay_seeds": replay_seeds,
        "final_metrics_by_method": _aggregate_final_metrics(seed_reports, reducer=np.mean),
        "final_metric_std_by_method": _aggregate_final_metrics(seed_reports, reducer=np.std),
        "cost_aware_method_available": first["cost_aware_method_available"],
    }


def _cost_metric_winners(final_metrics: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    metric_specs = {
        "cumulative_cost_proxy": True,
        "rmse_cd_per_cost_proxy": True,
        "rmse_cl_per_cost_proxy": True,
        "top_k_overlap_per_cost_proxy": False,
        "pareto_recall_per_cost_proxy": False,
        "regret_per_cost_proxy": True,
    }
    winners: dict[str, list[str]] = {}
    for metric, lower_is_better in metric_specs.items():
        values = {method: final_metrics[method][metric] for method in final_metrics}
        best = min(values.values()) if lower_is_better else max(values.values())
        winners[metric] = [
            method
            for method, value in values.items()
            if math.isclose(value, best, rel_tol=1.0e-9, abs_tol=1.0e-12)
        ]
    return winners


def _cost_learning_curve_area(seed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in seed_records:
        grouped.setdefault((int(record["seed"]), str(record["method"])), []).append(record)
    values_by_method_metric: dict[tuple[str, str], list[float]] = {}
    for (_seed, method), records in grouped.items():
        ordered = sorted(records, key=lambda item: float(item["cumulative_cost_proxy"]))
        x = np.asarray([float(item["cumulative_cost_proxy"]) for item in ordered], dtype=np.float64)
        for metric in DECISION_METRICS:
            y = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
            values_by_method_metric.setdefault((method, metric), []).append(
                float(np.trapezoid(y, x)),
            )
    rows: list[dict[str, Any]] = []
    for (method, metric), values in sorted(values_by_method_metric.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                "n": int(array.size),
            },
        )
    return rows


def run_cost_aware_decision_replay_v05(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    cost_proxy: CostProxy,
    *,
    split_modes: tuple[str, ...] = ("map_completion", "geometry_heldout"),
) -> dict[str, Any]:
    """Run cost-proxy-aware replay for one compact AeroMap dataset."""

    if cost_proxy.values.shape[0] != dataset.features.shape[0]:
        msg = "cost proxy count does not match dataset case count"
        raise ValueError(msg)
    split_reports = {
        split_mode: _run_cost_aware_replay_suite(
            dataset,
            config,
            cost_proxy.values,
            split_mode=split_mode,
        )
        for split_mode in split_modes
    }
    final_metrics = {
        split_mode: report["final_metrics_by_method"]
        for split_mode, report in split_reports.items()
    }
    return {
        "schema_version": AEROMAP_COST_REPLAY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AEROMAP_V0_5_COST_PROXY_AWARE_REPLAY",
        "dataset_classification": dataset.classification,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
            "f1_geometry": False,
            "live_cfd_savings": False,
            "field_prediction": False,
            "costs_are_proxy_values": True,
            "active_learning_claim": "offline_cost_proxy_aware_budget_replay",
        },
        "cost_proxy": {
            "kind": cost_proxy.kind,
            "source": cost_proxy.source,
            "description": cost_proxy.description,
            "available_for_all_cases": cost_proxy.available_for_all_cases,
            "lambda": COST_AWARE_LAMBDA,
            "policy_formula": (
                "engineering_decision_utility_v2_score - "
                f"{COST_AWARE_LAMBDA} * minmax_normalised_cost_proxy"
            ),
            "value_min": float(cost_proxy.values.min()),
            "value_median": float(np.median(cost_proxy.values)),
            "value_max": float(cost_proxy.values.max()),
            "evidence": cost_proxy.evidence,
        },
        "config": config.model_dump(),
        "split_modes": list(split_modes),
        "method_winners": {
            split_mode: _decision_metric_winners(metrics)
            for split_mode, metrics in final_metrics.items()
        },
        "cost_metric_winners": {
            split_mode: _cost_metric_winners(metrics)
            for split_mode, metrics in final_metrics.items()
        },
        "statistics": {
            split_mode: {
                **_split_replay_statistics(report),
                "cost_learning_curve_area": _cost_learning_curve_area(_report_seed_records(report)),
                "standard_label_curve_area": _learning_curve_area_statistics(
                    _report_seed_records(report),
                ),
            }
            for split_mode, report in split_reports.items()
        },
        "split_reports": split_reports,
    }


def write_cost_aware_decision_replay_v05(
    *,
    dataset_npz: Path,
    config: AeroMapConfig,
    dataset_name: str,
    out: Path,
    airfrans_processed_root: Path = Path("artifacts/benchmark/airfrans/processed"),
) -> Path:
    dataset = load_dataset_npz(dataset_npz)
    cost_proxy = _dataset_cost_proxy(
        dataset,
        dataset_name=dataset_name,
        airfrans_processed_root=airfrans_processed_root,
    )
    payload = run_cost_aware_decision_replay_v05(dataset, config, cost_proxy)
    atomic_write_json(out, payload)
    return out


def _field_array_stats(values: FloatArray, weights: FloatArray) -> dict[str, float]:
    weight_sum = float(weights.sum())
    if weight_sum <= EPSILON:
        weights = np.ones_like(values, dtype=np.float64)
        weight_sum = float(weights.sum())
    mean = float(np.sum(values * weights) / weight_sum)
    centered = values - mean
    rms = float(np.sqrt(np.sum(centered**2 * weights) / weight_sum))
    return {
        "weighted_mean": mean,
        "weighted_rms_about_mean": rms,
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _bin_field_by_axis(
    centers: FloatArray,
    values: FloatArray,
    weights: FloatArray,
    *,
    axis: int,
    bins: int,
) -> list[dict[str, float]]:
    coord = centers[:, axis]
    edges = np.linspace(float(coord.min()), float(coord.max()), bins + 1)
    rows: list[dict[str, float]] = []
    for idx in range(bins):
        if idx == bins - 1:
            mask = (coord >= edges[idx]) & (coord <= edges[idx + 1])
        else:
            mask = (coord >= edges[idx]) & (coord < edges[idx + 1])
        if not bool(np.any(mask)):
            continue
        stats = _field_array_stats(values[mask], weights[mask])
        rows.append(
            {
                "bin": idx,
                "coord_min": float(edges[idx]),
                "coord_max": float(edges[idx + 1]),
                "cell_count": int(mask.sum()),
                "area": float(weights[mask].sum()),
                **stats,
            },
        )
    return rows


def _write_surface_sample_png(
    *,
    centers: FloatArray,
    values: FloatArray,
    out: Path,
    title: str,
) -> None:
    rng = np.random.default_rng(20260629)
    count = min(SURFACE_SAMPLE_POINTS, centers.shape[0])
    selected = rng.choice(np.arange(centers.shape[0]), size=count, replace=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    scatter = ax.scatter(
        centers[selected, 0],
        centers[selected, 1],
        c=values[selected],
        s=0.5,
        cmap="coolwarm",
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=ax, label="CpMeanTrim")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _surface_field_case_summary(path: Path, *, visual_out: Path | None) -> dict[str, Any]:
    mesh = pv.read(path)
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    centers = cast("FloatArray", np.asarray(mesh.cell_centers().points, dtype=np.float64))
    fields = sorted(mesh.cell_data.keys())
    summary: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "cell_fields": fields,
        "point_fields": sorted(mesh.point_data.keys()),
        "area_total": float(areas.sum()),
    }
    for field_name in ("CpMeanTrim", "pMeanTrim", "pPrime2MeanTrim"):
        if field_name in mesh.cell_data:
            values = np.asarray(mesh.cell_data[field_name], dtype=np.float64)
            summary[f"{field_name}_stats"] = _field_array_stats(values, areas)
            summary[f"{field_name}_streamwise_bins"] = _bin_field_by_axis(
                centers,
                values,
                areas,
                axis=0,
                bins=8,
            )
            summary[f"{field_name}_left_right_bins"] = _bin_field_by_axis(
                centers,
                values,
                areas,
                axis=1,
                bins=2,
            )
    if "CpMeanTrim" in mesh.cell_data and visual_out is not None:
        values = np.asarray(mesh.cell_data["CpMeanTrim"], dtype=np.float64)
        _write_surface_sample_png(
            centers=centers,
            values=values,
            out=visual_out,
            title=f"{path.parent.name} cached DrivAerML Cp readiness sample",
        )
        summary["visual_png"] = str(visual_out)
    return summary


def write_surface_field_feasibility_precheck(
    *,
    drivaerml_cache_root: Path,
    out: Path,
    visual_dir: Path,
    max_cases: int = 1,
) -> Path:
    """Inspect cached DrivAerML boundary VTPs without downloading boundary data."""

    boundary_paths = sorted(drivaerml_cache_root.glob("run_*/boundary_*.vtp"))
    selected = boundary_paths[: min(max_cases, MAX_SURFACE_FIELD_CASES)]
    case_summaries: list[dict[str, Any]] = []
    for path in selected:
        visual_out = visual_dir / f"{path.parent.name}_cp_readiness.png"
        case_summaries.append(_surface_field_case_summary(path, visual_out=visual_out))
    payload = {
        "schema_version": AEROMAP_SURFACE_FIELD_READINESS_SCHEMA,
        "classification": (
            "AEROMAP_3D_SURFACE_FIELD_READINESS_SAMPLE"
            if case_summaries
            else "AEROMAP_3D_SURFACE_FIELD_METADATA_ONLY_PRECHECK"
        ),
        "claim_boundary": {
            "new_downloads": False,
            "boundary_fields_downloaded_in_this_goal": False,
            "volume_fields_downloaded": False,
            "field_prediction": False,
            "aerocliff_accuracy": False,
            "f1_accuracy": False,
            "surface_field_readiness_only": True,
        },
        "cache_root": str(drivaerml_cache_root),
        "cached_boundary_vtp_count": len(boundary_paths),
        "inspected_case_count": len(case_summaries),
        "inspected_cases": case_summaries,
        "decision": {
            "field_region_mini_proof_authorised_without_download": bool(case_summaries),
            "next_step_if_approved": (
                "Use a separately approved small boundary-field subset for region-aware "
                "surface metrics; do not infer field-prediction accuracy from this "
                "readiness sample."
            ),
        },
    }
    atomic_write_json(out, payload)
    return out


def write_cost_aware_report(
    *,
    airfrans_report: Path,
    drivaerml_report: Path,
    surface_report: Path,
    out: Path,
) -> Path:
    airfrans = json.loads(airfrans_report.read_text(encoding="utf-8"))
    drivaerml = json.loads(drivaerml_report.read_text(encoding="utf-8"))
    surface = json.loads(surface_report.read_text(encoding="utf-8"))
    lines = [
        "# AeroMap v0.5 Cost-Aware Acquisition and 3D Field Feasibility",
        "",
        "## Classification",
        "",
        "`AEROMAP_V0_5_COST_PROXY_AWARE_REPLAY` and `AEROMAP_3D_SURFACE_FIELD_READINESS_SAMPLE`.",
        "",
        "## Claim Boundaries",
        "",
        "- No CFD was run.",
        "- No EC2, NIM, cloud or new dataset download was used.",
        "- Cost values are proxies, not measured live solver savings.",
        "- DrivAerML boundary fields were inspected only because they were already cached locally.",
        "- No AeroCliff, F1, DoMINO accuracy or field-prediction claim is made.",
        "",
        "## Cost Proxy Sources",
        "",
        f"- AirfRANS: `{airfrans['cost_proxy']['kind']}` from {airfrans['cost_proxy']['source']}.",
        f"- DrivAerML: `{drivaerml['cost_proxy']['kind']}` from "
        f"{drivaerml['cost_proxy']['source']}.",
        "",
        "## Geometry-Heldout Winners",
        "",
        "AirfRANS final decision winners:",
        "",
        "```json",
        json.dumps(airfrans["method_winners"].get("geometry_heldout", {}), indent=2),
        "```",
        "",
        "AirfRANS cost-normalised winners:",
        "",
        "```json",
        json.dumps(airfrans["cost_metric_winners"].get("geometry_heldout", {}), indent=2),
        "```",
        "",
        "DrivAerML compact 3D scalar final decision winners:",
        "",
        "```json",
        json.dumps(drivaerml["method_winners"].get("geometry_heldout", {}), indent=2),
        "```",
        "",
        "DrivAerML compact 3D scalar cost-normalised winners:",
        "",
        "```json",
        json.dumps(drivaerml["cost_metric_winners"].get("geometry_heldout", {}), indent=2),
        "```",
        "",
        "## 3D Surface-Field Feasibility",
        "",
        f"- Classification: `{surface['classification']}`.",
        f"- Cached boundary VTPs found: {surface['cached_boundary_vtp_count']}.",
        f"- Cached boundary VTPs inspected: {surface['inspected_case_count']}.",
        "",
        "This proves local 3D surface-field ingestion/readiness only. It is not a surface-field "
        "prediction benchmark.",
        "",
    ]
    atomic_write_text(out, "\n".join(lines))
    return out

"""AirfRANS surface-pressure field baseline.

This module keeps the first field-level result intentionally small: one
point-wise MLP baseline, two transparent baselines and committed summary
artifacts only. The bulky AirfRANS VTK files stay in the local cache.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.benchmarks.aeromap import (
    AIRFRANS_CITATION_KEY,
    AIRFRANS_DOCS_URL,
    AIRFRANS_LICENSE,
    _airfoil_geometry_descriptor,
    _airfrans_manifest_path,
    airfrans_case_features_from_name,
)
from aeromap.io import atomic_write_json

FloatArray = NDArray[np.float64]

AIRFRANS_FIELD_BASELINE_SCHEMA = "aeromap_airfrans_surface_pressure_field_baseline_v0.1.0"
AIRFRANS_FIELD_CLASSIFICATION = "AEROMAP_AIRFRANS_SURFACE_PRESSURE_FIELD_BASELINE_V0_1"
DEFAULT_TRAIN_CASES = 80
DEFAULT_VAL_CASES = 16
DEFAULT_TEST_CASES = 32
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 8192
DEFAULT_HIDDEN_WIDTH = 64
DEFAULT_SEED = 20260630
VISUAL_PANEL_CASE_COUNT = 3
EPSILON = 1.0e-12


@dataclass(frozen=True)
class FieldCase:
    """Surface-pressure arrays for one AirfRANS case."""

    case_id: str
    features: FloatArray
    target: FloatArray
    weights: FloatArray
    xy: FloatArray
    case_feature_vector: FloatArray


@dataclass(frozen=True)
class FieldTable:
    """Flattened field dataset plus case slices."""

    features: FloatArray
    target: FloatArray
    weights: FloatArray
    case_ids: list[str]
    case_slices: dict[str, tuple[int, int]]
    xy_by_case: dict[str, FloatArray]
    target_by_case: dict[str, FloatArray]
    case_features: dict[str, FloatArray]


@dataclass(frozen=True)
class Standardizer:
    """Simple train-only affine standardizer."""

    mean: FloatArray
    scale: FloatArray

    def transform(self, values: FloatArray) -> FloatArray:
        return (values - self.mean) / self.scale

    def inverse_target(self, values: FloatArray) -> FloatArray:
        return values * self.scale + self.mean


class SurfacePressureMLP(torch.nn.Module):
    """Small point-wise MLP for surface pressure."""

    def __init__(self, input_dim: int, *, hidden_width: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.net(features).squeeze(-1))


def _load_manifest(root: Path) -> dict[str, list[str]]:
    manifest_path = _airfrans_manifest_path(root)
    return cast("dict[str, list[str]]", json.loads(manifest_path.read_text(encoding="utf-8")))


def _case_path(dataset_root: Path, case_id: str) -> Path:
    path = dataset_root / case_id / f"{case_id}_aerofoil.vtp"
    if not path.exists():
        msg = f"AirfRANS aerofoil surface file is missing for {case_id}"
        raise FileNotFoundError(msg)
    return path


def _dynamic_pressure_kinematic(case_id: str) -> float:
    inlet_velocity, _names = airfrans_case_features_from_name(case_id)
    u_inf = inlet_velocity[0]
    return 0.5 * u_inf**2


def _load_field_case(dataset_root: Path, case_id: str) -> FieldCase:
    path = _case_path(dataset_root, case_id)
    mesh = pv.read(path).compute_cell_sizes(area=False, volume=False)
    centers = np.asarray(mesh.cell_centers().points[:, :2], dtype=np.float64)
    if "p" not in mesh.cell_data or "Normals" not in mesh.cell_data:
        msg = f"AirfRANS aerofoil file lacks required cell fields for {case_id}"
        raise ValueError(msg)
    if "Length" not in mesh.cell_data:
        msg = f"AirfRANS aerofoil file lacks cell Length weights for {case_id}"
        raise ValueError(msg)
    normals = np.asarray(mesh.cell_data["Normals"][:, :2], dtype=np.float64)
    pressure = np.asarray(mesh.cell_data["p"], dtype=np.float64)
    weights = np.asarray(mesh.cell_data["Length"], dtype=np.float64)
    cp_like = pressure / max(_dynamic_pressure_kinematic(case_id), EPSILON)

    case_features, _base_names = airfrans_case_features_from_name(case_id)
    geometry_features, _geometry_names = _airfoil_geometry_descriptor(path)
    case_feature_vector = np.asarray([*case_features, *geometry_features], dtype=np.float64)
    repeated_case_features = np.repeat(case_feature_vector.reshape(1, -1), centers.shape[0], axis=0)
    features = np.column_stack([centers, normals, repeated_case_features]).astype(np.float64)
    return FieldCase(
        case_id=case_id,
        features=features,
        target=cp_like.astype(np.float64),
        weights=weights.astype(np.float64),
        xy=centers.astype(np.float64),
        case_feature_vector=case_feature_vector,
    )


def _build_table(cases: list[FieldCase]) -> FieldTable:
    features = np.concatenate([case.features for case in cases], axis=0)
    target = np.concatenate([case.target for case in cases], axis=0)
    weights = np.concatenate([case.weights for case in cases], axis=0)
    case_slices: dict[str, tuple[int, int]] = {}
    start = 0
    for case in cases:
        end = start + case.target.shape[0]
        case_slices[case.case_id] = (start, end)
        start = end
    return FieldTable(
        features=features,
        target=target,
        weights=weights,
        case_ids=[case.case_id for case in cases],
        case_slices=case_slices,
        xy_by_case={case.case_id: case.xy for case in cases},
        target_by_case={case.case_id: case.target for case in cases},
        case_features={case.case_id: case.case_feature_vector for case in cases},
    )


def _standardizer(values: FloatArray) -> Standardizer:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < EPSILON, 1.0, scale)
    return Standardizer(mean=mean.astype(np.float64), scale=scale.astype(np.float64))


def _target_standardizer(values: FloatArray) -> Standardizer:
    mean = np.asarray(values.mean(), dtype=np.float64)
    scale = np.asarray(max(float(values.std()), EPSILON), dtype=np.float64)
    return Standardizer(mean=mean.reshape(()), scale=scale.reshape(()))


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    return float(np.sum(values * weights) / max(float(np.sum(weights)), EPSILON))


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * float(cumulative[-1])
    return float(sorted_values[np.searchsorted(cumulative, threshold, side="left")])


def _metrics(y_true: FloatArray, y_pred: FloatArray, weights: FloatArray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = _weighted_mean(abs_err, weights)
    rmse = math.sqrt(_weighted_mean(err**2, weights))
    denom = max(
        _weighted_quantile(y_true, weights, 0.95) - _weighted_quantile(y_true, weights, 0.05),
        EPSILON,
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nrmse_p95_p05": float(rmse / denom),
        "max_abs_error": float(abs_err.max()),
    }


def _per_case_metrics(table: FieldTable, predictions: FloatArray) -> list[dict[str, Any]]:
    records = []
    for case_id in table.case_ids:
        start, end = table.case_slices[case_id]
        metrics = _metrics(
            table.target[start:end], predictions[start:end], table.weights[start:end]
        )
        records.append({"case_id": case_id, **metrics})
    return records


def _fit_mlp(
    train: FieldTable,
    val: FieldTable,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_width: int,
) -> tuple[SurfacePressureMLP, Standardizer, Standardizer, dict[str, float]]:
    torch.manual_seed(seed)
    x_scaler = _standardizer(train.features)
    y_scaler = _target_standardizer(train.target)
    x_train = torch.tensor(x_scaler.transform(train.features), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(train.target), dtype=torch.float32)
    weight = torch.tensor(
        train.weights / max(float(train.weights.mean()), EPSILON), dtype=torch.float32
    )
    x_val = torch.tensor(x_scaler.transform(val.features), dtype=torch.float32)
    y_val = torch.tensor(y_scaler.transform(val.target), dtype=torch.float32)

    model = SurfacePressureMLP(train.features.shape[1], hidden_width=hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    generator = torch.Generator().manual_seed(seed)
    sample_count = x_train.shape[0]
    for _epoch in range(epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            idx = permutation[start : start + batch_size]
            pred = model(x_train[idx])
            loss = torch.mean(weight[idx] * (pred - y_train[idx]) ** 2)
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
    with torch.no_grad():
        train_loss = float(torch.mean((model(x_train) - y_train) ** 2).item())
        val_loss = float(torch.mean((model(x_val) - y_val) ** 2).item())
    return (
        model,
        x_scaler,
        y_scaler,
        {"train_mse_standardized": train_loss, "val_mse_standardized": val_loss},
    )


def _predict_mlp(
    model: SurfacePressureMLP,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    table: FieldTable,
) -> FloatArray:
    with torch.no_grad():
        x = torch.tensor(x_scaler.transform(table.features), dtype=torch.float32)
        pred_scaled = model(x).numpy().astype(np.float64)
    return y_scaler.inverse_target(pred_scaled)


def _nearest_case_predictions(
    train: FieldTable, test: FieldTable
) -> tuple[FloatArray, dict[str, str]]:
    train_case_ids = train.case_ids
    train_matrix = np.vstack([train.case_features[case_id] for case_id in train_case_ids])
    scaler = _standardizer(train_matrix)
    train_scaled = scaler.transform(train_matrix)
    predictions: list[FloatArray] = []
    mapping: dict[str, str] = {}
    for test_case_id in test.case_ids:
        test_scaled = scaler.transform(test.case_features[test_case_id].reshape(1, -1))
        distances = np.linalg.norm(train_scaled - test_scaled, axis=1)
        nearest_id = train_case_ids[int(np.argmin(distances))]
        mapping[test_case_id] = nearest_id
        tree = cKDTree(train.xy_by_case[nearest_id])
        _dist, idx = tree.query(test.xy_by_case[test_case_id])
        predictions.append(train.target_by_case[nearest_id][idx].astype(np.float64))
    return np.concatenate(predictions).astype(np.float64), mapping


def _write_visual_panel(
    *,
    table: FieldTable,
    predictions: FloatArray,
    per_case: list[dict[str, Any]],
    out: Path,
) -> None:
    ranked = sorted(per_case, key=lambda item: float(item["rmse"]))
    candidates = (
        [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        if len(ranked) >= VISUAL_PANEL_CASE_COUNT
        else ranked
    )
    rows = len(candidates)
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, 3.0 * rows), dpi=160, squeeze=False)
    for row, record in enumerate(candidates):
        case_id = str(record["case_id"])
        start, end = table.case_slices[case_id]
        xy = table.xy_by_case[case_id]
        true = table.target[start:end]
        pred = predictions[start:end]
        err = pred - true
        values = [true, pred, err]
        titles = ["true Cp-like", "MLP prediction", "error"]
        for col, (value, title) in enumerate(zip(values, titles, strict=True)):
            ax = axes[row][col]
            scatter = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=value,
                s=4.0,
                cmap="coolwarm",
                linewidths=0.0,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}\n{case_id[:36]}", fontsize=8)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _write_summary_plot(metrics_by_method: dict[str, dict[str, float]], out: Path) -> None:
    labels = list(metrics_by_method)
    rmse = [metrics_by_method[label]["rmse"] for label in labels]
    mae = [metrics_by_method[label]["mae"] for label in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    ax.bar(x - width / 2, rmse, width, label="RMSE")
    ax.bar(x + width / 2, mae, width, label="MAE")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Cp-like error")
    ax.set_title("AirfRANS surface pressure baseline")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run_airfrans_surface_pressure_baseline(
    *,
    root: Path,
    out: Path,
    visual_out: Path,
    summary_plot_out: Path,
    train_cases: int = DEFAULT_TRAIN_CASES,
    val_cases: int = DEFAULT_VAL_CASES,
    test_cases: int = DEFAULT_TEST_CASES,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Train and evaluate a compact AirfRANS surface-pressure baseline."""

    manifest = _load_manifest(root)
    dataset_root = _airfrans_manifest_path(root).parent
    full_train = list(manifest["full_train"])
    full_test = list(manifest["full_test"])
    if len(full_train) < train_cases + val_cases:
        msg = "not enough AirfRANS full_train cases for requested train/validation split"
        raise ValueError(msg)
    if len(full_test) < test_cases:
        msg = "not enough AirfRANS full_test cases for requested test split"
        raise ValueError(msg)
    train_ids = full_train[:train_cases]
    val_ids = full_train[train_cases : train_cases + val_cases]
    test_ids = full_test[:test_cases]

    train_table = _build_table([_load_field_case(dataset_root, case_id) for case_id in train_ids])
    val_table = _build_table([_load_field_case(dataset_root, case_id) for case_id in val_ids])
    test_table = _build_table([_load_field_case(dataset_root, case_id) for case_id in test_ids])

    train_mean = _weighted_mean(train_table.target, train_table.weights)
    mean_pred = np.full_like(test_table.target, train_mean, dtype=np.float64)
    nearest_pred, nearest_mapping = _nearest_case_predictions(train_table, test_table)
    model, x_scaler, y_scaler, training_summary = _fit_mlp(
        train_table,
        val_table,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        hidden_width=hidden_width,
    )
    mlp_pred = _predict_mlp(model, x_scaler, y_scaler, test_table)

    metrics_by_method = {
        "train_mean": _metrics(test_table.target, mean_pred, test_table.weights),
        "nearest_case": _metrics(test_table.target, nearest_pred, test_table.weights),
        "pointwise_mlp": _metrics(test_table.target, mlp_pred, test_table.weights),
    }
    per_case_mlp = _per_case_metrics(test_table, mlp_pred)
    best_method_by_rmse = min(
        metrics_by_method, key=lambda method: metrics_by_method[method]["rmse"]
    )
    _write_visual_panel(
        table=test_table, predictions=mlp_pred, per_case=per_case_mlp, out=visual_out
    )
    _write_summary_plot(metrics_by_method, summary_plot_out)

    payload = {
        "schema_version": AIRFRANS_FIELD_BASELINE_SCHEMA,
        "classification": AIRFRANS_FIELD_CLASSIFICATION,
        "source": {
            "dataset": "AirfRANS",
            "license": AIRFRANS_LICENSE,
            "citation_key": AIRFRANS_CITATION_KEY,
            "docs_url": AIRFRANS_DOCS_URL,
            "local_source": "local AirfRANS processed cache; bulky VTK files are not committed",
        },
        "field_contract": {
            "surface_file": "<case>/<case>_aerofoil.vtp",
            "target_raw_field": "cell_data['p']",
            "target": "surface_pressure_coefficient_like",
            "target_formula": "Cp_like = p / (0.5 * U_inf**2)",
            "normalisation_basis": (
                "matches the AirfRANS package convention used for pressure in boundary_layer() "
                "with compressible=False"
            ),
            "inputs": [
                "surface cell center x",
                "surface cell center y",
                "surface normal x",
                "surface normal y",
                "case operating-condition features",
                "compact airfoil geometry descriptors",
            ],
            "weights": "surface cell Length; metrics are length-weighted",
        },
        "split": {
            "source_manifest": "AirfRANS full_train/full_test",
            "train_cases": train_cases,
            "validation_cases": val_cases,
            "test_cases": test_cases,
            "train_surface_points": int(train_table.target.shape[0]),
            "validation_surface_points": int(val_table.target.shape[0]),
            "test_surface_points": int(test_table.target.shape[0]),
            "train_case_ids_sha256_prefix": _case_id_hash(train_ids),
            "validation_case_ids_sha256_prefix": _case_id_hash(val_ids),
            "test_case_ids_sha256_prefix": _case_id_hash(test_ids),
        },
        "model": {
            "kind": "pointwise_mlp",
            "framework": "PyTorch",
            "hidden_width": hidden_width,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
            "feature_normalisation": "fit on training surface points only",
            "target_normalisation": "fit on training surface targets only",
            "training_summary": training_summary,
        },
        "baselines": {
            "train_mean": "length-weighted mean pressure coefficient from training surface points",
            "nearest_case": (
                "nearest training case in standardised case-feature space, then nearest "
                "surface coordinate on that case"
            ),
        },
        "metrics": {
            "formulae": {
                "mae": "sum_i w_i |prediction_i - target_i| / sum_i w_i",
                "rmse": "sqrt(sum_i w_i (prediction_i - target_i)^2 / sum_i w_i)",
                "nrmse_p95_p05": "rmse / (weighted_p95(target) - weighted_p05(target))",
            },
            "by_method": metrics_by_method,
            "best_method_by_rmse": best_method_by_rmse,
            "pointwise_mlp_per_case": per_case_mlp,
            "nearest_case_mapping_sample": dict(list(nearest_mapping.items())[:10]),
        },
        "artifacts": {
            "visual_panel": str(visual_out),
            "summary_plot": str(summary_plot_out),
        },
        "claim_boundary": {
            "field_level_baseline": True,
            "open_cfd_result": True,
            "airfrans_surface_pressure_only": True,
            "not_sota": True,
            "not_f1_geometry": True,
            "not_aerocliff_accuracy": True,
            "not_external_predictor_replacement": True,
            "not_live_cfd_savings": True,
        },
    }
    atomic_write_json(out, payload)
    return out


def _case_id_hash(case_ids: list[str]) -> str:
    encoded = "\n".join(case_ids).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]

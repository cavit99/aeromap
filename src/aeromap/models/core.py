"""Small project-owned PyTorch corrector core for interface smoke tests."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

CHECKPOINT_SCHEMA_VERSION = "aeromap_corrector_checkpoint_v0.1.0"
FEATURE_TENSOR_RANK = 2
VECTOR_FIELD_TENSOR_RANK = 2
CARTESIAN_VECTOR_WIDTH = 3
COLUMN_VECTOR_WIDTH = 1


@dataclass(frozen=True)
class CorrectorCoreConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 32
    depth: int = 2

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            msg = "input_dim and output_dim must be positive"
            raise ValueError(msg)
        if self.hidden_dim <= 0:
            msg = "hidden_dim must be positive"
            raise ValueError(msg)
        if self.depth < 1:
            msg = "depth must be at least one"
            raise ValueError(msg)


@dataclass(frozen=True)
class SurfaceLoadCoefficients:
    """Differentiable surface-integrated load coefficients from nondimensional fields."""

    pressure_xyz: Tensor
    viscous_xyz: Tensor
    total_xyz: Tensor
    drag: Tensor
    downforce: Tensor


@dataclass(frozen=True)
class SensitivityResult:
    """Prediction plus input gradient for local project-owned sensitivity probes."""

    prediction: Tensor
    input_gradient: Tensor


class ResidualCorrectorCore(nn.Module):
    """Compact MLP that predicts residual aerodynamic fields from prepared features."""

    def __init__(self, config: CorrectorCoreConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != FEATURE_TENSOR_RANK or features.shape[-1] != self.config.input_dim:
            msg = (
                "features must have shape [n_points, input_dim]; "
                f"got {tuple(features.shape)} for input_dim={self.config.input_dim}"
            )
            raise ValueError(msg)
        output = self.network(features)
        if not isinstance(output, Tensor):
            msg = "corrector network returned a non-Tensor value"
            raise TypeError(msg)
        return output


def residual_mse_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean-squared residual loss with optional boolean/float sample mask."""

    if prediction.shape != target.shape:
        msg = f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        raise ValueError(msg)
    residual = (prediction - target).square()
    if mask is None:
        return residual.mean()
    raw_weights = mask.to(dtype=residual.dtype, device=residual.device)
    if raw_weights.shape == residual.shape:
        weights = raw_weights
    elif raw_weights.ndim == 1 and raw_weights.shape[0] == residual.shape[0]:
        weights = raw_weights.reshape(-1, *([1] * (residual.ndim - 1))).expand_as(residual)
    elif (
        raw_weights.ndim == residual.ndim
        and raw_weights.shape[:-1] == residual.shape[:-1]
        and raw_weights.shape[-1] == 1
    ):
        weights = raw_weights.expand_as(residual)
    else:
        msg = (
            "mask must have shape [n_samples], [n_samples, 1], or match prediction shape; "
            f"got {tuple(raw_weights.shape)} for prediction {tuple(prediction.shape)}"
        )
        raise ValueError(msg)
    denominator = torch.clamp(weights.sum(), min=1.0)
    return (residual * weights).sum() / denominator


def _require_vector_field(name: str, value: Tensor, *, rows: int | None = None) -> Tensor:
    if value.ndim != VECTOR_FIELD_TENSOR_RANK or value.shape[-1] != CARTESIAN_VECTOR_WIDTH:
        msg = f"{name} must have shape [n_faces, 3]; got {tuple(value.shape)}"
        raise ValueError(msg)
    if rows is not None and value.shape[0] != rows:
        msg = f"{name} row count {value.shape[0]} does not match expected {rows}"
        raise ValueError(msg)
    return value


def _require_scalar_field(name: str, value: Tensor) -> Tensor:
    if value.ndim == 1:
        return value
    if value.ndim == VECTOR_FIELD_TENSOR_RANK and value.shape[-1] == COLUMN_VECTOR_WIDTH:
        return value.reshape(-1)
    msg = f"{name} must have shape [n_faces] or [n_faces, 1]; got {tuple(value.shape)}"
    raise ValueError(msg)


def _inlet_unit_vector(yaw_deg: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    yaw = torch.tensor(math.radians(yaw_deg), device=device, dtype=dtype)
    return torch.stack(
        (
            torch.cos(yaw),
            torch.sin(yaw),
            torch.zeros((), device=device, dtype=dtype),
        ),
    )


def surface_load_coefficients(
    *,
    surface_cp: Tensor,
    surface_cf: Tensor,
    surface_normals: Tensor,
    surface_area_m2: Tensor,
    yaw_deg: float = 0.0,
    a_ref_m2: float = 2.0,
) -> SurfaceLoadCoefficients:
    """Integrate nondimensional wall fields into drag/downforce coefficients.

    This mirrors AeroCliff's OpenFOAM wall-force sign convention in project-owned PyTorch:
    pressure contributes ``C_p * area_vector / A_ref`` and viscous load contributes
    ``-C_f * area / A_ref``.
    """

    if a_ref_m2 <= 0.0:
        msg = "a_ref_m2 must be positive"
        raise ValueError(msg)
    cp = _require_scalar_field("surface_cp", surface_cp)
    cf = _require_vector_field("surface_cf", surface_cf, rows=int(cp.shape[0]))
    normals = _require_vector_field("surface_normals", surface_normals, rows=int(cp.shape[0]))
    area = _require_scalar_field("surface_area_m2", surface_area_m2)
    if area.shape[0] != cp.shape[0]:
        msg = f"surface_area_m2 row count {area.shape[0]} does not match expected {cp.shape[0]}"
        raise ValueError(msg)
    area = area.to(dtype=cp.dtype, device=cp.device)
    cf = cf.to(dtype=cp.dtype, device=cp.device)
    normals = normals.to(dtype=cp.dtype, device=cp.device)

    pressure_xyz = torch.sum(cp[:, None] * normals * area[:, None], dim=0) / a_ref_m2
    viscous_xyz = -torch.sum(cf * area[:, None], dim=0) / a_ref_m2
    total_xyz = pressure_xyz + viscous_xyz
    inlet = _inlet_unit_vector(yaw_deg, device=total_xyz.device, dtype=total_xyz.dtype)
    return SurfaceLoadCoefficients(
        pressure_xyz=pressure_xyz,
        viscous_xyz=viscous_xyz,
        total_xyz=total_xyz,
        drag=-torch.dot(total_xyz, inlet),
        downforce=-total_xyz[2],
    )


def load_coefficient_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    """MSE for drag/downforce or full load-coefficient tensors."""

    return residual_mse_loss(prediction, target, weight)


def physics_residual_loss(
    residuals: Mapping[str, Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Weighted MSE over named physics residual tensors owned by the project model lane."""

    if not residuals:
        msg = "at least one physics residual tensor is required"
        raise ValueError(msg)
    weights = weights or {}
    unknown = sorted(set(weights) - set(residuals))
    if unknown:
        msg = f"weights supplied for unknown residuals: {unknown}"
        raise ValueError(msg)

    total: Tensor | None = None
    for name in sorted(residuals):
        residual = residuals[name]
        if not torch.is_floating_point(residual):
            msg = f"physics residual {name!r} must be a floating-point tensor"
            raise TypeError(msg)
        term = residual.square().mean() * float(weights.get(name, 1.0))
        total = term if total is None else total + term
    if total is None:
        msg = "at least one physics residual tensor is required"
        raise ValueError(msg)
    return total


def corrector_input_sensitivity(
    model: nn.Module,
    features: Tensor,
    *,
    output_weights: Tensor | None = None,
    create_graph: bool = False,
) -> SensitivityResult:
    """Return d(weighted output sum)/d(features) for a project-owned corrector."""

    track_feature_gradient = True
    probe = features.detach().clone().requires_grad_(track_feature_gradient)
    prediction = model(probe)
    if not isinstance(prediction, Tensor):
        msg = "sensitivity model returned a non-Tensor value"
        raise TypeError(msg)
    weights = torch.ones_like(prediction) if output_weights is None else output_weights
    if weights.shape != prediction.shape:
        msg = (
            "output_weights must match prediction shape; "
            f"got {tuple(weights.shape)} for prediction {tuple(prediction.shape)}"
        )
        raise ValueError(msg)
    objective = (prediction * weights.to(device=prediction.device, dtype=prediction.dtype)).sum()
    gradient = torch.autograd.grad(
        objective,
        probe,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return SensitivityResult(prediction=prediction, input_gradient=gradient)


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_device_neutral_checkpoint(
    path: Path,
    *,
    model: ResidualCorrectorCore,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint whose tensors are detached onto CPU for portable reloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": asdict(model.config),
        "metadata": metadata or {},
        "model_state": _cpu_state_dict(model),
    }
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".pt",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_device_neutral_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ResidualCorrectorCore, dict[str, Any]]:
    """Load a device-neutral checkpoint and return a ready corrector core plus metadata."""

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        msg = f"unsupported checkpoint schema: {payload.get('schema_version')}"
        raise ValueError(msg)
    config = CorrectorCoreConfig(**payload["config"])
    model = ResidualCorrectorCore(config)
    model.load_state_dict(payload["model_state"])
    model.to(map_location)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        msg = "checkpoint metadata must be a mapping"
        raise TypeError(msg)
    return model, metadata

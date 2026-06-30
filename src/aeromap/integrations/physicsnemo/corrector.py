"""CUDA-only PhysicsNeMo corrector smoke with profiling and checkpoint round-trip."""

from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.nn import functional

from aeromap.integrations.physicsnemo.preflight import (
    CudaPreflightBlocker,
    build_preflight_report,
    require_cuda_preflight,
)
from aeromap.io import atomic_write_json, sha256_file

AmpDtype = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    dtype: AmpDtype = "bf16"

    def __post_init__(self) -> None:
        if self.dtype not in {"fp32", "fp16", "bf16"}:
            msg = f"unsupported CUDA mixed precision dtype: {self.dtype}"
            raise ValueError(msg)

    @property
    def autocast_enabled(self) -> bool:
        return self.dtype != "fp32"

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "fp16":
            return torch.float16
        if self.dtype == "fp32":
            return torch.float32
        msg = f"unsupported CUDA mixed precision dtype: {self.dtype}"
        raise ValueError(msg)


@dataclass(frozen=True)
class CorrectorSmokeConfig:
    input_features: int = 16
    output_features: int = 8
    hidden_features: int = 64
    hidden_layers: int = 2
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    seed: int = 1729
    amp_dtype: AmpDtype = "bf16"


@dataclass(frozen=True)
class CorrectorRoundTripResult:
    checkpoint_path: Path
    manifest_path: Path
    loss: float
    latency_ms: float
    throughput_samples_s: float
    peak_vram_mb: float
    amp_dtype: AmpDtype
    model_api: str


def _patch_pandas_api_for_rapids_compat() -> None:
    try:
        pandas = importlib.import_module("pandas")
    except ModuleNotFoundError:
        return
    api_types = getattr(getattr(pandas, "api", None), "types", None)
    interval_cls = getattr(pandas, "Interval", None)
    if api_types is None or interval_cls is None or hasattr(api_types, "is_interval"):
        return

    def is_interval(value: object) -> bool:
        return isinstance(value, interval_cls)

    api_types.is_interval = is_interval


def _load_fully_connected_class() -> type[torch.nn.Module]:
    _patch_pandas_api_for_rapids_compat()
    try:
        module = importlib.import_module("physicsnemo.models.mlp.fully_connected")
    except ModuleNotFoundError as exc:
        msg = (
            "Official PhysicsNeMo FullyConnected model API is not importable: "
            "physicsnemo.models.mlp.fully_connected"
        )
        raise CudaPreflightBlocker(msg) from exc
    model_cls = getattr(module, "FullyConnected", None)
    if not isinstance(model_cls, type):
        msg = "physicsnemo.models.mlp.fully_connected.FullyConnected is absent or not a class."
        raise CudaPreflightBlocker(msg)
    return cast("type[torch.nn.Module]", model_cls)


def _new_model(config: CorrectorSmokeConfig, device: torch.device) -> torch.nn.Module:
    model_cls = _load_fully_connected_class()
    model = model_cls(
        in_features=config.input_features,
        layer_size=config.hidden_features,
        out_features=config.output_features,
        num_layers=config.hidden_layers,
    )
    return model.to(device)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CorrectorSmokeConfig,
    loss: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "loss": float(loss.detach().item()),
        "model_api": "physicsnemo.models.mlp.fully_connected.FullyConnected",
    }


def _write_corrector_manifest(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    config: CorrectorSmokeConfig,
    loss: torch.Tensor,
    latency_ms: float,
    throughput_samples_s: float,
    peak_vram_mb: float,
    min_vram_gb: int,
) -> None:
    atomic_write_json(
        manifest_path,
        {
            "schema": "aerocliff_cuda_corrector_round_trip_v0.1.0",
            "evidence_provenance": {
                "producer": "aeromap.integrations.physicsnemo.corrector.run_corrector_round_trip",
                "synthetic_fixture": False,
            },
            "preflight": build_preflight_report(min_vram_gb=min_vram_gb),
            "corrector": {
                "model_api": "physicsnemo.models.mlp.fully_connected.FullyConnected",
                "config": asdict(config),
                "mixed_precision": {
                    "amp_dtype": config.amp_dtype,
                    "autocast_enabled": MixedPrecisionPolicy(config.amp_dtype).autocast_enabled,
                },
                "forward_backward": "completed",
                "checkpoint_round_trip": "saved_and_reloaded",
                "loss": float(loss.detach().item()),
                "latency_ms": latency_ms,
                "throughput_samples_s": throughput_samples_s,
                "peak_vram_mb": peak_vram_mb,
            },
            "artifacts": {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "manifest_path": str(manifest_path),
            },
            "access_status": {
                "nim_access": "not_used_corrector_is_separate_from_frozen_nim",
                "checkpoint_access": "local_corrector_checkpoint_saved_and_reloaded",
            },
        },
    )


def run_corrector_round_trip(
    config: CorrectorSmokeConfig,
    out_dir: Path,
    *,
    min_vram_gb: int = 40,
) -> CorrectorRoundTripResult:
    """Run one official PhysicsNeMo CUDA forward/backward/checkpoint smoke."""

    require_cuda_preflight(
        min_vram_gb=min_vram_gb,
        require_ngc=False,
        require_docker=False,
        require_nvidia_smi=False,
        require_physicsnemo_cfd=False,
    )
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats(device)

    model = _new_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    x = torch.randn(config.batch_size, config.input_features, device=device)
    target = torch.randn(config.batch_size, config.output_features, device=device)
    policy = MixedPrecisionPolicy(config.amp_dtype)

    start_event = cast("Any", torch.cuda.Event)(enable_timing=True)
    end_event = cast("Any", torch.cuda.Event)(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=policy.torch_dtype,
        enabled=policy.autocast_enabled,
    ):
        prediction = model(x)
        loss = functional.mse_loss(prediction, target)
    cast("Any", loss).backward()
    optimizer.step()
    end_event.record()
    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start

    checkpoint_path = out_dir / "physicsnemo_corrector_round_trip.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, config, loss), checkpoint_path)

    restored_model = _new_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        msg = "corrector checkpoint payload is not a mapping"
        raise TypeError(msg)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        restored_prediction = restored_model(x)
        original_prediction = model(x)
    torch.testing.assert_close(restored_prediction, original_prediction)

    peak_vram_mb = float(torch.cuda.max_memory_allocated(device)) / float(1024**2)
    latency_ms = float(start_event.elapsed_time(end_event))
    throughput = float(config.batch_size) / max(wall_elapsed, 1.0e-9)
    manifest_path = out_dir / "manifest.json"
    _write_corrector_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        config=config,
        loss=loss,
        latency_ms=latency_ms,
        throughput_samples_s=throughput,
        peak_vram_mb=peak_vram_mb,
        min_vram_gb=min_vram_gb,
    )
    return CorrectorRoundTripResult(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        loss=float(loss.detach().item()),
        latency_ms=latency_ms,
        throughput_samples_s=throughput,
        peak_vram_mb=peak_vram_mb,
        amp_dtype=config.amp_dtype,
        model_api="physicsnemo.models.mlp.fully_connected.FullyConnected",
    )

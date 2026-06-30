"""Explicit DoMINO predictor and trainable corrector adapter boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeromap.runtime.device import DeviceSpec, resolve_device


@dataclass(frozen=True)
class PredictionCacheKey:
    geometry_sha256: str
    nim_image: str
    checkpoint_sha256: str
    params_hash: str


class NimPredictorAdapter:
    """Frozen NIM predictor wrapper for inference and cached base predictions only."""

    def __init__(self, *, endpoint: str, cache_dir: Path, image_ref: str) -> None:
        self.endpoint = endpoint
        self.cache_dir = cache_dir
        self.image_ref = image_ref

    def train(self, *_args: object, **_kwargs: object) -> None:
        message = "The DoMINO NIM is a frozen predictor; train the PhysicsNeMoCorrector instead."
        raise RuntimeError(message)

    def predict(self, geometry_path: Path, params: dict[str, Any]) -> Path:
        message = (
            "NimPredictorAdapter HTTP/container inference is CUDA-host integration work. "
            f"Would run frozen NIM {self.image_ref} for {geometry_path} with {params}."
        )
        raise RuntimeError(message)


class PhysicsNeMoCorrector:
    """Trainable PyTorch/PhysicsNeMo residual corrector boundary."""

    def __init__(self, *, config_path: Path, checkpoint_path: Path | None = None) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def require_cuda_device(self) -> DeviceSpec:
        return resolve_device("auto", require_physicsnemo=True)

    def train_step(self, *_args: object, **_kwargs: object) -> None:
        self.require_cuda_device()
        message = (
            "PhysicsNeMoCorrector forward/backward/checkpoint implementation starts "
            "at model-integration after CFD data conversion is stable."
        )
        raise RuntimeError(message)

    def save_checkpoint(self, path: Path) -> None:
        self.require_cuda_device()
        message = f"No trainable corrector state exists yet; cannot write checkpoint to {path}."
        raise RuntimeError(message)

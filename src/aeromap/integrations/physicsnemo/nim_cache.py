"""Frozen DoMINO NIM prediction cache adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aeromap.integrations.physicsnemo.preflight import (
    NIM_IMAGE_REF,
    build_preflight_report,
    load_domino_nim_caller,
    nvidia_smi_snapshot,
    require_cuda_preflight,
)
from aeromap.io import atomic_write_json, sha256_file


@dataclass(frozen=True)
class NimInferenceParams:
    endpoint: str = "http://localhost:8000/v1/infer"
    stream_velocity: float = 40.0
    stencil_size: int = 1
    point_cloud_size: int = 500_000

    def as_form_data(self) -> dict[str, str]:
        return {
            "stream_velocity": str(self.stream_velocity),
            "stencil_size": str(self.stencil_size),
            "point_cloud_size": str(self.point_cloud_size),
        }


@dataclass(frozen=True)
class CachedNimPrediction:
    cache_key: str
    prediction_path: Path
    manifest_path: Path


def _stable_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _client_peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(torch.device("cuda"))) / float(1024**2)


class FrozenNimCacheAdapter:
    """Cache frozen DoMINO NIM predictions without exposing trainable behavior."""

    def __init__(self, *, cache_dir: Path, image_ref: str = NIM_IMAGE_REF) -> None:
        self.cache_dir = cache_dir
        self.image_ref = image_ref

    def train(self, *_args: object, **_kwargs: object) -> None:
        msg = "The DoMINO NIM is a frozen predictor; train only the PhysicsNeMo corrector."
        raise RuntimeError(msg)

    def cache_key(self, stl_path: Path, params: NimInferenceParams) -> str:
        payload = {
            "image_ref": self.image_ref,
            "stl_sha256": sha256_file(stl_path),
            "params": asdict(params),
            "schema": "aerocliff_frozen_nim_cache_v0.1.0",
        }
        return _stable_key(payload)

    def cached_paths(self, stl_path: Path, params: NimInferenceParams) -> CachedNimPrediction:
        key = self.cache_key(stl_path, params)
        item_dir = self.cache_dir / key
        return CachedNimPrediction(
            cache_key=key,
            prediction_path=item_dir / "prediction.npz",
            manifest_path=item_dir / "manifest.json",
        )

    def capture_from_nim(
        self,
        stl_path: Path,
        params: NimInferenceParams,
        *,
        min_vram_gb: int = 40,
        geometry_source: str,
        prediction_classification: str = "FROZEN_NIM_PREDICTOR_CACHE",
    ) -> CachedNimPrediction:
        """Call NVIDIA's physicsnemo-cfd helper and write a deterministic cache item."""

        if not geometry_source.strip():
            msg = "geometry_source must explicitly identify the NIM input geometry source"
            raise ValueError(msg)
        if not prediction_classification.strip():
            msg = "prediction_classification must explicitly label the frozen predictor output"
            raise ValueError(msg)
        require_cuda_preflight(min_vram_gb=min_vram_gb)
        caller = load_domino_nim_caller()
        cached = self.cached_paths(stl_path, params)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
        nvidia_smi_before = nvidia_smi_snapshot()
        started = time.perf_counter()
        output_dict = caller(
            stl_path=str(stl_path),
            inference_api_url=params.endpoint,
            data=params.as_form_data(),
            verbose=True,
        )
        elapsed_s = time.perf_counter() - started
        nvidia_smi_after = nvidia_smi_snapshot()
        if not isinstance(output_dict, dict):
            msg = "DoMINO NIM helper did not return a mapping of arrays."
            raise TypeError(msg)
        _write_npz_atomic(cached.prediction_path, output_dict)
        atomic_write_json(
            cached.manifest_path,
            {
                "schema": "aerocliff_frozen_nim_cache_v0.1.0",
                "evidence_provenance": {
                    "producer": (
                        "aeromap.integrations.physicsnemo.nim_cache."
                        "FrozenNimCacheAdapter.capture_from_nim"
                    ),
                    "synthetic_fixture": False,
                },
                "cache_key": cached.cache_key,
                "image_ref": self.image_ref,
                "endpoint": params.endpoint,
                "params": params.as_form_data(),
                "stl_path": str(stl_path),
                "stl_sha256": sha256_file(stl_path),
                "prediction_path": str(cached.prediction_path),
                "prediction_sha256": sha256_file(cached.prediction_path),
                "geometry_source": geometry_source,
                "prediction_classification": prediction_classification,
                "agreement_claims_allowed": False,
                "agreement_claims_blocker": (
                    "No agreement claim is allowed until an accepted AeroCliff campaign CFD "
                    "reference exists for the same geometry/state."
                ),
                "frozen_predictor": True,
                "trainable": False,
                "preflight": build_preflight_report(min_vram_gb=min_vram_gb),
                "profile": {
                    "latency_ms": elapsed_s * 1000.0,
                    "throughput_cases_s": 1.0 / max(elapsed_s, 1.0e-9),
                    "throughput_points_s": float(params.point_cloud_size) / max(elapsed_s, 1.0e-9),
                    "client_peak_vram_mb": _client_peak_vram_mb(),
                    "nvidia_smi_before": nvidia_smi_before,
                    "nvidia_smi_after": nvidia_smi_after,
                },
                "access_status": {
                    "nim_inference": "completed",
                    "checkpoint_access": (
                        "NIM container/model access implied by successful inference; "
                        "no NGC checkpoint or weight artifact stored by AeroCliff"
                    ),
                },
            },
        )
        return cached

"""Central PyTorch device handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeviceRequest = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True)
class DeviceSpec:
    request: DeviceRequest
    resolved: Literal["cpu", "mps", "cuda"]
    torch_device: str
    cuda_available: bool
    mps_available: bool
    torch_version: str
    device_name: str


def _cuda_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(torch.cuda.get_device_name(0))


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(
    request: DeviceRequest = "auto",
    *,
    require_cuda: bool = False,
) -> DeviceSpec:
    """Resolve a PyTorch device request with optional CUDA-only enforcement."""

    if request not in {"auto", "cpu", "mps", "cuda"}:
        message = f"unknown device request: {request}"
        raise ValueError(message)

    cuda_available = torch.cuda.is_available()
    mps_available = _mps_available()

    if require_cuda:
        if request not in {"auto", "cuda"}:
            message = "This workload requires CUDA; requested device is not allowed."
            raise RuntimeError(message)
        if not cuda_available:
            message = "This workload requires CUDA; no CUDA device is visible."
            raise RuntimeError(message)
        resolved: Literal["cpu", "mps", "cuda"] = "cuda"
    elif request == "auto":
        if cuda_available:
            resolved = "cuda"
        elif mps_available:
            resolved = "mps"
        else:
            resolved = "cpu"
    elif request == "cuda":
        if not cuda_available:
            message = "Requested CUDA device, but torch.cuda.is_available() is false."
            raise RuntimeError(message)
        resolved = "cuda"
    elif request == "mps":
        if not mps_available:
            message = "Requested MPS device, but torch.backends.mps is unavailable."
            raise RuntimeError(message)
        resolved = "mps"
    elif request == "cpu":
        resolved = "cpu"

    device_name = {
        "cuda": _cuda_name(),
        "mps": "Apple MPS",
        "cpu": "CPU",
    }[resolved]

    return DeviceSpec(
        request=request,
        resolved=resolved,
        torch_device=resolved,
        cuda_available=cuda_available,
        mps_available=mps_available,
        torch_version=torch.__version__,
        device_name=device_name,
    )

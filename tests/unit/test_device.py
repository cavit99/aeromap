from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import aeromap.runtime.device as device_module
from aeromap.runtime.device import resolve_device


def test_auto_resolves_to_known_backend() -> None:
    spec = resolve_device("auto")
    assert spec.resolved in {"cpu", "mps", "cuda"}
    assert spec.torch_version


def test_physicsnemo_refuses_non_cuda_when_cuda_missing() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_device("cpu", require_physicsnemo=True)


def test_physicsnemo_auto_requires_cuda_when_cuda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)

    with pytest.raises(RuntimeError, match="requires a Linux NVIDIA CUDA host"):
        resolve_device("auto", require_physicsnemo=True)


def test_invalid_device_request_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown device request"):
        resolve_device(cast("Any", "gpu"))

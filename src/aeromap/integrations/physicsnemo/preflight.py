"""Fail-clear checks for the official CUDA PhysicsNeMo/DoMINO lane."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

NIM_IMAGE_REF = "nvcr.io/nim/nvidia/domino-automotive-aero:2.1.0-41313772"
CUDA_BLOCKER_EXIT_CODE = 86
NVIDIA_SMI_TIMEOUT_S = 10


class CudaPreflightBlocker(RuntimeError):  # noqa: N818
    """Raised when the official Linux NVIDIA CUDA path cannot be exercised."""


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required_for: tuple[str, ...]


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _secret_file_from_env(env: Mapping[str, str]) -> Path | None:
    value = env.get("NGC_API_KEY_FILE")
    return Path(value).expanduser() if value else None


def ngc_api_key_source(
    *,
    env: Mapping[str, str] | None = None,
    secret_file: Path | None = None,
) -> str:
    """Return a non-secret source label for NGC credential availability."""

    checked_env = os.environ if env is None else env
    if checked_env.get("NGC_API_KEY", "").strip():
        return "env:NGC_API_KEY"

    candidate = secret_file or _secret_file_from_env(checked_env)
    if candidate is None:
        return ""
    try:
        if candidate.is_file() and candidate.read_text(encoding="utf-8").strip():
            return f"file:{candidate}"
    except OSError:
        return f"unreadable-file:{candidate}"
    return ""


def _cuda_device_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(torch.cuda.get_device_name(0))


def _cuda_capability() -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def _cuda_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / float(1024**3)


def _cudnn_version() -> int | None:
    try:
        return cast("int | None", torch.backends.cudnn.version())  # type: ignore[no-untyped-call]
    except (AttributeError, RuntimeError):
        return None


def _distribution_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _first_distribution_version(distribution_names: tuple[str, ...]) -> str:
    for distribution_name in distribution_names:
        version = _distribution_version(distribution_name)
        if version:
            return version
    return ""


def package_versions() -> dict[str, str]:
    """Return relevant package versions without importing optional NVIDIA packages."""

    return {
        "torch": torch.__version__,
        "numpy": _distribution_version("numpy"),
        "nvidia-physicsnemo": _distribution_version("nvidia-physicsnemo"),
        "physicsnemo-cfd": _first_distribution_version(
            ("physicsnemo-cfd", "nvidia-physicsnemo-cfd"),
        ),
        "httpx": _distribution_version("httpx"),
        "trimesh": _distribution_version("trimesh"),
        "pyvista": _distribution_version("pyvista"),
        "vtk": _distribution_version("vtk"),
    }


def nvidia_smi_snapshot(*, timeout_s: int = NVIDIA_SMI_TIMEOUT_S) -> dict[str, Any]:
    """Capture safe NVIDIA driver/GPU details if nvidia-smi is available."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "available": False,
            "path": "",
            "query_returncode": None,
            "query_stdout": "",
            "query_stderr": "",
            "summary_returncode": None,
            "summary_stdout": "",
            "summary_stderr": "",
        }

    query_args = [
        nvidia_smi,
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    summary_args = [nvidia_smi]
    try:
        query = subprocess.run(  # noqa: S603 - binary path is resolved by shutil.which.
            query_args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        query_stdout = ""
        query_stderr = str(exc)
        query_returncode: int | None = None
    else:
        query_stdout = query.stdout.strip()
        query_stderr = query.stderr.strip()
        query_returncode = query.returncode

    try:
        summary = subprocess.run(  # noqa: S603 - binary path is resolved by shutil.which.
            summary_args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        summary_stdout = ""
        summary_stderr = str(exc)
        summary_returncode: int | None = None
    else:
        summary_stdout = summary.stdout.strip()
        summary_stderr = summary.stderr.strip()
        summary_returncode = summary.returncode

    return {
        "available": True,
        "path": nvidia_smi,
        "query_returncode": query_returncode,
        "query_stdout": query_stdout,
        "query_stderr": query_stderr,
        "summary_returncode": summary_returncode,
        "summary_stdout": summary_stdout,
        "summary_stderr": summary_stderr,
    }


def _check(
    name: str,
    *,
    ok: bool,
    detail: str,
    required_for: Sequence[str],
) -> PreflightCheck:
    return PreflightCheck(name=name, ok=ok, detail=detail, required_for=tuple(required_for))


def build_preflight_report(
    *,
    min_vram_gb: int = 40,
    env: Mapping[str, str] | None = None,
    secret_file: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe report without logging secrets or weakening CUDA requirements."""

    checked_env = os.environ if env is None else env
    system = platform.system()
    machine = platform.machine().lower()
    cuda_available = torch.cuda.is_available()
    vram_gb = _cuda_total_vram_gb()
    ngc_source = ngc_api_key_source(env=checked_env, secret_file=secret_file)
    physicsnemo_available = _module_available("physicsnemo")
    physicsnemo_cfd_available = _module_available("physicsnemo.cfd")
    nim_helper_available = _module_available("physicsnemo.cfd.evaluation.nims")

    checks = [
        _check(
            "linux",
            ok=system == "Linux",
            detail=f"platform.system()={system}",
            required_for=("corrector", "nim", "profile"),
        ),
        _check(
            "x86_64",
            ok=machine in {"x86_64", "amd64"},
            detail=f"platform.machine()={platform.machine()}",
            required_for=("nim",),
        ),
        _check(
            "cuda_visible",
            ok=cuda_available,
            detail=f"torch.cuda.is_available()={cuda_available}",
            required_for=("corrector", "nim", "profile"),
        ),
        _check(
            "min_vram",
            ok=cuda_available and vram_gb >= min_vram_gb,
            detail=f"cuda_vram_gb={vram_gb:.2f}, required_gb={min_vram_gb}",
            required_for=("nim", "profile"),
        ),
        _check(
            "nvidia_smi",
            ok=shutil.which("nvidia-smi") is not None,
            detail="nvidia-smi on PATH",
            required_for=("nim",),
        ),
        _check(
            "docker",
            ok=shutil.which("docker") is not None,
            detail="docker on PATH",
            required_for=("nim",),
        ),
        _check(
            "ngc_api_key",
            ok=bool(ngc_source),
            detail=(
                "NGC_API_KEY or NGC_API_KEY_FILE present" if ngc_source else "no NGC key source"
            ),
            required_for=("nim",),
        ),
        _check(
            "physicsnemo",
            ok=physicsnemo_available,
            detail="importable module: physicsnemo",
            required_for=("corrector",),
        ),
        _check(
            "physicsnemo_cfd",
            ok=physicsnemo_cfd_available,
            detail="importable module: physicsnemo.cfd",
            required_for=("nim",),
        ),
        _check(
            "physicsnemo_cfd_nims",
            ok=nim_helper_available,
            detail="importable module: physicsnemo.cfd.evaluation.nims",
            required_for=("nim",),
        ),
    ]

    return {
        "schema": "aerocliff_cuda_preflight_report_v0.1.0",
        "evidence_provenance": {
            "producer": "aeromap.integrations.physicsnemo.preflight.build_preflight_report",
            "synthetic_fixture": False,
        },
        "nim_image_ref": NIM_IMAGE_REF,
        "platform": {
            "system": system,
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_name": _cuda_device_name(),
            "cuda_capability": _cuda_capability(),
            "cuda_vram_gb": round(vram_gb, 3),
            "cuda_runtime_version": torch.version.cuda or "",
            "cudnn_version": _cudnn_version(),
        },
        "packages": package_versions(),
        "nvidia_smi": nvidia_smi_snapshot(),
        "tools": {
            "docker": shutil.which("docker") or "",
            "nvidia_smi": shutil.which("nvidia-smi") or "",
        },
        "credentials": {
            "ngc_api_key_available": bool(ngc_source),
            "ngc_api_key_source": ngc_source,
        },
        "checks": [asdict(item) for item in checks],
    }


def cuda_launch_blockers(
    *,
    min_vram_gb: int = 40,
    require_ngc: bool = True,
    require_docker: bool = True,
    require_nvidia_smi: bool = True,
    require_physicsnemo_cfd: bool = True,
    env: Mapping[str, str] | None = None,
    secret_file: Path | None = None,
) -> list[str]:
    """Return concrete blockers for the official CUDA path."""

    blockers: list[str] = []
    system = platform.system()
    machine = platform.machine().lower()
    cuda_available = torch.cuda.is_available()
    vram_gb = _cuda_total_vram_gb()

    if system != "Linux":
        blockers.append(f"Linux x86 NVIDIA host required; current platform is {system}.")
    if machine not in {"x86_64", "amd64"}:
        blockers.append(
            f"x86_64 host required for DoMINO NIM; current machine is {platform.machine()}.",
        )
    if not cuda_available:
        blockers.append("No CUDA device is visible to PyTorch; refusing CPU/MPS fallback.")
    elif vram_gb < min_vram_gb:
        blockers.append(
            "CUDA device has "
            f"{vram_gb:.2f} GB VRAM; DoMINO preflight requires >= {min_vram_gb} GB.",
        )
    if require_nvidia_smi and shutil.which("nvidia-smi") is None:
        blockers.append("nvidia-smi is not on PATH; NVIDIA driver/runtime visibility is unproven.")
    if require_docker and shutil.which("docker") is None:
        blockers.append("Docker is not on PATH; DoMINO NIM container launch cannot be verified.")
    if require_ngc and not ngc_api_key_source(env=env, secret_file=secret_file):
        blockers.append(
            "NGC_API_KEY or a non-empty NGC_API_KEY_FILE is required for NIM/checkpoint access.",
        )
    if not _module_available("physicsnemo"):
        blockers.append("Official NVIDIA PhysicsNeMo package is not importable.")
    if require_physicsnemo_cfd and not _module_available("physicsnemo.cfd"):
        blockers.append("Official NVIDIA physicsnemo-cfd package is not importable.")
    if require_physicsnemo_cfd and not _module_available("physicsnemo.cfd.evaluation.nims"):
        blockers.append(
            "physicsnemo.cfd.evaluation.nims is not importable; DoMINO helper API is absent.",
        )

    return blockers


def require_cuda_preflight(
    *,
    min_vram_gb: int = 40,
    require_ngc: bool = True,
    require_docker: bool = True,
    require_nvidia_smi: bool = True,
    require_physicsnemo_cfd: bool = True,
    env: Mapping[str, str] | None = None,
    secret_file: Path | None = None,
) -> None:
    """Raise a fail-clear blocker instead of falling back to CPU/MPS."""

    blockers = cuda_launch_blockers(
        min_vram_gb=min_vram_gb,
        require_ngc=require_ngc,
        require_docker=require_docker,
        require_nvidia_smi=require_nvidia_smi,
        require_physicsnemo_cfd=require_physicsnemo_cfd,
        env=env,
        secret_file=secret_file,
    )
    if blockers:
        bullet_list = "\n".join(f"- {item}" for item in blockers)
        msg = f"Official CUDA PhysicsNeMo/DoMINO path is blocked:\n{bullet_list}"
        raise CudaPreflightBlocker(msg)


def load_domino_nim_caller() -> Callable[..., object]:
    """Load NVIDIA's physicsnemo-cfd DoMINO helper or return an exact blocker."""

    try:
        module = importlib.import_module("physicsnemo.cfd.evaluation.nims")
    except ModuleNotFoundError as exc:
        msg = (
            "Official physicsnemo-cfd DoMINO helper is not importable: "
            "physicsnemo.cfd.evaluation.nims"
        )
        raise CudaPreflightBlocker(msg) from exc
    caller = getattr(module, "call_domino_nim", None)
    if not callable(caller):
        msg = "physicsnemo.cfd.evaluation.nims.call_domino_nim is absent or not callable."
        raise CudaPreflightBlocker(msg)
    return cast("Callable[..., object]", caller)

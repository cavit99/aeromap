"""CUDA-only PhysicsNeMo and DoMINO preflight helpers."""

from __future__ import annotations

from aeromap.integrations.physicsnemo.corrector import (
    CorrectorRoundTripResult,
    CorrectorSmokeConfig,
    MixedPrecisionPolicy,
    run_corrector_round_trip,
)
from aeromap.integrations.physicsnemo.drivaerml import (
    DRIVAERML_LICENSE,
    DRIVAERML_STL_URL,
    write_drivaerml_external_manifest,
)
from aeromap.integrations.physicsnemo.evidence import (
    validate_cuda_evidence_bundle,
)
from aeromap.integrations.physicsnemo.nim_cache import (
    FrozenNimCacheAdapter,
    NimInferenceParams,
)
from aeromap.integrations.physicsnemo.preflight import (
    CUDA_BLOCKER_EXIT_CODE,
    NIM_IMAGE_REF,
    CudaPreflightBlocker,
    build_preflight_report,
    cuda_launch_blockers,
    nvidia_smi_snapshot,
    package_versions,
    require_cuda_preflight,
)

__all__ = [
    "CUDA_BLOCKER_EXIT_CODE",
    "DRIVAERML_LICENSE",
    "DRIVAERML_STL_URL",
    "NIM_IMAGE_REF",
    "CorrectorRoundTripResult",
    "CorrectorSmokeConfig",
    "CudaPreflightBlocker",
    "FrozenNimCacheAdapter",
    "MixedPrecisionPolicy",
    "NimInferenceParams",
    "build_preflight_report",
    "cuda_launch_blockers",
    "nvidia_smi_snapshot",
    "package_versions",
    "require_cuda_preflight",
    "run_corrector_round_trip",
    "validate_cuda_evidence_bundle",
    "write_drivaerml_external_manifest",
]

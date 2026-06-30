"""Validation for returned CUDA PhysicsNeMo/DoMINO evidence bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from aeromap.io import sha256_file

EVIDENCE_SCHEMA = "aerocliff_cuda_domino_evidence_validation_v0.1.0"
PREFLIGHT_SCHEMA = "aerocliff_cuda_preflight_report_v0.1.0"
CORRECTOR_SCHEMA = "aerocliff_cuda_corrector_round_trip_v0.1.0"
DRIVAERML_SCHEMA = "aerocliff_external_drivaerml_smoke_v0.1.0"
NIM_CACHE_SCHEMA = "aerocliff_frozen_nim_cache_v0.1.0"
ARTIFACTS_CUDA_PREFIX_PARTS = 2
INVALID_ARTIFACT_SENTINEL = ".invalid-artifact-path"


@dataclass(frozen=True)
class EvidenceIssue:
    path: str
    message: str


def _issue(path: Path, message: str) -> EvidenceIssue:
    return EvidenceIssue(path=str(path), message=message)


def _load_json_mapping(path: Path, errors: list[EvidenceIssue]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(_issue(path, "required JSON evidence file is missing"))
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(_issue(path, f"invalid JSON: {exc}"))
        return {}
    if not isinstance(loaded, dict):
        errors.append(_issue(path, "JSON evidence file must contain an object"))
        return {}
    return cast("dict[str, Any]", loaded)


def _require_equal(
    errors: list[EvidenceIssue],
    path: Path,
    payload: Mapping[str, Any],
    key: str,
    *,
    expected: object,
) -> None:
    actual = payload.get(key)
    if actual != expected:
        errors.append(_issue(path, f"{key} must be {expected!r}; got {actual!r}"))


def _number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_positive(
    errors: list[EvidenceIssue],
    path: Path,
    payload: Mapping[str, Any],
    key: str,
) -> None:
    number = _number(payload, key)
    if number is None or number <= 0.0:
        errors.append(_issue(path, f"{key} must be a finite positive number"))


def _validate_evidence_provenance(
    errors: list[EvidenceIssue],
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_synthetic_fixture: bool,
) -> None:
    provenance = payload.get("evidence_provenance")
    if not isinstance(provenance, dict):
        errors.append(_issue(path, "evidence_provenance section is missing or invalid"))
        return
    producer = provenance.get("producer")
    if not isinstance(producer, str) or not producer.startswith(
        "aeromap.integrations.physicsnemo.",
    ):
        errors.append(_issue(path, "evidence_provenance.producer is missing or invalid"))
    synthetic_fixture = provenance.get("synthetic_fixture")
    if synthetic_fixture is True and not allow_synthetic_fixture:
        errors.append(_issue(path, "synthetic fixture evidence is not accepted for CUDA gates"))
    elif synthetic_fixture is not False and not allow_synthetic_fixture:
        errors.append(_issue(path, "evidence_provenance.synthetic_fixture must be false"))
    elif allow_synthetic_fixture and not isinstance(synthetic_fixture, bool):
        errors.append(_issue(path, "evidence_provenance.synthetic_fixture must be boolean"))


def _resolve_artifact_path(
    value: object,
    *,
    evidence_root: Path,
    errors: list[EvidenceIssue],
    manifest_path: Path,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        errors.append(_issue(manifest_path, f"{field_name} path is missing"))
        return evidence_root / INVALID_ARTIFACT_SENTINEL
    raw_path = Path(value)
    if raw_path.is_absolute():
        errors.append(_issue(manifest_path, f"{field_name} path must be relative to evidence root"))
        return evidence_root / INVALID_ARTIFACT_SENTINEL
    if ".." in raw_path.parts:
        errors.append(_issue(manifest_path, f"{field_name} path must not contain parent traversal"))
        return evidence_root / INVALID_ARTIFACT_SENTINEL
    if len(raw_path.parts) >= ARTIFACTS_CUDA_PREFIX_PARTS and raw_path.parts[
        :ARTIFACTS_CUDA_PREFIX_PARTS
    ] == ("artifacts", "cuda"):
        candidate = evidence_root.joinpath(*raw_path.parts[ARTIFACTS_CUDA_PREFIX_PARTS:])
    else:
        candidate = evidence_root / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(evidence_root.resolve(strict=False))
    except ValueError:
        errors.append(_issue(manifest_path, f"{field_name} path resolves outside evidence root"))
        return evidence_root / INVALID_ARTIFACT_SENTINEL
    return resolved


def _require_file_hash(
    errors: list[EvidenceIssue],
    manifest_path: Path,
    *,
    evidence_root: Path,
    artifact_value: object,
    expected_sha256: object,
    field_name: str,
) -> Path:
    artifact_path = _resolve_artifact_path(
        artifact_value,
        evidence_root=evidence_root,
        errors=errors,
        manifest_path=manifest_path,
        field_name=field_name,
    )
    if not artifact_path.is_file():
        errors.append(_issue(manifest_path, f"{field_name} does not exist: {artifact_path}"))
        return artifact_path
    if not isinstance(expected_sha256, str) or not expected_sha256:
        errors.append(_issue(manifest_path, f"{field_name} sha256 is missing"))
        return artifact_path
    actual = sha256_file(artifact_path)
    if actual != expected_sha256:
        errors.append(
            _issue(
                manifest_path,
                f"{field_name} sha256 mismatch for {artifact_path}: expected "
                f"{expected_sha256}, got {actual}",
            ),
        )
    return artifact_path


def _check_statuses(
    errors: list[EvidenceIssue],
    path: Path,
    checks: object,
    *,
    required_names: set[str],
) -> None:
    if not isinstance(checks, list):
        errors.append(_issue(path, "checks must be a list"))
        return
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in checks:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            by_name[item["name"]] = item
    for name in sorted(required_names):
        check = by_name.get(name)
        if check is None:
            errors.append(_issue(path, f"preflight check {name!r} is missing"))
        elif check.get("ok") is not True:
            errors.append(_issue(path, f"preflight check {name!r} is not ok"))


def _mapping_section(
    errors: list[EvidenceIssue],
    path: Path,
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    section = payload.get(key)
    if isinstance(section, dict):
        return section
    errors.append(_issue(path, f"{label} section is missing or invalid"))
    return {}


def _validate_platform_and_torch(
    errors: list[EvidenceIssue],
    path: Path,
    *,
    platform: Mapping[str, Any],
    torch_report: Mapping[str, Any],
    min_vram_gb: int,
) -> None:
    if platform.get("system") != "Linux":
        errors.append(_issue(path, "platform.system must be Linux for accepted CUDA evidence"))
    machine = str(platform.get("machine", "")).lower()
    if machine not in {"x86_64", "amd64"}:
        errors.append(_issue(path, "platform.machine must be x86_64/amd64"))
    if torch_report.get("cuda_available") is not True:
        errors.append(_issue(path, "torch.cuda_available must be true"))
    if not torch_report.get("cuda_device_name"):
        errors.append(_issue(path, "torch.cuda_device_name is required"))
    if not torch_report.get("cuda_runtime_version"):
        errors.append(_issue(path, "torch.cuda_runtime_version is required"))
    vram_gb = _number(torch_report, "cuda_vram_gb")
    if vram_gb is None or vram_gb < float(min_vram_gb):
        errors.append(_issue(path, f"torch.cuda_vram_gb must be >= {min_vram_gb}"))


def _validate_packages_and_credentials(
    errors: list[EvidenceIssue],
    path: Path,
    *,
    packages: Mapping[str, Any],
    credentials: Mapping[str, Any],
    require_ngc: bool,
    require_cfd: bool,
) -> None:
    if not packages.get("torch"):
        errors.append(_issue(path, "packages.torch is required"))
    if not packages.get("nvidia-physicsnemo"):
        errors.append(_issue(path, "packages.nvidia-physicsnemo is required"))
    if require_cfd and not packages.get("physicsnemo-cfd"):
        errors.append(_issue(path, "packages.physicsnemo-cfd is required"))

    if require_ngc and credentials.get("ngc_api_key_available") is not True:
        errors.append(_issue(path, "NGC credential source must be present"))
    if require_ngc and not credentials.get("ngc_api_key_source"):
        errors.append(_issue(path, "NGC credential source label is required"))


def _validate_nvidia_smi_section(
    errors: list[EvidenceIssue],
    path: Path,
    nvidia_smi: Mapping[str, Any],
    *,
    require_nvidia_smi: bool,
) -> None:
    if require_nvidia_smi:
        if nvidia_smi.get("available") is not True:
            errors.append(_issue(path, "nvidia-smi must be available"))
        if nvidia_smi.get("query_returncode") != 0:
            errors.append(_issue(path, "nvidia-smi query must return 0"))
        if not nvidia_smi.get("query_stdout"):
            errors.append(_issue(path, "nvidia-smi GPU query output is required"))
        if not nvidia_smi.get("summary_stdout"):
            errors.append(_issue(path, "nvidia-smi summary output is required"))


def _validate_preflight(
    errors: list[EvidenceIssue],
    path: Path,
    report: Mapping[str, Any],
    *,
    min_vram_gb: int,
    require_ngc: bool,
    require_cfd: bool,
    require_nvidia_smi: bool,
    allow_synthetic_fixture: bool,
) -> None:
    _require_equal(errors, path, report, "schema", expected=PREFLIGHT_SCHEMA)
    _validate_evidence_provenance(
        errors,
        path,
        report,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    platform = _mapping_section(errors, path, report, "platform", label="platform")
    torch_report = _mapping_section(errors, path, report, "torch", label="torch")
    packages = _mapping_section(errors, path, report, "packages", label="packages")
    credentials = _mapping_section(errors, path, report, "credentials", label="credentials")
    nvidia_smi = _mapping_section(errors, path, report, "nvidia_smi", label="nvidia_smi")
    _validate_platform_and_torch(
        errors,
        path,
        platform=platform,
        torch_report=torch_report,
        min_vram_gb=min_vram_gb,
    )
    _validate_packages_and_credentials(
        errors,
        path,
        packages=packages,
        credentials=credentials,
        require_ngc=require_ngc,
        require_cfd=require_cfd,
    )
    _validate_nvidia_smi_section(
        errors,
        path,
        nvidia_smi,
        require_nvidia_smi=require_nvidia_smi,
    )
    required_checks = {"cuda_visible", "linux", "min_vram", "physicsnemo", "x86_64"}
    if require_ngc:
        required_checks.add("ngc_api_key")
    if require_cfd:
        required_checks.update({"physicsnemo_cfd", "physicsnemo_cfd_nims"})
    if require_nvidia_smi:
        required_checks.update({"docker", "nvidia_smi"})
    _check_statuses(errors, path, report.get("checks"), required_names=required_checks)


def _validate_doctor(
    root: Path,
    errors: list[EvidenceIssue],
    checked: list[str],
    *,
    min_vram_gb: int,
    allow_synthetic_fixture: bool,
) -> None:
    path = root / "doctor.json"
    checked.append(str(path))
    report = _load_json_mapping(path, errors)
    if report:
        _validate_preflight(
            errors,
            path,
            report,
            min_vram_gb=min_vram_gb,
            require_ngc=True,
            require_cfd=True,
            require_nvidia_smi=True,
            allow_synthetic_fixture=allow_synthetic_fixture,
        )


def _validate_corrector(
    root: Path,
    errors: list[EvidenceIssue],
    checked: list[str],
    *,
    min_vram_gb: int,
    allow_synthetic_fixture: bool,
) -> None:
    path = root / "corrector_smoke" / "manifest.json"
    checked.append(str(path))
    manifest = _load_json_mapping(path, errors)
    if not manifest:
        return
    _require_equal(errors, path, manifest, "schema", expected=CORRECTOR_SCHEMA)
    _validate_evidence_provenance(
        errors,
        path,
        manifest,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    preflight = manifest.get("preflight")
    if isinstance(preflight, dict):
        _validate_preflight(
            errors,
            path,
            preflight,
            min_vram_gb=min_vram_gb,
            require_ngc=False,
            require_cfd=False,
            require_nvidia_smi=False,
            allow_synthetic_fixture=allow_synthetic_fixture,
        )
    else:
        errors.append(_issue(path, "preflight section is missing or invalid"))

    corrector = manifest.get("corrector")
    artifacts = manifest.get("artifacts")
    access_status = manifest.get("access_status")
    if not isinstance(corrector, dict):
        errors.append(_issue(path, "corrector section is missing or invalid"))
        corrector = {}
    if not isinstance(artifacts, dict):
        errors.append(_issue(path, "artifacts section is missing or invalid"))
        artifacts = {}
    if not isinstance(access_status, dict):
        errors.append(_issue(path, "access_status section is missing or invalid"))
        access_status = {}

    _require_equal(
        errors,
        path,
        corrector,
        "model_api",
        expected="physicsnemo.models.mlp.fully_connected.FullyConnected",
    )
    _require_equal(errors, path, corrector, "forward_backward", expected="completed")
    _require_equal(
        errors,
        path,
        corrector,
        "checkpoint_round_trip",
        expected="saved_and_reloaded",
    )
    _require_positive(errors, path, corrector, "latency_ms")
    _require_positive(errors, path, corrector, "throughput_samples_s")
    _require_positive(errors, path, corrector, "peak_vram_mb")
    if _number(corrector, "loss") is None:
        errors.append(_issue(path, "corrector.loss must be a finite number"))
    mixed_precision = corrector.get("mixed_precision")
    if not isinstance(mixed_precision, dict):
        errors.append(_issue(path, "mixed_precision section is missing or invalid"))
    elif mixed_precision.get("amp_dtype") not in {"bf16", "fp16", "fp32"}:
        errors.append(_issue(path, "mixed_precision.amp_dtype is invalid"))
    _require_equal(
        errors,
        path,
        access_status,
        "checkpoint_access",
        expected="local_corrector_checkpoint_saved_and_reloaded",
    )
    checkpoint = _require_file_hash(
        errors,
        path,
        evidence_root=root,
        artifact_value=artifacts.get("checkpoint_path"),
        expected_sha256=artifacts.get("checkpoint_sha256"),
        field_name="corrector checkpoint",
    )
    checked.append(str(checkpoint))


def _validate_drivaerml(
    root: Path,
    errors: list[EvidenceIssue],
    checked: list[str],
    *,
    allow_synthetic_fixture: bool,
) -> None:
    path = root / "drivaerml_external" / "manifest.json"
    checked.append(str(path))
    manifest = _load_json_mapping(path, errors)
    if not manifest:
        return
    _require_equal(errors, path, manifest, "schema", expected=DRIVAERML_SCHEMA)
    _validate_evidence_provenance(
        errors,
        path,
        manifest,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    _require_equal(errors, path, manifest, "source", expected="DrivAerML")
    _require_equal(errors, path, manifest, "license", expected="CC BY-SA 4.0")
    _require_equal(errors, path, manifest, "external_geometry", expected=True)
    _require_equal(errors, path, manifest, "aerocliff_geometry", expected=False)
    _require_equal(errors, path, manifest, "stl_present", expected=True)
    if "connectivity smoke only" not in str(manifest.get("permitted_use", "")):
        errors.append(_issue(path, "permitted_use must restrict DrivAerML to connectivity smoke"))
    stl_path = _require_file_hash(
        errors,
        path,
        evidence_root=root,
        artifact_value=manifest.get("stl_path"),
        expected_sha256=manifest.get("stl_sha256"),
        field_name="DrivAerML STL",
    )
    checked.append(str(stl_path))


def _validate_nim_cache_manifest(
    root: Path,
    manifest_path: Path,
    errors: list[EvidenceIssue],
    checked: list[str],
    *,
    min_vram_gb: int,
    allow_synthetic_fixture: bool,
) -> bool:
    checked.append(str(manifest_path))
    before_count = len(errors)
    manifest = _load_json_mapping(manifest_path, errors)
    if not manifest:
        return False
    _require_equal(errors, manifest_path, manifest, "schema", expected=NIM_CACHE_SCHEMA)
    _validate_evidence_provenance(
        errors,
        manifest_path,
        manifest,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    _require_equal(errors, manifest_path, manifest, "frozen_predictor", expected=True)
    _require_equal(errors, manifest_path, manifest, "trainable", expected=False)
    preflight = manifest.get("preflight")
    if isinstance(preflight, dict):
        _validate_preflight(
            errors,
            manifest_path,
            preflight,
            min_vram_gb=min_vram_gb,
            require_ngc=True,
            require_cfd=True,
            require_nvidia_smi=True,
            allow_synthetic_fixture=allow_synthetic_fixture,
        )
    else:
        errors.append(_issue(manifest_path, "preflight section is missing or invalid"))

    access_status = manifest.get("access_status")
    profile = manifest.get("profile")
    if not isinstance(access_status, dict):
        errors.append(_issue(manifest_path, "access_status section is missing or invalid"))
        access_status = {}
    if not isinstance(profile, dict):
        errors.append(_issue(manifest_path, "profile section is missing or invalid"))
        profile = {}
    _require_equal(errors, manifest_path, access_status, "nim_inference", expected="completed")
    _require_positive(errors, manifest_path, profile, "latency_ms")
    _require_positive(errors, manifest_path, profile, "throughput_cases_s")
    _require_positive(errors, manifest_path, profile, "throughput_points_s")
    for smi_key in ("nvidia_smi_before", "nvidia_smi_after"):
        snapshot = profile.get(smi_key)
        if not isinstance(snapshot, dict):
            errors.append(_issue(manifest_path, f"profile.{smi_key} is missing or invalid"))
        elif snapshot.get("available") is not True or not snapshot.get("query_stdout"):
            errors.append(_issue(manifest_path, f"profile.{smi_key} must prove GPU visibility"))
    if "DrivAerML" not in str(manifest.get("geometry_source", "")):
        errors.append(_issue(manifest_path, "geometry_source must label external DrivAerML smoke"))
    prediction = _require_file_hash(
        errors,
        manifest_path,
        evidence_root=root,
        artifact_value=manifest.get("prediction_path"),
        expected_sha256=manifest.get("prediction_sha256"),
        field_name="NIM prediction",
    )
    checked.append(str(prediction))
    return len(errors) == before_count


def _validate_nim_cache(
    root: Path,
    errors: list[EvidenceIssue],
    checked: list[str],
    *,
    min_vram_gb: int,
    allow_synthetic_fixture: bool,
) -> int:
    cache_root = root / "nim_cache"
    if not cache_root.is_dir():
        errors.append(_issue(cache_root, "NIM cache directory is missing"))
        return 0
    manifests = sorted(cache_root.glob("*/manifest.json"))
    if not manifests:
        errors.append(_issue(cache_root, "at least one NIM cache manifest is required"))
        return 0
    accepted = 0
    for manifest_path in manifests:
        if _validate_nim_cache_manifest(
            root,
            manifest_path,
            errors,
            checked,
            min_vram_gb=min_vram_gb,
            allow_synthetic_fixture=allow_synthetic_fixture,
        ):
            accepted += 1
    if accepted == 0:
        errors.append(_issue(cache_root, "no NIM cache manifest passed validation"))
    return accepted


def validate_cuda_evidence_bundle(
    evidence_root: Path,
    *,
    min_vram_gb: int = 40,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Validate that returned artifacts prove the CUDA/DoMINO lane ran for real."""

    root = evidence_root.resolve()
    errors: list[EvidenceIssue] = []
    checked: list[str] = []
    _validate_doctor(
        root,
        errors,
        checked,
        min_vram_gb=min_vram_gb,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    _validate_corrector(
        root,
        errors,
        checked,
        min_vram_gb=min_vram_gb,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    _validate_drivaerml(
        root,
        errors,
        checked,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    accepted_nim_caches = _validate_nim_cache(
        root,
        errors,
        checked,
        min_vram_gb=min_vram_gb,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "ok": not errors,
        "evidence_root": str(root),
        "accepted_nim_cache_manifests": accepted_nim_caches,
        "checked_artifacts": sorted(set(checked)),
        "errors": [asdict(item) for item in errors],
    }

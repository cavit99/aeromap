"""External DrivAerML pool-benchmark planning.

This module intentionally writes only compact planning manifests. It does not
download the DrivAerML dataset or turn it into AeroCliff campaign evidence.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyvista as pv
from pydantic import BaseModel, Field, PositiveInt, model_validator

from aeromap.io import atomic_write_json

DRIVAERML_DATASET_URL = "https://huggingface.co/datasets/neashton/drivaerml"
DRIVAERML_DATASET_PAGE = "https://neilashton.github.io/caemldatasets/drivaerml/"
DRIVAERML_LICENSE = "CC BY-SA 4.0"
DRIVAERML_BENCHMARK_CLASS: Literal["EXTERNAL_DRIVAERML_POOL_BENCHMARK"] = (
    "EXTERNAL_DRIVAERML_POOL_BENCHMARK"
)
SCHEMA_VERSION = "aerocliff_external_drivaerml_pool_benchmark_plan_v0.1.0"
ASSET_MANIFEST_SCHEMA_VERSION = "aerocliff_external_drivaerml_asset_manifest_v0.1.0"
FIXTURE_SAMPLE_SCHEMA_VERSION = "aerocliff_external_drivaerml_fixture_sample_v0.1.0"
ASSET_STATUS_DRY_RUN = "DRY_RUN_PLANNED"
ASSET_STATUS_PRESENT = "PRESENT_HASHED"
ASSET_STATUS_DOWNLOADED = "DOWNLOADED_HASHED"
ASSET_STATUS_MISSING = "MISSING_NOT_DOWNLOADED"
SAMPLE_MANIFEST_SCHEMA_VERSION = "aerocliff_external_drivaerml_sample_manifest_v0.1.0"
CUDA_BUNDLE_SCHEMA_VERSION = "aerocliff_external_drivaerml_cuda_bundle_v0.1.0"
SAMPLING_MANIFEST_SCHEMA_VERSION = "aerocliff_external_drivaerml_sampling_manifest_v0.1.0"
NORMALISATION_CONTRACT_SCHEMA_VERSION = (
    "aerocliff_external_drivaerml_normalisation_reference_contract_v0.1.0"
)
HELDOUT_PILOT_SPLIT_SCHEMA_VERSION = (
    "aerocliff_external_drivaerml_24_case_heldout_pilot_split_v0.1.0"
)
NIM_IMAGE_TAG = "nvcr.io/nim/nvidia/domino-automotive-aero:2.1.0-41313772"
DEFAULT_HELDOUT_PILOT_CASE_COUNT = 24
DEFAULT_HELDOUT_PILOT_TRAIN_COUNT = 16
DEFAULT_HELDOUT_PILOT_TEST_COUNT = 8
DRIVAERML_CASE_ID_RE = re.compile(r"^run_[0-9]+$")


def _training_eligibility() -> dict[str, bool]:
    return {
        "external_benchmark": True,
        "aerocliff_custom_model": False,
    }


def _claim_eligibility() -> dict[str, bool]:
    return {
        "external_label_efficiency": True,
        "aerocliff_accuracy": False,
        "aerocliff_solver_loop": False,
        "ground_effect_cliff": False,
    }


def _target_contract() -> dict[str, Any]:
    return {
        "surface_cp": {
            "source_field": "CpMeanTrim",
            "association": "cell",
            "query_positions": "cell_centres",
            "weights": "cell_area",
            "units": "dimensionless",
            "normalisation": "DrivAerML-native pressure coefficient",
            "interpolation_policy": "none_for_mvp",
        },
        "integrated_cd": {
            "source_column": "Cd",
            "units": "dimensionless",
            "sign_convention": "DrivAerML-native positive drag",
        },
        "integrated_cl": {
            "source_column": "Cl",
            "units": "dimensionless",
            "sign_convention": "positive_lift",
            "aerocliff_downforce_conversion": "C_DF = -integrated_cl",
        },
    }


def _drivaerml_operating_reference_conditions() -> dict[str, Any]:
    return {
        "source": "DrivAerML-native metadata for Automotive Aero NIM benchmark use",
        "yaw_deg": 0.0,
        "cross_flow_supported": False,
        "expected_velocity_range_m_per_s": [20.0, 50.0],
        "mvp_velocity_m_per_s": 38.89,
        "surface_pressure_field": "CpMeanTrim",
        "reference_note": (
            "CUDA execution must record the exact predictor/corrector reference "
            "conditions resolved from the official PhysicsNeMo/DoMINO workflow."
        ),
    }


def _normalisation_reference_contract() -> dict[str, Any]:
    return {
        "schema_version": NORMALISATION_CONTRACT_SCHEMA_VERSION,
        "domain_id": "drivaerml",
        "dataset_class": DRIVAERML_BENCHMARK_CLASS,
        "solver_fidelity": "high_fidelity_time_averaged_external_aerodynamics",
        "time_representation": "time_averaged",
        "adapter_id": "drivaerml_surface_corrector_v0",
        "coordinate_system": {
            "units": "metres",
            "x_axis": "downstream",
            "y_axis": "lateral",
            "z_axis": "upward",
            "transform_to_nim": "identity_pending_runtime_verification",
            "transform_from_nim": "identity_pending_runtime_verification",
            "coordinate_bounds": "case_surface_bounds_recorded_per_manifest",
        },
        "flow_reference": {
            "freestream_vector_m_per_s": [38.89, 0.0, 0.0],
            "yaw_deg": 0.0,
            "air_density_kg_m3": (
                "dataset_or_official_workflow_value_required_if_dimensional_pressure"
            ),
            "reference_pressure_pa": (
                "dataset_or_official_workflow_value_required_if_dimensional_pressure"
            ),
            "dynamic_pressure_pa": (
                "dataset_or_official_workflow_value_required_if_dimensional_pressure"
            ),
            "reference_area_m2": "dataset_or_official_workflow_value_required_for_scalar_loads",
            "reference_length_m": "dataset_or_official_workflow_value_required_for_moments",
        },
        "surface_target": {
            "name": "surface_cp",
            "source_field": "CpMeanTrim",
            "association": "cell",
            "query_location": "cell_centres",
            "weighting": "cell_area",
            "formula": "(p - p_ref) / q_ref",
            "normalisation": "DrivAerML-native pressure coefficient",
        },
        "predictor": {
            "input_normalisation": "official_automotive_aero_nim_contract",
            "output_units": "runtime_output_schema_must_be_recorded",
            "output_normalisation": "must_match_surface_target_before_residual_metrics",
            "checkpoint_identity": "captured_at_runtime",
        },
        "corrector": {
            "target": "residual_cp",
            "residual_normalisation_source": "training_split_only",
        },
        "fail_closed_rules": [
            "predictor and target Cp definitions must match before residual metrics",
            "coordinate transforms must match between prediction, mapping and labels",
            "cached predictions must record this contract hash",
            "residual statistics must not include validation or test cases",
        ],
    }


def _expected_cuda_output_schema() -> dict[str, Any]:
    return {
        "schema_version": "aerocliff_external_drivaerml_real_asset_cuda_outputs_v0.1.0",
        "classification": "REAL_ASSET_CUDA_PIPELINE_SMOKE",
        "required_outputs": [
            "runtime_environment_manifest",
            "resolved_nim_image_or_model_identity",
            "raw_frozen_nim_prediction_hashes",
            "prediction_to_cpmeantrim_mapping_report",
            "full_surface_area_weighted_predictor_metrics",
            "surface_residual_corrector_manifest_or_surface_only_fallback_manifest",
            "checkpoint_reload_verification",
            "gpu_cpu_ram_latency_throughput_cost_profile",
        ],
        "claim_flags": {
            "generalisation_established": False,
            "benchmark_accuracy_established": False,
            "aerocliff_accuracy_established": False,
            "aerocliff_claim_established": False,
        },
    }


class DrivAerMLSplitConfig(BaseModel):
    """Deterministic finite-pool split sizes for the external benchmark MVP."""

    initial_labelled: PositiveInt = 16
    acquisition_pool: PositiveInt = 64
    calibration: PositiveInt = 32
    heldout_test: PositiveInt = 32
    seed: int = 20260624

    @property
    def total_cases(self) -> int:
        return self.initial_labelled + self.acquisition_pool + self.calibration + self.heldout_test


class DrivAerMLAssetConfig(BaseModel):
    """Asset classes allowed in the bounded MVP plan."""

    stl: bool = True
    boundary_vtp: bool = True
    geometry_parameters: bool = True
    fixed_reference_forces: bool = True
    volume_vtu: bool = False
    images: bool = False


class DrivAerMLBenchmarkConfig(BaseModel):
    """External DrivAerML benchmark contract.

    The defaults follow the public MVP: 16 fixed initial labels,
    64 acquisition-pool cases, 32 calibration cases and 32 held-out cases.
    """

    name: str = "drivaerml_mvp_pool_v0"
    benchmark_class: Literal["EXTERNAL_DRIVAERML_POOL_BENCHMARK"] = DRIVAERML_BENCHMARK_CLASS
    source: Literal["DrivAerML"] = "DrivAerML"
    dataset_url: str = DRIVAERML_DATASET_URL
    dataset_page: str = DRIVAERML_DATASET_PAGE
    license: str = DRIVAERML_LICENSE
    citation_key: str = "ashton2024drivaer"
    first_run_id: PositiveInt = 1
    split: DrivAerMLSplitConfig = Field(default_factory=DrivAerMLSplitConfig)
    assets: DrivAerMLAssetConfig = Field(default_factory=DrivAerMLAssetConfig)
    target_fields: tuple[str, ...] = (
        "surface_cp",
        "integrated_cd",
        "integrated_cl",
    )
    excluded_targets: tuple[str, ...] = (
        "integrated_cm_pitch",
        "volume_fields",
        "wall_shear",
        "separation_fraction",
        "ground_effect_cliff",
    )
    acquisition_methods: tuple[str, ...] = (
        "random",
        "maximum_ensemble_uncertainty",
        "geometry_parameter_maximin",
        "uncertainty_plus_diversity",
    )
    ensemble_members: PositiveInt = 3
    initial_label_count: PositiveInt = 16
    acquisition_batch_size: PositiveInt = 8
    maximum_label_count: PositiveInt = 48
    acquisition_seeds: tuple[int, ...] = (11, 17, 23)

    @model_validator(mode="after")
    def _check_counts(self) -> DrivAerMLBenchmarkConfig:
        if self.initial_label_count != self.split.initial_labelled:
            msg = "initial_label_count must match split.initial_labelled"
            raise ValueError(msg)
        if self.maximum_label_count <= self.initial_label_count:
            msg = "maximum_label_count must exceed initial_label_count"
            raise ValueError(msg)
        acquired = self.maximum_label_count - self.initial_label_count
        if acquired > self.split.acquisition_pool:
            msg = "maximum_label_count requires more acquired labels than acquisition_pool provides"
            raise ValueError(msg)
        if acquired % self.acquisition_batch_size != 0:
            msg = (
                "maximum_label_count - initial_label_count must be divisible by "
                "acquisition_batch_size"
            )
            raise ValueError(msg)
        if self.assets.volume_vtu:
            msg = "volume_vtu is disabled for the bounded MVP plan"
            raise ValueError(msg)
        return self


def _run_id(run_number: int) -> str:
    return f"run_{run_number}"


def _case_asset_plan(run_number: int, assets: DrivAerMLAssetConfig) -> dict[str, str]:
    run = _run_id(run_number)
    prefix = f"{DRIVAERML_DATASET_URL}/resolve/main/{run}"
    plan: dict[str, str] = {}
    if assets.stl:
        plan["stl"] = f"{prefix}/drivaer_{run_number}.stl"
    if assets.boundary_vtp:
        plan["boundary_vtp"] = f"{prefix}/boundary_{run_number}.vtp"
    if assets.geometry_parameters:
        plan["geometry_parameters"] = f"{prefix}/geo_parameters_{run_number}.csv"
    if assets.fixed_reference_forces:
        plan["fixed_reference_forces"] = f"{prefix}/force_mom_constref_{run_number}.csv"
    return plan


def _split_case_ids(config: DrivAerMLBenchmarkConfig) -> dict[str, list[int]]:
    cases = list(
        range(config.first_run_id, config.first_run_id + config.split.total_cases),
    )
    random.Random(config.split.seed).shuffle(cases)  # noqa: S311 - deterministic split seed.
    start = 0
    stop = config.split.initial_labelled
    initial = cases[start:stop]
    start = stop
    stop += config.split.acquisition_pool
    pool = cases[start:stop]
    start = stop
    stop += config.split.calibration
    calibration = cases[start:stop]
    start = stop
    stop += config.split.heldout_test
    test = cases[start:stop]
    return {
        "initial_labelled": sorted(initial),
        "acquisition_pool": sorted(pool),
        "calibration": sorted(calibration),
        "heldout_test": sorted(test),
    }


def _case_records(case_ids: list[int], assets: DrivAerMLAssetConfig) -> list[dict[str, Any]]:
    return [
        {
            "case_id": _run_id(case_id),
            "run_number": case_id,
            "geometry_group": _run_id(case_id),
            "assets": _case_asset_plan(case_id, assets),
            "hash_status": "PENDING_DOWNLOAD",
        }
        for case_id in case_ids
    ]


def build_drivaerml_benchmark_plan(config: DrivAerMLBenchmarkConfig) -> dict[str, Any]:
    """Return the deterministic external benchmark planning manifest."""

    split_ids = _split_case_ids(config)
    all_ids = [case_id for split in split_ids.values() for case_id in split]
    if len(set(all_ids)) != len(all_ids):
        msg = "DrivAerML split contains duplicate case IDs"
        raise ValueError(msg)
    acquisition_rounds = (
        config.maximum_label_count - config.initial_label_count
    ) // config.acquisition_batch_size
    return {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "benchmark_class": config.benchmark_class,
        "source": config.source,
        "dataset_url": config.dataset_url,
        "dataset_page": config.dataset_page,
        "license": config.license,
        "citation_key": config.citation_key,
        "dataset_class": DRIVAERML_BENCHMARK_CLASS,
        "aerocliff_geometry": False,
        "aerocliff_campaign_cfd": False,
        "live_solver_loop": False,
        "offline_pool_replay": True,
        "training_eligibility": _training_eligibility(),
        "claim_eligibility": _claim_eligibility(),
        "permitted_use": (
            "External open-data benchmark for frozen DoMINO correction, uncertainty "
            "and pool-based active-learning replay."
        ),
        "prohibited_claims": [
            "AeroCliff campaign CFD",
            "AeroCliff custom geometry evidence",
            "AeroCliff Venturi-underfloor cliff discovery",
            "live CFD solver-in-the-loop result",
            "ground truth for AeroCliff",
        ],
        "split": {
            "seed": config.split.seed,
            "counts": {
                "initial_labelled": config.split.initial_labelled,
                "acquisition_pool": config.split.acquisition_pool,
                "calibration": config.split.calibration,
                "heldout_test": config.split.heldout_test,
                "total": config.split.total_cases,
            },
            "case_ids": split_ids,
        },
        "cases": {
            split_name: _case_records(case_ids, config.assets)
            for split_name, case_ids in split_ids.items()
        },
        "assets": config.assets.model_dump(),
        "target_fields": list(config.target_fields),
        "target_contract": _target_contract(),
        "excluded_targets": list(config.excluded_targets),
        "operating_reference_conditions": _drivaerml_operating_reference_conditions(),
        "model_strategy": {
            "authoritative_model": "official PhysicsNeMo/DoMINO trainable corrector",
            "baseline": "frozen DoMINO predictor",
            "ensemble_members": config.ensemble_members,
            "local_dev_model": "project-owned lightweight PyTorch residual head",
        },
        "active_learning": {
            "initial_label_count": config.initial_label_count,
            "acquisition_batch_size": config.acquisition_batch_size,
            "maximum_label_count": config.maximum_label_count,
            "acquisition_rounds": acquisition_rounds,
            "seeds": list(config.acquisition_seeds),
            "methods": list(config.acquisition_methods),
            "sobol_reserved_for_aerocliff_parametric_campaign": True,
        },
        "download_policy": {
            "status": "PLAN_ONLY_NO_DOWNLOAD",
            "full_dataset_size_note": "Do not download the full DrivAerML dataset for MVP.",
            "initial_assets_only": [
                key for key, enabled in config.assets.model_dump().items() if enabled
            ],
        },
        "cost_policy": {
            "ec2_usage": "not_required_for_plan_generation",
            "launch_gpu_only_after": [
                "deterministic split manifest exists",
                "selected assets are downloaded and hashed",
                "CUDA command bundle verifies sample semantics locally",
                "spot price and expected runtime are recorded",
            ],
            "avoid": [
                "idle spot instances",
                "full dataset downloads",
                "volume VTU downloads before surface/load MVP",
                "CPU-bound preprocessing on GPU instances",
            ],
            "gpu_instance_policy": {
                "starting_family": "g6e",
                "minimum_gpu_memory_gb": 48,
                "resize_by_evidence": True,
                "prefer_lower_total_cost_over_smaller_instance": True,
            },
            "required_runtime_observability": [
                "gpu_model",
                "driver_version",
                "cuda_version",
                "gpu_utilization_percent",
                "vram_used_mb",
                "cpu_utilization_percent",
                "system_ram_used_mb",
                "batch_size",
                "latency_ms",
                "throughput_samples_per_second",
                "wall_clock_seconds",
                "estimated_usd",
            ],
        },
    }


def write_drivaerml_benchmark_plan(
    config: DrivAerMLBenchmarkConfig,
    out_path: Path,
) -> Path:
    """Write the deterministic external DrivAerML benchmark plan."""

    atomic_write_json(out_path, build_drivaerml_benchmark_plan(config))
    return out_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_case_id(case_id: str) -> str:
    stripped = case_id.strip()
    if stripped.isdigit():
        return _run_id(int(stripped))
    return stripped


def _validate_case_id(case_id: str) -> str:
    normalised = _normalise_case_id(case_id)
    if not DRIVAERML_CASE_ID_RE.fullmatch(normalised):
        msg = f"Invalid DrivAerML case_id: {case_id}"
        raise ValueError(msg)
    return normalised


def _url_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        msg = f"DrivAerML asset URL does not contain a filename: {url}"
        raise ValueError(msg)
    return name


def _validate_https_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        msg = f"DrivAerML asset URL must use https: {url}"
        raise ValueError(msg)
    expected = urllib.parse.urlparse(DRIVAERML_DATASET_URL)
    if parsed.netloc != expected.netloc:
        msg = f"DrivAerML asset URL host is not trusted: {url}"
        raise ValueError(msg)
    expected_prefix = f"{expected.path.rstrip('/')}/resolve/main/"
    if not parsed.path.startswith(expected_prefix):
        msg = f"DrivAerML asset URL path is outside the expected dataset revision path: {url}"
        raise ValueError(msg)


def _load_plan(plan_path: Path) -> dict[str, Any]:
    try:
        payload = cast("dict[str, Any]", json.loads(plan_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read DrivAerML plan {plan_path}: {exc}"
        raise ValueError(msg) from exc
    if payload.get("benchmark_class") != DRIVAERML_BENCHMARK_CLASS:
        msg = "DrivAerML asset manifests require EXTERNAL_DRIVAERML_POOL_BENCHMARK plans"
        raise ValueError(msg)
    return payload


def _case_lookup(plan: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    cases_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for split_name, records in plan.get("cases", {}).items():
        for record in records:
            case_id = _validate_case_id(str(record["case_id"]))
            if case_id in cases_by_id:
                msg = f"Duplicate DrivAerML case ID in plan: {case_id}"
                raise ValueError(msg)
            cases_by_id[case_id] = (str(split_name), dict(record))
    return cases_by_id


def _selected_case_records(
    plan: dict[str, Any],
    *,
    splits: tuple[str, ...],
    case_ids: tuple[str, ...],
    max_cases: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    cases_by_id = _case_lookup(plan)
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    split_names = splits

    for split_name in split_names:
        records = plan.get("cases", {}).get(split_name)
        if records is None:
            msg = f"Unknown DrivAerML split: {split_name}"
            raise ValueError(msg)
        for record in records:
            case_id = _validate_case_id(str(record["case_id"]))
            if case_id not in seen:
                selected.append((split_name, dict(record)))
                seen.add(case_id)

    for raw_case_id in case_ids:
        case_id = _validate_case_id(raw_case_id)
        if case_id not in cases_by_id:
            msg = f"Unknown DrivAerML case_id: {raw_case_id}"
            raise ValueError(msg)
        if case_id not in seen:
            split_name, record = cases_by_id[case_id]
            selected.append((split_name, dict(record)))
            seen.add(case_id)

    if max_cases is not None:
        if max_cases <= 0:
            msg = "max_cases must be positive when supplied"
            raise ValueError(msg)
        selected = selected[:max_cases]
    return selected


def _asset_local_path(
    *,
    cache_dir: Path,
    case_id: str,
    asset_name: str,
    url: str,
) -> Path:
    case_id = _validate_case_id(case_id)
    _validate_https_url(url)
    filename = _url_filename(url)
    if asset_name == "fixed_reference_forces" and filename == "force_mom_all.csv":
        candidate = cache_dir / "_global" / filename
    else:
        candidate = cache_dir / case_id / filename
    root = cache_dir.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        msg = f"DrivAerML asset path escapes cache_dir: {candidate}"
        raise ValueError(msg)
    return candidate


def _download_asset(url: str, local_path: Path) -> None:
    _validate_https_url(url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{local_path.name}.",
        dir=local_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:  # noqa: S310
            shutil.copyfileobj(response, handle)
        tmp_path.replace(local_path)
    except (OSError, TimeoutError, urllib.error.URLError):
        tmp_path.unlink(missing_ok=True)
        raise


def _content_length(url: str) -> int | None:
    _validate_https_url(url)
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            length = response.headers.get("Content-Length")
    except (OSError, TimeoutError, urllib.error.URLError):
        return None
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def build_drivaerml_asset_manifest(
    plan_path: Path,
    *,
    cache_dir: Path,
    splits: tuple[str, ...] = ("initial_labelled",),
    case_ids: tuple[str, ...] = (),
    max_cases: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build a selected DrivAerML asset manifest without network access by default."""

    plan = _load_plan(plan_path)
    selected = _selected_case_records(
        plan,
        splits=splits,
        case_ids=case_ids,
        max_cases=max_cases,
    )
    if plan.get("assets", {}).get("volume_vtu"):
        msg = "DrivAerML MVP asset manifests prohibit volume_vtu"
        raise ValueError(msg)

    asset_records: list[dict[str, Any]] = []
    downloaded_bytes = 0
    known_estimated_bytes = 0
    unknown_byte_count_assets = 0
    for split_name, case in selected:
        case_id = str(case["case_id"])
        for asset_name, url in case.get("assets", {}).items():
            if asset_name == "volume_vtu":
                msg = "DrivAerML MVP asset manifests prohibit volume_vtu"
                raise ValueError(msg)
            local_path = _asset_local_path(
                cache_dir=cache_dir,
                case_id=case_id,
                asset_name=str(asset_name),
                url=str(url),
            )
            sha256: str | None = None
            byte_count: int | None = None
            estimated_byte_count: int | None = None
            if local_path.exists():
                sha256 = _file_sha256(local_path)
                byte_count = local_path.stat().st_size
                estimated_byte_count = byte_count
                status = ASSET_STATUS_PRESENT
            elif dry_run:
                status = ASSET_STATUS_DRY_RUN
            else:
                estimated_byte_count = _content_length(str(url))
                _download_asset(str(url), local_path)
                sha256 = _file_sha256(local_path)
                byte_count = local_path.stat().st_size
                downloaded_bytes += byte_count
                status = ASSET_STATUS_DOWNLOADED

            byte_count_for_estimate = (
                estimated_byte_count if estimated_byte_count is not None else byte_count
            )
            if byte_count_for_estimate is None:
                unknown_byte_count_assets += 1
            else:
                known_estimated_bytes += byte_count_for_estimate

            asset_records.append(
                {
                    "case_id": case_id,
                    "run_number": case["run_number"],
                    "geometry_group": case["geometry_group"],
                    "split": split_name,
                    "asset_name": str(asset_name),
                    "url": str(url),
                    "expected_path": str(local_path),
                    "estimated_byte_count": estimated_byte_count,
                    "byte_count": byte_count,
                    "sha256": sha256,
                    "status": status,
                },
            )

    selected_case_ids = [str(case["case_id"]) for _, case in selected]
    selected_split_names = list(dict.fromkeys(split_name for split_name, _ in selected))
    manifest = {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "benchmark_class": plan["benchmark_class"],
        "source": plan["source"],
        "dataset_url": plan["dataset_url"],
        "dataset_page": plan["dataset_page"],
        "license": plan["license"],
        "citation_key": plan["citation_key"],
        "dataset_class": plan["dataset_class"],
        "plan_path": str(plan_path),
        "plan_sha256": _json_sha256(plan),
        "selected_splits": selected_split_names,
        "selected_case_ids": selected_case_ids,
        "selected_case_count": len(selected_case_ids),
        "aerocliff_geometry": False,
        "aerocliff_campaign_cfd": False,
        "live_solver_loop": False,
        "training_eligibility": _training_eligibility(),
        "claim_eligibility": _claim_eligibility(),
        "offline_pool_replay": True,
        "no_volume_vtu_confirmed": True,
        "target_contract": plan["target_contract"],
        "operating_reference_conditions": plan["operating_reference_conditions"],
        "assets": asset_records,
        "storage": {
            "cache_dir": str(cache_dir),
            "known_estimated_bytes": known_estimated_bytes,
            "unknown_byte_count_assets": unknown_byte_count_assets,
            "downloaded_bytes": downloaded_bytes,
            "local_download_budget_bytes": 0 if dry_run else None,
        },
        "download_policy": {
            "dry_run": dry_run,
            "download_requested": not dry_run,
            "network_access": "disabled_by_default" if dry_run else "explicit_download_requested",
            "no_full_dataset_download": True,
            "no_volume_vtu_download": True,
        },
        "cost_policy": {
            **plan["cost_policy"],
            "ec2_usage": "not_required_for_asset_manifest",
            "default_local_download_budget_bytes": 0,
        },
        "data_ready": not dry_run
        and all(
            asset["status"] in {ASSET_STATUS_PRESENT, ASSET_STATUS_DOWNLOADED}
            for asset in asset_records
        ),
        "plan_evidence_ready": True,
    }
    validation = validate_drivaerml_asset_manifest_payload(manifest)
    manifest["validation"] = validation
    return manifest


def write_drivaerml_asset_manifest(
    plan_path: Path,
    out_path: Path,
    *,
    cache_dir: Path,
    splits: tuple[str, ...] = ("initial_labelled",),
    case_ids: tuple[str, ...] = (),
    max_cases: int | None = None,
    dry_run: bool = True,
) -> Path:
    """Write a selected DrivAerML asset manifest."""

    atomic_write_json(
        out_path,
        build_drivaerml_asset_manifest(
            plan_path,
            cache_dir=cache_dir,
            splits=splits,
            case_ids=case_ids,
            max_cases=max_cases,
            dry_run=dry_run,
        ),
    )
    return out_path


def validate_drivaerml_asset_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a DrivAerML asset manifest without making dry-run assets data-ready."""

    issues: list[str] = []
    if manifest.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        issues.append("unexpected schema_version")
    if manifest.get("benchmark_class") != DRIVAERML_BENCHMARK_CLASS:
        issues.append("benchmark_class must be EXTERNAL_DRIVAERML_POOL_BENCHMARK")
    issues.extend(
        f"missing {key}"
        for key in (
            "source",
            "license",
            "citation_key",
            "dataset_url",
            "dataset_page",
        )
        if not manifest.get(key)
    )
    issues.extend(
        f"{flag} must be false"
        for flag in (
            "aerocliff_geometry",
            "aerocliff_campaign_cfd",
            "live_solver_loop",
        )
        if manifest.get(flag) is not False
    )
    if manifest.get("dataset_class") != DRIVAERML_BENCHMARK_CLASS:
        issues.append("dataset_class must be EXTERNAL_DRIVAERML_POOL_BENCHMARK")
    if manifest.get("training_eligibility") != _training_eligibility():
        issues.append("training_eligibility must restrict training to external_benchmark only")
    if manifest.get("claim_eligibility") != _claim_eligibility():
        issues.append("claim_eligibility must prohibit AeroCliff accuracy/solver/cliff claims")
    if manifest.get("target_contract") != _target_contract():
        issues.append("target_contract must preserve DrivAerML Cp/Cd/Cl semantics")
    if manifest.get("offline_pool_replay") is not True:
        issues.append("offline_pool_replay must be true")
    if manifest.get("no_volume_vtu_confirmed") is not True:
        issues.append("no_volume_vtu_confirmed must be true")

    selected_case_ids = [str(case_id) for case_id in manifest.get("selected_case_ids", [])]
    if len(selected_case_ids) != len(set(selected_case_ids)):
        issues.append("selected_case_ids contains duplicates")
    if not selected_case_ids:
        issues.append("selected_case_ids is empty")

    allowed_status = {
        ASSET_STATUS_DRY_RUN,
        ASSET_STATUS_PRESENT,
        ASSET_STATUS_DOWNLOADED,
        ASSET_STATUS_MISSING,
    }
    for asset in manifest.get("assets", []):
        asset_name = str(asset.get("asset_name", ""))
        if asset_name == "volume_vtu":
            issues.append("volume_vtu asset is prohibited")
        status = str(asset.get("status", ""))
        if status not in allowed_status:
            issues.append(f"unknown status for {asset.get('case_id')}:{asset_name}")
        if str(asset.get("case_id", "")) not in selected_case_ids:
            issues.append(f"asset references unselected case {asset.get('case_id')}")
        if status in {ASSET_STATUS_PRESENT, ASSET_STATUS_DOWNLOADED}:
            expected_path = Path(str(asset.get("expected_path", "")))
            sha256 = asset.get("sha256")
            if not expected_path.exists():
                issues.append(f"hashed asset missing on disk: {expected_path}")
            elif not isinstance(sha256, str) or _file_sha256(expected_path) != sha256:
                issues.append(f"hashed asset integrity mismatch: {expected_path}")

    data_ready = all(
        str(asset.get("status", "")) in {ASSET_STATUS_PRESENT, ASSET_STATUS_DOWNLOADED}
        for asset in manifest.get("assets", [])
    )
    if manifest.get("download_policy", {}).get("dry_run") and manifest.get("data_ready"):
        issues.append("dry-run manifest cannot be data_ready")
    return {
        "ok": not issues,
        "data_ready": data_ready and not manifest.get("download_policy", {}).get("dry_run", False),
        "dry_run_plan_evidence": bool(manifest.get("download_policy", {}).get("dry_run", False)),
        "issues": issues,
    }


def validate_drivaerml_asset_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and validate a DrivAerML asset manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "data_ready": False, "issues": [str(exc)]}
    return validate_drivaerml_asset_manifest_payload(manifest)


def _load_asset_manifest(asset_manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = cast(
            "dict[str, Any]",
            json.loads(asset_manifest_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read DrivAerML asset manifest {asset_manifest_path}: {exc}"
        raise ValueError(msg) from exc
    validation = validate_drivaerml_asset_manifest_payload(manifest)
    if not validation["ok"]:
        msg = f"DrivAerML asset manifest is invalid: {validation['issues']}"
        raise ValueError(msg)
    if not validation["data_ready"]:
        msg = "DrivAerML asset manifest is not data-ready"
        raise ValueError(msg)
    return manifest


def _assets_by_case(asset_manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for asset in asset_manifest.get("assets", []):
        case_id = str(asset["case_id"])
        asset_name = str(asset["asset_name"])
        grouped.setdefault(case_id, {})[asset_name] = dict(asset)
    return grouped


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        msg = f"Expected exactly one row in {path}, found {len(rows)}"
        raise ValueError(msg)
    return rows[0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_number_from_case_id(case_id: str) -> int:
    normalised = _normalise_case_id(case_id)
    try:
        return int(normalised.replace("run_", ""))
    except ValueError:
        return 0


def _read_numeric_geometry_parameters(path: Path) -> dict[str, float]:
    row = _read_single_row_csv(path)
    excluded = {
        "id",
        "case",
        "caseid",
        "run",
        "runnumber",
        "index",
    }
    values: dict[str, float] = {}
    for key, value in row.items():
        normalised = _normalise_column(key)
        if normalised in excluded or not value.strip():
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        if math.isfinite(numeric):
            values[str(key)] = numeric
    if not values:
        msg = f"No finite non-label geometry parameters found in {path}"
        raise ValueError(msg)
    return values


def _standardise_feature_matrix(
    records: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    feature_names = sorted(
        {
            name
            for record in records
            for name in cast("dict[str, float]", record["geometry_parameters"])
        },
    )
    if not feature_names:
        msg = "No geometry features available for DrivAerML held-out pilot split"
        raise ValueError(msg)
    matrix = np.empty((len(records), len(feature_names)), dtype=np.float64)
    missing: list[str] = []
    for row_index, record in enumerate(records):
        params = cast("dict[str, float]", record["geometry_parameters"])
        for column_index, feature_name in enumerate(feature_names):
            if feature_name not in params:
                missing.append(f"{record['case_id']}:{feature_name}")
                matrix[row_index, column_index] = np.nan
            else:
                matrix[row_index, column_index] = params[feature_name]
    if missing:
        msg = "Geometry parameter columns must be present for every selected case: " + ", ".join(
            missing[:10],
        )
        raise ValueError(msg)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    zero_variance = std == 0.0
    safe_std = np.where(zero_variance, 1.0, std)
    standardised = (matrix - mean) / safe_std
    stats = {
        "feature_names": feature_names,
        "mean": {name: float(value) for name, value in zip(feature_names, mean, strict=True)},
        "std": {name: float(value) for name, value in zip(feature_names, safe_std, strict=True)},
        "zero_variance_features": [
            name for name, is_zero in zip(feature_names, zero_variance, strict=True) if is_zero
        ],
    }
    return feature_names, standardised, stats


def _maximin_case_selection(
    *,
    case_ids: list[str],
    features: np.ndarray,
    count: int,
) -> list[str]:
    if count <= 0:
        msg = "maximin selection count must be positive"
        raise ValueError(msg)
    if count > len(case_ids):
        msg = "Cannot select requested number of cases from available candidates"
        raise ValueError(msg)
    if count == len(case_ids):
        return sorted(case_ids, key=_run_number_from_case_id)

    selected_indices: list[int] = []
    remaining = set(range(len(case_ids)))
    centroid = features.mean(axis=0)
    first_index = max(
        remaining,
        key=lambda index: (
            float(np.linalg.norm(features[index] - centroid)),
            -_run_number_from_case_id(case_ids[index]),
        ),
    )
    selected_indices.append(first_index)
    remaining.remove(first_index)

    while len(selected_indices) < count:
        next_index = max(
            remaining,
            key=lambda index: (
                float(
                    np.min(
                        np.linalg.norm(features[index] - features[selected_indices], axis=1),
                    ),
                ),
                -_run_number_from_case_id(case_ids[index]),
            ),
        )
        selected_indices.append(next_index)
        remaining.remove(next_index)
    return [case_ids[index] for index in selected_indices]


def build_drivaerml_heldout_pilot_split(
    asset_manifest_path: Path,
    *,
    pilot_case_count: int = DEFAULT_HELDOUT_PILOT_CASE_COUNT,
    train_count: int = DEFAULT_HELDOUT_PILOT_TRAIN_COUNT,
    test_count: int = DEFAULT_HELDOUT_PILOT_TEST_COUNT,
) -> dict[str, Any]:
    """Build the immutable 24-case held-out pilot split from geometry parameters only."""

    if train_count + test_count != pilot_case_count:
        msg = "train_count + test_count must equal pilot_case_count"
        raise ValueError(msg)
    asset_manifest = _load_asset_manifest(asset_manifest_path)
    grouped_assets = _assets_by_case(asset_manifest)
    records: list[dict[str, Any]] = []
    for case_id in asset_manifest["selected_case_ids"]:
        assets = grouped_assets[str(case_id)]
        geometry_record = _case_asset_record(assets, "geometry_parameters")
        geometry_path = _case_asset_path(assets, "geometry_parameters")
        records.append(
            {
                "case_id": str(case_id),
                "run_number": int(geometry_record["run_number"]),
                "geometry_group": str(geometry_record["geometry_group"]),
                "split_source": str(geometry_record["split"]),
                "geometry_parameters_path": str(geometry_path),
                "geometry_parameters_sha256": str(geometry_record["sha256"]),
                "geometry_parameters": _read_numeric_geometry_parameters(geometry_path),
            },
        )
    if len(records) < pilot_case_count:
        msg = (
            f"DrivAerML held-out pilot requires at least {pilot_case_count} "
            f"geometry-parameter cases, found {len(records)}"
        )
        raise ValueError(msg)
    case_ids = [str(record["case_id"]) for record in records]
    _feature_names, standardised, selection_standardisation = _standardise_feature_matrix(records)
    pool_ids = _maximin_case_selection(
        case_ids=case_ids,
        features=standardised,
        count=pilot_case_count,
    )
    pool_indices = [case_ids.index(case_id) for case_id in pool_ids]
    pool_features = standardised[pool_indices]
    heldout_ids = _maximin_case_selection(
        case_ids=pool_ids,
        features=pool_features,
        count=test_count,
    )
    train_candidates = [case_id for case_id in pool_ids if case_id not in set(heldout_ids)]
    train_features = standardised[[case_ids.index(case_id) for case_id in train_candidates]]
    train_ids = _maximin_case_selection(
        case_ids=train_candidates,
        features=train_features,
        count=train_count,
    )
    if set(train_ids) & set(heldout_ids):
        msg = "DrivAerML held-out pilot split produced overlapping train/test cases"
        raise ValueError(msg)

    records_by_case = {str(record["case_id"]): record for record in records}
    _train_feature_names, _train_standardised, model_standardisation = _standardise_feature_matrix(
        [records_by_case[case_id] for case_id in train_ids],
    )
    split_payload = {
        "schema_version": HELDOUT_PILOT_SPLIT_SCHEMA_VERSION,
        "experiment_id": "DRIVAERML_24_CASE_HELDOUT_PILOT",
        "dataset_class": DRIVAERML_BENCHMARK_CLASS,
        "domain_id": "drivaerml",
        "aerocliff_claim_eligible": False,
        "active_learning_claim_eligible": False,
        "exact_nvidia_reproduction": False,
        "protocol": "NVIDIA-inspired predictor-corrector protocol",
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": _file_sha256(asset_manifest_path),
        "selection_method": {
            "candidate_pool": "maximin_farthest_first_over_standardised_geometry_parameters",
            "heldout_test": "maximin_farthest_first_over_selected_pool",
            "train": "maximin_farthest_first_over_remaining_pool",
            "file_order_used_for_split": False,
            "selection_standardisation_scope": "available_geometry_cases_for_selection_only",
            "model_feature_standardisation_scope": "training_split_only",
        },
        "counts": {
            "available_geometry_cases": len(records),
            "pilot_cases": pilot_case_count,
            "train": train_count,
            "heldout_test": test_count,
        },
        "case_ids": {
            "pilot_pool_selection_order": pool_ids,
            "train_selection_order": train_ids,
            "heldout_test_selection_order": heldout_ids,
            "train": sorted(train_ids, key=_run_number_from_case_id),
            "heldout_test": sorted(heldout_ids, key=_run_number_from_case_id),
            "all_pilot": sorted(pool_ids, key=_run_number_from_case_id),
        },
        "geometry_parameter_selection_standardisation": selection_standardisation,
        "geometry_parameter_selection_standardisation_sha256": _json_sha256(
            selection_standardisation,
        ),
        "geometry_parameter_standardisation": model_standardisation,
        "geometry_parameter_standardisation_scope": "training_split_only",
        "geometry_parameter_standardisation_sha256": _json_sha256(model_standardisation),
        "cases": [
            {
                key: value
                for key, value in records_by_case[case_id].items()
                if key != "geometry_parameters"
            }
            | {"role": "train" if case_id in set(train_ids) else "heldout_test"}
            for case_id in sorted(pool_ids, key=_run_number_from_case_id)
        ],
        "label_access_policy": {
            "test_labels_may_be_used_for_preprocessing_statistics": False,
            "test_labels_may_be_used_for_hyperparameter_selection": False,
            "test_labels_may_be_used_for_early_stopping": False,
            "test_labels_may_be_used_for_checkpoint_selection": False,
            "residual_normalisation_source": "training_split_only",
            "epoch_budget": "fixed_predeclared",
        },
        "target_scope": {
            "primary_target": "surface_cp",
            "diagnostics_only": ["Cd", "Cl", "Clf", "Clr"],
            "excluded": [
                "scalar_residual_heads",
                "pitch_moment",
                "volume_fields",
                "wall_shear",
                "active_learning",
                "uncertainty_ensemble",
            ],
        },
        "pass_criteria": {
            "corrected_mean_heldout_nrmse_below_predictor_only": True,
            "corrected_median_heldout_nrmse_below_predictor_only": True,
            "minimum_improved_heldout_cases": 6,
            "maximum_allowed_case_degradation_fraction": 0.25,
            "checkpoint_reload_required": True,
            "complete_cuda_provenance_cost_evidence_required": True,
        },
    }
    split_payload["split_manifest_sha256"] = _json_sha256(split_payload)
    return split_payload


def write_drivaerml_heldout_pilot_split(
    asset_manifest_path: Path,
    out_path: Path,
    *,
    pilot_case_count: int = DEFAULT_HELDOUT_PILOT_CASE_COUNT,
    train_count: int = DEFAULT_HELDOUT_PILOT_TRAIN_COUNT,
    test_count: int = DEFAULT_HELDOUT_PILOT_TEST_COUNT,
) -> Path:
    """Write the DrivAerML 24-case held-out pilot split manifest."""

    atomic_write_json(
        out_path,
        build_drivaerml_heldout_pilot_split(
            asset_manifest_path,
            pilot_case_count=pilot_case_count,
            train_count=train_count,
            test_count=test_count,
        ),
    )
    return out_path


def _normalise_column(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _find_column(row: dict[str, str], candidates: set[str]) -> str | None:
    for key in row:
        if _normalise_column(key) in candidates:
            return key
    return None


def _find_force_row(rows: list[dict[str, str]], *, case_id: str, run_number: int) -> dict[str, str]:
    run_values = {str(run_number), case_id, case_id.replace("run_", "")}
    for row in rows:
        for key, value in row.items():
            normalised = _normalise_column(key)
            if normalised in {"run", "runnumber", "case", "caseid", "id"} and value in run_values:
                return row
    if len(rows) == 1:
        return rows[0]
    msg = f"Could not find force/moment row for {case_id}"
    raise ValueError(msg)


def _float_from_row(row: dict[str, str], candidates: set[str], *, target_name: str) -> float:
    key = _find_column(row, candidates)
    if key is None:
        msg = f"Could not find {target_name} in columns {sorted(row)}"
        raise ValueError(msg)
    return float(row[key])


def _optional_float_from_row(
    row: dict[str, str], candidates: set[str]
) -> tuple[float | None, str | None]:
    key = _find_column(row, candidates)
    if key is None:
        return None, None
    return float(row[key]), key


def _surface_cp_summary(surface: pv.DataSet) -> dict[str, Any]:
    field_name = "CpMeanTrim"
    if field_name not in surface.cell_data:
        msg = (
            "DrivAerML MVP boundary VTP must expose cell-associated CpMeanTrim. "
            f"Point arrays: {list(surface.point_data.keys())}; "
            f"cell arrays: {list(surface.cell_data.keys())}"
        )
        raise ValueError(msg)
    data = np.asarray(surface.cell_data[field_name])
    finite = np.isfinite(data)
    if not finite.any():
        msg = f"DrivAerML Cp field {field_name} has no finite values"
        raise ValueError(msg)
    finite_values = data[finite]
    return {
        "surface_cp": {
            "source_field": field_name,
            "association": "cell",
            "query_positions": "cell_centres",
            "weights": "cell_area",
            "normalisation": "DrivAerML-native pressure coefficient",
            "units": "dimensionless",
            "tuple_count": int(data.shape[0]),
            "component_count": int(data.shape[1]) if data.ndim > 1 else 1,
            "finite_count": int(finite.sum()),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
        },
    }


def _case_asset_path(assets: dict[str, dict[str, Any]], asset_name: str) -> Path:
    try:
        return Path(str(assets[asset_name]["expected_path"]))
    except KeyError as exc:
        msg = f"Missing required DrivAerML asset {asset_name}"
        raise ValueError(msg) from exc


def _case_asset_record(assets: dict[str, dict[str, Any]], asset_name: str) -> dict[str, Any]:
    try:
        return assets[asset_name]
    except KeyError as exc:
        msg = f"Missing required DrivAerML asset {asset_name}"
        raise ValueError(msg) from exc


def build_drivaerml_sample_manifest(asset_manifest_path: Path) -> dict[str, Any]:
    """Build a compact sample manifest from downloaded DrivAerML MVP assets."""

    asset_manifest = _load_asset_manifest(asset_manifest_path)
    grouped_assets = _assets_by_case(asset_manifest)
    force_rows_by_path: dict[Path, list[dict[str, str]]] = {}
    cases: list[dict[str, Any]] = []

    for case_id in asset_manifest["selected_case_ids"]:
        assets = grouped_assets[str(case_id)]
        boundary_path = _case_asset_path(assets, "boundary_vtp")
        geometry_path = _case_asset_path(assets, "geometry_parameters")
        fixed_forces_path = _case_asset_path(assets, "fixed_reference_forces")
        surface = pv.read(boundary_path)
        geometry_row = _read_single_row_csv(geometry_path)
        force_rows = force_rows_by_path.setdefault(
            fixed_forces_path, _read_csv_rows(fixed_forces_path)
        )
        run_number = int(_case_asset_record(assets, "boundary_vtp")["run_number"])
        force_row = _find_force_row(force_rows, case_id=str(case_id), run_number=run_number)
        cd_value, cd_source = _optional_float_from_row(
            force_row,
            {
                "cd",
                "drag",
                "dragcoefficient",
                "coefdrag",
                "coefficientdrag",
                "integratedcd",
            },
        )
        vertical_value, vertical_source = _optional_float_from_row(
            force_row,
            {
                "cl",
                "lift",
                "liftcoefficient",
                "coefficientlift",
                "integratedcl",
            },
        )
        clf_value, clf_source = _optional_float_from_row(
            force_row,
            {
                "clf",
                "frontlift",
                "liftfront",
                "integratedclf",
            },
        )
        clr_value, clr_source = _optional_float_from_row(
            force_row,
            {
                "clr",
                "rearlift",
                "liftrear",
                "integratedclr",
            },
        )
        if cd_value is None:
            msg = f"Could not find integrated_cd in columns {sorted(force_row)}"
            raise ValueError(msg)
        if vertical_value is None:
            msg = f"Could not find integrated_cl in columns {sorted(force_row)}"
            raise ValueError(msg)
        integrated_targets = {
            "integrated_cd": cd_value,
            "integrated_cl": vertical_value,
        }
        integrated_target_sources = {
            "integrated_cd": {"source_column": cd_source, "available": True},
            "integrated_cl": {
                "source_column": vertical_source,
                "available": True,
                "note": (
                    "External DrivAerML lift coefficient; sign convention is positive_lift. "
                    "AeroCliff positive downforce is C_DF = -integrated_cl and is not "
                    "stored as the external target."
                ),
            },
        }
        scalar_diagnostics = {
            "integrated_cl_front": clf_value,
            "integrated_cl_rear": clr_value,
        }
        scalar_diagnostic_sources = {
            "integrated_cl_front": {
                "source_column": clf_source,
                "available": clf_value is not None,
            },
            "integrated_cl_rear": {
                "source_column": clr_source,
                "available": clr_value is not None,
            },
            "integrated_cm_pitch": {
                "source_column": None,
                "available": False,
                "note": (
                    "Pitching moment is unavailable in the selected DrivAerML fixed-reference "
                    "force CSV and is disabled for the MVP target set until a documented "
                    "moment convention is implemented."
                ),
            },
        }
        cases.append(
            {
                "case_id": str(case_id),
                "run_number": run_number,
                "geometry_group": str(_case_asset_record(assets, "boundary_vtp")["geometry_group"]),
                "split": str(_case_asset_record(assets, "boundary_vtp")["split"]),
                "source": asset_manifest["source"],
                "license": asset_manifest["license"],
                "citation_key": asset_manifest["citation_key"],
                "dataset_class": asset_manifest["dataset_class"],
                "training_eligibility": _training_eligibility(),
                "claim_eligibility": _claim_eligibility(),
                "assets": {
                    name: {
                        "path": str(_case_asset_path(assets, name)),
                        "sha256": str(_case_asset_record(assets, name)["sha256"]),
                        "byte_count": int(_case_asset_record(assets, name)["byte_count"]),
                    }
                    for name in (
                        "stl",
                        "boundary_vtp",
                        "geometry_parameters",
                        "fixed_reference_forces",
                    )
                },
                "geometry_parameters": geometry_row,
                "force_columns": sorted(force_row),
                "surface": {
                    "path": str(boundary_path),
                    "point_count": int(surface.n_points),
                    "cell_count": int(surface.n_cells),
                    "point_arrays": list(surface.point_data.keys()),
                    "cell_arrays": list(surface.cell_data.keys()),
                },
                "surface_targets": _surface_cp_summary(surface),
                "integrated_targets": integrated_targets,
                "integrated_target_sources": integrated_target_sources,
                "scalar_diagnostics": scalar_diagnostics,
                "scalar_diagnostic_sources": scalar_diagnostic_sources,
            },
        )

    return {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "benchmark_class": DRIVAERML_BENCHMARK_CLASS,
        "source": asset_manifest["source"],
        "dataset_url": asset_manifest["dataset_url"],
        "dataset_page": asset_manifest["dataset_page"],
        "license": asset_manifest["license"],
        "citation_key": asset_manifest["citation_key"],
        "dataset_class": asset_manifest["dataset_class"],
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": _file_sha256(asset_manifest_path),
        "selected_case_count": len(cases),
        "aerocliff_geometry": False,
        "aerocliff_campaign_cfd": False,
        "live_solver_loop": False,
        "training_eligibility": _training_eligibility(),
        "claim_eligibility": _claim_eligibility(),
        "offline_pool_replay": True,
        "no_volume_vtu_confirmed": True,
        "target_contract": _target_contract(),
        "operating_reference_conditions": asset_manifest["operating_reference_conditions"],
        "targets": [
            "surface_cp",
            "integrated_cd",
            "integrated_cl",
        ],
        "excluded_targets": [
            "integrated_cm_pitch",
            "volume_fields",
            "wall_shear",
            "separation_fraction",
            "ground_effect_cliff",
        ],
        "cases": cases,
        "data_ready": True,
    }


def write_drivaerml_sample_manifest(asset_manifest_path: Path, out_path: Path) -> Path:
    """Write a compact sample manifest from downloaded DrivAerML MVP assets."""

    atomic_write_json(out_path, build_drivaerml_sample_manifest(asset_manifest_path))
    return out_path


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _surface_cell_areas(path: Path) -> np.ndarray:
    surface = pv.read(path)
    sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
    if "Area" not in sized.cell_data:
        msg = f"{path} did not produce cell-area data"
        raise ValueError(msg)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    if areas.shape != (surface.n_cells,):
        msg = f"{path} generated unexpected area array shape {areas.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        msg = f"{path} generated non-positive or non-finite cell areas"
        raise ValueError(msg)
    return areas


def build_drivaerml_sampling_manifest(
    sample_manifest_path: Path,
    *,
    seed: int = 20260624,
    sample_count: int = 65_536,
) -> dict[str, Any]:
    """Build deterministic area-weighted sampled-cell manifests for CUDA training."""

    if sample_count <= 0:
        msg = "sample_count must be positive"
        raise ValueError(msg)
    sample_manifest = _load_json_object(sample_manifest_path, label="sample manifest")
    if sample_manifest.get("data_ready") is not True:
        msg = "DrivAerML sampling manifest requires a data-ready sample manifest"
        raise ValueError(msg)
    if sample_manifest.get("dataset_class") != DRIVAERML_BENCHMARK_CLASS:
        msg = "DrivAerML sampling manifest requires EXTERNAL_DRIVAERML_POOL_BENCHMARK"
        raise ValueError(msg)

    contract = _normalisation_reference_contract()
    cases: list[dict[str, Any]] = []
    for case_index, case in enumerate(sample_manifest.get("cases", [])):
        case_id = str(case["case_id"])
        boundary_path = Path(str(case["assets"]["boundary_vtp"]["path"]))
        areas = _surface_cell_areas(boundary_path)
        weights = areas / float(areas.sum())
        effective_count = min(int(sample_count), int(areas.shape[0]))
        rng = np.random.default_rng(seed + case_index)
        sampled_ids = np.sort(
            rng.choice(
                areas.shape[0],
                size=effective_count,
                replace=False,
                p=weights,
            ).astype(np.int64),
        )
        sampled_probabilities = weights[sampled_ids].astype(np.float64)
        sampled_areas = areas[sampled_ids].astype(np.float64)
        sample_payload = {
            "case_id": case_id,
            "surface_path": str(boundary_path),
            "surface_sha256": case["assets"]["boundary_vtp"]["sha256"],
            "cell_count": int(areas.shape[0]),
            "sample_count_requested": int(sample_count),
            "sample_count": int(effective_count),
            "seed": int(seed + case_index),
            "sampling": "area_weighted_without_replacement",
            "sampled_cell_ids": sampled_ids.tolist(),
            "sampled_cell_ids_sha256": _array_sha256(sampled_ids),
            "sampling_probabilities": sampled_probabilities.tolist(),
            "sampling_probabilities_sha256": _array_sha256(sampled_probabilities),
            "sampled_cell_areas": sampled_areas.tolist(),
            "sampled_cell_areas_sha256": _array_sha256(sampled_areas),
            "area_sum_full_surface": float(areas.sum()),
            "area_sum_sampled_cells": float(sampled_areas.sum()),
        }
        sample_payload["preprocessing_hash"] = _json_sha256(
            {
                "case_id": case_id,
                "surface_sha256": sample_payload["surface_sha256"],
                "sampled_cell_ids_sha256": sample_payload["sampled_cell_ids_sha256"],
                "sampling_probabilities_sha256": sample_payload["sampling_probabilities_sha256"],
                "sampled_cell_areas_sha256": sample_payload["sampled_cell_areas_sha256"],
                "normalisation_reference_contract_sha256": _json_sha256(contract),
            },
        )
        cases.append(sample_payload)

    return {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "created_at": _now_utc(),
        "dataset_class": DRIVAERML_BENCHMARK_CLASS,
        "domain_id": "drivaerml",
        "adapter_id": "drivaerml_surface_corrector_v0",
        "sample_manifest_path": str(sample_manifest_path),
        "sample_manifest_sha256": _file_sha256(sample_manifest_path),
        "normalisation_reference_contract": contract,
        "normalisation_reference_contract_sha256": _json_sha256(contract),
        "training_sampling": {
            "deterministic_seed": int(seed),
            "sample_count": int(sample_count),
            "sampled_cell_ids_recorded": True,
            "sampling_probabilities_recorded": True,
            "cell_areas_retained": True,
            "checkpoint_round_trip_sample_set": "same_sampled_cells",
        },
        "evaluation": {
            "surface": "full_surface",
            "execution": "bounded_chunks",
            "metrics": "area_weighted_all_cells",
            "training_sample_metrics_are_not_final_metrics": True,
        },
        "cases": cases,
        "case_count": len(cases),
    }


def write_drivaerml_sampling_manifest(
    sample_manifest_path: Path,
    out_path: Path,
    *,
    seed: int = 20260624,
    sample_count: int = 65_536,
) -> Path:
    """Write deterministic sampled-cell manifests for CUDA training smoke."""

    atomic_write_json(
        out_path,
        build_drivaerml_sampling_manifest(
            sample_manifest_path,
            seed=seed,
            sample_count=sample_count,
        ),
    )
    return out_path


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_head_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    result = subprocess.run(  # noqa: S603 - fixed git command; no user-controlled args.
        [git, "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read {label} {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{label} must be a JSON object: {path}"
        raise TypeError(msg)
    return cast("dict[str, Any]", payload)


def _verify_surface_cell_geometry(path: Path) -> dict[str, Any]:
    surface = pv.read(path)
    if "CpMeanTrim" not in surface.cell_data:
        msg = f"{path} does not expose cell-data CpMeanTrim"
        raise ValueError(msg)
    cp = np.asarray(surface.cell_data["CpMeanTrim"])
    finite = np.isfinite(cp)
    if not finite.all():
        msg = f"{path} CpMeanTrim contains non-finite values"
        raise ValueError(msg)
    centers = surface.cell_centers(vertex=False)
    sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
    if "Area" not in sized.cell_data:
        msg = f"{path} did not produce cell-area data"
        raise ValueError(msg)
    areas = np.asarray(sized.cell_data["Area"])
    positive_areas = np.isfinite(areas) & (areas > 0.0)
    if centers.n_points != surface.n_cells:
        msg = f"{path} generated {centers.n_points} cell centres for {surface.n_cells} cells"
        raise ValueError(msg)
    if not positive_areas.all():
        msg = f"{path} generated non-positive or non-finite cell areas"
        raise ValueError(msg)
    return {
        "path": str(path),
        "point_count": int(surface.n_points),
        "cell_count": int(surface.n_cells),
        "cp_field": "CpMeanTrim",
        "cp_association": "cell",
        "cp_finite_count": int(finite.sum()),
        "cp_min": float(cp.min()),
        "cp_max": float(cp.max()),
        "cell_centres_generated": True,
        "cell_centre_count": int(centers.n_points),
        "cell_areas_generated": True,
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
        "area_sum": float(areas.sum()),
    }


def _bundle_pin(value: str | None) -> dict[str, Any]:
    if value:
        return {"value": value, "status": "pinned"}
    return {
        "value": None,
        "status": "required_before_cuda_launch",
    }


def _runtime_capture(name: str) -> dict[str, str]:
    return {"name": name, "status": "captured_at_runtime"}


def _post_training_capture(name: str) -> dict[str, str]:
    return {"name": name, "status": "captured_after_training"}


def _drivaerml_remote_commands(bundle_path: Path) -> list[dict[str, str]]:
    return [
        {
            "name": "remote_cuda_execution",
            "command": (
                "CUDA/NIM execution is outside the public compact release. "
                "Use this bundle as the data/contract input for a private GPU runner."
            ),
        },
        {
            "name": "bundle_semantics_to_preserve",
            "command": (
                "uv run aeromap benchmark drivaerml-cuda-bundle "
                "--asset-manifest "
                "artifacts/benchmark/drivaerml_assets_manifest_initial2_downloaded.json "
                "--sample-manifest artifacts/benchmark/drivaerml_sample_manifest_initial2.json "
                f"--out {bundle_path.as_posix()}"
            ),
        },
        {
            "name": "sampling_manifest_to_preserve",
            "command": (
                "uv run aeromap benchmark drivaerml-sampling "
                "--sample-manifest artifacts/benchmark/drivaerml_sample_manifest_initial2.json "
                "--out artifacts/benchmark/drivaerml_sampling_manifest_initial2.json"
            ),
        },
    ]


def build_drivaerml_cuda_bundle(
    asset_manifest_path: Path,
    sample_manifest_path: Path,
    *,
    out_path: Path,
    dataset_revision: str | None = None,
    physicsnemo_commit: str | None = None,
    physicsnemo_cfd_commit: str | None = None,
    nim_image_tag: str = NIM_IMAGE_TAG,
    nim_image_digest: str | None = None,
    predictor_checkpoint_digest: str | None = None,
    corrector_initialisation_checkpoint_digest: str | None = None,
    corrector_checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    """Build a verified local command bundle for the future CUDA DrivAerML smoke."""

    asset_manifest = _load_json_object(asset_manifest_path, label="asset manifest")
    asset_validation = validate_drivaerml_asset_manifest_payload(asset_manifest)
    if not asset_validation["ok"] or not asset_validation["data_ready"]:
        msg = f"DrivAerML asset manifest is not data-ready: {asset_validation}"
        raise ValueError(msg)
    sample_manifest = _load_json_object(sample_manifest_path, label="sample manifest")
    issues: list[str] = []
    if sample_manifest.get("data_ready") is not True:
        issues.append("sample manifest must be data_ready")
    if sample_manifest.get("asset_manifest_sha256") != _file_sha256(asset_manifest_path):
        issues.append("sample manifest asset_manifest_sha256 does not match current asset manifest")
    for payload_name, payload in (
        ("asset manifest", asset_manifest),
        ("sample manifest", sample_manifest),
    ):
        if payload.get("dataset_class") != DRIVAERML_BENCHMARK_CLASS:
            issues.append(f"{payload_name} dataset_class is not EXTERNAL_DRIVAERML_POOL_BENCHMARK")
        if payload.get("training_eligibility") != _training_eligibility():
            issues.append(f"{payload_name} training_eligibility is incorrect")
        if payload.get("claim_eligibility") != _claim_eligibility():
            issues.append(f"{payload_name} claim_eligibility is incorrect")
        if payload.get("target_contract") != _target_contract():
            issues.append(f"{payload_name} target_contract is incorrect")
        if payload.get("no_volume_vtu_confirmed") is not True:
            issues.append(f"{payload_name} must confirm no volume VTU")

    case_surface_checks = []
    for case in sample_manifest.get("cases", []):
        surface_targets = case.get("surface_targets", {}).get("surface_cp", {})
        if surface_targets.get("source_field") != "CpMeanTrim":
            issues.append(f"{case.get('case_id')} surface_cp source_field must be CpMeanTrim")
        if surface_targets.get("association") != "cell":
            issues.append(f"{case.get('case_id')} surface_cp must be cell-associated")
        targets = case.get("integrated_targets", {})
        if "integrated_cl" not in targets:
            issues.append(f"{case.get('case_id')} missing integrated_cl")
        if "integrated_cz_or_cdf" in targets or "integrated_cm_pitch" in targets:
            issues.append(f"{case.get('case_id')} contains disabled/ambiguous integrated target")
        boundary_path = Path(str(case.get("assets", {}).get("boundary_vtp", {}).get("path", "")))
        try:
            case_surface_checks.append(_verify_surface_cell_geometry(boundary_path))
        except ValueError as exc:
            issues.append(str(exc))

    normalisation_contract = _normalisation_reference_contract()
    expected_output_schema = _expected_cuda_output_schema()
    run_config = {
        "classification": "REAL_ASSET_CUDA_PIPELINE_SMOKE",
        "selected_case_ids": [
            str(case.get("case_id")) for case in sample_manifest.get("cases", [])
        ],
        "sample_manifest_sha256": _file_sha256(sample_manifest_path),
        "asset_manifest_sha256": _file_sha256(asset_manifest_path),
        "target_contract": _target_contract(),
        "normalisation_reference_contract_sha256": _json_sha256(
            normalisation_contract,
        ),
        "expected_output_schema_sha256": _json_sha256(expected_output_schema),
        "no_volume_vtu_confirmed": True,
    }
    pre_launch_requirements = {
        "nim_image_tag": _bundle_pin(nim_image_tag),
        "huggingface_dataset_revision": _bundle_pin(dataset_revision),
        "physicsnemo_commit": _bundle_pin(physicsnemo_commit),
        "physicsnemo_cfd_commit": _bundle_pin(physicsnemo_cfd_commit),
        "asset_manifest_sha256": _bundle_pin(_file_sha256(asset_manifest_path)),
        "sample_manifest_sha256": _bundle_pin(_file_sha256(sample_manifest_path)),
        "run_config_hash": _bundle_pin(_json_sha256(run_config)),
        "normalisation_reference_contract_hash": _bundle_pin(
            _json_sha256(normalisation_contract),
        ),
        "expected_output_schema_hash": _bundle_pin(_json_sha256(expected_output_schema)),
        "preprocessing_code_commit": _bundle_pin(_git_head_sha()),
    }
    runtime_captures = {
        "resolved_nim_image_digest": _bundle_pin(nim_image_digest)
        if nim_image_digest
        else _runtime_capture("resolved_nim_image_digest"),
        "nim_model_or_version_identity": _runtime_capture("nim_model_or_version_identity"),
        "predictor_artifact_or_model_manifest_identity": _bundle_pin(
            predictor_checkpoint_digest,
        )
        if predictor_checkpoint_digest
        else _runtime_capture("predictor_artifact_or_model_manifest_identity"),
        "cuda_gpu_package_environment_manifest": _runtime_capture(
            "cuda_gpu_package_environment_manifest",
        ),
    }
    post_training_outputs = {
        "corrector_initialisation_checkpoint_digest": _bundle_pin(
            corrector_initialisation_checkpoint_digest,
        )
        if corrector_initialisation_checkpoint_digest
        else {"value": None, "status": "not_used_for_this_launch_unless_supplied"},
        "trained_corrector_checkpoint_digest": _bundle_pin(corrector_checkpoint_digest)
        if corrector_checkpoint_digest
        else _post_training_capture("trained_corrector_checkpoint_digest"),
        "optimizer_state_digest": _post_training_capture("optimizer_state_digest"),
        "final_config_digest": _post_training_capture("final_config_digest"),
    }
    missing_prelaunch = [
        name for name, pin in pre_launch_requirements.items() if pin["status"] != "pinned"
    ]
    verification_ok = not issues
    return {
        "schema_version": CUDA_BUNDLE_SCHEMA_VERSION,
        "created_at": _now_utc(),
        "bundle_class": "REAL_ASSET_CUDA_PIPELINE_SMOKE_PREP",
        "classification": "EXTERNAL_DRIVAERML_POOL_BENCHMARK",
        "local_preparation_complete": verification_ok,
        "cuda_launch_ready": verification_ok and not missing_prelaunch,
        "cuda_launch_blockers": [
            f"missing pre-launch requirement: {name}" for name in missing_prelaunch
        ],
        "source_git_sha": _git_head_sha(),
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": _file_sha256(asset_manifest_path),
        "sample_manifest_path": str(sample_manifest_path),
        "sample_manifest_sha256": _file_sha256(sample_manifest_path),
        "selected_case_ids": [
            str(case.get("case_id")) for case in sample_manifest.get("cases", [])
        ],
        "selected_case_count": sample_manifest.get("selected_case_count"),
        "dataset_class": DRIVAERML_BENCHMARK_CLASS,
        "training_eligibility": _training_eligibility(),
        "claim_eligibility": _claim_eligibility(),
        "target_contract": _target_contract(),
        "normalisation_reference_contract": normalisation_contract,
        "normalisation_reference_contract_sha256": _json_sha256(
            normalisation_contract,
        ),
        "operating_reference_conditions": _drivaerml_operating_reference_conditions(),
        "run_config": run_config,
        "run_config_sha256": _json_sha256(run_config),
        "expected_output_schema": expected_output_schema,
        "expected_output_schema_sha256": _json_sha256(expected_output_schema),
        "surface_checks": case_surface_checks,
        "pre_launch_requirements": pre_launch_requirements,
        "runtime_captures": runtime_captures,
        "post_training_outputs": post_training_outputs,
        "source_pins": pre_launch_requirements,
        "verification": {
            "ok": verification_ok,
            "issues": issues,
            "asset_manifest_validation": asset_validation,
            "checked_no_volume_vtu": True,
            "checked_hashes": True,
            "checked_cp_cell_data": True,
            "checked_cell_centres_and_areas": True,
            "checked_secret_policy_environment_only": True,
        },
        "secret_policy": {
            "ngc_api_key_sources": ["NGC_API_KEY", "NGC_API_KEY_FILE", "AWS SSM SecureString"],
            "secrets_written_to_manifest": False,
            "repo_secret_storage_allowed": False,
        },
        "cost_and_observability_policy": {
            "starting_instance_family": "g6e",
            "starting_gpu": "L40S",
            "minimum_vram_gb": 48,
            "compare_before_launch": ["g6e.xlarge", "g6e.2xlarge"],
            "prefer_same_region_s3_bucket": True,
            "resize_decision": (
                "Use measured VRAM, GPU utilization, CPU utilization, batch size, "
                "throughput, wall time and spot price to minimize total USD, not "
                "instance hourly price alone."
            ),
            "required_logs": [
                "nvidia-smi query before/during/after",
                "GPU utilization and VRAM at fixed interval",
                "CPU utilization and system memory at fixed interval",
                "batch size and point count",
                "predictor latency and throughput",
                "corrector forward/backward latency and throughput",
                "wall-clock seconds",
                "spot price USD/hour",
                "estimated USD/run",
                "estimated USD/case",
            ],
        },
        "remote_commands": _drivaerml_remote_commands(out_path),
        "outputs_to_hash_after_cuda": [
            "frozen NIM prediction cache manifest",
            "surface pressure prediction",
            "volume velocity/pressure prediction if produced by official workflow",
            "area-weighted Cp metric JSON",
            "surface-only corrector checkpoint",
            "VRAM/latency/throughput profile JSON",
        ],
        "prohibited_claims": [
            "AeroCliff custom geometry accuracy",
            "AeroCliff solver-loop completion",
            "ground-effect cliff discovery",
            "training eligibility for AeroCliff campaign labels",
        ],
    }


def write_drivaerml_cuda_bundle(
    asset_manifest_path: Path,
    sample_manifest_path: Path,
    out_path: Path,
    *,
    dataset_revision: str | None = None,
    physicsnemo_commit: str | None = None,
    physicsnemo_cfd_commit: str | None = None,
    nim_image_tag: str = NIM_IMAGE_TAG,
    nim_image_digest: str | None = None,
    predictor_checkpoint_digest: str | None = None,
    corrector_initialisation_checkpoint_digest: str | None = None,
    corrector_checkpoint_digest: str | None = None,
) -> Path:
    """Write the verified local command bundle for future CUDA execution."""

    atomic_write_json(
        out_path,
        build_drivaerml_cuda_bundle(
            asset_manifest_path,
            sample_manifest_path,
            out_path=out_path,
            dataset_revision=dataset_revision,
            physicsnemo_commit=physicsnemo_commit,
            physicsnemo_cfd_commit=physicsnemo_cfd_commit,
            nim_image_tag=nim_image_tag,
            nim_image_digest=nim_image_digest,
            predictor_checkpoint_digest=predictor_checkpoint_digest,
            corrector_initialisation_checkpoint_digest=corrector_initialisation_checkpoint_digest,
            corrector_checkpoint_digest=corrector_checkpoint_digest,
        ),
    )
    return out_path


def load_drivaerml_fixture_sample(case_dir: Path) -> dict[str, Any]:
    """Load a tiny synthetic DrivAerML fixture sample used by local interface tests."""

    manifest_path = case_dir / "case_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read DrivAerML fixture manifest {manifest_path}: {exc}"
        raise ValueError(msg) from exc

    if manifest.get("benchmark_class") != DRIVAERML_BENCHMARK_CLASS:
        msg = "DrivAerML fixture must use EXTERNAL_DRIVAERML_POOL_BENCHMARK"
        raise ValueError(msg)
    if manifest.get("source") != "DrivAerML" or manifest.get("license") != DRIVAERML_LICENSE:
        msg = "DrivAerML fixture must preserve source and CC BY-SA 4.0 licence metadata"
        raise ValueError(msg)

    surface_path = case_dir / str(manifest["surface_vtp"])
    loads_path = case_dir / str(manifest["loads_csv"])
    surface = pv.read(surface_path)
    if "CpMeanTrim" in surface.cell_data:
        cp = surface.cell_data["CpMeanTrim"]
    elif "surface_cp" in surface.cell_data:
        cp = surface.cell_data["surface_cp"]
    else:
        msg = "DrivAerML fixture surface must contain cell-associated CpMeanTrim or surface_cp"
        raise ValueError(msg)

    with loads_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        msg = "DrivAerML fixture loads CSV must contain exactly one data row"
        raise ValueError(msg)
    row = rows[0]
    vertical_key = next(
        (key for key in ("integrated_cl", "Cl", "cl") if key in row),
        None,
    )
    if vertical_key is None:
        msg = "DrivAerML fixture loads CSV must contain an integrated_cl target"
        raise ValueError(msg)
    return {
        "schema_version": FIXTURE_SAMPLE_SCHEMA_VERSION,
        "case_id": manifest["case_id"],
        "geometry_group": manifest["geometry_group"],
        "split": manifest["split"],
        "benchmark_class": manifest["benchmark_class"],
        "source": manifest["source"],
        "license": manifest["license"],
        "citation_key": manifest["citation_key"],
        "surface_targets": {"surface_cp": cp.tolist()},
        "surface_target_contract": {
            "surface_cp": {
                "association": "cell",
                "query_positions": "cell_centres",
                "weights": "cell_area",
            },
        },
        "integrated_targets": {
            "integrated_cd": float(row["integrated_cd"]),
            "integrated_cl": float(row[vertical_key]),
        },
        "training_eligibility": _training_eligibility(),
        "claim_eligibility": _claim_eligibility(),
    }

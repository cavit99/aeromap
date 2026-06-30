"""AeroMap Mission Control open-CFD budget benchmark utilities.

The first implementation is deliberately compact: it proves the acquisition
protocol, target governance and reporting loop locally before any larger model,
cloud run or custom AeroCliff CFD label is introduced.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, PositiveInt, model_validator

from aeromap.io import atomic_write_json, atomic_write_text, sha256_file

AEROMAP_CLASS: Literal["OPEN_CFD_AEROMAP_BUDGET_BENCHMARK"] = "OPEN_CFD_AEROMAP_BUDGET_BENCHMARK"
AEROMAP_PLAN_SCHEMA = "aerocliff_aeromap_budget_plan_v0.1.0"
AEROMAP_FEASIBILITY_SCHEMA = "aerocliff_airfrans_feasibility_v0.1.0"
AEROMAP_DATASET_SCHEMA = "aerocliff_aeromap_fixture_dataset_v0.1.0"
AEROMAP_AIRFRANS_SCALAR_SCHEMA = "aerocliff_aeromap_airfrans_scalar_dataset_v0.1.0"
AEROMAP_AIRFRANS_GEOMETRY_SCHEMA = "aerocliff_aeromap_airfrans_geometry_dataset_v0.2.0"
AEROMAP_REPLAY_SCHEMA = "aerocliff_aeromap_active_replay_v0.1.0"
AEROMAP_DECISION_REPLAY_SCHEMA = "aerocliff_aeromap_decision_replay_v0.2.0"
AEROMAP_DECISION_REPLAY_V03_SCHEMA = "aerocliff_aeromap_decision_replay_v0.3.0"
AEROMAP_AUDIT_SCHEMA = "aerocliff_aeromap_airfrans_audit_v0.3.0"

AIRFRANS_DOCS_URL = "https://airfrans.readthedocs.io/en/latest/notes/dataset.html"
AIRFRANS_DATASET_URL = "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip"
AIRFRANS_LICENSE = "ODbL-1.0"
AIRFRANS_CITATION_KEY = "bonnet2022airfrans"
DEFAULT_TOP_K_FRACTION = 0.10
RIDGE_LAMBDA = 1.0e-3
EPSILON = 1.0e-12
DECISION_METRICS = (
    "rmse_cd",
    "rmse_cl",
    "top_k_efficiency_overlap",
    "pareto_recall",
    "spearman_efficiency",
    "best_design_regret",
)
ERROR_METRICS = ("rmse_cd", "rmse_cl")
LOWER_IS_BETTER_METRICS = {"rmse_cd", "rmse_cl", "mae_cd", "mae_cl", "best_design_regret"}
HEADLINE_READY_DECISION_METRIC_COUNT = 3
ARRAY_RANK_TWO = 2
MIN_XY_COLUMNS = 2
MIN_LABELLED_CASES = 2
MIN_AIRFRANS_NAME_TOKENS = 5
MAX_SHAPE_PARAMETERS = 6
GEOMETRY_STATION_COUNT = 16
GEOMETRY_COSINE_MODES = 6

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
ForceCoefficient = tuple[tuple[float, float, float], tuple[float, float, float]]


class AirfransSimulation(Protocol):
    """Subset of the AirfRANS Simulation API needed for scalar targets."""

    def force_coefficient(self, *, reference: bool) -> ForceCoefficient: ...


SimulationFactory = Callable[[Path, str], AirfransSimulation]


class AeroMapConfig(BaseModel):
    """Local open-CFD benchmark plan for the Mission Control MVP."""

    name: str = "aeromap_mission_control_mvp"
    benchmark_class: Literal["OPEN_CFD_AEROMAP_BUDGET_BENCHMARK"] = AEROMAP_CLASS
    preferred_dataset: Literal["AirfRANS", "DrivAerML", "HiLiftAeroML", "AhmedML"] = "AirfRANS"
    seed: int = 20260628
    fixture_case_count: PositiveInt = 240
    initial_labels: PositiveInt = 24
    acquisition_batch: PositiveInt = 12
    max_labels: PositiveInt = 96
    test_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    ensemble_members: PositiveInt = 5
    replay_seeds: tuple[int, ...] = ()
    acquisition_methods: tuple[str, ...] = (
        "random",
        "diversity",
        "uncertainty",
        "uncertainty_plus_diversity",
        "engineering_utility",
    )

    @model_validator(mode="after")
    def _validate_budget(self) -> AeroMapConfig:
        if self.max_labels <= self.initial_labels:
            msg = "max_labels must exceed initial_labels"
            raise ValueError(msg)
        if self.acquisition_batch >= self.max_labels:
            msg = "acquisition_batch must be smaller than max_labels"
            raise ValueError(msg)
        if self.fixture_case_count < self.max_labels + 20:
            msg = "fixture_case_count must leave enough train/pool/test cases"
            raise ValueError(msg)
        unknown = set(self.acquisition_methods) - {
            "random",
            "diversity",
            "uncertainty",
            "uncertainty_plus_diversity",
            "engineering_utility",
            "engineering_decision_utility_v1",
            "engineering_decision_utility_v2_regret_aware",
            "engineering_decision_utility_v2_cost_aware",
        }
        if unknown:
            msg = f"unknown acquisition methods: {sorted(unknown)}"
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class DatasetArrays:
    case_ids: list[str]
    features: FloatArray
    targets: FloatArray
    feature_names: list[str]
    target_names: list[str]
    classification: str = "AEROMAP_AIRFRANS_CONTRACT_FIXTURE"
    open_cfd_result: bool = False
    group_ids: list[str] | None = None


@dataclass(frozen=True)
class SplitArrays:
    train_pool_indices: IndexArray
    test_indices: IndexArray


@dataclass(frozen=True)
class StandardScaler:
    mean: FloatArray
    scale: FloatArray

    def transform(self, values: FloatArray) -> FloatArray:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class RidgeEnsemble:
    weights: FloatArray
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def predict_members(self, features: FloatArray) -> FloatArray:
        x_scaled = self.x_scaler.transform(features)
        design = polynomial_features(x_scaled)
        y_scaled = np.einsum("nd,mdt->mnt", design, self.weights)
        return cast(
            "FloatArray",
            y_scaled * self.y_scaler.scale.reshape(1, 1, -1) + self.y_scaler.mean.reshape(1, 1, -1),
        )

    def predict(self, features: FloatArray) -> FloatArray:
        return cast("FloatArray", self.predict_members(features).mean(axis=0))

    def uncertainty(self, features: FloatArray) -> FloatArray:
        members = self.predict_members(features)
        return cast("FloatArray", np.linalg.norm(members.std(axis=0), axis=1))


def build_aeromap_plan(config: AeroMapConfig) -> dict[str, Any]:
    """Build the local Mission Control benchmark plan."""

    return {
        "schema_version": AEROMAP_PLAN_SCHEMA,
        "name": config.name,
        "benchmark_class": config.benchmark_class,
        "headline": "AeroMap Mission Control",
        "purpose": "positive_budgeted_learning_result",
        "lanes": {
            "lane_a": {
                "name": "AEROMAP_BUDGET_BENCHMARK",
                "purpose": "learn aerodynamic behaviour under limited CFD labels",
            },
            "lane_b": {
                "name": "AEROCLIFF_CORE_EXTENSION",
                "purpose": "custom underfloor Core benchmark and future 3D transfer lane",
            },
        },
        "dataset_order": [
            "AirfRANS first if scalar coefficient contract is defensible",
            "3D automotive open data only after the first learning curve",
            "AeroCliff Core extension after custom targets are available",
        ],
        "cost_policy": {
            "cloud": "forbidden_for_this_goal",
            "ec2": "forbidden_for_this_goal",
            "nim": "forbidden_for_this_goal",
            "custom_cfd_solves": "forbidden_for_this_goal",
        },
        "claim_boundaries": {
            "not_aerocliff_cfd": True,
            "not_f1_geometry": True,
            "aerocliff_accuracy_claim": False,
            "ground_effect_cliff_claim": False,
            "live_solver_savings_claim": False,
        },
        "open_dataset": {
            "preferred": config.preferred_dataset,
            "airfrans_docs": AIRFRANS_DOCS_URL,
            "airfrans_dataset_url": AIRFRANS_DATASET_URL,
            "license": AIRFRANS_LICENSE,
            "citation_key": AIRFRANS_CITATION_KEY,
            "mandatory_gate": (
                "verify scalar lift/drag target availability before real-data benchmark"
            ),
        },
        "budget_protocol": {
            "fixture_case_count": config.fixture_case_count,
            "initial_labels": config.initial_labels,
            "acquisition_batch": config.acquisition_batch,
            "max_labels": config.max_labels,
            "test_fraction": config.test_fraction,
            "ensemble_members": config.ensemble_members,
            "acquisition_methods": list(config.acquisition_methods),
        },
        "engineering_metrics": [
            "rmse_cd",
            "rmse_cl",
            "mae_cd",
            "mae_cl",
            "spearman_efficiency",
            "top_k_efficiency_overlap",
            "pareto_recall",
            "learning_curve_area",
        ],
        "stop_conditions": [
            "no custom AeroCliff CFD",
            "no cloud",
            "no NIM",
            "no DrivAerML scale-up",
            "custom AeroCliff CFD labels remain outside this benchmark",
            "stop if AirfRANS scalar targets are ambiguous",
        ],
    }


def write_aeromap_plan(config: AeroMapConfig, out: Path) -> Path:
    payload = build_aeromap_plan(config)
    atomic_write_json(out, payload)
    return out


def _airfrans_package_status() -> dict[str, Any]:
    try:
        airfrans = importlib.import_module("airfrans")
        simulation = importlib.import_module("airfrans.simulation")
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "module_path": None,
            "simulation_force_coefficient_available": False,
            "error": "airfrans package is not importable",
        }

    return {
        "installed": True,
        "version": str(getattr(airfrans, "__version__", "unknown")),
        "module_path": str(getattr(airfrans, "__file__", "")),
        "simulation_force_coefficient_available": hasattr(
            getattr(simulation, "Simulation", object),
            "force_coefficient",
        ),
        "error": None,
    }


def _dataset_content_length(url: str) -> int | None:
    request = urllib.request.Request(  # noqa: S310 - trusted AirfRANS HTTPS constant.
        url,
        method="HEAD",
        headers={"User-Agent": "AeroCliff/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            value = response.headers.get("content-length")
    except (OSError, urllib.error.URLError):
        return None
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def write_airfrans_feasibility(out: Path, *, data_root: Path | None = None) -> Path:
    """Write a fail-closed AirfRANS feasibility report without downloading data."""

    package = _airfrans_package_status()
    content_length = _dataset_content_length(AIRFRANS_DATASET_URL)
    root = data_root or Path("artifacts/benchmark/airfrans")
    manifest_path = root / "manifest.json"
    dataset_present = manifest_path.exists()
    real_data_ready = bool(
        dataset_present
        and package["simulation_force_coefficient_available"]
        and package["installed"],
    )
    target_status = (
        "AVAILABLE_VIA_AIRFRANS_SIMULATION_FORCE_COEFFICIENT_IF_DATASET_MATERIALISED"
        if package["simulation_force_coefficient_available"]
        else "BLOCKED_PACKAGE_API_NOT_AVAILABLE"
    )
    payload = {
        "schema_version": AEROMAP_FEASIBILITY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AIRFRANS_FEASIBILITY",
        "source": {
            "name": "AirfRANS",
            "docs_url": AIRFRANS_DOCS_URL,
            "dataset_url": AIRFRANS_DATASET_URL,
            "license": AIRFRANS_LICENSE,
            "citation_key": AIRFRANS_CITATION_KEY,
        },
        "package": package,
        "dataset_archive": {
            "content_length_bytes": content_length,
            "content_length_gib": None if content_length is None else content_length / (1024.0**3),
            "download_attempted": False,
            "reason_not_downloaded": (
                "processed AirfRANS archive is about 10 GiB before extraction; "
                "explicit approval required before materialising it"
            ),
        },
        "local_data": {
            "data_root": str(root),
            "manifest_path": str(manifest_path),
            "manifest_present": dataset_present,
            "real_data_ready": real_data_ready,
        },
        "target_contract": {
            "candidate_targets": ["integrated_cd", "integrated_cl"],
            "target_status": target_status,
            "derivation": (
                "AirfRANS Simulation.force_coefficient computes drag/lift from pressure "
                "and wall shear over the airfoil when the dataset files are present."
            ),
            "fail_closed": not real_data_ready,
        },
        "mvp_decision": {
            "real_airfrans_benchmark_ready": real_data_ready,
            "fixture_replay_allowed": True,
            "fixture_claim_boundary": (
                "fixture replay tests code path and decision logic only; it is not an "
                "open-CFD result"
            ),
        },
    }
    atomic_write_json(out, payload)
    return out


def extract_airfrans_archive(archive: Path, root: Path) -> Path:
    """Extract the processed AirfRANS archive and return the manifest path."""

    if not archive.exists():
        msg = f"AirfRANS archive does not exist: {archive}"
        raise FileNotFoundError(msg)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(root)
    manifest = root / "manifest.json"
    if not manifest.exists():
        nested = next(root.rglob("manifest.json"), None)
        if nested is None:
            msg = f"AirfRANS manifest.json not found after extracting {archive}"
            raise FileNotFoundError(msg)
        return nested
    return manifest


def airfrans_case_features_from_name(name: str) -> tuple[list[float], list[str]]:
    """Parse AirfRANS simulation-name features without reading the mesh."""

    tokens = name.split("_")
    if len(tokens) < MIN_AIRFRANS_NAME_TOKENS:
        msg = f"AirfRANS simulation name is too short to parse: {name}"
        raise ValueError(msg)
    try:
        inlet_velocity = float(tokens[2])
        aoa_deg = float(tokens[3])
    except ValueError as exc:
        msg = f"AirfRANS simulation name lacks numeric velocity/AoA fields: {name}"
        raise ValueError(msg) from exc
    raw_shape: list[float] = []
    for token in tokens[4:-1]:
        try:
            raw_shape.append(float(token))
        except ValueError:
            continue
    shape = (raw_shape + [0.0] * MAX_SHAPE_PARAMETERS)[:MAX_SHAPE_PARAMETERS]
    reynolds = inlet_velocity / 1.56e-5
    features = [
        inlet_velocity,
        aoa_deg,
        math.sin(math.radians(aoa_deg)),
        math.cos(math.radians(aoa_deg)),
        math.log10(max(reynolds, 1.0)),
        *shape,
    ]
    names = [
        "inlet_velocity_m_s",
        "aoa_deg",
        "sin_aoa",
        "cos_aoa",
        "log10_reynolds",
        *[f"shape_param_{idx}" for idx in range(MAX_SHAPE_PARAMETERS)],
    ]
    return features, names


def _airfrans_manifest_path(root: Path) -> Path:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        nested = next(root.rglob("manifest.json"), None)
        if nested is None:
            msg = f"AirfRANS manifest.json not found under {root}"
            raise FileNotFoundError(msg)
        return nested
    return manifest_path


def _airfrans_case_names(root: Path, *, task: str, split: str) -> list[str]:
    manifest_path = _airfrans_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = [f"{task}_train", f"{task}_test"] if split == "both" else [f"{task}_{split}"]
    names: list[str] = []
    for key in keys:
        if key not in manifest:
            msg = f"AirfRANS manifest key {key!r} is missing"
            raise ValueError(msg)
        names.extend(str(item) for item in manifest[key])
    return sorted(set(names))


def _default_airfrans_simulation_factory(root: Path, name: str) -> AirfransSimulation:
    try:
        simulation_module = importlib.import_module("airfrans.simulation")
    except ImportError as exc:
        msg = "airfrans package is required to build the real AirfRANS scalar dataset"
        raise RuntimeError(msg) from exc
    simulation_cls = cast("Callable[..., AirfransSimulation]", simulation_module.Simulation)
    return simulation_cls(root=str(root), name=name)


def build_airfrans_scalar_dataset(
    root: Path,
    out: Path,
    *,
    task: str = "full",
    split: str = "both",
    max_cases: int | None = None,
    simulation_factory: SimulationFactory = _default_airfrans_simulation_factory,
) -> Path:
    """Build a real AirfRANS scalar dataset from documented force coefficients."""

    manifest_path = _airfrans_manifest_path(root)
    dataset_root = manifest_path.parent
    case_names = _airfrans_case_names(dataset_root, task=task, split=split)
    if max_cases is not None:
        case_names = case_names[:max_cases]
    if not case_names:
        msg = "no AirfRANS cases selected"
        raise ValueError(msg)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    feature_names: list[str] | None = None
    failed: list[dict[str, str]] = []
    for case_name in case_names:
        try:
            parsed_features, parsed_feature_names = airfrans_case_features_from_name(case_name)
            simulation = simulation_factory(dataset_root, case_name)
            cd_tuple, cl_tuple = simulation.force_coefficient(reference=True)
            cd = float(cd_tuple[0])
            cl = float(cl_tuple[0])
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            failed.append({"case_id": case_name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if feature_names is None:
            feature_names = parsed_feature_names
        features.append(parsed_features)
        targets.append([cd, cl])

    if not features:
        msg = "AirfRANS scalar dataset build failed for every selected case"
        raise ValueError(msg)
    feature_array = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    completed_case_ids = [
        case for case in case_names if case not in {item["case_id"] for item in failed}
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        features=feature_array,
        targets=target_array,
        case_ids=np.array(completed_case_ids),
        feature_names=np.array(feature_names or []),
        target_names=np.array(["integrated_cd", "integrated_cl"]),
        classification=np.array("AIRFRANS_REAL_SCALAR_DATASET"),
        open_cfd_result=np.ones((), dtype=np.bool_),
    )
    payload = {
        "schema_version": AEROMAP_AIRFRANS_SCALAR_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AIRFRANS_REAL_SCALAR_DATASET",
        "source": {
            "name": "AirfRANS",
            "license": AIRFRANS_LICENSE,
            "citation_key": AIRFRANS_CITATION_KEY,
            "docs_url": AIRFRANS_DOCS_URL,
        },
        "root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "task": task,
        "split": split,
        "selected_case_count": len(case_names),
        "completed_case_count": len(completed_case_ids),
        "failed_case_count": len(failed),
        "failed_cases": failed[:20],
        "feature_names": feature_names or [],
        "target_names": ["integrated_cd", "integrated_cl"],
        "target_derivation": (
            "AirfRANS Simulation.force_coefficient(reference=True), returning "
            "((cd, cdp, cdv), (cl, clp, clv)); stored targets use cd and cl."
        ),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "claim_boundary": {
            "open_cfd_result": True,
            "aerocliff_result": False,
            "f1_geometry": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _airfrans_aerofoil_vtp(dataset_root: Path, case_id: str) -> Path:
    path = dataset_root / case_id / f"{case_id}_aerofoil.vtp"
    if not path.exists():
        msg = f"AirfRANS aerofoil VTP is missing for {case_id}: {path}"
        raise FileNotFoundError(msg)
    return path


def _cosine_coefficients(values: FloatArray, *, modes: int) -> list[float]:
    idx = np.arange(values.shape[0], dtype=np.float64)
    coeffs = []
    for mode in range(modes):
        basis = np.cos(np.pi * mode * (idx + 0.5) / values.shape[0])
        coeffs.append(float(np.mean(values * basis)))
    return coeffs


def _airfoil_geometry_descriptor(vtp_path: Path) -> tuple[list[float], list[str]]:
    try:
        pyvista = importlib.import_module("pyvista")
    except ImportError as exc:
        msg = "pyvista is required to extract AirfRANS geometry descriptors"
        raise RuntimeError(msg) from exc

    mesh = pyvista.read(vtp_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.ndim != ARRAY_RANK_TWO or points.shape[1] < MIN_XY_COLUMNS:
        msg = f"unexpected AirfRANS aerofoil point array shape in {vtp_path}: {points.shape}"
        raise ValueError(msg)

    x = points[:, 0]
    y = points[:, 1]
    chord = float(x.max() - x.min())
    if chord <= EPSILON:
        msg = f"degenerate AirfRANS chord in {vtp_path}"
        raise ValueError(msg)
    x_norm = (x - float(x.min())) / chord
    y_norm = y / chord

    stations = np.linspace(0.04, 0.96, GEOMETRY_STATION_COUNT, dtype=np.float64)
    half_width = 0.035
    upper: list[float] = []
    lower: list[float] = []
    for station in stations:
        mask = np.abs(x_norm - station) <= half_width
        if not bool(np.any(mask)):
            nearest = np.argsort(np.abs(x_norm - station))[:8]
            values = y_norm[nearest]
        else:
            values = y_norm[mask]
        upper.append(float(values.max()))
        lower.append(float(values.min()))

    upper_arr = np.asarray(upper, dtype=np.float64)
    lower_arr = np.asarray(lower, dtype=np.float64)
    thickness = upper_arr - lower_arr
    camber = 0.5 * (upper_arr + lower_arr)
    features = [
        *_cosine_coefficients(thickness, modes=GEOMETRY_COSINE_MODES),
        *_cosine_coefficients(camber, modes=GEOMETRY_COSINE_MODES),
        float(thickness.max()),
        float(np.max(np.abs(camber))),
        float(thickness[0]),
        float(thickness[-1]),
    ]
    names = [
        *[f"geom_thickness_cos_{idx}" for idx in range(GEOMETRY_COSINE_MODES)],
        *[f"geom_camber_cos_{idx}" for idx in range(GEOMETRY_COSINE_MODES)],
        "geom_max_thickness",
        "geom_max_abs_camber",
        "geom_leading_station_thickness",
        "geom_trailing_station_thickness",
    ]
    return features, names


def _stable_feature_hash(values: list[float]) -> str:
    rounded = ",".join(f"{value:.8f}" for value in values)
    return hashlib.sha256(rounded.encode("utf-8")).hexdigest()[:16]


def build_airfrans_geometry_dataset(
    root: Path,
    scalar_npz: Path,
    out: Path,
    *,
    feature_contract_out: Path | None = None,
) -> Path:
    """Append deterministic airfoil-shape descriptors to the AirfRANS scalar dataset."""

    manifest_path = _airfrans_manifest_path(root)
    dataset_root = manifest_path.parent
    scalar = load_dataset_npz(scalar_npz)
    geometry_features: list[list[float]] = []
    geometry_hashes: list[str] = []
    geometry_feature_names: list[str] | None = None
    for case_id in scalar.case_ids:
        features, names = _airfoil_geometry_descriptor(
            _airfrans_aerofoil_vtp(dataset_root, case_id)
        )
        if geometry_feature_names is None:
            geometry_feature_names = names
        geometry_features.append(features)
        geometry_hashes.append(_stable_feature_hash(features))

    geometry_array = np.asarray(geometry_features, dtype=np.float64)
    enriched_features = np.concatenate([scalar.features, geometry_array], axis=1)
    feature_names = [*scalar.feature_names, *(geometry_feature_names or [])]
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        features=enriched_features,
        targets=scalar.targets,
        case_ids=np.array(scalar.case_ids),
        feature_names=np.array(feature_names),
        target_names=np.array(scalar.target_names),
        classification=np.array("AIRFRANS_REAL_GEOMETRY_SCALAR_DATASET"),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.array(geometry_hashes),
    )
    payload = {
        "schema_version": AEROMAP_AIRFRANS_GEOMETRY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AIRFRANS_REAL_GEOMETRY_SCALAR_DATASET",
        "source_scalar_npz": str(scalar_npz),
        "root": str(dataset_root),
        "case_count": len(scalar.case_ids),
        "feature_count": int(enriched_features.shape[1]),
        "base_feature_count": int(scalar.features.shape[1]),
        "geometry_feature_count": int(geometry_array.shape[1]),
        "unique_geometry_group_count": len(set(geometry_hashes)),
        "geometry_descriptor": {
            "method": "fixed-station upper/lower boundary resampling plus cosine coefficients",
            "station_count": GEOMETRY_STATION_COUNT,
            "cosine_modes_per_signal": GEOMETRY_COSINE_MODES,
            "signals": ["thickness", "camber"],
            "normalisation": "x and y coordinates normalised by chord length per case",
        },
        "feature_names": feature_names,
        "target_names": scalar.target_names,
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "claim_boundary": {
            "open_cfd_result": True,
            "aerocliff_result": False,
            "f1_geometry": False,
        },
    }
    atomic_write_json(out, payload)
    if feature_contract_out is not None:
        write_airfrans_feature_contract(payload, feature_contract_out)
    return out


def write_airfrans_feature_contract(dataset_manifest: dict[str, Any], out: Path) -> Path:
    """Write the immutable AirfRANS v0.2 feature contract."""

    feature_names = [str(item) for item in dataset_manifest["feature_names"]]
    records = []
    for name in feature_names:
        if name == "inlet_velocity_m_s":
            meaning, units, category = (
                "freestream inlet velocity parsed from AirfRANS case ID",
                "m/s",
                "operating_condition",
            )
        elif name == "aoa_deg":
            meaning, units, category = (
                "angle of attack parsed from AirfRANS case ID",
                "deg",
                "operating_condition",
            )
        elif name in {"sin_aoa", "cos_aoa"}:
            meaning, units, category = (
                "trigonometric angle-of-attack encoding",
                "dimensionless",
                "derived_operating_condition",
            )
        elif name == "log10_reynolds":
            meaning, units, category = (
                "log10 Reynolds proxy from inlet velocity and fixed AirfRANS kinematic viscosity",
                "dimensionless",
                "derived_operating_condition",
            )
        elif name.startswith("shape_param_"):
            meaning, units, category = (
                "AirfRANS shape token parsed from case ID",
                "dataset_native",
                "geometry_descriptor",
            )
        elif name.startswith("geom_thickness_cos_"):
            meaning, units, category = (
                "cosine coefficient of chord-normalised thickness distribution",
                "dimensionless",
                "geometry_descriptor",
            )
        elif name.startswith("geom_camber_cos_"):
            meaning, units, category = (
                "cosine coefficient of chord-normalised camber distribution",
                "dimensionless",
                "geometry_descriptor",
            )
        elif name == "geom_max_thickness":
            meaning, units, category = (
                "maximum chord-normalised sampled thickness",
                "dimensionless",
                "geometry_descriptor",
            )
        elif name == "geom_max_abs_camber":
            meaning, units, category = (
                "maximum absolute chord-normalised sampled camber",
                "dimensionless",
                "geometry_descriptor",
            )
        elif name == "geom_leading_station_thickness":
            meaning, units, category = (
                "sampled thickness near x/c=0.04",
                "dimensionless",
                "geometry_descriptor",
            )
        elif name == "geom_trailing_station_thickness":
            meaning, units, category = (
                "sampled thickness near x/c=0.96",
                "dimensionless",
                "geometry_descriptor",
            )
        else:
            meaning, units, category = "unclassified feature", "unknown", "unknown"
        records.append(
            {
                "name": name,
                "meaning": meaning,
                "units": units,
                "category": category,
                "target_leakage": False,
                "stored_normalisation": "raw",
                "model_normalisation": "fit on labelled training subset inside each replay round",
            },
        )
    payload = {
        "schema_version": "aerocliff_airfrans_feature_contract_v0.2.0",
        "dataset_classification": dataset_manifest["classification"],
        "feature_count": len(records),
        "features": records,
        "target_contract": {
            "targets": dataset_manifest["target_names"],
            "source": "AirfRANS Simulation.force_coefficient(reference=True)",
            "target_leakage_into_features": False,
        },
        "split_policy": {
            "map_completion": "random case split over the full scalar pool",
            "geometry_heldout": (
                "deterministic farthest-first held-out selection over geometry descriptors"
            ),
            "exact_repeated_geometry_groups": (
                "not present in the materialised 1000-case AirfRANS scalar set"
            ),
        },
        "claim_boundary": dataset_manifest["claim_boundary"],
    }
    atomic_write_json(out, payload)
    return out


def make_fixture_dataset(case_count: int, *, seed: int) -> DatasetArrays:
    """Create a deterministic AirfRANS-contract fixture with aerodynamic structure."""

    rng = np.random.default_rng(seed)
    camber = rng.uniform(0.0, 0.06, size=case_count)
    thickness = rng.uniform(0.07, 0.16, size=case_count)
    aoa_deg = rng.uniform(-4.0, 14.0, size=case_count)
    reynolds = rng.uniform(2.0e5, 6.0e6, size=case_count)
    aoa_rad = np.radians(aoa_deg)
    log_re = np.log10(reynolds)

    stall_softening = 1.0 / (1.0 + np.exp((aoa_deg - (11.0 - 35.0 * camber)) / 1.4))
    lift_slope = 2.0 * np.pi * (1.0 + 1.8 * camber - 0.5 * (thickness - 0.10))
    cl_linear = lift_slope * (aoa_rad + 1.7 * camber)
    cl = cl_linear * (0.72 + 0.28 * stall_softening)
    cl += 0.03 * np.sin(8.0 * camber + 0.7 * aoa_rad)

    skin_friction = 0.0065 + 0.0015 * (6.0 - log_re)
    form_drag = 0.008 + 0.9 * (thickness - 0.09) ** 2 + 0.08 * camber**2
    induced_drag = 0.018 * cl**2
    stall_drag = 0.08 * (1.0 - stall_softening) ** 2
    cd = np.clip(skin_friction + form_drag + induced_drag + stall_drag, 0.004, None)

    features = np.column_stack(
        [
            camber,
            thickness,
            aoa_deg,
            np.log10(reynolds),
            camber * aoa_deg,
            thickness * aoa_deg,
        ],
    ).astype(np.float64)
    targets = np.column_stack([cd, cl]).astype(np.float64)
    case_ids = [f"airfrans_fixture_{idx:04d}" for idx in range(case_count)]
    return DatasetArrays(
        case_ids=case_ids,
        features=features,
        targets=targets,
        feature_names=[
            "camber",
            "thickness",
            "aoa_deg",
            "log10_reynolds",
            "camber_x_aoa",
            "thickness_x_aoa",
        ],
        target_names=["integrated_cd", "integrated_cl"],
    )


def write_fixture_dataset(config: AeroMapConfig, out: Path) -> Path:
    dataset = make_fixture_dataset(config.fixture_case_count, seed=config.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        features=dataset.features,
        targets=dataset.targets,
        case_ids=np.array(dataset.case_ids),
        feature_names=np.array(dataset.feature_names),
        target_names=np.array(dataset.target_names),
        classification=np.array("AEROMAP_AIRFRANS_CONTRACT_FIXTURE"),
        open_cfd_result=np.zeros((), dtype=np.bool_),
    )
    manifest = {
        "schema_version": AEROMAP_DATASET_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AEROMAP_AIRFRANS_CONTRACT_FIXTURE",
        "case_count": len(dataset.case_ids),
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "claim_boundary": {
            "open_cfd_result": False,
            "aerocliff_result": False,
            "purpose": "exercise Mission Control active-learning code path locally",
        },
    }
    atomic_write_json(out, manifest)
    return out


def load_dataset_npz(path: Path) -> DatasetArrays:
    with np.load(path, allow_pickle=False) as data:
        features = cast("FloatArray", np.asarray(data["features"], dtype=np.float64))
        targets = cast("FloatArray", np.asarray(data["targets"], dtype=np.float64))
        case_ids = [str(item) for item in data["case_ids"].tolist()]
        feature_names = [str(item) for item in data["feature_names"].tolist()]
        target_names = [str(item) for item in data["target_names"].tolist()]
        classification = (
            str(data["classification"].item())
            if "classification" in data.files
            else "AEROMAP_AIRFRANS_CONTRACT_FIXTURE"
        )
        open_cfd_result = (
            bool(data["open_cfd_result"].item()) if "open_cfd_result" in data.files else False
        )
        group_ids = (
            [str(item) for item in data["group_ids"].tolist()]
            if "group_ids" in data.files
            else None
        )
    if features.ndim != ARRAY_RANK_TWO or targets.ndim != ARRAY_RANK_TWO:
        msg = "features and targets must be rank-2 arrays"
        raise ValueError(msg)
    if features.shape[0] != targets.shape[0] or len(case_ids) != features.shape[0]:
        msg = "case, feature and target counts differ"
        raise ValueError(msg)
    return DatasetArrays(
        case_ids=case_ids,
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=target_names,
        classification=classification,
        open_cfd_result=open_cfd_result,
        group_ids=group_ids,
    )


def split_dataset(dataset: DatasetArrays, *, test_fraction: float, seed: int) -> SplitArrays:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset.features.shape[0], dtype=np.int64)
    rng.shuffle(indices)
    test_count = max(1, round(float(indices.shape[0]) * test_fraction))
    test_indices = np.sort(indices[:test_count])
    train_pool_indices = np.sort(indices[test_count:])
    return SplitArrays(train_pool_indices=train_pool_indices, test_indices=test_indices)


def split_dataset_by_mode(
    dataset: DatasetArrays,
    *,
    test_fraction: float,
    seed: int,
    split_mode: str,
) -> SplitArrays:
    if split_mode == "map_completion":
        return split_dataset(dataset, test_fraction=test_fraction, seed=seed)
    if split_mode != "geometry_heldout":
        msg = f"unknown split mode: {split_mode}"
        raise ValueError(msg)

    geometry_columns = [
        idx
        for idx, name in enumerate(dataset.feature_names)
        if name.startswith(("geom_", "shape_param_"))
    ]
    if not geometry_columns:
        msg = "geometry_heldout split requires geometry descriptor features"
        raise ValueError(msg)
    geometry = dataset.features[:, geometry_columns]
    geometry = _fit_scaler(geometry).transform(geometry)
    test_count = max(1, round(float(geometry.shape[0]) * test_fraction))
    centroid = geometry.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(geometry - centroid, axis=1)))]
    while len(selected) < test_count:
        selected_array = np.array(selected, dtype=np.int64)
        diff = geometry[:, None, :] - geometry[selected_array][None, :, :]
        min_dist = np.linalg.norm(diff, axis=2).min(axis=1)
        min_dist[selected_array] = -np.inf
        selected.append(int(np.argmax(min_dist)))
    test_indices = np.sort(np.array(selected, dtype=np.int64))
    all_indices = np.arange(geometry.shape[0], dtype=np.int64)
    train_pool_indices = np.sort(np.setdiff1d(all_indices, test_indices, assume_unique=True))
    return SplitArrays(train_pool_indices=train_pool_indices, test_indices=test_indices)


def polynomial_features(features: FloatArray) -> FloatArray:
    columns = [np.ones((features.shape[0], 1), dtype=np.float64), features]
    quadratic = [
        (features[:, left] * features[:, right]).reshape(-1, 1)
        for left in range(features.shape[1])
        for right in range(left, features.shape[1])
    ]
    columns.extend(quadratic)
    return np.concatenate(columns, axis=1)


def _fit_scaler(values: FloatArray) -> StandardScaler:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < EPSILON, 1.0, scale)
    return StandardScaler(mean=mean.astype(np.float64), scale=scale.astype(np.float64))


def fit_ridge_ensemble(
    features: FloatArray,
    targets: FloatArray,
    *,
    members: int,
    seed: int,
) -> RidgeEnsemble:
    if features.shape[0] < MIN_LABELLED_CASES:
        msg = "at least two labelled cases are required"
        raise ValueError(msg)
    x_scaler = _fit_scaler(features)
    y_scaler = _fit_scaler(targets)
    x_scaled = x_scaler.transform(features)
    y_scaled = y_scaler.transform(targets)
    design = polynomial_features(x_scaled)
    weights = []
    rng = np.random.default_rng(seed)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    for _ in range(members):
        boot = rng.choice(features.shape[0], size=features.shape[0], replace=True)
        x_boot = design[boot]
        y_boot = y_scaled[boot]
        lhs = x_boot.T @ x_boot + regularizer
        rhs = x_boot.T @ y_boot
        weights.append(np.linalg.solve(lhs, rhs))
    return RidgeEnsemble(
        weights=np.stack(weights, axis=0).astype(np.float64),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def _safe_normalize(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _diversity_score(pool: FloatArray, labelled: FloatArray) -> FloatArray:
    diff = pool[:, None, :] - labelled[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return cast("FloatArray", distances.min(axis=1))


def _pareto_mask(cd: FloatArray, cl: FloatArray) -> NDArray[np.bool_]:
    mask = np.ones(cd.shape[0], dtype=np.bool_)
    for idx in range(cd.shape[0]):
        dominates = (cd <= cd[idx]) & (cl >= cl[idx]) & ((cd < cd[idx]) | (cl > cl[idx]))
        if bool(np.any(dominates)):
            mask[idx] = False
    return mask


def _rankdata(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def _spearman(a: FloatArray, b: FloatArray) -> float:
    if a.size < MIN_LABELLED_CASES:
        return 0.0
    ar = _rankdata(a)
    br = _rankdata(b)
    ar -= ar.mean()
    br -= br.mean()
    denominator = float(np.linalg.norm(ar) * np.linalg.norm(br))
    if denominator < EPSILON:
        return 0.0
    return float(np.dot(ar, br) / denominator)


def _top_k_overlap(true_scores: FloatArray, pred_scores: FloatArray, *, fraction: float) -> float:
    count = max(1, round(true_scores.shape[0] * fraction))
    true_top = set(np.argsort(true_scores)[-count:].tolist())
    pred_top = set(np.argsort(pred_scores)[-count:].tolist())
    return len(true_top & pred_top) / float(count)


def evaluate_predictions(targets: FloatArray, predictions: FloatArray) -> dict[str, float]:
    residual = predictions - targets
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    true_eff = targets[:, 1] / np.maximum(targets[:, 0], 1.0e-12)
    pred_eff = predictions[:, 1] / np.maximum(predictions[:, 0], 1.0e-12)
    true_pareto = _pareto_mask(targets[:, 0], targets[:, 1])
    pred_pareto = _pareto_mask(predictions[:, 0], predictions[:, 1])
    pareto_denominator = max(1, int(true_pareto.sum()))
    pareto_recall = float(np.logical_and(true_pareto, pred_pareto).sum() / pareto_denominator)
    predicted_best_idx = int(np.argmax(pred_eff))
    best_design_regret = float(true_eff.max() - true_eff[predicted_best_idx])
    return {
        "rmse_cd": float(rmse[0]),
        "rmse_cl": float(rmse[1]),
        "mae_cd": float(mae[0]),
        "mae_cl": float(mae[1]),
        "spearman_efficiency": _spearman(true_eff, pred_eff),
        "top_k_efficiency_overlap": _top_k_overlap(
            true_eff,
            pred_eff,
            fraction=DEFAULT_TOP_K_FRACTION,
        ),
        "pareto_recall": pareto_recall,
        "best_design_regret": best_design_regret,
    }


def _select_batch(
    *,
    method: str,
    ensemble: RidgeEnsemble,
    features: FloatArray,
    labelled_indices: IndexArray,
    pool_indices: IndexArray,
    batch_size: int,
    max_labels: int | None,
    rng: np.random.Generator,
) -> IndexArray:
    count = min(batch_size, int(pool_indices.shape[0]))
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if method == "random":
        return cast("IndexArray", np.sort(rng.choice(pool_indices, size=count, replace=False)))

    pool_features = features[pool_indices]
    x_scaled = ensemble.x_scaler.transform(features)
    diversity = _diversity_score(x_scaled[pool_indices], x_scaled[labelled_indices])
    if method == "diversity":
        score = diversity
    else:
        uncertainty = ensemble.uncertainty(pool_features)
        if method == "uncertainty":
            score = uncertainty
        elif method == "uncertainty_plus_diversity":
            score = _safe_normalize(uncertainty) + 0.35 * _safe_normalize(diversity)
        elif method == "engineering_utility":
            prediction = ensemble.predict(pool_features)
            efficiency = prediction[:, 1] / np.maximum(prediction[:, 0], 1.0e-12)
            high_value = _safe_normalize(efficiency)
            score = (
                0.45 * _safe_normalize(uncertainty)
                + 0.35 * high_value
                + 0.20 * _safe_normalize(diversity)
            )
        elif method == "engineering_decision_utility_v1":
            prediction = ensemble.predict(pool_features)
            efficiency = prediction[:, 1] / np.maximum(prediction[:, 0], 1.0e-12)
            pareto_relevance = _pareto_mask(prediction[:, 0], prediction[:, 1]).astype(np.float64)
            cd_gradient_proxy = np.abs(prediction[:, 0] - np.median(prediction[:, 0]))
            score = (
                0.30 * _safe_normalize(uncertainty)
                + 0.25 * _safe_normalize(efficiency)
                + 0.15 * pareto_relevance
                + 0.10 * _safe_normalize(cd_gradient_proxy)
                + 0.20 * _safe_normalize(diversity)
            )
        elif method == "engineering_decision_utility_v2_regret_aware":
            prediction = ensemble.predict(pool_features)
            efficiency = prediction[:, 1] / np.maximum(prediction[:, 0], 1.0e-12)
            labelled_prediction = ensemble.predict(features[labelled_indices])
            labelled_efficiency = labelled_prediction[:, 1] / np.maximum(
                labelled_prediction[:, 0],
                1.0e-12,
            )
            current_best = float(labelled_efficiency.max())
            expected_improvement = np.maximum(efficiency - current_best, 0.0)
            pareto_relevance = _pareto_mask(prediction[:, 0], prediction[:, 1]).astype(np.float64)
            ood_penalty = np.maximum(_safe_normalize(diversity) - 0.85, 0.0) / 0.15
            progress = (
                min(1.0, float(labelled_indices.shape[0]) / float(max_labels))
                if max_labels is not None and max_labels > 0
                else 0.5
            )
            early_weight = 1.0 - progress
            late_weight = progress
            exploration_score = (
                0.45 * _safe_normalize(diversity)
                + 0.35 * _safe_normalize(uncertainty)
                + 0.20 * pareto_relevance
            )
            exploitation_score = (
                0.20 * _safe_normalize(uncertainty)
                + 0.10 * _safe_normalize(diversity)
                + 0.20 * pareto_relevance
                + 0.30 * _safe_normalize(efficiency)
                + 0.20 * _safe_normalize(expected_improvement)
            )
            score = early_weight * exploration_score + late_weight * exploitation_score
            score = score - 0.05 * np.clip(ood_penalty, 0.0, 1.0)
        else:
            msg = f"unknown acquisition method: {method}"
            raise ValueError(msg)
    selected_local = np.argsort(score)[-count:]
    return cast("IndexArray", np.sort(pool_indices[selected_local]))


def run_active_learning_replay(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_mode: str = "map_completion",
) -> dict[str, Any]:
    split = split_dataset_by_mode(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.seed,
        split_mode=split_mode,
    )
    if split.train_pool_indices.shape[0] < config.max_labels:
        msg = "train/pool split is smaller than max_labels"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    final_by_method: dict[str, dict[str, float]] = {}
    for method in config.acquisition_methods:
        rng = np.random.default_rng(config.seed + _stable_method_seed(method))
        labelled = cast(
            "IndexArray",
            np.sort(
                rng.choice(
                    split.train_pool_indices,
                    size=int(config.initial_labels),
                    replace=False,
                ),
            ),
        )
        pool = cast(
            "IndexArray",
            np.setdiff1d(split.train_pool_indices, labelled, assume_unique=True),
        )
        budget = int(labelled.shape[0])
        round_idx = 0
        while True:
            ensemble = fit_ridge_ensemble(
                dataset.features[labelled],
                dataset.targets[labelled],
                members=int(config.ensemble_members),
                seed=config.seed + round_idx + len(method),
            )
            predictions = ensemble.predict(dataset.features[split.test_indices])
            metrics = evaluate_predictions(dataset.targets[split.test_indices], predictions)
            record = {
                "method": method,
                "round": round_idx,
                "label_count": budget,
                **metrics,
            }
            records.append(record)
            final_by_method[method] = metrics
            if budget >= int(config.max_labels) or pool.shape[0] == 0:
                break
            new_indices = _select_batch(
                method=method,
                ensemble=ensemble,
                features=dataset.features,
                labelled_indices=labelled,
                pool_indices=pool,
                batch_size=int(config.acquisition_batch),
                max_labels=int(config.max_labels),
                rng=rng,
            )
            labelled = cast("IndexArray", np.sort(np.concatenate([labelled, new_indices])))
            pool = cast("IndexArray", np.setdiff1d(pool, new_indices, assume_unique=True))
            budget = int(labelled.shape[0])
            round_idx += 1

    area_by_method = _learning_curve_area(records, metric="rmse_cd")
    best_method = min(final_by_method, key=lambda item: final_by_method[item]["rmse_cd"])
    classification = (
        "AIRFRANS_REAL_SCALAR_ACTIVE_REPLAY"
        if dataset.open_cfd_result
        else "AEROMAP_CONTRACT_FIXTURE_ACTIVE_REPLAY"
    )
    active_learning_claim = (
        "open_cfd_budget_benchmark_replay_not_aerocliff"
        if dataset.open_cfd_result
        else "code_path_only_until_real_airfrans_data_runs"
    )
    return {
        "schema_version": AEROMAP_REPLAY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": classification,
        "dataset_classification": dataset.classification,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
            "active_learning_claim": active_learning_claim,
        },
        "config": config.model_dump(),
        "split": {
            "mode": split_mode,
            "train_pool_count": int(split.train_pool_indices.shape[0]),
            "test_count": int(split.test_indices.shape[0]),
            "test_case_ids": [dataset.case_ids[int(idx)] for idx in split.test_indices[:10]],
        },
        "records": records,
        "final_metrics_by_method": final_by_method,
        "learning_curve_area_rmse_cd": area_by_method,
        "best_method_by_final_rmse_cd": best_method,
        "engineering_utility_vs_random": _method_delta(
            final_by_method,
            "engineering_utility",
            "random",
        ),
    }


def _stable_method_seed(method: str) -> int:
    digest = hashlib.sha256(method.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % 10_000


def _learning_curve_area(records: list[dict[str, Any]], *, metric: str) -> dict[str, float]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)
    areas: dict[str, float] = {}
    for method, items in by_method.items():
        ordered = sorted(items, key=lambda item: int(item["label_count"]))
        x = np.array([float(item["label_count"]) for item in ordered], dtype=np.float64)
        y = np.array([float(item[metric]) for item in ordered], dtype=np.float64)
        areas[method] = float(np.trapezoid(y, x))
    return areas


def _method_delta(
    final_by_method: dict[str, dict[str, float]],
    left: str,
    right: str,
) -> dict[str, float | None]:
    if left not in final_by_method or right not in final_by_method:
        return {"rmse_cd_relative_delta": None, "rmse_cl_relative_delta": None}
    left_metrics = final_by_method[left]
    right_metrics = final_by_method[right]
    return {
        "rmse_cd_relative_delta": _relative_delta(
            left_metrics["rmse_cd"],
            right_metrics["rmse_cd"],
        ),
        "rmse_cl_relative_delta": _relative_delta(
            left_metrics["rmse_cl"],
            right_metrics["rmse_cl"],
        ),
    }


def _relative_delta(value: float, reference: float) -> float:
    if math.isclose(reference, 0.0, abs_tol=1.0e-15):
        return 0.0
    return (value - reference) / abs(reference)


def run_active_learning_replay_suite(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_mode: str = "map_completion",
) -> dict[str, Any]:
    """Run the active-learning replay over configured acquisition seeds."""

    replay_seeds = list(config.replay_seeds or (config.seed,))
    seed_reports = [
        run_active_learning_replay(
            dataset,
            config.model_copy(update={"seed": int(seed)}),
            split_mode=split_mode,
        )
        for seed in replay_seeds
    ]
    if len(seed_reports) == 1:
        payload = seed_reports[0]
        payload["seed_count"] = 1
        payload["replay_seeds"] = replay_seeds
        return payload

    seed_records: list[dict[str, Any]] = []
    for seed, report in zip(replay_seeds, seed_reports, strict=True):
        seed_records.extend({"seed": int(seed), **record} for record in report["records"])
    final_metrics = _aggregate_final_metrics(seed_reports, reducer=np.mean)
    best_method = min(final_metrics, key=lambda item: final_metrics[item]["rmse_cd"])
    first = seed_reports[0]
    return {
        "schema_version": AEROMAP_REPLAY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": first["classification"],
        "dataset_classification": first["dataset_classification"],
        "claim_boundary": first["claim_boundary"],
        "config": {**config.model_dump(), "seed": replay_seeds[0]},
        "seed_count": len(replay_seeds),
        "replay_seeds": replay_seeds,
        "split": first["split"],
        "records": _mean_records(seed_records),
        "seed_records": seed_records,
        "final_metrics_by_method": final_metrics,
        "final_metric_std_by_method": _aggregate_final_metrics(seed_reports, reducer=np.std),
        "learning_curve_area_rmse_cd": _aggregate_area(seed_reports, reducer=np.mean),
        "learning_curve_area_rmse_cd_std": _aggregate_area(seed_reports, reducer=np.std),
        "best_method_by_final_rmse_cd": best_method,
        "engineering_utility_vs_random": _method_delta(
            final_metrics,
            "engineering_utility",
            "random",
        ),
    }


def _mean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["method"]), int(record["label_count"]))
        grouped.setdefault(key, []).append(record)
    averaged: list[dict[str, Any]] = []
    for (method, label_count), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        numeric_keys = [
            key
            for key, value in items[0].items()
            if key not in {"seed", "method", "round", "label_count"}
            and isinstance(value, int | float)
        ]
        row: dict[str, Any] = {
            "method": method,
            "label_count": label_count,
            "seed_count": len(items),
        }
        for metric_name in numeric_keys:
            row[metric_name] = float(np.mean([float(item[metric_name]) for item in items]))
        averaged.append(row)
    return averaged


def _aggregate_final_metrics(
    seed_reports: list[dict[str, Any]],
    *,
    reducer: Callable[[FloatArray], np.float64],
) -> dict[str, dict[str, float]]:
    methods = sorted(seed_reports[0]["final_metrics_by_method"])
    aggregate: dict[str, dict[str, float]] = {}
    for method in methods:
        metric_names = sorted(seed_reports[0]["final_metrics_by_method"][method])
        aggregate[method] = {}
        for metric in metric_names:
            values = np.array(
                [
                    float(report["final_metrics_by_method"][method][metric])
                    for report in seed_reports
                ],
                dtype=np.float64,
            )
            aggregate[method][metric] = float(reducer(values))
    return aggregate


def _aggregate_area(
    seed_reports: list[dict[str, Any]],
    *,
    reducer: Callable[[FloatArray], np.float64],
) -> dict[str, float]:
    methods = sorted(seed_reports[0]["learning_curve_area_rmse_cd"])
    aggregate: dict[str, float] = {}
    for method in methods:
        values = np.array(
            [float(report["learning_curve_area_rmse_cd"][method]) for report in seed_reports],
            dtype=np.float64,
        )
        aggregate[method] = float(reducer(values))
    return aggregate


def run_decision_replay_v02(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_modes: tuple[str, ...] = ("map_completion", "geometry_heldout"),
) -> dict[str, Any]:
    split_reports = {
        split_mode: run_active_learning_replay_suite(dataset, config, split_mode=split_mode)
        for split_mode in split_modes
    }
    return {
        "schema_version": AEROMAP_DECISION_REPLAY_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": (
            "AIRFRANS_REAL_SCALAR_DECISION_REPLAY_V02"
            if dataset.open_cfd_result
            else "AEROMAP_FIXTURE_DECISION_REPLAY_V02"
        ),
        "dataset_classification": dataset.classification,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
            "active_learning_claim": (
                "decision_quality_open_cfd_offline_replay_not_aerocliff"
                if dataset.open_cfd_result
                else "fixture_decision_replay_only"
            ),
        },
        "config": config.model_dump(),
        "split_modes": list(split_modes),
        "method_winners": {
            split_mode: _decision_metric_winners(report["final_metrics_by_method"])
            for split_mode, report in split_reports.items()
        },
        "engineering_decision_utility_v1_assessment": {
            split_mode: _engineering_decision_assessment(report["final_metrics_by_method"])
            for split_mode, report in split_reports.items()
        },
        "split_reports": split_reports,
    }


def run_decision_replay_v03(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_modes: tuple[str, ...] = ("map_completion", "geometry_heldout"),
) -> dict[str, Any]:
    """Run the v0.3 decision replay with richer statistics and v2 acquisition."""

    split_reports = {
        split_mode: run_active_learning_replay_suite(dataset, config, split_mode=split_mode)
        for split_mode in split_modes
    }
    final_metrics = {
        split_mode: report["final_metrics_by_method"]
        for split_mode, report in split_reports.items()
    }
    if dataset.classification == "AEROMAP_3D_SCALAR_BRIDGE_DATASET":
        classification = "AEROMAP_3D_OPEN_CFD_SCALAR_BRIDGE"
        active_learning_claim = "compact_3d_open_cfd_scalar_bridge_offline_replay"
    elif dataset.open_cfd_result:
        classification = "AIRFRANS_AEROMAP_V0_3_DECISION_REPLAY"
        active_learning_claim = "decision_quality_open_cfd_offline_replay_not_aerocliff"
    else:
        classification = "AEROMAP_FIXTURE_DECISION_REPLAY_V03"
        active_learning_claim = "fixture_decision_replay_only"

    return {
        "schema_version": AEROMAP_DECISION_REPLAY_V03_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": classification,
        "dataset_classification": dataset.classification,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
            "f1_geometry": False,
            "live_cfd_savings": False,
            "domino_accuracy": False,
            "active_learning_claim": active_learning_claim,
        },
        "config": config.model_dump(),
        "split_modes": list(split_modes),
        "method_winners": {
            split_mode: _decision_metric_winners(metrics)
            for split_mode, metrics in final_metrics.items()
        },
        "engineering_decision_utility_assessment": {
            split_mode: {
                "engineering_decision_utility_v1": _engineering_decision_assessment(metrics),
                "engineering_decision_utility_v2_regret_aware": _engineering_decision_assessment(
                    metrics,
                    method="engineering_decision_utility_v2_regret_aware",
                ),
            }
            for split_mode, metrics in final_metrics.items()
        },
        "headline_readiness": _headline_readiness(final_metrics),
        "statistics": {
            split_mode: _split_replay_statistics(report)
            for split_mode, report in split_reports.items()
        },
        "split_reports": split_reports,
    }


def _headline_readiness(
    final_metrics_by_split: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    geometry_metrics = final_metrics_by_split.get("geometry_heldout")
    if geometry_metrics is None:
        return {"release_headline_ready": False, "reason": "geometry_heldout split absent"}
    winners = _decision_metric_winners(geometry_metrics)
    methods = sorted(geometry_metrics)
    win_counts = {
        method: sum(1 for metric in DECISION_METRICS if method in winners[metric])
        for method in methods
    }
    best_count = max(win_counts.values()) if win_counts else 0
    recommended = [method for method, count in win_counts.items() if count == best_count]
    return {
        "release_headline_ready": bool(best_count >= HEADLINE_READY_DECISION_METRIC_COUNT),
        "criterion": "one method leads or ties at least three geometry-heldout decision metrics",
        "recommended_methods": recommended,
        "geometry_heldout_decision_metric_wins_or_ties": win_counts,
        "method_winners": winners,
        "claim_note": (
            "Headline may be geometry-heldout decision quality only; report metrics where "
            "the recommended method loses."
        ),
    }


def _split_replay_statistics(report: dict[str, Any]) -> dict[str, Any]:
    seed_records = _report_seed_records(report)
    return {
        "confidence_method": "mean, sample standard deviation, standard error, normal 95% CI",
        "budget_statistics": _budget_statistics(seed_records),
        "paired_differences": _paired_difference_statistics(
            seed_records,
            baselines=("random", "diversity", "uncertainty_plus_diversity"),
        ),
        "learning_curve_area": _learning_curve_area_statistics(seed_records),
    }


def _report_seed_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "seed_records" in report:
        return [dict(item) for item in report["seed_records"]]
    seed = int(report.get("config", {}).get("seed", 0))
    return [{"seed": seed, **dict(item)} for item in report["records"]]


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if array.size else 0.0
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    sem = std / math.sqrt(float(array.size)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "n": int(array.size),
    }


def _budget_statistics(seed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for record in seed_records:
        for metric in DECISION_METRICS:
            key = (str(record["method"]), int(record["label_count"]), metric)
            grouped.setdefault(key, []).append(float(record[metric]))
    rows = []
    for (method, label_count, metric), values in sorted(grouped.items()):
        rows.append(
            {
                "method": method,
                "label_count": label_count,
                "metric": metric,
                "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
                **_stats(values),
            },
        )
    return rows


def _paired_difference_statistics(
    seed_records: list[dict[str, Any]],
    *,
    baselines: tuple[str, ...],
) -> list[dict[str, Any]]:
    indexed = {
        (int(record["seed"]), str(record["method"]), int(record["label_count"])): record
        for record in seed_records
    }
    methods = sorted({str(record["method"]) for record in seed_records})
    seeds = sorted({int(record["seed"]) for record in seed_records})
    budgets = sorted({int(record["label_count"]) for record in seed_records})
    rows = []
    for baseline in baselines:
        if baseline not in methods:
            continue
        for method in methods:
            if method == baseline:
                continue
            for budget in budgets:
                for metric in DECISION_METRICS:
                    deltas = []
                    for seed in seeds:
                        left = indexed.get((seed, method, budget))
                        right = indexed.get((seed, baseline, budget))
                        if left is not None and right is not None:
                            deltas.append(float(left[metric]) - float(right[metric]))
                    if deltas:
                        rows.append(
                            {
                                "method": method,
                                "baseline": baseline,
                                "label_count": budget,
                                "metric": metric,
                                "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
                                "delta_interpretation": (
                                    "negative_is_better"
                                    if metric in LOWER_IS_BETTER_METRICS
                                    else "positive_is_better"
                                ),
                                **_stats(deltas),
                            },
                        )
    return rows


def _learning_curve_area_statistics(seed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in seed_records:
        grouped.setdefault((int(record["seed"]), str(record["method"])), []).append(record)

    by_method_metric: dict[tuple[str, str], list[float]] = {}
    for (_seed, method), records in grouped.items():
        ordered = sorted(records, key=lambda item: int(item["label_count"]))
        x = np.asarray([float(item["label_count"]) for item in ordered], dtype=np.float64)
        for metric in DECISION_METRICS:
            y = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
            by_method_metric.setdefault((method, metric), []).append(float(np.trapezoid(y, x)))

    return [
        {
            "method": method,
            "metric": metric,
            "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
            **_stats(values),
        }
        for (method, metric), values in sorted(by_method_metric.items())
    ]


def _decision_metric_winners(final_metrics: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    return {
        "rmse_cd": _metric_leaders(final_metrics, "rmse_cd", lower_is_better=True),
        "rmse_cl": _metric_leaders(final_metrics, "rmse_cl", lower_is_better=True),
        "top_k_efficiency_overlap": _metric_leaders(
            final_metrics,
            "top_k_efficiency_overlap",
            lower_is_better=False,
        ),
        "pareto_recall": _metric_leaders(final_metrics, "pareto_recall", lower_is_better=False),
        "best_design_regret": _metric_leaders(
            final_metrics,
            "best_design_regret",
            lower_is_better=True,
        ),
        "spearman_efficiency": _metric_leaders(
            final_metrics,
            "spearman_efficiency",
            lower_is_better=False,
        ),
    }


def _metric_leaders(
    final_metrics: dict[str, dict[str, float]],
    metric: str,
    *,
    lower_is_better: bool,
) -> list[str]:
    values = {method: final_metrics[method][metric] for method in final_metrics}
    best = min(values.values()) if lower_is_better else max(values.values())
    return [
        method
        for method, value in values.items()
        if math.isclose(value, best, rel_tol=1.0e-9, abs_tol=1.0e-12)
    ]


def _engineering_decision_assessment(
    final_metrics: dict[str, dict[str, float]],
    *,
    method: str = "engineering_decision_utility_v1",
) -> dict[str, Any]:
    if method not in final_metrics:
        return {"available": False, "decision_metric_wins_or_ties": 0}
    winners = _decision_metric_winners(final_metrics)
    decision_metrics = [
        "top_k_efficiency_overlap",
        "pareto_recall",
        "best_design_regret",
        "spearman_efficiency",
    ]
    wins_or_ties = [metric for metric in decision_metrics if method in winners[metric]]
    strict_wins = [metric for metric in decision_metrics if winners[metric] == [method]]
    return {
        "available": True,
        "method": method,
        "decision_metric_wins_or_ties": len(wins_or_ties),
        "decision_metric_strict_wins": len(strict_wins),
        "winning_or_tied_decision_metrics": wins_or_ties,
        "strict_winning_decision_metrics": strict_wins,
        "beats_random": {
            "rmse_cd_relative_delta": _relative_delta(
                final_metrics[method]["rmse_cd"],
                final_metrics["random"]["rmse_cd"],
            ),
            "rmse_cl_relative_delta": _relative_delta(
                final_metrics[method]["rmse_cl"],
                final_metrics["random"]["rmse_cl"],
            ),
            "best_design_regret_relative_delta": _relative_delta(
                final_metrics[method]["best_design_regret"],
                final_metrics["random"]["best_design_regret"],
            ),
        },
    }


def _linear_ridge_predict(
    train_x: FloatArray,
    train_y: FloatArray,
    test_x: FloatArray,
) -> FloatArray:
    x_scaler = _fit_scaler(train_x)
    y_scaler = _fit_scaler(train_y)
    x_train = x_scaler.transform(train_x)
    y_train = y_scaler.transform(train_y)
    x_test = x_scaler.transform(test_x)
    design_train = np.concatenate(
        [np.ones((x_train.shape[0], 1), dtype=np.float64), x_train],
        axis=1,
    )
    design_test = np.concatenate(
        [np.ones((x_test.shape[0], 1), dtype=np.float64), x_test],
        axis=1,
    )
    regularizer = np.eye(design_train.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(
        design_train.T @ design_train + regularizer,
        design_train.T @ y_train,
    )
    return (design_test @ weights) * y_scaler.scale + y_scaler.mean


def _torch_mlp_predict(
    train_x: FloatArray,
    train_y: FloatArray,
    test_x: FloatArray,
    *,
    seed: int,
    epochs: int = 250,
) -> FloatArray:
    torch = cast("Any", importlib.import_module("torch"))
    x_scaler = _fit_scaler(train_x)
    y_scaler = _fit_scaler(train_y)
    x_train = torch.tensor(x_scaler.transform(train_x), dtype=torch.float32)
    y_train = torch.tensor(y_scaler.transform(train_y), dtype=torch.float32)
    x_test = torch.tensor(x_scaler.transform(test_x), dtype=torch.float32)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, train_y.shape[1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test).detach().cpu().numpy().astype(np.float64)
    return cast("FloatArray", pred_scaled * y_scaler.scale + y_scaler.mean)


def run_model_baselines_v02(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_modes: tuple[str, ...] = ("map_completion", "geometry_heldout"),
) -> dict[str, Any]:
    replay_seeds = list(config.replay_seeds or (config.seed,))
    reports: dict[str, Any] = {}
    for split_mode in split_modes:
        split = split_dataset_by_mode(
            dataset,
            test_fraction=config.test_fraction,
            seed=config.seed,
            split_mode=split_mode,
        )
        train_x = dataset.features[split.train_pool_indices]
        train_y = dataset.targets[split.train_pool_indices]
        test_x = dataset.features[split.test_indices]
        test_y = dataset.targets[split.test_indices]
        linear_metrics = evaluate_predictions(
            test_y, _linear_ridge_predict(train_x, train_y, test_x)
        )
        ridge = fit_ridge_ensemble(
            train_x,
            train_y,
            members=int(config.ensemble_members),
            seed=config.seed,
        )
        ridge_metrics = evaluate_predictions(test_y, ridge.predict(test_x))
        mlp_predictions = [
            _torch_mlp_predict(train_x, train_y, test_x, seed=int(seed)) for seed in replay_seeds
        ]
        mlp_metrics = [evaluate_predictions(test_y, prediction) for prediction in mlp_predictions]
        mlp_ensemble_prediction = np.mean(np.stack(mlp_predictions, axis=0), axis=0)
        reports[split_mode] = {
            "split": {
                "mode": split_mode,
                "train_count": int(split.train_pool_indices.shape[0]),
                "test_count": int(split.test_indices.shape[0]),
            },
            "models": {
                "linear_ridge": linear_metrics,
                "polynomial_ridge_ensemble": ridge_metrics,
                "torch_mlp_mean": _mean_metric_dicts(mlp_metrics),
                "torch_mlp_std": _std_metric_dicts(mlp_metrics),
                "torch_mlp_ensemble": evaluate_predictions(test_y, mlp_ensemble_prediction),
                "random_forest_or_gradient_boosting": {
                    "status": "skipped",
                    "reason": "scikit-learn is not installed in the local project environment",
                },
            },
        }
    return {
        "schema_version": "aerocliff_aeromap_model_baselines_v0.2.0",
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AIRFRANS_REAL_SCALAR_MODEL_BASELINES_V02"
        if dataset.open_cfd_result
        else "AEROMAP_FIXTURE_MODEL_BASELINES_V02",
        "dataset_classification": dataset.classification,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
        },
        "replay_seeds": replay_seeds,
        "split_modes": list(split_modes),
        "split_reports": reports,
    }


def _mean_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in items])) for key in sorted(items[0])}


def _std_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.std([item[key] for item in items])) for key in sorted(items[0])}


def write_model_baselines_v02(dataset_npz: Path, config: AeroMapConfig, out: Path) -> Path:
    dataset = load_dataset_npz(dataset_npz)
    payload = run_model_baselines_v02(dataset, config)
    atomic_write_json(out, payload)
    return out


def write_active_learning_replay(
    dataset_npz: Path,
    config: AeroMapConfig,
    out: Path,
    *,
    svg_out: Path | None = None,
) -> Path:
    dataset = load_dataset_npz(dataset_npz)
    payload = run_active_learning_replay_suite(dataset, config)
    atomic_write_json(out, payload)
    if svg_out is not None:
        _write_learning_curve_svg(payload["records"], svg_out)
    return out


def write_decision_replay_v02(
    dataset_npz: Path,
    config: AeroMapConfig,
    out: Path,
    *,
    svg_dir: Path | None = None,
) -> Path:
    dataset = load_dataset_npz(dataset_npz)
    payload = run_decision_replay_v02(dataset, config)
    atomic_write_json(out, payload)
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)
        for split_mode, report in payload["split_reports"].items():
            _write_metric_curve_svg(
                report["records"],
                svg_dir / f"airfrans_decision_{split_mode}_topk.svg",
                metric="top_k_efficiency_overlap",
                title=f"AeroMap {split_mode}: top-k overlap vs CFD labels",
                y_label="top-k overlap",
                claim_note="Open-CFD AirfRANS offline replay; compact benchmark scope.",
            )
            _write_metric_curve_svg(
                report["records"],
                svg_dir / f"airfrans_decision_{split_mode}_regret.svg",
                metric="best_design_regret",
                title=f"AeroMap {split_mode}: best-design regret vs CFD labels",
                y_label="best-design regret",
                claim_note="Lower is better. Open-CFD AirfRANS offline replay.",
            )
    return out


def write_decision_replay_v03(
    dataset_npz: Path,
    config: AeroMapConfig,
    out: Path,
    *,
    svg_dir: Path | None = None,
) -> Path:
    dataset = load_dataset_npz(dataset_npz)
    payload = run_decision_replay_v03(dataset, config)
    atomic_write_json(out, payload)
    if svg_dir is not None:
        if payload["classification"] == "AEROMAP_3D_OPEN_CFD_SCALAR_BRIDGE":
            prefix = "aeromap3d_v03"
            claim_note = "Open-CFD 3D scalar replay; compact benchmark scope."
        else:
            prefix = "airfrans_v03"
            claim_note = "Open-CFD AirfRANS offline replay; compact benchmark scope."
        svg_dir.mkdir(parents=True, exist_ok=True)
        for split_mode, report in payload["split_reports"].items():
            _write_metric_curve_svg(
                report["records"],
                svg_dir / f"{prefix}_{split_mode}_topk.svg",
                metric="top_k_efficiency_overlap",
                title=f"AeroMap v0.3 {split_mode}: top-k overlap vs CFD labels",
                y_label="top-k overlap",
                claim_note=claim_note,
            )
            _write_metric_curve_svg(
                report["records"],
                svg_dir / f"{prefix}_{split_mode}_regret.svg",
                metric="best_design_regret",
                title=f"AeroMap v0.3 {split_mode}: best-design regret vs CFD labels",
                y_label="best-design regret",
                claim_note=f"Lower is better. {claim_note}",
            )
    return out


def write_airfrans_v02_audit(
    dataset_npz: Path,
    feature_contract: Path,
    config: AeroMapConfig,
    out: Path,
    *,
    decision_report: Path | None = None,
) -> Path:
    """Write a compact feature/split/statistics audit for the v0.2/v0.3 dataset."""

    dataset = load_dataset_npz(dataset_npz)
    contract = json.loads(feature_contract.read_text(encoding="utf-8"))
    feature_records = [dict(item) for item in contract["features"]]
    leaked = [item["name"] for item in feature_records if bool(item.get("target_leakage"))]
    categories: dict[str, int] = {}
    for item in feature_records:
        category = str(item.get("category", "unknown"))
        categories[category] = categories.get(category, 0) + 1
    split_audits = {
        split_mode: _split_audit(dataset, config, split_mode=split_mode)
        for split_mode in ("map_completion", "geometry_heldout")
    }
    decision_payload: dict[str, Any] | None = None
    if decision_report is not None and decision_report.exists():
        raw = json.loads(decision_report.read_text(encoding="utf-8"))
        decision_payload = {
            "classification": raw.get("classification"),
            "schema_version": raw.get("schema_version"),
            "method_winners": raw.get("method_winners"),
            "headline_readiness": raw.get("headline_readiness"),
        }

    payload = {
        "schema_version": AEROMAP_AUDIT_SCHEMA,
        "benchmark_class": AEROMAP_CLASS,
        "classification": "AIRFRANS_AEROMAP_V0_3_AUDIT",
        "dataset": {
            "npz_path": str(dataset_npz),
            "npz_sha256": sha256_file(dataset_npz),
            "case_count": len(dataset.case_ids),
            "feature_count": int(dataset.features.shape[1]),
            "target_count": int(dataset.targets.shape[1]),
            "classification": dataset.classification,
            "open_cfd_result": dataset.open_cfd_result,
        },
        "feature_contract": {
            "path": str(feature_contract),
            "feature_count": len(feature_records),
            "categories": categories,
            "target_leakage_features": leaked,
            "passes_no_target_leakage_check": len(leaked) == 0,
            "normalisation_policy": (
                "raw features are stored; replay model scalers are fit on the labelled "
                "subset inside each acquisition round"
            ),
            "geometry_descriptor_uses_labels": False,
        },
        "split_audit": split_audits,
        "decision_report": decision_payload,
        "claim_boundary": {
            "open_cfd_result": dataset.open_cfd_result,
            "aerocliff_result": False,
            "f1_geometry": False,
            "live_cfd_savings": False,
            "domino_accuracy": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _split_audit(
    dataset: DatasetArrays,
    config: AeroMapConfig,
    *,
    split_mode: str,
) -> dict[str, Any]:
    split = split_dataset_by_mode(
        dataset,
        test_fraction=config.test_fraction,
        seed=config.seed,
        split_mode=split_mode,
    )
    train_groups = (
        {dataset.group_ids[int(idx)] for idx in split.train_pool_indices}
        if dataset.group_ids is not None
        else set()
    )
    test_groups = (
        {dataset.group_ids[int(idx)] for idx in split.test_indices}
        if dataset.group_ids is not None
        else set()
    )
    overlap = sorted(train_groups & test_groups)
    return {
        "mode": split_mode,
        "train_pool_count": int(split.train_pool_indices.shape[0]),
        "test_count": int(split.test_indices.shape[0]),
        "group_ids_available": dataset.group_ids is not None,
        "train_test_group_overlap_count": len(overlap),
        "train_test_group_overlap_sample": overlap[:10],
        "same_geometry_leakage_detected": len(overlap) > 0,
        "test_case_sample": [dataset.case_ids[int(idx)] for idx in split.test_indices[:10]],
    }


def _write_learning_curve_svg(records: list[dict[str, Any]], out: Path) -> None:
    _write_metric_curve_svg(
        records,
        out,
        metric="rmse_cd",
        title="AeroMap replay: drag RMSE vs CFD labels",
        y_label="C_D RMSE",
        claim_note="Open-CFD only when the input dataset is classified as real AirfRANS.",
    )


def _write_metric_curve_svg(
    records: list[dict[str, Any]],
    out: Path,
    *,
    metric: str,
    title: str,
    y_label: str,
    claim_note: str,
) -> None:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(str(record["method"]), []).append(record)
    all_x = np.array([float(record["label_count"]) for record in records], dtype=np.float64)
    all_y = np.array([float(record[metric]) for record in records], dtype=np.float64)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    width = 960
    height = 540
    left = 80
    right = 40
    top = 40
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    palette = {
        "random": "#69707a",
        "diversity": "#2f6fed",
        "uncertainty": "#d55e00",
        "uncertainty_plus_diversity": "#009e73",
        "engineering_utility": "#8b3fd1",
        "engineering_decision_utility_v1": "#c33c7a",
        "engineering_decision_utility_v2_regret_aware": "#5f3dc4",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Arial" font-size="22">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#111" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#111" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2 - 70}" y="{height - 20}" font-family="Arial" '
        'font-size="15">labelled CFD cases</text>',
        f'<text x="18" y="{top + plot_h / 2 + 60}" transform="rotate(-90 18 '
        f'{top + plot_h / 2 + 60})" font-family="Arial" font-size="15">{y_label}</text>',
    ]
    legend_y = 64
    for method, items in sorted(by_method.items()):
        ordered = sorted(items, key=lambda item: int(item["label_count"]))
        points = " ".join(
            f"{sx(float(item['label_count'])):.1f},{sy(float(item[metric])):.1f}"
            for item in ordered
        )
        color = palette.get(method, "#222222")
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>',
        )
        lines.append(
            f'<circle cx="{sx(float(ordered[-1]["label_count"])):.1f}" '
            f'cy="{sy(float(ordered[-1][metric])):.1f}" r="4" fill="{color}"/>',
        )
        lines.append(
            f'<rect x="{width - 310}" y="{legend_y - 11}" width="14" height="4" fill="{color}"/>',
        )
        lines.append(
            f'<text x="{width - 288}" y="{legend_y}" font-family="Arial" font-size="13">'
            f"{method}</text>",
        )
        legend_y += 20
    lines.append(
        f'<text x="{left}" y="{height - 46}" font-family="Arial" font-size="12" fill="#555">'
        f"{claim_note}</text>",
    )
    lines.append("</svg>")
    atomic_write_text(out, "\n".join(lines) + "\n")

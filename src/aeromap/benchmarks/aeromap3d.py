"""Compact 3D open-CFD bridge utilities for AeroMap Mission Control."""

from __future__ import annotations

import csv
import json
import math
import struct
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from aeromap.io import atomic_write_json, sha256_file

FloatArray = NDArray[np.float64]

AEROMAP3D_TRIAGE_SCHEMA = "aerocliff_aeromap3d_practicability_triage_v0.1.0"
AEROMAP3D_DATASET_SCHEMA = "aerocliff_aeromap3d_scalar_bridge_dataset_v0.1.0"
AEROMAP3D_GEOMETRY_SCHEMA = "aerocliff_aeromap3d_geometry_readiness_v0.1.0"
AEROMAP3D_CLASSIFICATION = "AEROMAP_3D_SCALAR_BRIDGE_DATASET"
DRIVAERML_MIN_JOINED_ROWS = 50
MAX_GEOMETRY_SAMPLE_STLS = 5
EPSILON_STD = 1.0e-12
STL_VERTEX_FIELD_COUNT = 4
STL_VERTICES_PER_TRIANGLE = 3
BINARY_STL_TRIANGLE_BYTES = 50


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    licence: str
    citation: str
    source_url: str
    full_size_note: str
    case_count_note: str
    scalar_files: tuple[str, ...]
    geometry_parameter_file: str | None
    representative_stl: str | None
    motorsport_vehicle_relevance: int
    wing_relevance: int
    ease_of_compact_download: int
    scalar_label_availability: int
    geometry_descriptor_availability: int
    time_to_positive_result: int


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="HiLiftAeroML",
        repo_id="nvidia/HiLiftAeroML",
        licence="cc-by-4.0",
        citation="NVIDIA HiLiftAeroML dataset card / NASA CRM-HL high-lift dataset",
        source_url="https://huggingface.co/datasets/nvidia/HiLiftAeroML",
        full_size_note="README reports about 63 TB compressed and 190 TB unzipped.",
        case_count_note="1,800 samples: 180 geometry variants across 10 angles of attack.",
        scalar_files=("force_mom_all.csv",),
        geometry_parameter_file="geo_values_all.csv",
        representative_stl="geo_LHC001_AoA_10/geo_LHC001_AoA_10.stl",
        motorsport_vehicle_relevance=2,
        wing_relevance=5,
        ease_of_compact_download=4,
        scalar_label_availability=5,
        geometry_descriptor_availability=4,
        time_to_positive_result=4,
    ),
    DatasetSpec(
        name="DrivAerML",
        repo_id="neashton/drivaerml",
        licence="cc-by-sa-4.0",
        citation="DrivAerML dataset card / Ashton et al. scale-resolving DrivAer CFD dataset",
        source_url="https://huggingface.co/datasets/neashton/drivaerml",
        full_size_note="README reports about 31 TB for the full dataset.",
        case_count_note=(
            "500 morphed DrivAer notchback geometries; compact force file has 484 labelled rows."
        ),
        scalar_files=("force_mom_all.csv", "force_mom_constref_all.csv"),
        geometry_parameter_file="geo_parameters_all.csv",
        representative_stl="run_1/drivaer_1.stl",
        motorsport_vehicle_relevance=5,
        wing_relevance=2,
        ease_of_compact_download=5,
        scalar_label_availability=5,
        geometry_descriptor_availability=5,
        time_to_positive_result=5,
    ),
    DatasetSpec(
        name="AhmedML",
        repo_id="neashton/ahmedml",
        licence="cc-by-sa-4.0",
        citation="AhmedML dataset card / Ashton et al. Ahmed-body CFD dataset",
        source_url="https://huggingface.co/datasets/neashton/ahmedml",
        full_size_note="README reports about 2 TB for the full dataset.",
        case_count_note="500 Ahmed-body geometry variants.",
        scalar_files=("force_mom_all.csv", "force_mom_varref_all.csv"),
        geometry_parameter_file="geo_parameters_all.csv",
        representative_stl="run_1/ahmed_1.stl",
        motorsport_vehicle_relevance=4,
        wing_relevance=1,
        ease_of_compact_download=5,
        scalar_label_availability=5,
        geometry_descriptor_availability=5,
        time_to_positive_result=4,
    ),
)


def _resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _head_size(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            size = response.headers.get("content-length")
            return {
                "ok": True,
                "content_length_bytes": int(size) if size is not None else None,
                "content_type": response.headers.get("content-type"),
            }
    except (OSError, TimeoutError) as exc:  # pragma: no cover - exercised by network state.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _api_file_summary(repo_id: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
        files = [str(item.get("rfilename", "")) for item in payload.get("siblings", [])]
        return {
            "ok": True,
            "hf_file_count": len(files),
            "stl_count": sum(name.endswith(".stl") for name in files),
            "boundary_count": sum("boundary" in name.lower() for name in files),
            "volume_count": sum("volume" in name.lower() for name in files),
            "split_files": [name for name in files if "split" in name.lower()][:20],
            "global_csvs": [
                name
                for name in files
                if name.endswith(".csv") and "/" not in name and "all" in name.lower()
            ],
        }
    except (
        OSError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:  # pragma: no cover - exercised by network state.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _score_spec(spec: DatasetSpec) -> int:
    return (
        spec.motorsport_vehicle_relevance
        + spec.wing_relevance
        + spec.ease_of_compact_download
        + spec.scalar_label_availability
        + spec.geometry_descriptor_availability
        + spec.time_to_positive_result
    )


def _metadata_size(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0


def write_aeromap3d_metadata_triage(out: Path) -> Path:
    """Inspect 3D bridge candidates without downloading large data."""

    candidates: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        scalar_checks = {
            filename: _head_size(_resolve_url(spec.repo_id, filename))
            for filename in (
                *spec.scalar_files,
                *(() if spec.geometry_parameter_file is None else (spec.geometry_parameter_file,)),
            )
        }
        scalar_bytes = sum(
            _metadata_size(check.get("content_length_bytes"))
            for check in scalar_checks.values()
            if check.get("ok")
        )
        stl_check = (
            _head_size(_resolve_url(spec.repo_id, spec.representative_stl))
            if spec.representative_stl is not None
            else {"ok": False, "error": "no representative STL configured"}
        )
        stl_bytes = _metadata_size(stl_check.get("content_length_bytes"))
        candidates.append(
            {
                "name": spec.name,
                "repo_id": spec.repo_id,
                "source_url": spec.source_url,
                "licence": spec.licence,
                "citation": spec.citation,
                "dataset_size": spec.full_size_note,
                "case_count": spec.case_count_note,
                "api_file_summary": _api_file_summary(spec.repo_id),
                "scalar_metadata_files": list(spec.scalar_files),
                "geometry_parameter_file": spec.geometry_parameter_file,
                "scalar_metadata_download_bytes": scalar_bytes,
                "representative_stl": spec.representative_stl,
                "representative_stl_head": stl_check,
                "estimated_3_stl_sample_bytes": stl_bytes * 3 if stl_bytes else None,
                "scalar_replay_feasible_without_volume_or_boundary": all(
                    check.get("ok") for check in scalar_checks.values()
                ),
                "ranking_inputs": {
                    "motorsport_vehicle_relevance": spec.motorsport_vehicle_relevance,
                    "wing_relevance": spec.wing_relevance,
                    "ease_of_compact_download": spec.ease_of_compact_download,
                    "scalar_label_availability": spec.scalar_label_availability,
                    "geometry_descriptor_availability": spec.geometry_descriptor_availability,
                    "time_to_positive_result": spec.time_to_positive_result,
                },
                "ranking_score": _score_spec(spec),
            }
        )

    by_name = {item["name"]: item for item in candidates}
    if by_name["DrivAerML"]["scalar_replay_feasible_without_volume_or_boundary"]:
        selected = "DrivAerML"
        selection_reason = (
            "DrivAerML satisfies the explicit selection rule: compact all-case geometry "
            "parameters and force/moment summaries are accessible."
        )
    elif by_name["HiLiftAeroML"]["scalar_replay_feasible_without_volume_or_boundary"]:
        selected = "HiLiftAeroML"
        selection_reason = "HiLiftAeroML compact scalar/geometry metadata is accessible."
    elif by_name["AhmedML"]["scalar_replay_feasible_without_volume_or_boundary"]:
        selected = "AhmedML"
        selection_reason = "AhmedML is the safe compact fallback."
    else:
        selected = None
        selection_reason = "No candidate exposed compact scalar metadata cleanly."

    payload = {
        "schema_version": AEROMAP3D_TRIAGE_SCHEMA,
        "classification": "AEROMAP_3D_COMPACT_PRACTICABILITY_TRIAGE",
        "download_policy": {
            "full_dataset_clone": False,
            "volume_data_downloaded": False,
            "boundary_field_downloaded": False,
            "cloud_used": False,
            "nim_used": False,
            "aerocliff_cfd_run": False,
        },
        "candidates": candidates,
        "ranked_by_positive_result_likelihood": [
            item["name"]
            for item in sorted(candidates, key=lambda item: -int(item["ranking_score"]))
        ],
        "selected_dataset": selected,
        "selection_reason": selection_reason,
    }
    atomic_write_json(out, payload)
    return out


def _download_text(url: str, path: Path, *, max_bytes: int = 2_000_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        msg = f"refusing oversized compact metadata download from {url}"
        raise RuntimeError(msg)
    path.write_bytes(content)
    return path


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return [
        {str(key).strip(): str(value).strip() for key, value in row.items()}
        for row in csv.DictReader(text.splitlines())
    ]


def _float(value: str) -> float:
    return float(value.strip())


def build_drivaerml_scalar_bridge_dataset(cache_dir: Path, out: Path) -> Path:
    """Download compact DrivAerML CSVs and write the 3D scalar bridge NPZ."""

    force_path = _download_text(
        _resolve_url("neashton/drivaerml", "force_mom_constref_all.csv"),
        cache_dir / "force_mom_constref_all.csv",
    )
    force_varref_path = _download_text(
        _resolve_url("neashton/drivaerml", "force_mom_all.csv"),
        cache_dir / "force_mom_all.csv",
    )
    geo_path = _download_text(
        _resolve_url("neashton/drivaerml", "geo_parameters_all.csv"),
        cache_dir / "geo_parameters_all.csv",
    )

    return build_drivaerml_scalar_bridge_dataset_from_paths(
        force_constref_path=force_path,
        force_varref_path=force_varref_path,
        geo_path=geo_path,
        out=out,
    )


def build_drivaerml_scalar_bridge_dataset_from_paths(
    *,
    force_constref_path: Path,
    force_varref_path: Path,
    geo_path: Path,
    out: Path,
) -> Path:
    """Write the DrivAerML scalar bridge NPZ from already available compact CSVs."""

    force_rows = {int(row["run"]): row for row in _read_csv_dicts(force_constref_path)}
    geo_rows = {int(row["Run"]): row for row in _read_csv_dicts(geo_path)}
    common_runs = sorted(set(force_rows) & set(geo_rows))
    if len(common_runs) < DRIVAERML_MIN_JOINED_ROWS:
        msg = "DrivAerML compact scalar bridge has too few joined rows"
        raise ValueError(msg)

    raw_geo_names = [name for name in geo_rows[common_runs[0]] if name != "Run"]
    feature_names = [f"geom_{_normalise_name(name)}" for name in raw_geo_names]
    target_names = ["integrated_cd", "integrated_cl"]
    features = np.asarray(
        [[_float(geo_rows[run][name]) for name in raw_geo_names] for run in common_runs],
        dtype=np.float64,
    )
    targets = np.asarray(
        [[_float(force_rows[run]["cd"]), _float(force_rows[run]["cl"])] for run in common_runs],
        dtype=np.float64,
    )
    case_ids = [f"drivaerml_run_{run}" for run in common_runs]
    group_ids = [f"drivaerml_geometry_{run}" for run in common_runs]

    npz_path = out.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        features=features,
        targets=targets,
        case_ids=np.asarray(case_ids),
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
        classification=np.asarray(AEROMAP3D_CLASSIFICATION),
        open_cfd_result=np.ones((), dtype=np.bool_),
        group_ids=np.asarray(group_ids),
    )
    payload = {
        "schema_version": AEROMAP3D_DATASET_SCHEMA,
        "classification": AEROMAP3D_CLASSIFICATION,
        "benchmark_class": "OPEN_CFD_AEROMAP_3D_BRIDGE",
        "source_dataset": "DrivAerML",
        "source_url": "https://huggingface.co/datasets/neashton/drivaerml",
        "licence": "cc-by-sa-4.0",
        "citation": "DrivAerML dataset card / Ashton et al. scale-resolving DrivAer CFD dataset",
        "case_count": len(common_runs),
        "feature_count": int(features.shape[1]),
        "target_count": int(targets.shape[1]),
        "feature_names": feature_names,
        "target_names": target_names,
        "split_policy": "geometry-disjoint over DrivAerML run/design parameters",
        "target_contract": {
            "primary_targets": target_names,
            "source_force_file": "force_mom_constref_all.csv",
            "reason": (
                "constant-reference coefficients keep scalar labels comparable across "
                "geometry variants"
            ),
            "force_mom_all_csv_retained_as_diagnostic": True,
        },
        "source_files": {
            "force_mom_constref_all.csv": {
                "path": str(force_constref_path),
                "sha256": sha256_file(force_constref_path),
                "bytes": force_constref_path.stat().st_size,
            },
            "force_mom_all.csv": {
                "path": str(force_varref_path),
                "sha256": sha256_file(force_varref_path),
                "bytes": force_varref_path.stat().st_size,
            },
            "geo_parameters_all.csv": {
                "path": str(geo_path),
                "sha256": sha256_file(geo_path),
                "bytes": geo_path.stat().st_size,
            },
        },
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "claim_boundary": {
            "open_cfd_result": True,
            "compact_3d_scalar_bridge": True,
            "aerocliff_result": False,
            "f1_geometry": False,
            "field_prediction": False,
            "live_cfd_savings": False,
            "external_predictor_accuracy": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _normalise_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def write_geometry_readiness_sample(
    stl_paths: Iterable[Path],
    out: Path,
    *,
    aerocliff_stl: Path | None = None,
    sample_points_per_geometry: int = 512,
) -> Path:
    """Compute compact descriptors for a few 3D aero STLs."""

    stl_list = list(stl_paths)
    if not 1 <= len(stl_list) <= MAX_GEOMETRY_SAMPLE_STLS:
        msg = "geometry readiness sample must contain between 1 and 5 STLs"
        raise ValueError(msg)

    summaries = [
        _stl_descriptor(path, sample_points_per_geometry=sample_points_per_geometry)
        for path in stl_list
    ]
    aerocliff_summary = (
        _stl_descriptor(aerocliff_stl, sample_points_per_geometry=sample_points_per_geometry)
        if aerocliff_stl is not None and aerocliff_stl.exists()
        else None
    )
    point_payload = {
        f"points_{idx}": item.pop("sample_points") for idx, item in enumerate(summaries)
    }
    if aerocliff_summary is not None:
        point_payload["points_aerocliff"] = aerocliff_summary.pop("sample_points")

    embedding_keys = [
        "bbox_dx",
        "bbox_dy",
        "bbox_dz",
        "surface_area",
        "normal_abs_x_mean",
        "normal_abs_y_mean",
        "normal_abs_z_mean",
        "triangle_area_mean",
        "triangle_area_std",
    ]
    embeddings = np.asarray(
        [[float(item["embedding"][key]) for key in embedding_keys] for item in summaries],
        dtype=np.float64,
    )
    if aerocliff_summary is not None:
        aero_embedding = np.asarray(
            [float(aerocliff_summary["embedding"][key]) for key in embedding_keys],
            dtype=np.float64,
        )
        mean = embeddings.mean(axis=0)
        scale = np.where(embeddings.std(axis=0) < EPSILON_STD, 1.0, embeddings.std(axis=0))
        aerocliff_summary["ood_distance_to_drivaerml_sample"] = float(
            np.linalg.norm((aero_embedding - mean) / scale)
        )

    npz_path = out.with_suffix(".points.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **point_payload)
    payload = {
        "schema_version": AEROMAP3D_GEOMETRY_SCHEMA,
        "classification": "AEROMAP_3D_GEOMETRY_READINESS_SAMPLE",
        "source_dataset": "DrivAerML",
        "stl_count": len(stl_list),
        "stl_summaries": summaries,
        "aerocliff_comparison": aerocliff_summary,
        "points_npz_path": str(npz_path),
        "points_npz_sha256": sha256_file(npz_path),
        "claim_boundary": {
            "geometry_ingestion": True,
            "accuracy_result": False,
            "field_prediction": False,
            "f1_accuracy": False,
            "aerocliff_accuracy": False,
        },
    }
    atomic_write_json(out, payload)
    return out


def _stl_descriptor(path: Path, *, sample_points_per_geometry: int) -> dict[str, Any]:
    if not path.exists():
        msg = f"STL does not exist: {path}"
        raise FileNotFoundError(msg)
    if _looks_ascii_stl(path):
        descriptor = _ascii_stl_descriptor(path, sample_points_per_geometry)
    else:
        descriptor = _binary_stl_descriptor(path, sample_points_per_geometry)
    descriptor["path"] = str(path)
    descriptor["sha256"] = sha256_file(path)
    descriptor["bytes"] = path.stat().st_size
    return descriptor


def _looks_ascii_stl(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(256)
    return header.lstrip().lower().startswith(b"solid") and b"facet" in header.lower()


def _empty_accumulators(sample_count: int) -> dict[str, Any]:
    return {
        "triangle_count": 0,
        "area_sum": 0.0,
        "area_sq_sum": 0.0,
        "area_max": 0.0,
        "bbox_min": np.full(3, np.inf, dtype=np.float64),
        "bbox_max": np.full(3, -np.inf, dtype=np.float64),
        "normal_abs_sum": np.zeros(3, dtype=np.float64),
        "normal_abs_sq_sum": np.zeros(3, dtype=np.float64),
        "samples": np.zeros((sample_count, 3), dtype=np.float64),
        "sample_count": 0,
    }


def _update_triangle(acc: dict[str, Any], vertices: FloatArray) -> None:
    edge_a = vertices[1] - vertices[0]
    edge_b = vertices[2] - vertices[0]
    cross = np.cross(edge_a, edge_b)
    area = 0.5 * float(np.linalg.norm(cross))
    if area <= 0.0:
        return
    normal = cross / (2.0 * area)
    centroid = vertices.mean(axis=0)
    idx = int(acc["triangle_count"])
    acc["triangle_count"] = idx + 1
    acc["area_sum"] = float(acc["area_sum"]) + area
    acc["area_sq_sum"] = float(acc["area_sq_sum"]) + area * area
    acc["area_max"] = max(float(acc["area_max"]), area)
    acc["bbox_min"] = np.minimum(acc["bbox_min"], vertices.min(axis=0))
    acc["bbox_max"] = np.maximum(acc["bbox_max"], vertices.max(axis=0))
    normal_abs = np.abs(normal)
    acc["normal_abs_sum"] = acc["normal_abs_sum"] + normal_abs
    acc["normal_abs_sq_sum"] = acc["normal_abs_sq_sum"] + normal_abs * normal_abs
    sample_limit = acc["samples"].shape[0]
    if idx < sample_limit:
        acc["samples"][idx] = centroid
        acc["sample_count"] = idx + 1
    else:
        # Deterministic reservoir replacement without retaining all triangles.
        replace_idx = (idx * 2_654_435_761) % (idx + 1)
        if replace_idx < sample_limit:
            acc["samples"][replace_idx] = centroid


def _finalise_descriptor(acc: dict[str, Any], *, source_format: str) -> dict[str, Any]:
    count = int(acc["triangle_count"])
    if count <= 0:
        msg = "STL contained no nondegenerate triangles"
        raise ValueError(msg)
    bbox_min = cast("FloatArray", acc["bbox_min"])
    bbox_max = cast("FloatArray", acc["bbox_max"])
    bbox_size = bbox_max - bbox_min
    area_mean = float(acc["area_sum"]) / count
    area_variance = max(0.0, float(acc["area_sq_sum"]) / count - area_mean * area_mean)
    normal_abs_mean = cast("FloatArray", acc["normal_abs_sum"]) / count
    normal_abs_sq_mean = cast("FloatArray", acc["normal_abs_sq_sum"]) / count
    normal_abs_std = np.sqrt(np.maximum(0.0, normal_abs_sq_mean - normal_abs_mean**2))
    sample_count = int(acc["sample_count"])
    samples = cast("FloatArray", acc["samples"])[:sample_count].copy()
    embedding = {
        "bbox_dx": float(bbox_size[0]),
        "bbox_dy": float(bbox_size[1]),
        "bbox_dz": float(bbox_size[2]),
        "surface_area": float(acc["area_sum"]),
        "normal_abs_x_mean": float(normal_abs_mean[0]),
        "normal_abs_y_mean": float(normal_abs_mean[1]),
        "normal_abs_z_mean": float(normal_abs_mean[2]),
        "triangle_area_mean": area_mean,
        "triangle_area_std": math.sqrt(area_variance),
    }
    return {
        "source_format": source_format,
        "triangle_count": count,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size": bbox_size.tolist(),
        "scale_diagonal": float(np.linalg.norm(bbox_size)),
        "surface_area": float(acc["area_sum"]),
        "triangle_area_mean": area_mean,
        "triangle_area_std": math.sqrt(area_variance),
        "triangle_area_max": float(acc["area_max"]),
        "normal_abs_mean": normal_abs_mean.tolist(),
        "normal_abs_std": normal_abs_std.tolist(),
        "sample_point_count": sample_count,
        "embedding": embedding,
        "sample_points": samples,
    }


def _ascii_stl_descriptor(path: Path, sample_count: int) -> dict[str, Any]:
    acc = _empty_accumulators(sample_count)
    vertices: list[list[float]] = []
    with path.open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            parts = stripped.split()
            if len(parts) != STL_VERTEX_FIELD_COUNT:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(vertices) == STL_VERTICES_PER_TRIANGLE:
                _update_triangle(acc, np.asarray(vertices, dtype=np.float64))
                vertices.clear()
    return _finalise_descriptor(acc, source_format="ascii_stl")


def _binary_stl_descriptor(path: Path, sample_count: int) -> dict[str, Any]:
    acc = _empty_accumulators(sample_count)
    with path.open("rb") as handle:
        handle.seek(80)
        tri_count = struct.unpack("<I", handle.read(4))[0]
        for _ in range(tri_count):
            raw = handle.read(BINARY_STL_TRIANGLE_BYTES)
            if len(raw) != BINARY_STL_TRIANGLE_BYTES:
                break
            values = struct.unpack("<12fH", raw)
            vertices = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
            _update_triangle(acc, vertices)
    return _finalise_descriptor(acc, source_format="binary_stl")

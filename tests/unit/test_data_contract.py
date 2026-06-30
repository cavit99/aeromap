from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import trimesh

from aeromap.constants import REF
from aeromap.data.converter import (
    ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA,
    ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS,
    ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA,
    ACCEPTED_SMOKE_EXPORTED_VTU_CELLS,
    ACCEPTED_SMOKE_SIMULATION_ID,
    ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS,
    convert_case_to_sample,
)
from aeromap.data.loader import (
    CampaignSampleDataset,
    TrainingEligibilityError,
    aggregate_source_cell_field,
    batch_samples,
    build_campaign_dataloader,
    collate_variable_samples,
    load_sample,
)
from aeromap.data.sampling import DataSamplingConfig, build_sample_selection
from aeromap.data.schema import DataSampleManifest
from aeromap.data.splits import (
    GroupedSplitEntry,
    GroupedSplitManifest,
    build_grouped_split_manifest,
)
from aeromap.data.vtk_workflow import load_geometry_stl, workflow_manifest
from aeromap.io import sha256_file


def _write_surface(path: Path, *, pressure: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    surface_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    surface = pv.PolyData(surface_points, np.array([4, 0, 1, 2, 3]))
    surface.cell_data["p"] = np.array([pressure], dtype=np.float64)
    surface.cell_data["wallShearStress"] = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
    surface.cell_data["surface_region_id"] = np.array([2], dtype=np.int32)
    surface.cell_data["surface_region"] = np.array(["underfloor"])
    surface.cell_data["local_face_area_m2"] = np.array([1.0], dtype=np.float64)
    surface.save(path)


def _simple_volume(
    *,
    cell_ids: list[int],
    pressure: list[float] | None = None,
    include_cell_id: bool = True,
    field_association: str = "cell",
) -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cell_count = len(cell_ids)
    volume = pv.UnstructuredGrid(
        np.tile(np.array([4, 0, 1, 2, 3], dtype=np.int64), cell_count),
        np.full(cell_count, pv.CellType.TETRA, dtype=np.uint8),
        points,
    )
    if include_cell_id:
        volume.cell_data["cellID"] = np.asarray(cell_ids, dtype=np.int64)
    pressure_values = np.asarray(
        pressure if pressure is not None else [float(cell_id) for cell_id in cell_ids],
        dtype=np.float64,
    )
    velocity_values = np.column_stack(
        [
            np.full(cell_count, 40.0, dtype=np.float64),
            np.zeros(cell_count, dtype=np.float64),
            np.zeros(cell_count, dtype=np.float64),
        ],
    )
    if field_association == "cell":
        volume.cell_data["p"] = pressure_values
        volume.cell_data["U"] = velocity_values
    elif field_association == "point":
        volume.point_data["p"] = np.linspace(0.0, 3.0, len(points), dtype=np.float64)
        volume.point_data["U"] = np.column_stack(
            [
                np.full(len(points), 40.0, dtype=np.float64),
                np.zeros(len(points), dtype=np.float64),
                np.zeros(len(points), dtype=np.float64),
            ],
        )
    else:
        raise ValueError(field_association)
    return volume


def _accepted_smoke_volume() -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells: list[int] = []
    cell_types: list[int] = []
    cell_ids: list[int] = []
    single_source_count = ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS - ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA
    for source_id in range(single_source_count):
        cells.extend([8, 0, 1, 2, 3, 4, 5, 6, 7])
        cell_types.append(int(pv.CellType.HEXAHEDRON))
        cell_ids.append(source_id)

    child_counts = np.full(ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA, 14, dtype=np.int64)
    child_counts[:166] += 1
    tetrahedra_remaining = ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA
    for source_id, child_count in zip(
        range(single_source_count, ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS),
        child_counts,
        strict=True,
    ):
        for _ in range(int(child_count)):
            if tetrahedra_remaining > 0:
                cells.extend([4, 0, 1, 2, 4])
                cell_types.append(int(pv.CellType.TETRA))
                tetrahedra_remaining -= 1
            else:
                cells.extend([5, 0, 1, 2, 3, 4])
                cell_types.append(int(pv.CellType.PYRAMID))
            cell_ids.append(source_id)

    volume = pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        points,
    )
    ids = np.asarray(cell_ids, dtype=np.int64)
    values = ids.astype(np.float64)
    volume.cell_data["cellID"] = ids
    volume.cell_data["p"] = values
    volume.cell_data["U"] = np.column_stack([values, values + 1.0, values + 2.0])
    volume.cell_data["k"] = values * 0.01
    volume.cell_data["omega"] = values * 0.02
    volume.cell_data["nut"] = values * 0.03
    assert len(ids) == ACCEPTED_SMOKE_EXPORTED_VTU_CELLS
    assert tetrahedra_remaining == 0
    assert np.count_nonzero(np.asarray(cell_types) == pv.CellType.PYRAMID) == (
        ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS
    )
    return volume


def _write_synthetic_case(
    case_dir: Path,
    *,
    case_class: str,
    accepted: bool | str = True,
    mapped_wall_vtp: str | None = None,
    volume: pv.UnstructuredGrid | None = None,
    source_cell_count: int = 1,
    simulation_id: str = "simulation_test",
    geometry_id: str = "geometry_test",
    state_id: str = "state_test",
) -> None:
    quality = case_dir / "quality"
    outputs = case_dir / "outputs"
    quality.mkdir(parents=True)
    outputs.mkdir()

    _write_surface(outputs / "article_wall_regions.vtp", pressure=2.0)
    (volume or _simple_volume(cell_ids=list(range(source_cell_count)))).save(outputs / "volume.vtu")

    (case_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case_test",
                "geometry_id": geometry_id,
                "state_id": state_id,
                "simulation_id": simulation_id,
                "attempt_id": f"attempt_{simulation_id}",
                "git_sha": "abc123",
                "surface_export_id": "surface_export_test",
                "cfd_config": {"quality": {"case_class": case_class}},
            },
        ),
        encoding="utf-8",
    )
    (quality / "status.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "accepted_scope": case_class,
                "artifacts": {
                    "mapped_wall_vtp": mapped_wall_vtp or str(outputs / "article_wall_regions.vtp"),
                    "volume_vtu": str(outputs / "volume.vtu"),
                    "convergence": str(quality / "convergence.json"),
                    "force_integration": str(quality / "force_integration.json"),
                    "mesh": str(quality / "mesh.json"),
                },
            },
        ),
        encoding="utf-8",
    )
    (quality / "convergence.json").write_text('{"force_stable": true}', encoding="utf-8")
    (quality / "force_integration.json").write_text(
        '{"independent_total_n": [1.0, 0.0, 0.0]}',
        encoding="utf-8",
    )
    (quality / "mesh.json").write_text(
        json.dumps({"mesh_ok": True, "cells": source_cell_count}),
        encoding="utf-8",
    )
    (quality / "region_mapping.json").write_text('{"coverage": 1.0}', encoding="utf-8")


def _manifest_payload(*, case_class: str, training_eligible: bool) -> dict[str, object]:
    return {
        "sample_id": "sample_bad",
        "geometry_id": "geometry_test",
        "state_id": "state_test",
        "simulation_id": "simulation_test",
        "attempt_id": "attempt_test",
        "case_class": case_class,
        "training_eligible": training_eligible,
        "source_case_dir": "case",
        "arrays_path": "arrays.npz",
        "arrays_sha256": "abc",
        "reference": {},
        "counts": {},
        "array_names": [],
        "vtk_workflow": {
            "surface_adapter": "surface",
            "volume_adapter": "volume",
            "surface_path": "wall.vtp",
            "volume_path": "volume.vtu",
            "semantics": {"surface": "wall", "volume": "cellID"},
        },
        "volume_provenance": {
            "source_openfoam_cell_count": 1,
            "exported_vtu_cell_count": 1,
            "cellid_count": 1,
            "cellid_unique_source_count": 1,
            "cellid_missing_source_count": 0,
            "cellid_min": 0,
            "cellid_max": 0,
            "cellid_maps_all_exported_cells": True,
            "cellid_covers_all_source_cells": True,
            "duplicated_source_cell_count": 0,
            "duplicated_exported_child_cell_count": 0,
            "foam_to_vtk_decomposition": {
                "source_polyhedra_decomposed": 0,
                "child_tetrahedra": 0,
                "child_pyramids": 0,
                "exported_child_cells": 0,
                "net_exported_cell_increase": 0,
            },
            "duplicated_child_field_validation": {},
            "source_reduction_semantics": "aggregate through cellID",
        },
        "field_validation": {
            "checks": {
                "surface_cp": {
                    "equation": "rho * p / q_inf",
                    "dimensional_array": "surface_pressure_kinematic",
                    "nondimensional_array": "surface_cp",
                    "max_abs_error": 0.0,
                    "tolerance": 1e-12,
                    "passed": True,
                },
            },
        },
        "loads": {},
        "quality": {},
    }


def test_convert_smoke_case_to_nondimensional_sample_and_reject_training(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(case_dir, case_class="NON_CAMPAIGN_ENGINEERING_SMOKE")

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["case_class"] == "NON_CAMPAIGN_ENGINEERING_SMOKE"
    assert manifest["training_eligible"] is False
    assert manifest["geometry_id"] == "geometry_test"
    assert manifest["counts"] == {
        "surface_points": 4,
        "surface_faces": 1,
        "volume_points": 4,
        "volume_cells": 1,
        "volume_exported_vtu_cells": 1,
        "volume_source_openfoam_cells": 1,
        "volume_unique_cell_ids": 1,
    }
    assert manifest["vtk_workflow"]["surface_format"] == "VTP"
    assert manifest["vtk_workflow"]["volume_format"] == "VTU"
    assert manifest["volume_provenance"]["source_openfoam_cell_count"] == 1
    assert manifest["volume_provenance"]["exported_vtu_cell_count"] == 1
    assert manifest["field_validation"]["checks"]["surface_cp"]["passed"] is True

    with np.load(artifacts.arrays_path, allow_pickle=False) as arrays:
        assert arrays["surface_points_nd"][1].tolist() == pytest.approx([0.5, 0.0, 0.0])
        assert arrays["surface_cp"][0] == pytest.approx((REF.rho_kg_m3 * 2.0) / REF.q_inf_pa)
        assert arrays["surface_cf"][0].tolist() == pytest.approx(
            [(REF.rho_kg_m3 * 0.5) / REF.q_inf_pa, 0.0, 0.0],
        )
        assert arrays["volume_cell_id"].tolist() == [0]
        assert arrays["volume_velocity_nd"][0].tolist() == pytest.approx([1.0, 0.0, 0.0])

    with pytest.raises(TrainingEligibilityError):
        load_sample(artifacts.sample_dir)

    sample = load_sample(artifacts.sample_dir, allow_non_campaign=True)
    batch = batch_samples([sample, sample])
    assert batch["surface_points_nd"].shape == (2, 4, 3)


def test_training_loader_accepts_campaign_samples_by_default(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(case_dir, case_class="CAMPAIGN_REFERENCE_CFD")

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")
    sample = load_sample(artifacts.sample_dir)

    assert sample.manifest.training_eligible is True


def test_rejected_campaign_sample_is_not_training_eligible(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="CAMPAIGN_REFERENCE_CFD",
        accepted=False,
    )

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")
    sample = load_sample(artifacts.sample_dir, allow_non_campaign=True)

    assert sample.manifest.case_class == "CAMPAIGN_REFERENCE_CFD"
    assert sample.manifest.training_eligible is False
    with pytest.raises(TrainingEligibilityError):
        load_sample(artifacts.sample_dir)


def test_converter_refuses_existing_sample_id_and_preserves_sample(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    samples_dir = tmp_path / "samples"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        simulation_id="simulation_preserve",
    )
    artifacts = convert_case_to_sample(case_dir, samples_dir)
    original_manifest = artifacts.manifest_path.read_text(encoding="utf-8")
    original_arrays_sha256 = sha256_file(artifacts.arrays_path)

    _write_surface(case_dir / "outputs" / "article_wall_regions.vtp", pressure=99.0)
    with pytest.raises(FileExistsError, match="refusing to overwrite immutable sample"):
        convert_case_to_sample(case_dir, samples_dir)

    assert artifacts.manifest_path.read_text(encoding="utf-8") == original_manifest
    assert sha256_file(artifacts.arrays_path) == original_arrays_sha256
    sample = load_sample(artifacts.sample_dir, allow_non_campaign=True)
    assert sample.arrays["surface_cp"][0] == pytest.approx((REF.rho_kg_m3 * 2.0) / REF.q_inf_pa)


def test_converter_rejects_non_boolean_accepted_status(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="CAMPAIGN_REFERENCE_CFD",
        accepted="false",
    )

    with pytest.raises(TypeError, match="accepted must be a boolean"):
        convert_case_to_sample(case_dir, tmp_path / "samples")


def test_manifest_forbids_training_eligible_smoke() -> None:
    payload = _manifest_payload(
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        training_eligible=True,
    )

    with pytest.raises(ValueError, match="only CAMPAIGN_REFERENCE_CFD"):
        DataSampleManifest.model_validate(payload)


def test_manifest_rejects_failed_field_validation() -> None:
    payload = _manifest_payload(case_class="CAMPAIGN_REFERENCE_CFD", training_eligible=True)
    field_validation = payload["field_validation"]
    assert isinstance(field_validation, dict)
    checks = field_validation["checks"]
    assert isinstance(checks, dict)
    surface_cp = checks["surface_cp"]
    assert isinstance(surface_cp, dict)
    surface_cp["passed"] = False

    with pytest.raises(ValueError, match="field validation failed"):
        DataSampleManifest.model_validate(payload)


def test_workflow_manifest_names_official_stl_vtp_vtu_adapters(tmp_path: Path) -> None:
    stl_path = tmp_path / "article.stl"
    trimesh.creation.box().export(stl_path)

    workflow = workflow_manifest(
        tmp_path / "article_wall_regions.vtp",
        tmp_path / "volume.vtu",
        geometry_path=stl_path,
    )
    mesh = load_geometry_stl(stl_path)

    assert workflow.geometry_format == "STL"
    assert workflow.surface_format == "VTP"
    assert workflow.volume_format == "VTU"
    assert workflow.geometry_adapter == "aeromap.data.vtk_workflow.load_geometry_stl"
    assert workflow.surface_adapter == "aeromap.data.vtk_workflow.load_wall_vtp"
    assert workflow.volume_adapter == "aeromap.data.vtk_workflow.load_volume_vtu"
    assert workflow.geometry_path == str(stl_path)
    assert "geometry" in workflow.semantics
    assert len(mesh.faces) > 0


def test_geometry_stl_adapter_rejects_non_stl_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="STL"):
        load_geometry_stl(tmp_path / "article.obj")


def test_converter_resolves_relative_artifacts_against_case_dir_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        mapped_wall_vtp="outputs/article_wall_regions.vtp",
    )
    other_cwd = tmp_path / "other"
    _write_surface(other_cwd / "outputs" / "article_wall_regions.vtp", pressure=99.0)
    monkeypatch.chdir(other_cwd)

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")

    with np.load(artifacts.arrays_path, allow_pickle=False) as arrays:
        assert arrays["surface_cp"][0] == pytest.approx((REF.rho_kg_m3 * 2.0) / REF.q_inf_pa)


def test_converter_requires_cellid_for_volume_vtu(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        volume=_simple_volume(cell_ids=[0], include_cell_id=False),
    )

    with pytest.raises(ValueError, match="cellID"):
        convert_case_to_sample(case_dir, tmp_path / "samples")


def test_converter_rejects_duplicate_child_field_spread(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        source_cell_count=2,
        volume=_simple_volume(cell_ids=[0, 1, 1], pressure=[0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="duplicated child field validation"):
        convert_case_to_sample(case_dir, tmp_path / "samples")


def test_source_cell_aggregation_uses_cellid(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        source_cell_count=2,
        volume=_simple_volume(cell_ids=[0, 1, 1], pressure=[4.0, 8.0, 8.0]),
    )

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")
    sample = load_sample(artifacts.sample_dir, allow_non_campaign=True)

    reduced = aggregate_source_cell_field(sample, "volume_pressure_kinematic")

    assert reduced.tolist() == pytest.approx([4.0, 8.0])


def test_variable_collation_accepts_different_volume_sizes(tmp_path: Path) -> None:
    first_case = tmp_path / "first"
    second_case = tmp_path / "second"
    _write_synthetic_case(
        first_case,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        simulation_id="simulation_variable_first",
    )
    _write_synthetic_case(
        second_case,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        source_cell_count=2,
        simulation_id="simulation_variable_second",
        volume=_simple_volume(cell_ids=[0, 1]),
    )
    first = load_sample(
        convert_case_to_sample(first_case, tmp_path / "samples").sample_dir,
        allow_non_campaign=True,
    )
    second = load_sample(
        convert_case_to_sample(second_case, tmp_path / "samples").sample_dir,
        allow_non_campaign=True,
    )

    variable_batch = collate_variable_samples([first, second])

    surface_faces = variable_batch.arrays["surface_faces"]
    first_surface_start, first_surface_end = variable_batch.offsets["surface_faces"][0].tolist()
    second_surface_start, second_surface_end = variable_batch.offsets["surface_faces"][1].tolist()
    assert variable_batch.offsets["surface_faces"].tolist() == [[0, 5], [5, 10]]
    assert surface_faces[first_surface_start:first_surface_end].tolist() == [4, 0, 1, 2, 3]
    assert surface_faces[second_surface_start:second_surface_end].tolist() == [4, 4, 5, 6, 7]
    assert variable_batch.offsets["surface_face_values"].tolist() == [[0, 1], [1, 2]]
    assert variable_batch.arrays["volume_celltypes"].shape == (3,)
    assert variable_batch.offsets["volume_exported_vtu_cells"].tolist() == [[0, 1], [1, 3]]
    assert variable_batch.offsets["volume_source_openfoam_cells"].tolist() == [[0, 1], [1, 3]]
    assert variable_batch.arrays["volume_exported_cell_sample_index"].tolist() == [0, 1, 1]
    assert variable_batch.arrays["volume_cell_id"].tolist() == [0, 0, 1]
    assert variable_batch.arrays["volume_global_cell_id"].tolist() == [0, 1, 2]
    assert variable_batch.identifiers["sample_id"] == (
        first.manifest.sample_id,
        second.manifest.sample_id,
    )
    assert variable_batch.identifiers["geometry_id"] == ("geometry_test", "geometry_test")
    assert variable_batch.identifiers["case_class"] == (
        "NON_CAMPAIGN_ENGINEERING_SMOKE",
        "NON_CAMPAIGN_ENGINEERING_SMOKE",
    )
    assert variable_batch.identifiers["training_eligible"] == (False, False)
    assert "volume_global_cell_id" in variable_batch.semantics["volume_reduction"]


def test_variable_collation_records_point_field_offsets(tmp_path: Path) -> None:
    first_case = tmp_path / "first"
    second_case = tmp_path / "second"
    _write_synthetic_case(
        first_case,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        simulation_id="simulation_point_first",
        volume=_simple_volume(cell_ids=[0], field_association="point"),
    )
    _write_synthetic_case(
        second_case,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        simulation_id="simulation_point_second",
        volume=_simple_volume(cell_ids=[0], field_association="point"),
    )
    first = load_sample(
        convert_case_to_sample(first_case, tmp_path / "samples").sample_dir,
        allow_non_campaign=True,
    )
    second = load_sample(
        convert_case_to_sample(second_case, tmp_path / "samples").sample_dir,
        allow_non_campaign=True,
    )

    variable_batch = collate_variable_samples([first, second])

    assert variable_batch.arrays["volume_pressure_association"].tolist() == ["point"]
    assert variable_batch.offsets["volume_pressure_values"].tolist() == [[0, 4], [4, 8]]
    assert variable_batch.offsets["volume_velocity_values"].tolist() == [[0, 4], [4, 8]]
    assert variable_batch.arrays["volume_pressure_sample_index"].tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]


def test_accepted_smoke_cell_count_contract_is_encoded(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="NON_CAMPAIGN_ENGINEERING_SMOKE",
        source_cell_count=ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS,
        simulation_id=ACCEPTED_SMOKE_SIMULATION_ID,
        volume=_accepted_smoke_volume(),
    )

    artifacts = convert_case_to_sample(case_dir, tmp_path / "samples")
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["volume_provenance"]

    assert provenance["source_openfoam_cell_count"] == ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS
    assert provenance["exported_vtu_cell_count"] == ACCEPTED_SMOKE_EXPORTED_VTU_CELLS
    assert provenance["cellid_unique_source_count"] == ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS
    assert provenance["cellid_missing_source_count"] == 0
    assert provenance["duplicated_source_cell_count"] == ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA
    assert provenance["foam_to_vtk_decomposition"] == {
        "source_polyhedra_decomposed": ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA,
        "child_tetrahedra": ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA,
        "child_pyramids": ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS,
        "exported_child_cells": ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA
        + ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS,
        "net_exported_cell_increase": ACCEPTED_SMOKE_EXPORTED_VTU_CELLS
        - ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS,
    }
    for field_name in ("p", "U", "k", "omega", "nut"):
        report = provenance["duplicated_child_field_validation"][field_name]
        assert report["max_abs_spread"] == 0.0
        assert report["passed"] is True


def test_campaign_dataset_rejects_smoke_by_default_and_override_is_test_scoped(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(case_dir, case_class="NON_CAMPAIGN_ENGINEERING_SMOKE")
    sample_dir = convert_case_to_sample(case_dir, tmp_path / "samples").sample_dir

    dataset = CampaignSampleDataset([sample_dir])
    with pytest.raises(TrainingEligibilityError):
        _ = dataset[0]

    test_dataset = CampaignSampleDataset([sample_dir], allow_non_campaign_for_tests=True)
    loaded = test_dataset[0]

    assert loaded.sample.manifest.case_class == "NON_CAMPAIGN_ENGINEERING_SMOKE"
    assert loaded.sample.manifest.training_eligible is False


def test_campaign_dataloader_collates_campaign_samples_and_preserves_identifiers(
    tmp_path: Path,
) -> None:
    first_case = tmp_path / "first"
    second_case = tmp_path / "second"
    _write_synthetic_case(
        first_case,
        case_class="CAMPAIGN_REFERENCE_CFD",
        simulation_id="simulation_first",
        geometry_id="geometry_a",
        state_id="state_a1",
    )
    _write_synthetic_case(
        second_case,
        case_class="CAMPAIGN_REFERENCE_CFD",
        simulation_id="simulation_second",
        geometry_id="geometry_b",
        state_id="state_b1",
    )
    first = convert_case_to_sample(first_case, tmp_path / "samples").sample_dir
    second = convert_case_to_sample(second_case, tmp_path / "samples").sample_dir

    loader = build_campaign_dataloader([first, second], batch_size=2, num_workers=0)
    batch = next(iter(loader))

    assert batch.variable.identifiers["geometry_id"] == ("geometry_a", "geometry_b")
    assert batch.variable.identifiers["state_id"] == ("state_a1", "state_b1")
    assert batch.variable.identifiers["case_class"] == (
        "CAMPAIGN_REFERENCE_CFD",
        "CAMPAIGN_REFERENCE_CFD",
    )
    assert batch.variable.identifiers["training_eligible"] == (True, True)
    assert batch.selections == (None, None)


def test_deterministic_sampling_plan_preserves_source_cell_semantics(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    _write_synthetic_case(
        case_dir,
        case_class="CAMPAIGN_REFERENCE_CFD",
        source_cell_count=3,
        volume=_simple_volume(cell_ids=[0, 1, 1, 2], pressure=[4.0, 8.0, 8.0, 12.0]),
    )
    sample = load_sample(convert_case_to_sample(case_dir, tmp_path / "samples").sample_dir)
    config = DataSamplingConfig(
        seed=11,
        surface_face_count=1,
        volume_source_cell_count=1,
    )

    first = build_sample_selection(sample, config)
    second = build_sample_selection(sample, config)

    assert first is not None
    assert second is not None
    assert first.surface_face_indices is not None
    assert second.surface_face_indices is not None
    assert first.volume_source_cell_ids is not None
    assert second.volume_source_cell_ids is not None
    assert first.volume_exported_cell_indices is not None
    assert first.surface_face_indices.tolist() == second.surface_face_indices.tolist()
    assert first.volume_source_cell_ids.tolist() == second.volume_source_cell_ids.tolist()
    selected_source_ids = set(first.volume_source_cell_ids.tolist())
    selected_exported_ids = sample.arrays["volume_cell_id"][first.volume_exported_cell_indices]
    assert set(selected_exported_ids.tolist()) == selected_source_ids
    assert "cellID" in first.semantics["volume_source_cells"]


def test_grouped_split_manifest_rejects_geometry_family_leakage() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        GroupedSplitManifest(
            entries=[
                GroupedSplitEntry(
                    sample_id="sample_a",
                    geometry_id="geometry_family_a",
                    state_id="state_a",
                    simulation_id="simulation_a",
                    attempt_id="attempt_a",
                    case_class="CAMPAIGN_REFERENCE_CFD",
                    training_eligible=True,
                    split="train",
                ),
                GroupedSplitEntry(
                    sample_id="sample_b",
                    geometry_id="geometry_family_a",
                    state_id="state_b",
                    simulation_id="simulation_b",
                    attempt_id="attempt_b",
                    case_class="CAMPAIGN_REFERENCE_CFD",
                    training_eligible=True,
                    split="test",
                ),
            ],
        )


def test_build_grouped_split_manifest_uses_geometry_id_groups(tmp_path: Path) -> None:
    first_case = tmp_path / "first"
    second_case = tmp_path / "second"
    _write_synthetic_case(
        first_case,
        case_class="CAMPAIGN_REFERENCE_CFD",
        simulation_id="simulation_first",
        geometry_id="geometry_shared",
        state_id="state_1",
    )
    _write_synthetic_case(
        second_case,
        case_class="CAMPAIGN_REFERENCE_CFD",
        simulation_id="simulation_second",
        geometry_id="geometry_shared",
        state_id="state_2",
    )
    first = load_sample(convert_case_to_sample(first_case, tmp_path / "samples").sample_dir)
    second = load_sample(convert_case_to_sample(second_case, tmp_path / "samples").sample_dir)

    manifest = build_grouped_split_manifest(
        [first, second],
        {"geometry_shared": "calibration"},
        notes=["geometry-family held-out split fixture"],
    )

    assert {entry.split for entry in manifest.entries} == {"calibration"}
    assert {entry.geometry_id for entry in manifest.entries} == {"geometry_shared"}
    assert [entry.state_id for entry in manifest.entries] == ["state_1", "state_2"]

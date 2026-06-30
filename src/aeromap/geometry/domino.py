"""DoMINO-specific geometry export adapter."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
import trimesh

from aeromap.geometry.generator import build_article
from aeromap.geometry.validate import validate_mesh
from aeromap.io import atomic_write_json, sha256_file
from aeromap.parameters import AeroParams
from aeromap.transforms import apply_ride_height_pitch

MAX_REFINEMENT_STEPS = 3
CAD_CHORD_TOLERANCE_M = 0.00015
CAD_ANGULAR_TOLERANCE_RAD = 0.006
NIM_GROUND_PLANE_Z_M = -0.318469


@dataclass(frozen=True)
class DominoGeometryExport:
    stl_path: Path
    manifest_path: Path
    surface_point_count: int
    meets_target_surface_points: bool
    subdivision_steps: int
    nim_input_contract_valid: bool


class DominoGeometryAdapter:
    """Export NIM-ready high-resolution single-part geometry in metres."""

    def __init__(self, *, target_surface_points: int = 300_000) -> None:
        if target_surface_points <= 0:
            message = "target_surface_points must be positive"
            raise ValueError(message)
        self.target_surface_points = target_surface_points

    def subdivide_to_target(self, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
        """Use subdivision only as a final point-sampling aid after CAD retessellation."""

        refined = mesh
        steps = 0
        while len(refined.vertices) < self.target_surface_points and steps < MAX_REFINEMENT_STEPS:
            vertices, faces = trimesh.remesh.subdivide(  # type: ignore[no-untyped-call]
                np.asarray(refined.vertices, dtype=np.float64),
                np.asarray(refined.faces, dtype=np.int64),
            )
            refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            refined.remove_unreferenced_vertices()
            steps += 1
        return refined, steps

    def refine_to_target(self, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
        """Backward-compatible alias for the bounded sampling-aid subdivision step."""

        return self.subdivide_to_target(mesh)

    def export(self, params: AeroParams, output_dir: Path) -> DominoGeometryExport:
        article = build_article(params)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "domino_article.stl"
            cq.exporters.export(
                article,
                str(tmp_path),
                tolerance=CAD_CHORD_TOLERANCE_M,
                angularTolerance=CAD_ANGULAR_TOLERANCE_RAD,
            )
            mesh = trimesh.load_mesh(tmp_path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.vertices = apply_ride_height_pitch(
            np.asarray(mesh.vertices, dtype=np.float64),
            ride_height_mm=params.ride_height_mm,
            pitch_deg=params.pitch_deg,
        )
        validation = validate_mesh(mesh)
        cad_to_nim_transform = np.eye(4, dtype=np.float64)
        cad_to_nim_transform[2, 3] = NIM_GROUND_PLANE_Z_M
        nim_to_cad_transform = np.linalg.inv(cad_to_nim_transform)
        mesh.apply_transform(cad_to_nim_transform)
        mesh.fix_normals()
        mesh, subdivision_steps = self.subdivide_to_target(mesh)
        mesh.fix_normals()

        stl_path = output_dir / "domino_article_highres.stl"
        mesh.export(stl_path, file_type="stl")
        surface_point_count = len(mesh.vertices)
        meets_target_surface_points = surface_point_count >= self.target_surface_points
        nim_input_contract_valid = bool(validation.valid and meets_target_surface_points)
        manifest: dict[str, Any] = {
            "adapter": "DominoGeometryAdapter",
            "purpose": "DoMINO/NIM geometry inference input, not CFD meshing or preview",
            "units": "metres",
            "single_part": bool(len(mesh.split(only_watertight=False)) == 1),
            "yaw_deg_required_for_primary_benchmark": 0.0,
            "nim_ground_plane_z_m": NIM_GROUND_PLANE_Z_M,
            "cad_ground_plane_z_m": 0.0,
            "cad_to_nim_transform_4x4": cad_to_nim_transform.tolist(),
            "nim_to_cad_transform_4x4": nim_to_cad_transform.tolist(),
            "cad_retessellation": {
                "chord_tolerance_m": CAD_CHORD_TOLERANCE_M,
                "angular_tolerance_rad": CAD_ANGULAR_TOLERANCE_RAD,
            },
            "target_surface_points": self.target_surface_points,
            "surface_point_count": surface_point_count,
            "meets_target_surface_points": meets_target_surface_points,
            "subdivision_used_as_sampling_aid": subdivision_steps > 0,
            "subdivision_steps": subdivision_steps,
            "nim_input_contract_valid": nim_input_contract_valid,
            "validation": validation.model_dump(mode="json"),
            "params": params.model_dump(),
            "stl_sha256": sha256_file(stl_path),
        }
        manifest_path = output_dir / "domino_geometry_manifest.json"
        atomic_write_json(manifest_path, manifest)
        return DominoGeometryExport(
            stl_path=stl_path,
            manifest_path=manifest_path,
            surface_point_count=surface_point_count,
            meets_target_surface_points=meets_target_surface_points,
            subdivision_steps=subdivision_steps,
            nim_input_contract_valid=nim_input_contract_valid,
        )

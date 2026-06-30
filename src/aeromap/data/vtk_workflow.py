"""Explicit VTP/VTU adapters for OpenFOAM export interface artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyvista as pv
import trimesh

from aeromap.data.schema import VtkWorkflowManifest


def load_geometry_stl(path: Path) -> trimesh.Trimesh:
    """Load the source deterministic AeroCliff STL used by the OpenFOAM case."""

    if path.suffix.lower() != ".stl":
        msg = f"expected an STL geometry export, got {path}"
        raise ValueError(msg)
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        scene = cast("trimesh.Scene", loaded)
        mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        msg = f"expected Trimesh-compatible STL, got {type(mesh).__name__}: {path}"
        raise TypeError(msg)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        msg = f"STL geometry export is empty: {path}"
        raise ValueError(msg)
    return mesh


def load_wall_vtp(path: Path) -> pv.PolyData:
    """Load the mapped wall VTP used for surface fields and force checks."""

    if path.suffix.lower() != ".vtp":
        msg = f"expected a VTP wall export, got {path}"
        raise ValueError(msg)
    loaded = pv.read(path)
    if isinstance(loaded, pv.MultiBlock):
        msg = f"expected PolyData, got MultiBlock: {path}"
        raise TypeError(msg)
    return loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface()


def load_volume_vtu(path: Path) -> pv.UnstructuredGrid:
    """Load the full-volume VTU export produced through foamToVTK/PyVista."""

    if path.suffix.lower() != ".vtu":
        msg = f"expected a VTU volume export, got {path}"
        raise ValueError(msg)
    loaded = pv.read(path)
    if not isinstance(loaded, pv.UnstructuredGrid):
        msg = f"expected UnstructuredGrid, got {type(loaded).__name__}: {path}"
        raise TypeError(msg)
    return loaded


def workflow_manifest(
    surface_path: Path,
    volume_path: Path,
    *,
    geometry_path: Path | None = None,
) -> VtkWorkflowManifest:
    """Describe the official wall-VTP plus volume-VTU conversion path."""

    return VtkWorkflowManifest(
        surface_adapter="aeromap.data.vtk_workflow.load_wall_vtp",
        volume_adapter="aeromap.data.vtk_workflow.load_volume_vtu",
        geometry_path=str(geometry_path) if geometry_path is not None else None,
        surface_path=str(surface_path),
        volume_path=str(volume_path),
        semantics={
            "geometry": (
                "Source STL is the deterministic AeroCliff OpenFOAM case geometry. It is "
                "recorded as geometry provenance only; VTP/VTU exports provide the converted "
                "surface and volume field arrays."
            ),
            "surface": (
                "Wall VTP cells carry mapped surface fields and local face areas; wall-force "
                "integration uses this surface artifact and is independent of volume-cell "
                "decomposition."
            ),
            "volume": (
                "Volume VTU cells are preserved as exported by foamToVTK. Source-cell-aware "
                "reductions must aggregate through the cellID array rather than treating raw "
                "VTU cell count as the OpenFOAM source mesh count."
            ),
        },
    )

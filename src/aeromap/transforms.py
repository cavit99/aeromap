"""Nondimensionalisation, signs, and body/inlet transforms."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from aeromap.constants import REF, ReferenceConditions

FloatArray = NDArray[np.float64]
POINT_ARRAY_NDIM = 2
XYZ_COMPONENTS = 3
PITCH_COS_MIN = 1e-12


def nondim_coords(coords_m: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(coords_m, dtype=np.float64) / ref.l_ref_m


def dim_coords(coords_nd: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(coords_nd, dtype=np.float64) * ref.l_ref_m


def nondim_velocity(velocity_m_s: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(velocity_m_s, dtype=np.float64) / ref.u_inf_m_s


def dim_velocity(velocity_nd: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(velocity_nd, dtype=np.float64) * ref.u_inf_m_s


def pressure_coefficient(
    pressure_pa: FloatArray,
    *,
    p_inf_pa: float = REF.p_inf_pa,
    ref: ReferenceConditions = REF,
) -> FloatArray:
    return (np.asarray(pressure_pa, dtype=np.float64) - p_inf_pa) / ref.q_inf_pa


def pressure_from_cp(
    cp: FloatArray,
    *,
    p_inf_pa: float = REF.p_inf_pa,
    ref: ReferenceConditions = REF,
) -> FloatArray:
    return np.asarray(cp, dtype=np.float64) * ref.q_inf_pa + p_inf_pa


def nondim_wall_shear(shear_pa: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(shear_pa, dtype=np.float64) / ref.q_inf_pa


def dim_wall_shear(shear_nd: FloatArray, ref: ReferenceConditions = REF) -> FloatArray:
    return np.asarray(shear_nd, dtype=np.float64) * ref.q_inf_pa


def downforce_coefficient(force_z_n: float, ref: ReferenceConditions = REF) -> float:
    return -force_z_n / (ref.q_inf_pa * ref.a_ref_m2)


def drag_coefficient(
    force_xyz_n: FloatArray, yaw_deg: float, ref: ReferenceConditions = REF
) -> float:
    force = np.asarray(force_xyz_n, dtype=np.float64)
    return float(-np.dot(force, inlet_unit_vector(yaw_deg)) / (ref.q_inf_pa * ref.a_ref_m2))


def moment_coefficient(moment_n_m: float, ref: ReferenceConditions = REF) -> float:
    return moment_n_m / (ref.q_inf_pa * ref.a_ref_m2 * ref.l_ref_m)


def inlet_unit_vector(yaw_deg: float) -> FloatArray:
    """Return inlet direction; positive yaw points freestream toward vehicle left (+y)."""

    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


def apply_ride_height_pitch(
    vertices_m: FloatArray,
    *,
    ride_height_mm: float,
    pitch_deg: float,
) -> FloatArray:
    """Rotate about body y, then translate so the minimum z equals ride height."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    pitch = math.radians(pitch_deg)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)

    rotation = np.array(
        [
            [cos_p, 0.0, -sin_p],
            [0.0, 1.0, 0.0],
            [sin_p, 0.0, cos_p],
        ],
        dtype=np.float64,
    )
    transformed = vertices @ rotation.T
    transformed[:, 2] += ride_height_mm / 1000.0 - float(np.min(transformed[:, 2]))
    return transformed


def _pitch_rotation_matrix(pitch_deg: float) -> FloatArray:
    pitch = math.radians(pitch_deg)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    return np.array(
        [
            [cos_p, 0.0, -sin_p],
            [0.0, 1.0, 0.0],
            [sin_p, 0.0, cos_p],
        ],
        dtype=np.float64,
    )


def infer_ride_height_pitch_z_shift(
    posed_vertices_m: FloatArray,
    *,
    pitch_deg: float,
) -> float:
    """Infer the z-translation used by :func:`apply_ride_height_pitch`.

    AeroCliff body-datum geometry has ``min(z) == 0`` before pose.  Given a
    posed reference surface, this solves the translation that makes the
    inverse-rotated reference surface return to that body datum.
    """

    vertices = np.asarray(posed_vertices_m, dtype=np.float64)
    if (
        vertices.ndim != POINT_ARRAY_NDIM
        or vertices.shape[1] != XYZ_COMPONENTS
        or len(vertices) == 0
    ):
        msg = "posed_vertices_m must be a non-empty (n, 3) array"
        raise ValueError(msg)
    pitch = math.radians(pitch_deg)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    if abs(cos_p) < PITCH_COS_MIN:
        msg = "pitch is too close to 90 degrees for ride-height inversion"
        raise ValueError(msg)
    return float(np.min(-sin_p * vertices[:, 0] + cos_p * vertices[:, 2]) / cos_p)


def inverse_ride_height_pitch(
    points_m: FloatArray,
    *,
    pitch_deg: float,
    z_shift_m: float,
) -> FloatArray:
    """Transform posed coordinates back to AeroCliff body-local coordinates."""

    points = np.asarray(points_m, dtype=np.float64)
    rotation = _pitch_rotation_matrix(pitch_deg)
    shifted = points.copy()
    shifted[:, 2] -= z_shift_m
    return shifted @ rotation


def inverse_pitch_normals(normals: FloatArray, *, pitch_deg: float) -> FloatArray:
    """Rotate posed normals back to body-local orientation."""

    normal_array = np.asarray(normals, dtype=np.float64)
    rotation = _pitch_rotation_matrix(pitch_deg)
    body_normals = normal_array @ rotation
    lengths = np.linalg.norm(body_normals, axis=1)
    valid = lengths > 0.0
    body_normals[valid] /= lengths[valid, None]
    return body_normals

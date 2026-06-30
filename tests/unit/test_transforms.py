from __future__ import annotations

import numpy as np
import pytest

from aeromap.constants import REF
from aeromap.transforms import (
    apply_ride_height_pitch,
    dim_coords,
    downforce_coefficient,
    drag_coefficient,
    infer_ride_height_pitch_z_shift,
    inlet_unit_vector,
    inverse_pitch_normals,
    inverse_ride_height_pitch,
    nondim_coords,
    pressure_coefficient,
    pressure_from_cp,
)


def test_reference_conditions_match_authoritative_brief() -> None:
    assert REF.u_inf_m_s == pytest.approx(40.0)
    assert REF.q_inf_pa == pytest.approx(980.0)


def test_coordinate_round_trip() -> None:
    coords = np.array([[0.0, 0.5, 0.1], [2.0, -0.5, 0.2]], dtype=np.float64)
    assert np.allclose(dim_coords(nondim_coords(coords)), coords)
    assert np.allclose(
        nondim_coords(np.array([[REF.l_ref_m, REF.l_ref_m / 2.0, -REF.l_ref_m]])),
        np.array([[1.0, 0.5, -1.0]]),
    )


def test_pressure_coefficient_round_trip() -> None:
    pressure = np.array([REF.p_inf_pa, REF.p_inf_pa + REF.q_inf_pa], dtype=np.float64)
    assert np.allclose(pressure_from_cp(pressure_coefficient(pressure)), pressure)
    assert np.allclose(pressure_coefficient(pressure), np.array([0.0, 1.0]))


def test_downforce_positive_downward() -> None:
    assert downforce_coefficient(-REF.q_inf_pa * REF.a_ref_m2) == pytest.approx(1.0)


def test_drag_positive_opposes_freestream() -> None:
    force = np.array([-REF.q_inf_pa * REF.a_ref_m2, 0.0, 0.0], dtype=np.float64)
    assert drag_coefficient(force, yaw_deg=0.0) == pytest.approx(1.0)


def test_positive_yaw_points_inlet_toward_vehicle_left() -> None:
    inlet = inlet_unit_vector(3.0)
    assert inlet[0] > 0.0
    assert inlet[1] > 0.0
    assert np.linalg.norm(inlet) == pytest.approx(1.0)
    assert inlet[0] == pytest.approx(np.cos(np.deg2rad(3.0)))
    assert inlet[1] == pytest.approx(np.sin(np.deg2rad(3.0)))


def test_positive_pitch_raises_rear() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    transformed = apply_ride_height_pitch(vertices, ride_height_mm=40.0, pitch_deg=1.0)
    assert transformed[1, 2] > transformed[0, 2]
    assert np.min(transformed[:, 2]) == pytest.approx(0.04)
    assert transformed[1, 2] - transformed[0, 2] == pytest.approx(2.0 * np.sin(np.deg2rad(1.0)))


def test_inverse_ride_height_pitch_recovers_body_local_coordinates() -> None:
    body = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.3, 0.08]],
        dtype=np.float64,
    )
    posed = apply_ride_height_pitch(body, ride_height_mm=60.0, pitch_deg=0.4)
    z_shift = infer_ride_height_pitch_z_shift(posed, pitch_deg=0.4)

    recovered = inverse_ride_height_pitch(posed, pitch_deg=0.4, z_shift_m=z_shift)

    assert recovered == pytest.approx(body)


def test_inverse_pitch_normals_recovers_body_local_orientation() -> None:
    body_normals = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    pitch = np.deg2rad(0.4)
    rotation = np.array(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    posed_normals = body_normals @ rotation.T

    recovered = inverse_pitch_normals(posed_normals, pitch_deg=0.4)

    assert recovered == pytest.approx(body_normals)

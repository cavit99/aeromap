from __future__ import annotations

import numpy as np
import pytest

from aeromap.geometry.diagnostics import cad_topology_summary, triangle_metrics


class _FakeBox:
    xmin = 0.0
    ymin = 0.0
    zmin = 0.0
    xmax = 1.0
    ymax = 1.0
    zmax = 1.0


class _FakeVector:
    def __init__(self, values: tuple[float, float, float]) -> None:
        self._values = values

    def toTuple(self) -> tuple[float, float, float]:  # noqa: N802
        return self._values


class _FakeEdge:
    def __init__(self, length: float, center: tuple[float, float, float]) -> None:
        self._length = length
        self._center = center

    def Length(self) -> float:  # noqa: N802
        return self._length

    def BoundingBox(self) -> _FakeBox:  # noqa: N802
        return _FakeBox()

    def Center(self) -> _FakeVector:  # noqa: N802
        return _FakeVector(self._center)

    def geomType(self) -> str:  # noqa: N802
        return "LINE"


class _FakeFace:
    def __init__(
        self,
        area: float,
        center: tuple[float, float, float],
        edges: list[_FakeEdge],
    ) -> None:
        self._area = area
        self._center = center
        self._edges = edges

    def Area(self) -> float:  # noqa: N802
        return self._area

    def Edges(self) -> list[_FakeEdge]:  # noqa: N802
        return self._edges

    def BoundingBox(self) -> _FakeBox:  # noqa: N802
        return _FakeBox()

    def Center(self) -> _FakeVector:  # noqa: N802
        return _FakeVector(self._center)

    def geomType(self) -> str:  # noqa: N802
        return "PLANE"

    def tessellate(
        self,
        _tolerance: float,
        _angularTolerance: float = 0.1,  # noqa: N803
    ) -> tuple[list[_FakeVector], list[tuple[int, int, int]]]:
        return ([], [])


class _FakeShape:
    def Faces(self) -> list[_FakeFace]:  # noqa: N802
        return [
            _FakeFace(
                2.0e-4,
                (0.2, 0.0, 0.1),
                [
                    _FakeEdge(2.0e-3, (0.1, 0.0, 0.0)),
                    _FakeEdge(4.0e-4, (0.2, 0.0, 0.0)),
                ],
            ),
            _FakeFace(5.0e-5, (0.8, 0.0, 0.1), [_FakeEdge(3.0e-3, (0.9, 0.0, 0.0))]),
        ]

    def isValid(self) -> bool:  # noqa: N802
        return True


def test_triangle_metrics_reports_unit_quality_for_equilateral_triangle() -> None:
    height = np.sqrt(3.0) / 2.0
    triangles = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]]])

    metrics = triangle_metrics(triangles)

    assert metrics["triangle_quality"][0] == pytest.approx(1.0)
    assert metrics["area_m2"][0] == pytest.approx(height / 2.0)
    assert metrics["aspect_ratio"][0] == pytest.approx(1.0)


def test_triangle_metrics_exposes_degenerate_sliver_triangle() -> None:
    triangles = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]])

    metrics = triangle_metrics(triangles)

    assert metrics["triangle_quality"][0] == pytest.approx(0.0)
    assert metrics["min_edge_m"][0] < 1e-6
    assert metrics["aspect_ratio"][0] > 1e6


def test_cad_topology_summary_reports_minimum_locations_and_threshold_counts() -> None:
    summary = cad_topology_summary(_FakeShape())

    assert summary["valid"] is True
    assert summary["min_edge"]["length_m"] == pytest.approx(4.0e-4)
    assert summary["min_edge"]["center_m"] == (0.2, 0.0, 0.0)
    assert summary["min_face"]["area_m2"] == pytest.approx(5.0e-5)
    assert summary["min_face"]["center_m"] == (0.8, 0.0, 0.1)
    assert summary["edge_length_threshold_counts"][-1]["count_below"] == 1
    assert summary["face_area_threshold_counts"][-1]["count_below"] == 1

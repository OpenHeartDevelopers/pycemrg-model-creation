"""Unit tests for `utilities.geodesic`.

Path-length and direction tests use synthetic polylines. The geodesic search uses
a small pyvista plane, so it exercises the real `geodesic` call without test data.
"""

import numpy as np
import pyvista as pv
import pytest

from pycemrg_model_creation.utilities.geodesic import (
    compute_path_length,
    find_closest_point,
    find_midway_point,
    find_point_along_direction,
)


def _polyline(points: np.ndarray) -> pv.PolyData:
    """Builds a pyvista polyline through the given points."""
    return pv.lines_from_points(np.asarray(points, dtype=float))


class TestComputePathLength:
    def test_straight_line(self):
        path = _polyline([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        assert compute_path_length(path) == pytest.approx(5.0)

    def test_sums_multiple_segments(self):
        path = _polyline(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        assert compute_path_length(path) == pytest.approx(3.0)

    def test_single_point_has_zero_length(self):
        path = pv.PolyData(np.array([[1.0, 2.0, 3.0]]))
        assert compute_path_length(path) == 0.0


class TestFindMidwayPoint:
    def test_midpoint_of_a_uniform_line(self):
        path = _polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_allclose(find_midway_point(path), [1.0, 0.0, 0.0])

    def test_returns_an_existing_vertex_not_an_interpolation(self):
        points = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = find_midway_point(_polyline(points))

        assert np.any(np.all(np.isclose(points, result), axis=1))

    def test_accounts_for_uneven_spacing(self):
        # Half of the 10-unit total is 5.0; the vertex at x=9 is nearer to that
        # than the vertex at x=1, so it is not simply the middle index.
        path = _polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        np.testing.assert_allclose(find_midway_point(path), [1.0, 0.0, 0.0])

    def test_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="empty path"):
            find_midway_point(pv.PolyData())


class TestFindClosestPoint:
    def test_picks_the_nearer_candidate(self):
        mesh = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
        source = 0
        candidates = [1, mesh.n_points - 1]

        closest, path = find_closest_point(mesh, source, candidates)

        assert closest in candidates
        assert compute_path_length(path) > 0

    def test_returns_the_path_to_the_winner(self):
        mesh = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
        closest, path = find_closest_point(mesh, 0, [1, 2, 3])

        expected = compute_path_length(mesh.geodesic(0, closest))
        assert compute_path_length(path) == pytest.approx(expected)

    def test_single_candidate_is_the_winner(self):
        mesh = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
        closest, _ = find_closest_point(mesh, 0, [5])

        assert closest == 5

    def test_raises_on_empty_candidates(self):
        mesh = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
        with pytest.raises(ValueError, match="No candidate vertices"):
            find_closest_point(mesh, 0, [])


class TestFindPointAlongDirection:
    @pytest.fixture
    def points(self):
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
            ]
        )

    def test_furthest_takes_the_largest_projection(self, points):
        result = find_point_along_direction(
            points, [0, 1, 2, 3], np.array([1.0, 0.0, 0.0]), np.zeros(3)
        )
        assert result == 2

    def test_closest_takes_the_smallest_non_negative(self, points):
        result = find_point_along_direction(
            points,
            [1, 2, 3],
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            mode="closest",
        )
        assert result == 1

    def test_closest_ignores_candidates_behind_the_reference(self, points):
        # Index 3 sits at x=-3, a negative projection, so it must not win.
        result = find_point_along_direction(
            points, [2, 3], np.array([1.0, 0.0, 0.0]), np.zeros(3), mode="closest"
        )
        assert result == 2

    def test_reference_point_shifts_the_result(self, points):
        result = find_point_along_direction(
            points,
            [0, 1, 2],
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            mode="closest",
        )
        assert result == 2

    def test_raises_when_nothing_lies_forward(self, points):
        with pytest.raises(ValueError, match="positive direction"):
            find_point_along_direction(
                points,
                [3],
                np.array([1.0, 0.0, 0.0]),
                np.zeros(3),
                mode="closest",
            )

    def test_rejects_unknown_mode(self, points):
        with pytest.raises(ValueError, match="furthest"):
            find_point_along_direction(
                points, [0], np.array([1.0, 0.0, 0.0]), np.zeros(3), mode="sideways"
            )

    def test_raises_on_empty_candidates(self, points):
        with pytest.raises(ValueError, match="No candidate vertices"):
            find_point_along_direction(
                points, [], np.array([1.0, 0.0, 0.0]), np.zeros(3)
            )

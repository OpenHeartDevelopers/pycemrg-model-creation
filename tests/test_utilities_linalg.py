"""Unit tests for `utilities.linalg` — pure transforms, no test data required."""

import numpy as np
import pytest

from pycemrg_model_creation.utilities.linalg import (
    compute_normal_from_surface,
    create_csys,
    normalise_vectors,
    rotation_matrix,
    vector_cprod,
    vector_sprod,
)


class TestRotationMatrix:
    def test_identity_at_zero_angle(self):
        R = rotation_matrix(np.array([0.0, 0.0, 1.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_is_orthogonal_with_unit_determinant(self):
        axis = normalise_vectors(np.array([[1.0, 2.0, 3.0]]))[0]
        R = rotation_matrix(axis, 0.7)

        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_quarter_turn_about_z(self):
        R = rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        np.testing.assert_allclose(
            R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12
        )

    def test_axis_is_invariant(self):
        axis = normalise_vectors(np.array([[1.0, -2.0, 0.5]]))[0]
        R = rotation_matrix(axis, 1.3)
        np.testing.assert_allclose(R @ axis, axis, atol=1e-12)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            rotation_matrix(np.array([1.0, 0.0]), 0.5)


class TestNormaliseVectors:
    def test_rows_become_unit_length(self):
        result = normalise_vectors(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])

    def test_preserves_direction(self):
        result = normalise_vectors(np.array([[3.0, 4.0, 0.0]]))
        np.testing.assert_allclose(result[0], [0.6, 0.8, 0.0])

    def test_zero_row_survives_without_nan(self):
        result = normalise_vectors(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert not np.isnan(result).any()
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0])

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            normalise_vectors(np.array([1.0, 0.0, 0.0]))


class TestVectorProducts:
    def test_cprod_matches_numpy(self):
        a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(vector_cprod(a, b), [0.0, 0.0, 1.0])

    def test_sprod_matches_numpy(self):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        assert vector_sprod(a, b) == pytest.approx(32.0)

    def test_sprod_returns_a_scalar(self):
        assert isinstance(vector_sprod(np.ones(3), np.ones(3)), float)


class TestCreateCsys:
    @pytest.mark.parametrize(
        "vec",
        [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.577350, 0.577350, 0.577350]),
        ],
    )
    def test_triple_is_orthonormal(self, vec):
        vec0, vec1, vec2 = create_csys(vec)

        for v in (vec0, vec1, vec2):
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

        assert vector_sprod(vec0, vec1) == pytest.approx(0.0, abs=1e-9)
        assert vector_sprod(vec0, vec2) == pytest.approx(0.0, abs=1e-9)
        assert vector_sprod(vec1, vec2) == pytest.approx(0.0, abs=1e-9)

    def test_third_axis_is_the_input(self):
        vec = np.array([0.0, 0.0, 1.0])
        _, _, third = create_csys(vec)
        np.testing.assert_allclose(third, vec)

    def test_branch_avoids_division_by_zero_on_z_axis(self):
        # vec = [0,0,1] takes the first branch, where tmp = sqrt(y^2 + z^2) = 1.
        vec0, vec1, _ = create_csys(np.array([0.0, 0.0, 1.0]))
        assert not np.isnan(vec0).any()
        assert not np.isnan(vec1).any()

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            create_csys(np.array([1.0, 0.0]))


class TestComputeNormalFromSurface:
    def test_unit_length_normals(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
        )
        triangles = np.array([[0, 1, 2]])

        normals = compute_normal_from_surface(points, triangles)
        assert np.linalg.norm(normals[0]) == pytest.approx(1.0)

    def test_sign_convention_is_inverted_cross_product(self):
        # Counter-clockwise in the xy-plane gives a raw cross product of +z;
        # the upstream convention this port preserves returns -z.
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
        )
        triangles = np.array([[0, 1, 2]])

        normals = compute_normal_from_surface(points, triangles)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, -1.0], atol=1e-12)

    def test_handles_multiple_triangles(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [1, 3, 2]])

        normals = compute_normal_from_surface(points, triangles)
        assert normals.shape == (2, 3)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), [1.0, 1.0])

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            compute_normal_from_surface(np.zeros((3, 3)), np.array([[0, 1]]))

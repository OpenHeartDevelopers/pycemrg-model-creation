# src/pycemrg_model_creation/utilities/linalg.py

"""
Vector and rotation primitives.

Pure array-in/array-out transforms with no file access. These sit underneath the
fibre and coordinate stages: `rotation_matrix` is used both by atrial sheet
orthogonalisation and by atrial apex selection.
"""

import logging
import numpy as np

from numpy.typing import NDArray
from typing import Tuple

logger = logging.getLogger(__name__)


def rotation_matrix(axis: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """
    Builds a rotation matrix of `theta` radians about `axis` (Rodrigues' formula).

    Args:
        axis: A 3-element unit vector. Not normalised internally — pass a unit
            vector or the result is not a rotation.
        theta: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError(f"axis must have shape (3,); got {axis.shape}.")

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1.0 - cos_t
    x, y, z = axis

    return np.array(
        [
            [
                x * x + cos_t * (1 - x * x),
                one_minus_cos * x * y - z * sin_t,
                one_minus_cos * x * z + y * sin_t,
            ],
            [
                one_minus_cos * x * y + z * sin_t,
                y * y + cos_t * (1 - y * y),
                one_minus_cos * y * z - x * sin_t,
            ],
            [
                one_minus_cos * x * z - y * sin_t,
                one_minus_cos * y * z + x * sin_t,
                z * z + cos_t * (1 - z * z),
            ],
        ],
        dtype=float,
    )


def normalise_vectors(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Scales each row to unit length.

    Rows of zero length are returned unchanged rather than producing NaN.

    Args:
        vectors: Array of shape (n, d).

    Returns:
        Array of the same shape with unit-length rows.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D; got shape {vectors.shape}.")

    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, magnitudes, where=magnitudes != 0)


def vector_cprod(
    vec1: NDArray[np.float64], vec2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Cross product of two 3-vectors.

    Args:
        vec1: First 3-vector.
        vec2: Second 3-vector.

    Returns:
        The cross product as a 3-vector.
    """
    return np.cross(np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float))


def vector_sprod(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
    """
    Scalar (dot) product of two 3-vectors.

    Args:
        vec1: First 3-vector.
        vec2: Second 3-vector.

    Returns:
        The scalar product.
    """
    return float(
        np.dot(np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float))
    )


def create_csys(
    vec: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Builds an orthonormal coordinate system whose third axis is `vec`.

    The branch chooses whichever pair of components gives the numerically larger
    denominator, avoiding a near-zero divide when `vec` is aligned with an axis.

    Args:
        vec: A 3-element unit vector.

    Returns:
        (vec0, vec1, vec) — a right-handed orthonormal triple.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"vec must have shape (3,); got {vec.shape}.")

    if (vec[0] < 0.5) and (vec[1] < 0.5):
        tmp = np.sqrt(vec[1] * vec[1] + vec[2] * vec[2])
        vec1 = np.array([0.0, -vec[2] / tmp, vec[1] / tmp])
    else:
        tmp = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        vec1 = np.array([vec[1] / tmp, -vec[0] / tmp, 0.0])

    vec0 = vector_cprod(vec, vec1)

    return vec0, vec1, vec


def compute_normal_from_surface(
    points: NDArray[np.float64], triangles: NDArray[np.int_]
) -> NDArray[np.float64]:
    """
    Computes unit normals for each triangle of a surface.

    The sign is inverted relative to the raw cross product, matching the upstream
    convention this port preserves. Callers that need the other orientation should
    negate the result rather than changing it here.

    Unlike the original, this writes no file — it returns the array only.

    Args:
        points: Array of shape (n_points, 3).
        triangles: Array of shape (n_triangles, 3) of vertex indices.

    Returns:
        Array of shape (n_triangles, 3) of unit normals.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"triangles must have shape (n, 3); got {triangles.shape}."
        )

    v10 = points[triangles[:, 1], :] - points[triangles[:, 0], :]
    v20 = points[triangles[:, 2], :] - points[triangles[:, 0], :]

    normals = -np.cross(normalise_vectors(v10), normalise_vectors(v20))

    return normalise_vectors(normals)

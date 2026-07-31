# src/pycemrg_model_creation/utilities/geodesic.py

"""
Geodesic path and directional-extremum primitives.

These are the substrate underneath atrial landmark detection: locating a point
part-way along a surface path, finding the geodesically nearest member of a
candidate set, and picking the extremum along a direction.

Paths are pyvista polylines as returned by `pyvista.PolyData.geodesic`.
"""

import logging
import numpy as np
import pyvista as pv

from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def compute_path_length(path: pv.PolyData) -> float:
    """
    Total length of a polyline path, summed over consecutive point pairs.

    Args:
        path: A pyvista polyline, e.g. the result of `mesh.geodesic(a, b)`.

    Returns:
        The path length in the mesh's coordinate units.
    """
    points = np.asarray(path.points, dtype=float)

    if points.shape[0] < 2:
        return 0.0

    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def find_midway_point(path: pv.PolyData) -> NDArray[np.float64]:
    """
    Finds the point on a path closest to half its total arc length.

    Returns an existing path vertex rather than interpolating between two.

    Args:
        path: A pyvista polyline.

    Returns:
        The 3D coordinates of the midway vertex.

    Raises:
        ValueError: If the path has no points.
    """
    points = np.asarray(path.points, dtype=float)

    if points.shape[0] == 0:
        raise ValueError("Cannot find the midway point of an empty path.")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    half_length = 0.5 * cumulative[-1]

    return points[int(np.argmin(np.abs(cumulative - half_length))), :]


def find_closest_point(
    pv_mesh: pv.PolyData, idx: int, vtx: Sequence[int]
) -> Tuple[int, pv.PolyData]:
    """
    Finds which of several candidate vertices is geodesically closest to `idx`.

    Distance is measured along the surface, not through space, so this is not
    equivalent to a nearest-neighbour search on coordinates.

    Args:
        pv_mesh: The surface to walk.
        idx: Index of the source vertex.
        vtx: Candidate vertex indices.

    Returns:
        (closest_vertex_index, geodesic_path_to_it).

    Raises:
        ValueError: If `vtx` is empty.
    """
    if len(vtx) == 0:
        raise ValueError("No candidate vertices supplied.")

    closest: Optional[int] = None
    best_path: Optional[pv.PolyData] = None
    best_distance = np.inf

    for candidate in vtx:
        path = pv_mesh.geodesic(idx, candidate)
        distance = compute_path_length(path)

        if distance < best_distance:
            best_distance = distance
            best_path = path
            closest = candidate

    return int(closest), best_path


def find_point_along_direction(
    points: NDArray[np.float64],
    vtx: Sequence[int],
    direction: NDArray[np.float64],
    cog: NDArray[np.float64],
    mode: str = "furthest",
) -> int:
    """
    Picks the candidate vertex at an extremum along a direction.

    Each candidate is scored by the projection of its offset from `cog` onto
    `direction`. `furthest` takes the largest projection; `closest` takes the
    smallest non-negative one, so candidates behind the plane through `cog` are
    ignored.

    Args:
        points: Array of shape (n_points, 3).
        vtx: Candidate vertex indices into `points`.
        direction: A 3-element direction vector.
        cog: The 3D reference point projections are measured from.
        mode: Either `"furthest"` or `"closest"`.

    Returns:
        The winning vertex index.

    Raises:
        ValueError: If `mode` is unrecognised, `vtx` is empty, or `closest` finds
            no candidate with a non-negative projection.
    """
    if mode not in ("furthest", "closest"):
        raise ValueError(
            f"mode must be 'furthest' or 'closest'; got {mode!r}."
        )

    candidates = np.asarray(vtx, dtype=int)
    if candidates.size == 0:
        raise ValueError("No candidate vertices supplied.")

    projections = (points[candidates, :] - np.asarray(cog, dtype=float)) @ np.asarray(
        direction, dtype=float
    )

    if mode == "furthest":
        return int(candidates[np.argmax(projections)])

    forward = projections >= 0
    if not forward.any():
        raise ValueError(
            "No candidate lies in the positive direction from the reference point."
        )

    masked = np.where(forward, projections, np.inf)
    return int(candidates[np.argmin(masked)])

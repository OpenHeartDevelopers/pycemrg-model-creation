# src/pycemrg_model_creation/utilities/geometry.py

import logging
import warnings

import numpy as np

from pathlib import Path
from typing import List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


def compute_surface_center_of_gravity(pts: np.ndarray) -> np.ndarray:
    """
    Compute center of gravity for a set of surface points.

    Args:
        pts: Nx3 array of point coordinates

    Returns:
        3D coordinate of center of gravity

    """
    return np.mean(pts, axis=0)


def compute_mesh_region_cog(
    mesh_pts: np.ndarray, mesh_elem: np.ndarray, tag_value: int
) -> np.ndarray:
    """
    Compute center of gravity for all elements with a specific tag.

    Args:
        mesh_pts: Nx3 array of mesh points
        mesh_elem: Mx5 array of elements (4 connectivity + 1 tag)
        tag_value: Tag value to filter elements

    Returns:
        3D coordinate of center of gravity for tagged region

    """
    mesh_region_pts_idx = []
    for i, e in enumerate(mesh_elem):
        if e[4] == int(tag_value):
            mesh_region_pts_idx.extend([int(e[0]), int(e[1]), int(e[2]), int(e[3])])

    mesh_region_pts_idx = np.unique(mesh_region_pts_idx)
    mesh_region_pts = mesh_pts[mesh_region_pts_idx]

    mesh_region_cog = np.mean(mesh_region_pts, axis=0)
    return mesh_region_cog


def outward_normal_fraction(
    pts: np.ndarray, surf: np.ndarray, reference_point: np.ndarray
) -> float:
    """
    Fraction of a surface's triangles whose normals point away from a reference.

    This measures; it does not classify. Callers apply whatever threshold and
    comparison their question needs — a closed surface enclosing the reference
    tends toward 0.0, one facing away from it toward 1.0 — and different
    anatomical questions legitimately want different cut-offs. Keep it that
    way: a threshold baked in here would be wrong for the next caller.

    A triangle counts as outward when the vector from its first vertex to the
    reference point opposes the triangle normal. Triangles whose dot product is
    exactly zero — degenerate or exactly edge-on — count as *not* outward.

    Args:
        pts: Nx3 array of surface points
        surf: Mx3 array of triangle connectivity
        reference_point: 3D reference point (e.g., chamber center)

    Returns:
        Fraction of triangles with outward-pointing normals (0.0 to 1.0)

    """

    is_outward = np.zeros((surf.shape[0],), dtype=int)

    for i, t in enumerate(surf):
        p0, p1, p2 = pts[t[0]], pts[t[1]], pts[t[2]]
        v0 = p1 - p0
        v0 /= np.linalg.norm(v0)

        v1 = p2 - p0
        v1 /= np.linalg.norm(v1)

        n = np.cross(v0, v1)
        n /= np.linalg.norm(n)

        dot_prod = np.dot(reference_point - p0, n)
        is_outward[i] = 1 if dot_prod < 0 else 0

    outward_fraction = np.sum(is_outward) / surf.shape[0]
    return outward_fraction


def identify_surface_orientation(
    pts: np.ndarray, surf: np.ndarray, reference_point: np.ndarray
) -> float:
    """
    Deprecated alias for :func:`outward_normal_fraction`.

    The old name reads as though it classifies a surface, when it returns a
    ratio and leaves the classifying to the caller. Behaviour is unchanged.
    """
    warnings.warn(
        "identify_surface_orientation is deprecated; use outward_normal_fraction. "
        "It returns a fraction rather than an orientation, and the caller applies "
        "its own threshold. Behaviour is identical.",
        DeprecationWarning,
        stacklevel=2,
    )
    return outward_normal_fraction(pts, surf, reference_point)


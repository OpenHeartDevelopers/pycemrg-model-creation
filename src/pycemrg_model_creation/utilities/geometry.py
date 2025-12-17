# src/pycemrg_model_creation/utilities/geometry.py

import logging
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
    for i,e in enumerate(mesh_elem):
        if e[4] == int(tag_value[0]):  
            mesh_region_pts_idx.extend([int(e[0]), int(e[1]), int(e[2]), int(e[3])])
    
    mesh_region_pts_idx = np.unique(mesh_region_pts_idx)
    mesh_region_pts = mesh_pts[mesh_region_pts_idx]

    mesh_region_cog = np.mean(mesh_region_pts,axis=0)
    return mesh_region_cog


def identify_surface_orientation(
    pts: np.ndarray, surf: np.ndarray, reference_point: np.ndarray
) -> float:
    """
    Determine if surface normals point outward from reference point.

    Args:
        pts: Nx3 array of surface points
        surf: Mx3 array of triangle connectivity
        reference_point: 3D reference point (e.g., chamber center)

    Returns:
        Fraction of triangles with outward-pointing normals (0.0 to 1.0)

    """
    
    is_outward = np.zeros((surf.shape[0],),dtype=int)
    
    for i,t in enumerate(surf):
        p0, p1, p2 = pts[t[0]], pts[t[1]], pts[t[2]]
        v0 = p1 - p0
        v0 /= np.linalg.norm(v0)
        
        v1 = p2 - p0
        v1 /= np.linalg.norm(v1)
        
        n = np.cross(v0,v1)
        n /= np.linalg.norm(n)
        
        dot_prod = np.dot(reference_point - p0,n) 
        is_outward[i] = 1 if dot_prod < 0 else 0
    
    outward_fraction = np.sum(is_outward)/surf.shape[0]
    return outward_fraction
    
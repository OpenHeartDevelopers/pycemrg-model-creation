# src/pycemrg_model_creation/utilities/mesh_utils.py

import numpy as np
import logging

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ElemType(Enum):
    """
    Defined CARP element types and their corresponding column indices for connectivity.
    Values are Tuples of integers representing column indices in the .elem file
    """

    Tt = (1, 2, 3, 4)  # Tetrahedra
    Tr = (1, 2, 3)  # Triangles
    Ln = (1, 2)  # Lines


def read_carp_mesh(
    mesh_base_path: Path, elem_type: ElemType = ElemType.Tt, read_tags: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read CARP mesh files.

    Args:
        mesh_path: Base path to mesh (without extensions)
        elem_type: Element type ("Tr" for triangles, "Tt" for tetrahedra)
        read_tags: Whether to read element tags

    Returns:
        (points_array, elements_array) where elements include connectivity
        and optionally tags

    """
    logger.info(f"Reading CARP mesh from base: {mesh_base_path}")

    pts_path = mesh_base_path.with_suffix(".pts")
    elem_path = mesh_base_path.with_suffix(".elem")

    if not pts_path.exists() or not elem_path.exists():
        raise FileNotFoundError(
            f"Could not find both .pts and .elem files for {mesh_base_path}"
        )
    points = read_pts(pts_path)
    elements = read_elem(elem_path, elem_type=elem_type, read_tags=read_tags)

    return points, elements


def read_pts(pts_path: Path) -> np.ndarray:
    """
    Read CARP points file
    """
    logging.info(f"Reading {pts_path.name} pts")
    return np.loadtxt(pts_path, dtype=float, skiprows=1)


def read_elem(
    elem_path: Path,
    elem_type: ElemType = ElemType.Tt,
    read_tags: bool = False,
    check_format: bool = True,
) -> np.ndarray:
    """
    Read CARP element file.

    Args:
        elem_path: Path to .elem file
        el_type: Element type ("Tr" for triangles, "Tt" for tetrahedra)
        tags: Whether to read tags column

    Returns:
        Array of element connectivity (and tags if requested)
    """
    if check_format:
        elem_path = elem_path.with_suffix(
            ".elem"
        )  # TODO: try to remove this to make library more robust

    logger.info(f"Reading elements from {elem_path.name} (type: {elem_type.name})")
    cols = elem_type.value
    if read_tags:
        tag_col_index = max(cols) + 1
        cols = cols + (tag_col_index,)

    try:
        return np.loadtxt(elem_path, dtype=int, skiprows=1, usecols=cols)
    except IndexError:
        logger.error(
            f"Failed to read tags gtom {elem_path.name}."
            "File may not contain a tags column. Try running with read_tags=False"
        )
        raise
    except Exception:
        logger.error(f"An unexpected error occurred while reading {elem_path.name}.")
        raise


def read_tets(tets_path: Path) -> np.ndarray:
    return read_elem(tets_path, elem_type=ElemType.Tt, read_tags=True)


def read_lon(lon_path: Path) -> np.ndarray:
    logger.info(f"Reading lon file {lon_path.name}")
    return np.loadtxt(lon_path, dtype=float, skiprows=1)


def find_connected_components(surface_path: Path, output_dir: Path) -> List[str]:
    """
    Find all connected components files matching a pattern.

    Args:
        surface_path: Base path for surface
        output_dir: Directory to search in

    Returns:
        List of connected component base names (without extensions)

    TODO: Implement in pycemrg_model_creation.utils.surface_utils
    """
    raise NotImplementedError("find_connected_components must be implemented")


def keep_largest_n_components(
    component_names: List[str], tmp_dir: Path, n: int
) -> List[str]:
    """
    Keep only the n largest connected components by element count.

    Args:
        component_names: List of component base names
        tmp_dir: Directory containing the components
        n: Number of components to keep

    Returns:
        List of n largest component base names, sorted by size (descending)

    TODO: Implement in pycemrg_model_creation.utils.surface_utils
    """
    raise NotImplementedError("keep_largest_n_components must be implemented")


def compute_surface_center_of_gravity(pts: np.ndarray) -> np.ndarray:
    """
    Compute center of gravity for a set of surface points.

    Args:
        pts: Nx3 array of point coordinates

    Returns:
        3D coordinate of center of gravity

    TODO: Implement in pycemrg_model_creation.utils.geometry_utils
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

    TODO: Implement in pycemrg_model_creation.utils.geometry_utils
    """
    raise NotImplementedError("compute_mesh_region_cog must be implemented")


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

    TODO: Implement in pycemrg_model_creation.utils.geometry_utils
    """
    raise NotImplementedError("identify_surface_orientation must be implemented")


def prepare_vtx_files_for_uvc(
    surface_dir: Path, output_vtx_paths: Dict[str, Path]
) -> None:
    """
    Prepare VTX (vertex) files required for UVC computation.

    This involves converting surface mesh boundaries to VTX format.

    Args:
        surface_dir: Directory containing surface meshes
        output_vtx_paths: Mapping of surface names to output VTX paths

    TODO: Implement in pycemrg_model_creation.utils.vtx_utils
    """
    raise NotImplementedError("prepare_vtx_files_for_uvc must be implemented")


def remove_septum_from_endo(
    lv_endo_path: Path, septum_path: Path, output_path: Path
) -> None:
    """
    Remove septum surface from LV endocardium surface.

    Args:
        lv_endo_path: Path to LV endocardium surface
        septum_path: Path to septum surface
        output_path: Path for output (LV endo without septum)

    TODO: Implement in pycemrg_model_creation.utils.surface_utils
    """
    raise NotImplementedError("remove_septum_from_endo must be implemented")

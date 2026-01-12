# src/pycemrg_model_creation/utilities/mesh.py

import logging
import numpy as np
import pyvista as pv

from numpy.typing import NDArray
from pathlib import Path
from typing import List, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


CARP_COMMON_EXTENSIONS = [".pts", ".elem", ".lon", ".nod", ".eidx", ".vtk"]


class ElemType(Enum):
    """
    Defined CARP element types and their corresponding column indices for connectivity.
    Values are Tuples of integers representing column indices in the .elem file
    """

    Tt = (1, 2, 3, 4)  # Tetrahedra
    Tr = (1, 2, 3)  # Triangles
    Ln = (1, 2)  # Lines


# READING FUNCTIONS
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
    # Validating we didn't receive a path with CARP extensions
    if mesh_base_path.suffix in CARP_COMMON_EXTENSIONS:
        raise ValueError(
            f"mesh_base_path should not include CARP file extensions. "
            f"Got {mesh_base_path}, expected base path without extensions."
        )

    logger.info(f"Reading CARP mesh from base: {mesh_base_path}")

    pts_path = mesh_base_path.parent / f"{mesh_base_path}.pts"
    elem_path = mesh_base_path.parent / f"{mesh_base_path}.elem"

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


def read_surf(surface_path: Path) -> np.ndarray:
    """
    Reads a CARP .surf file, returning only the triangle connectivity.

    Args:
        surface_path: Path to the .surf file.

    Returns:
        A NumPy array of shape (n_triangles, 3) with vertex indices.
    """
    logger.info(f"Reading surface connectivity from {surface_path.name}")
    # The .surf file format is "Tr vtx1 vtx2 vtx3". We skip the "Tr" column.
    return np.loadtxt(
        surface_path.with_suffix(".surf"), dtype=int, skiprows=1, usecols=[1, 2, 3]
    )


# WRITING FUNCTIONS


def write_surf(
    surface_cells: NDArray[np.int_], output_path: Path
) -> None:  # refactor note: use for write_surface
    """
    Writes a surface connectivity array to a CARP .surf file.

    Args:
        surface_cells: A NumPy array of shape (n_triangles, 3) with vertex indices.
        output_path: Path to the output .surf file.
    """
    assert surface_cells.ndim == 2 and surface_cells.shape[1] == 3, (
        "Input array must be of shape (n, 3)."
    )

    header = str(surface_cells.shape[0])
    # The format string 'Tr %d %d %d' prepends 'Tr' to each line of integer data.
    np.savetxt(
        output_path, surface_cells, fmt="Tr %d %d %d", header=header, comments=""
    )
    logger.info(f"Successfully wrote surface to {output_path}")


def write_vtx(vertex_indices: NDArray[np.int_], output_path: Path) -> None:
    """
    Writes a vertex array to a CARP .vtx file.

    Args:
        vertex_indices: A 1D NumPy array of vertex indices.
        output_path: Path to the output .vtx file.
    """
    assert vertex_indices.ndim == 1, "Input array must be 1-dimensional."

    # The header for a .vtx file includes the count and the 'intra' keyword.
    header = f"{vertex_indices.shape[0]}\nintra"
    np.savetxt(output_path, vertex_indices, fmt="%d", header=header, comments="")
    logger.info(f"Successfully wrote vertices to {output_path}")


def write_pts(
    points: NDArray[np.float64], output_path: Path
) -> None:  # refactor note: use for write_pnts
    """
    Writes a points coordinate array to a CARP .pts file.

    Args:
        points: A NumPy array of shape (n_points, 3) with vertex coordinates.
        output_path: Path to the output .pts file.
    """
    assert points.ndim == 2 and points.shape[1] == 3, (
        "Input array must be of shape (n, 3)."
    )

    header = str(points.shape[0])
    # Using '%.8f' for precision, consistent with `write_dat`.
    # Using a space delimiter as it's more standard than tab.
    np.savetxt(
        output_path, points, fmt="%.8f", delimiter=" ", header=header, comments=""
    )
    logger.info(f"Successfully wrote points to {output_path}")


def write_dat(data: NDArray, output_path: Path) -> None:
    """
    Writes a NumPy array to a .dat file with specified precision.

    Args:
        data: The NumPy array to save.
        output_path: The path for the output file.
    """
    np.savetxt(output_path, data, fmt="%.8f")
    logger.info(f"Successfully wrote data to {output_path}")


def connected_component_to_surface(
    eidx_path: Path, input_surface_path: Path, output_surface_path: Path
) -> None:
    eidx = np.fromfile(eidx_path.with_suffix(".eidx"), dtype=int, count=-1)
    nod = np.fromfile(eidx_path.with_suffix(".nod"), dtype=int, count=-1)
    surf = read_surf(input_surface_path)
    vtx = surf2vtx(surf)

    subsurf = surf[eidx, :]
    subvtx = vtx[nod]

    write_surf(subsurf, output_surface_path.with_suffix(".surf"))
    write_vtx(subvtx, output_surface_path.with_suffix(".vtx"))


# CONVERSION FUNCTIONS
def surf2vtx(surf: np.ndarray) -> np.ndarray:
    return np.unique(surf.flatten())


def surf2vtk(mesh_base_path: Path, surface_path: Path, output_vtk_path: Path) -> None:
    """
    Convert a CARP surface mesh to a VTK PolyData file.

    Args:
        mesh_base_path: Path to the base mesh (e.g., 'mesh/my_mesh'),
                        used to locate the corresponding .pts file.
        surface_path: Path to the source surface mesh (.surf file).
        output_vtk_path: Path for the output VTK file (.vtk).
    """
    logger.info(f"Converting surface {surface_path.name} to VTK format")

    # 1. Read input files using existing helpers
    points_all = read_pts(mesh_base_path.with_suffix(".pts"))
    surface_cells_original = read_surf(surface_path)

    # 2. Extract surface-specific vertices and re-index cells
    unique_vertex_indices = surf2vtx(surface_cells_original)
    surface_points = points_all[unique_vertex_indices]

    # 3. Vectorized Re-indexing: Fast and efficient
    # Create a lookup map from old vertex index to new surface-local index
    index_map = np.full(unique_vertex_indices.max() + 1, -1, dtype=int)
    index_map[unique_vertex_indices] = np.arange(len(unique_vertex_indices))

    # Apply the map to the entire cell array in one operation
    surface_cells_reindexed = index_map[surface_cells_original]

    # 4. Create and save PyVista PolyData object
    # PyVista requires a faces array in the format [3, v0, v1, v2, 3, v3, v4, v5, ...]
    num_cells = surface_cells_reindexed.shape[0]
    padding = np.full((num_cells, 1), 3)
    faces = np.hstack((padding, surface_cells_reindexed)).flatten()

    surface_mesh = pv.PolyData(surface_points, faces=faces)
    surface_mesh.save(output_vtk_path, binary=False)
    logger.info(f"Successfully saved VTK surface to {output_vtk_path}")

# def vtx2pts(vtx_path: Path, pts_path: Path, output_pts_path: Path) -> None:
#     """
#     Convert VTX file of indices into a .pts readable by paraview 
    
#     :param vtx_path: Description
#     :type vtx_path: Path
#     :param pts_path: Description
#     :type pts_path: Path
#     :param output_pts_path: Description
#     :type output_pts_path: Path
#     """

#     vtx = np.loadtxt(vtx_path, dtype=int, skiprows=2)
#     pts = read_pts(pts_path)

    

# SURFACE OPERATIONS
def find_numbered_parts(directory: Path, base_prefix: str) -> List[str]:
    """
    Find all numbered part files matching a base prefix pattern.

    Searches for files with pattern: {base_prefix}.part{N}.elem
    where N starts at 0 and increments sequentially.

    Args:
        directory: Directory to search for part files
        base_prefix: Base prefix to match (e.g., "epi_endo_CC")

    Returns:
        List of base names without extensions (e.g., ["epi_endo_CC.part0", "epi_endo_CC.part1"])

    Example:
        >>> find_numbered_parts(Path("/tmp"), "mesh_CC")
        ["mesh_CC.part0", "mesh_CC.part1", "mesh_CC.part2"]
    """
    parts = []
    part_idx = 0

    part_name = f"{base_prefix}.part{part_idx}"
    elem_file = directory / f"{part_name}.elem"

    while elem_file.exists():
        parts.append(part_name)
        part_idx += 1

        part_name = f"{base_prefix}.part{part_idx}"
        elem_file = directory / f"{part_name}.elem"

    return parts


def keep_largest_n_components(
    component_names: List[str],
    directory: Path,
    keep_n: int,
    delete_smaller: bool = True,
) -> List[str]:
    """
    Identifies the N largest mesh components and deletes the others.

    Component size is determined by the number of elements in the .elem file.
    This function deletes all associated files (e.g., .elem, .pts) for the
    smaller, discarded components.

    Args:
        component_names: List of component base names (e.g., "mesh.part0").
        directory: The directory containing the component files.
        keep_n: The number of largest components to keep.
        delete_smaller: If True, all files for smaller components are deleted.

    Returns:
        A list of the base names of the components that were kept, sorted
        by size from largest to smallest.

    Raises:
        ValueError: If `keep_n` is greater than the number of components found.
    """
    if len(component_names) < keep_n:
        raise ValueError(
            f"Cannot keep {keep_n} components: only {len(component_names)} found."
        )

    component_data = []
    for name in component_names:
        elem_path = directory / f"{name}.elem"
        if not elem_path.is_file():
            logger.warning(f"Component file not found, skipping: {elem_path}")
            continue

        # Correctly call read_elem using the ElemType enum
        elements = read_elem(elem_path, elem_type=ElemType.Tr, read_tags=False)
        component_data.append({"name": name, "size": elements.shape[0]})

    # Sort components by size in descending order
    component_data.sort(key=lambda x: x["size"], reverse=True)

    kept_components = [cd["name"] for cd in component_data[:keep_n]]
    components_to_delete = [cd["name"] for cd in component_data[keep_n:]]

    logger.info(f"Keeping the {keep_n} largest of {len(component_data)} components:")
    for i, cd in enumerate(component_data[:keep_n]):
        logger.info(f"  {i + 1}. {cd['name']} (size: {cd['size']} triangles)")

    if delete_smaller and components_to_delete:
        logger.info(f"Deleting {len(components_to_delete)} smaller components...")
        for comp_name in components_to_delete:
            for file_path in directory.glob(f"{comp_name}.*"):
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted {file_path.name}")
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e}")

    return kept_components


def remove_septum_from_endo(
    endo_surface_path: Path,
    septum_surface_path: Path,
    output_path: Path,
) -> None:
    """
    Removes a septal surface from an endocardial surface.

    This function identifies all triangles in the endocardial surface where all
    three vertices are also part of the septal surface, and removes them.
    The result is the endocardial free wall.

    Args:
        endo_surface_path: Path to the endocardial .surf file.
        septum_surface_path: Path to the septal .surf file.
        output_path: Path for the output .surf file (the free wall).
    """
    logger.info(
        f"Removing septum {septum_surface_path.name} from "
        f"endocardium {endo_surface_path.name}"
    )

    # 1. Read surface and septum data using existing helpers
    endo_cells: NDArray[np.int_] = read_surf(endo_surface_path)
    septum_cells: NDArray[np.int_] = read_surf(septum_surface_path)

    # Get the unique vertex indices that define the septum
    septum_vtx: NDArray[np.int_] = surf2vtx(septum_cells)

    # 2. Vectorized triangle removal (replaces the slow for-loop)
    # Create a boolean array of the same shape as endo_cells,
    # where True indicates a vertex is part of the septum.
    is_septum_vertex_mask = np.isin(endo_cells, septum_vtx)

    # Count how many vertices in each triangle are septal vertices.
    # A triangle is part of the septum if this count is 3.
    septum_vertex_count_per_triangle = np.sum(is_septum_vertex_mask, axis=1)

    # The free wall triangles are those with fewer than 3 septal vertices.
    freewall_mask = septum_vertex_count_per_triangle < 3
    freewall_triangles = endo_cells[freewall_mask]

    logger.info(
        f"Identified {len(freewall_triangles)} free wall triangles "
        f"(removed {len(endo_cells) - len(freewall_triangles)})."
    )

    # 3. Write the resulting free wall surface
    write_surf(freewall_triangles, output_path)


# VTX UTILS
def generate_vtx_from_surf(input_surf_path: Path, output_vtx_path: Path) -> None:
    """
    Generates a .vtx file from a .surf file.

    Reads a surface file, extracts the unique vertex indices, and writes them
    to a .vtx file. This is a direct one-to-one conversion.

    Args:
        input_surf_path: Path to the source .surf file.
        output_vtx_path: Path for the destination .vtx file.
    """
    logger.info(
        f"Generating VTX file for {input_surf_path.name} -> {output_vtx_path.name}"
    )
    try:
        surface_cells = read_surf(input_surf_path)
        vertex_indices = surf2vtx(surface_cells)
        write_vtx(vertex_indices, output_vtx_path)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_surf_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to generate VTX from {input_surf_path.name}: {e}")
        raise


def identify_epi_from_endo(
    component1_base_path: Path,
    component2_base_path: Path,
) -> Tuple[Path, Path]:
    """
    Identifies epicardial and endocardial surfaces from two mesh components.

    The method assumes the epicardium is the outer surface. It calculates the
    surface normals for one component and checks their orientation relative to the
    component's center of gravity (CoG). If the majority of normals point inward
    (towards the CoG), it is classified as the epicardium.

    Args:
        component1_base_path: The base path to the first connected component
                              (e.g., '.../tmp/epi_endo_CC.part0').
        component2_base_path: The base path to the second connected component.

    Returns:
        A tuple of (epicardium_base_path, endocardium_base_path).
    """
    logger.info(
        f"Identifying epi/endo between {component1_base_path.name} "
        f"and {component2_base_path.name}"
    )

    # Analyze the first component to determine its identity
    points = read_pts(component1_base_path.with_suffix(".pts"))
    cells = read_elem(component1_base_path.with_suffix(".elem"), elem_type=ElemType.Tr)

    center_of_gravity = np.mean(points, axis=0)

    # --- Vectorized Normal Calculation ---
    v0 = points[cells[:, 1]] - points[cells[:, 0]]
    v1 = points[cells[:, 2]] - points[cells[:, 0]]
    normals = np.cross(v0, v1)

    # Normalize the normal vectors to unit length
    norms_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_normalized = np.divide(normals, norms_magnitude, where=norms_magnitude != 0)

    # Vector from a vertex on each triangle to the center of gravity
    vertex_to_cog = center_of_gravity - points[cells[:, 0]]

    # Dot product for all triangles at once. A positive dot product means the
    # normal and the vector to the CoG are in the same general direction
    # (i.e., the normal points inward), which is characteristic of an outer surface.
    dot_products = np.sum(normals_normalized * vertex_to_cog, axis=1)

    inward_pointing_ratio = np.mean(dot_products > 0)

    # The original code used a 0.7 threshold, but >0.5 is sufficient
    if inward_pointing_ratio > 0.5:
        logger.info(f"'{component1_base_path.name}' identified as Epicardium.")
        return component1_base_path, component2_base_path
    else:
        logger.info(f"'{component1_base_path.name}' identified as Endocardium.")
        return component2_base_path, component1_base_path

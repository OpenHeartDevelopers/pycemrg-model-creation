# src/pycemrg_model_creation/utilities/mesh.py

import logging
import numpy as np
import pyvista as pv

from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, Iterable, Tuple
from enum import Enum

from ..meshpaths import CarpMesh, SubmeshParts
from .geometry import outward_normal_fraction
from .topology import SurfaceTopology, SplitComponent

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
        surface_path: Path to the .surf file, extension included. The path is
            taken literally -- no extension is added or replaced. `write_surf`
            reads its argument the same way, so the two agree and a caller can
            hand the same path to both.

    Returns:
        A NumPy array of shape (n_triangles, 3) with vertex indices.
    """
    logger.info(f"Reading surface connectivity from {surface_path.name}")
    # The .surf file format is "Tr vtx1 vtx2 vtx3". We skip the "Tr" column.
    return np.loadtxt(surface_path, dtype=int, skiprows=1, usecols=[1, 2, 3])


# WRITING FUNCTIONS


def write_surf(
    surface_cells: NDArray[np.int_], output_path: Path
) -> None:  # refactor note: use for write_surface
    """
    Writes a surface connectivity array to a CARP .surf file.

    Args:
        surface_cells: A NumPy array of shape (n_triangles, 3) with vertex indices.
        output_path: Path to the output .surf file, extension included. Taken
            literally, matching `read_surf`.
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
    # The two path arguments are not the same shape, which is easy to get
    # wrong: `input_surface_path` arrives with `.surf` already on it, while
    # `eidx_path` and `output_surface_path` are stems that this function
    # extends. So the read is literal and needs no conversion.
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
def find_numbered_parts(directory: Path, base_prefix: str) -> SubmeshParts:
    """
    Find all numbered part files matching a base prefix pattern.

    Searches for files with pattern: {base_prefix}.part{N}.elem
    where N starts at 0 and increments sequentially.

    The arity is a property of the data rather than of the invocation, which
    is why it has to be discovered rather than predicted.

    Args:
        directory: Directory to search for part files
        base_prefix: Base prefix to match (e.g., "epi_endo_CC")

    Returns:
        The components found, as resolved handles. Returning handles rather
        than bare names is what keeps a component's name and its directory
        together; callers used to re-join the two themselves at every site.

    Example:
        >>> find_numbered_parts(Path("/tmp"), "mesh_CC")[1].elem
        PosixPath('/tmp/mesh_CC.part1.elem')
    """
    parts = []
    part_idx = 0

    part_name = f"{base_prefix}.part{part_idx}"
    elem_file = directory / f"{part_name}.elem"

    while elem_file.exists():
        parts.append(CarpMesh(directory / part_name))
        part_idx += 1

        part_name = f"{base_prefix}.part{part_idx}"
        elem_file = directory / f"{part_name}.elem"

    return SubmeshParts(tuple(parts))


def keep_largest_n_components(
    components: Iterable[CarpMesh],
    keep_n: int,
    delete_smaller: bool = True,
) -> SubmeshParts:
    """
    Identifies the N largest mesh components and deletes the others.

    Component size is determined by the number of elements in the .elem file.
    This function deletes all associated files (e.g., .elem, .pts) for the
    smaller, discarded components.

    Takes no `directory` argument: each component carries its own, so the
    two cannot be handed in disagreeing with each other.

    Strict in what it returns, liberal in what it accepts. A `SubmeshParts`
    from `find_numbered_parts` is the usual input, but any iterable of
    handles will do -- a caller holding a plain list need not wrap it first.

    This lives here rather than as a `SubmeshParts` method on purpose. It
    reads every `.elem` to measure size and then deletes files; the handles
    in `meshpaths` never touch the file system, and `meshpaths` cannot
    import `read_elem` from this module without a cycle. The function sits
    where the I/O sits.

    Args:
        components: The components to choose among, as handles.
        keep_n: The number of largest components to keep.
        delete_smaller: If True, all files for smaller components are deleted.

    Returns:
        The components that were kept, sorted by size from largest to
        smallest.

    Raises:
        ValueError: If `keep_n` is greater than the number of components found.
    """
    # Materialise first: the argument may be any iterable, and this is
    # counted before it is walked.
    candidates = tuple(components)

    if len(candidates) < keep_n:
        raise ValueError(
            f"Cannot keep {keep_n} components: only {len(candidates)} found."
        )

    component_data = []
    for component in candidates:
        if not component.elem.is_file():
            logger.warning(f"Component file not found, skipping: {component.elem}")
            continue

        # Correctly call read_elem using the ElemType enum
        elements = read_elem(component.elem, elem_type=ElemType.Tr, read_tags=False)
        component_data.append({"mesh": component, "size": elements.shape[0]})

    # Sort components by size in descending order
    component_data.sort(key=lambda x: x["size"], reverse=True)

    kept_components = [cd["mesh"] for cd in component_data[:keep_n]]
    components_to_delete = [cd["mesh"] for cd in component_data[keep_n:]]

    logger.info(f"Keeping the {keep_n} largest of {len(component_data)} components:")
    for i, cd in enumerate(component_data[:keep_n]):
        logger.info(f"  {i + 1}. {cd['mesh'].name} (size: {cd['size']} triangles)")

    if delete_smaller and components_to_delete:
        logger.info(f"Deleting {len(components_to_delete)} smaller components...")
        for component in components_to_delete:
            # Glob on the handle's own directory and basename: the family is
            # every sibling of the stem, whatever extensions meshtool wrote.
            for file_path in component.directory.glob(f"{component.name}.*"):
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted {file_path.name}")
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e}")

    return SubmeshParts(tuple(kept_components))


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

    The method assumes the epicardium is the outer surface. It measures the
    first component's normals against that component's *own* centre of gravity:
    if the majority point inward, toward the CoG, it is the epicardium.

    The measuring is delegated to
    :func:`..geometry.outward_normal_fraction`, which is the single
    implementation of this calculation. That function returns the *outward*
    fraction, so the test here is inverted — "mostly inward" is
    ``outward_fraction < 0.5``. Note the reference point: passing the
    component's own centroid asks about winding order, which is a different
    question from passing an anatomical landmark, and is why this is not
    interchangeable with the ventricular surface classification in
    ``logic/surfaces.py``.

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

    # A handle, not with_suffix: these stems are '.partN' from
    # `meshtool extract unreachable`, and with_suffix would eat the '.part0'.
    component1 = CarpMesh(component1_base_path)
    points = read_pts(component1.pts)
    cells = read_elem(component1.elem, elem_type=ElemType.Tr)

    center_of_gravity = np.mean(points, axis=0)
    outward_fraction = outward_normal_fraction(points, cells, center_of_gravity)

    # The original code used a 0.7 threshold on the inward ratio; >0.5 is
    # sufficient, and inverts to <0.5 on the outward fraction returned above.
    if outward_fraction < 0.5:
        logger.info(f"'{component1_base_path.name}' identified as Epicardium.")
        return component1_base_path, component2_base_path
    else:
        logger.info(f"'{component1_base_path.name}' identified as Endocardium.")
        return component2_base_path, component1_base_path


def relabel_carp_elem_file(
    input_elem_path: Path, output_elem_path: Path, tag_mapping: Dict[int, int]
) -> None:
    """
    Reads a CARP .elem file, replaces element tags, and writes a new file.
    This version correctly preserves the element type string in the first column.
    """
    if not input_elem_path.is_file():
        raise FileNotFoundError(f"Input element file not found: {input_elem_path}")

    logger.info(f"Relabeling element tags from {input_elem_path.name}")

    with open(input_elem_path, "r") as f_in:
        header = f_in.readline().strip()
        lines = f_in.readlines()

    if not lines:
        raise ValueError(f"No data found in {input_elem_path} after header.")

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        elem_type = parts[0]
        # All other parts are expected to be integers
        connectivity_and_tag = [int(p) for p in parts[1:]]

        # The tag is the last element
        old_tag = connectivity_and_tag[-1]

        # Replace the tag if it's in the mapping
        connectivity_and_tag[-1] = tag_mapping.get(old_tag, old_tag)

        # Reconstruct the line
        new_line_parts = [elem_type] + [str(p) for p in connectivity_and_tag]
        new_lines.append(" ".join(new_line_parts) + "\n")

    output_elem_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_elem_path, "w") as f_out:
        f_out.write(header + "\n")
        f_out.writelines(new_lines)

    logger.info(f"Successfully wrote relabeled element file to {output_elem_path.name}")


def read_vtx(vtx_path: Path, header_rows: int = 2) -> NDArray[np.int_]:
    """
    Reads a CARP .vtx file, returning the vertex indices.

    The default of two header rows matches the file `write_vtx` produces: a count
    line followed by a group keyword line (`intra`).

    Args:
        vtx_path: Path to the .vtx file.
        header_rows: Number of leading lines to skip before the indices.

    Returns:
        A 1D array of vertex indices.
    """
    logger.info(f"Reading vtx file {vtx_path.name}")
    return np.loadtxt(vtx_path, dtype=int, skiprows=header_rows, ndmin=1)


def read_dat(dat_path: Path) -> NDArray[np.float64]:
    """
    Reads a CARP .dat file of per-node or per-element scalar values.

    A .dat file carries no header, matching what `write_dat` produces.

    Args:
        dat_path: Path to the .dat file.

    Returns:
        An array of the values in file order.
    """
    logger.info(f"Reading dat file {dat_path.name}")
    return np.loadtxt(dat_path, dtype=float, ndmin=1)


def write_elem(
    elements: NDArray[np.int_],
    tags: NDArray[np.int_],
    output_path: Path,
    elem_type: ElemType = ElemType.Tt,
) -> None:
    """
    Writes element connectivity and tags to a CARP .elem file.

    Args:
        elements: Array of shape (n_elements, n_nodes) of vertex indices.
        tags: 1D array of n_elements integer region tags.
        output_path: Path to the output .elem file.
        elem_type: Element type determining the expected node count and row prefix.

    Raises:
        ValueError: If the connectivity width does not match `elem_type`, or if the
            tag count does not match the element count.
    """
    expected_nodes = len(elem_type.value)
    if elements.ndim != 2 or elements.shape[1] != expected_nodes:
        raise ValueError(
            f"{elem_type.name} elements must have shape (n, {expected_nodes}); "
            f"got {elements.shape}."
        )

    tags = np.asarray(tags).ravel()
    if tags.shape[0] != elements.shape[0]:
        raise ValueError(
            f"Tag count ({tags.shape[0]}) does not match element count "
            f"({elements.shape[0]})."
        )

    rows = np.column_stack([elements, tags])
    fmt = f"{elem_type.name}" + " %d" * (expected_nodes + 1)
    np.savetxt(
        output_path, rows, fmt=fmt, header=str(elements.shape[0]), comments=""
    )
    logger.info(f"Successfully wrote {elements.shape[0]} elements to {output_path}")


def write_lon(fibres: NDArray[np.float64], output_path: Path) -> None:
    """
    Writes a fibre array to a CARP .lon file.

    The .lon header is the number of direction vectors per element, not the number
    of elements: 1 for fibre only, 2 for fibre plus sheet.

    Args:
        fibres: Array of shape (n_elements, 3) or (n_elements, 6).
        output_path: Path to the output .lon file.

    Raises:
        ValueError: If the column count is not a multiple of three.
    """
    if fibres.ndim != 2 or fibres.shape[1] % 3 != 0:
        raise ValueError(
            f"Fibre array must have shape (n, 3k); got {fibres.shape}."
        )

    n_directions = fibres.shape[1] // 3
    np.savetxt(
        output_path, fibres, fmt="%.8f", header=str(n_directions), comments=""
    )
    logger.info(
        f"Successfully wrote {n_directions}-direction fibres to {output_path}"
    )


def read_nod_eidx(index_path: Path, dtype: type = np.int64) -> NDArray[np.int_]:
    """
    Reads a binary CARP .nod or .eidx submesh index file.

    These files map submesh node or element indices back onto the parent mesh.

    The default dtype reproduces the original implementation's `dtype=int`, which
    resolves to a 64-bit integer on macOS and Linux. This has not been verified
    against a file emitted by meshtool; if indices come back implausible, a 32-bit
    dtype is the first thing to try.

    Args:
        index_path: Path to the .nod or .eidx file.
        dtype: Integer type the file was written with.

    Returns:
        A 1D array of parent-mesh indices.
    """
    logger.info(f"Reading binary index file {index_path.name}")
    return np.fromfile(index_path, dtype=dtype, count=-1)


def reindex_vtx(
    vtx: NDArray[np.int_], vtx_surf: NDArray[np.int_]
) -> NDArray[np.int_]:
    """
    Re-expresses vertex indices in the local numbering of a surface's vertex list.

    Indices in `vtx` that do not appear in `vtx_surf` are dropped.

    Args:
        vtx: Vertex indices in the parent numbering.
        vtx_surf: The surface's vertex list, defining the local numbering.

    Returns:
        The surviving indices, expressed as positions within `vtx_surf`.
    """
    common = np.intersect1d(vtx, vtx_surf)
    order = np.argsort(vtx_surf)
    positions = np.searchsorted(vtx_surf, common, sorter=order)
    return order[positions]


def reindex_surf(
    vtx_surf: NDArray[np.int_], surf: NDArray[np.int_]
) -> NDArray[np.int_]:
    """
    Re-expresses triangle connectivity in the local numbering of a vertex list.

    Args:
        vtx_surf: The surface's vertex list, defining the local numbering.
        surf: Triangle connectivity of shape (n_triangles, 3) in parent numbering.

    Returns:
        Triangle connectivity of the same shape, in local numbering.

    Raises:
        ValueError: If any vertex referenced by `surf` is absent from `vtx_surf`.
    """
    order = np.argsort(vtx_surf)
    positions = np.searchsorted(vtx_surf, surf, sorter=order)

    if positions.max(initial=0) >= vtx_surf.shape[0]:
        raise ValueError("Surface references a vertex absent from vtx_surf.")

    reindexed = order[positions]
    if not np.array_equal(vtx_surf[reindexed], surf):
        raise ValueError("Surface references a vertex absent from vtx_surf.")

    return reindexed


def extract_points_by_vtx(
    points: NDArray[np.float64], vtx: NDArray[np.int_]
) -> NDArray[np.float64]:
    """
    Selects the subset of points named by a vertex index list.

    Args:
        points: Array of shape (n_points, 3).
        vtx: 1D array of vertex indices into `points`.

    Returns:
        Array of shape (len(vtx), 3), in the order given by `vtx`.
    """
    return points[vtx, :]


def read_neubc(neubc_path: Path) -> NDArray[np.int_]:
    """
    Reads a CARP .neubc Neumann boundary condition file.

    Args:
        neubc_path: Path to the .neubc file.

    Returns:
        An integer array of the first six columns.
    """
    logger.info(f"Reading neubc file {neubc_path.name}")
    return np.loadtxt(
        neubc_path, dtype=int, skiprows=1, usecols=(0, 1, 2, 3, 4, 5), ndmin=2
    )


def read_clipper(clipper_path: Path) -> Tuple[NDArray[np.float64], float]:
    """
    Reads a clipper geometry file and derives its centre and radius.

    The radius is taken as the distance from the centroid to the first point, which
    assumes the points lie on a circle.

    Args:
        clipper_path: Path to a VTK-readable clipper file.

    Returns:
        (centre, radius).
    """
    logger.info(f"Reading clipper file {clipper_path.name}")
    points = np.asarray(pv.read(clipper_path).points, dtype=float)

    if points.shape[0] == 0:
        raise ValueError(f"Clipper file {clipper_path} contains no points.")

    center = np.mean(points, axis=0)
    radius = float(np.linalg.norm(points[0, :] - center))

    return center, radius



def write_topology_vtk(
    points: NDArray[np.float64],
    surf: NDArray[np.int_],
    topology: SurfaceTopology,
    output_path: Path,
) -> None:
    """
    Write a surface as VTK, carrying its topology report as cell data.

    This is the file a human opens when the component count comes out wrong.
    Both arrays are **cell** data, not point data: a pinch is a property of a
    vertex, but a vertex cannot be shaded on its own, so what gets coloured is
    every triangle that touches one.

    Colour by `component` to judge whether the proposed edge-adjacency split is
    the right one, and by `pinch` to find the defects that caused the failure.

    Args:
        points: Coordinates of the parent mesh, as `read_pts` returns them.
            Only the vertices `surf` refers to are written.
        surf: Triangle connectivity, (n, 3), indexing into `points`.
        topology: The report for that same `surf`.
        output_path: Path to the output .vtk file, extension included.

    Raises:
        ValueError: If `topology` was built from a different surface.
    """
    if topology.n_triangles != surf.shape[0]:
        raise ValueError(
            f"Topology describes {topology.n_triangles} triangles but the "
            f"surface has {surf.shape[0]}."
        )

    # Same reindex-then-PolyData shape as `surf2vtk`, which this deliberately
    # does not call: that one reads its inputs from disk, and everything here
    # is already in memory by the time a failure is being diagnosed.
    unique_vertex_indices = surf2vtx(surf)
    surface_points = points[unique_vertex_indices]

    index_map = np.full(unique_vertex_indices.max() + 1, -1, dtype=int)
    index_map[unique_vertex_indices] = np.arange(len(unique_vertex_indices))
    cells = index_map[surf]

    padding = np.full((cells.shape[0], 1), 3)
    mesh = pv.PolyData(surface_points, faces=np.hstack((padding, cells)).flatten())

    mesh.cell_data["component"] = topology.triangle_component
    mesh.cell_data["pinch"] = topology.pinched_triangles(surf).astype(np.int8)

    mesh.save(output_path, binary=False)
    logger.info(
        f"Wrote topology diagnostic to {output_path} "
        f"({topology.n_components} components, "
        f"{len(topology.pinch_vertices)} pinch vertices)"
    )


def write_split_component(
    component: SplitComponent,
    points: NDArray[np.float64],
    surf: NDArray[np.int_],
    tags: NDArray[np.int_] = None,
    write_vtk: bool = True,
) -> None:
    """
    Write one component as the CARP submesh family `meshtool` would have left.

    The four files are what the pipeline reads back, and each is already implied
    by the component's triangle list:

    * ``.eidx`` -- rows of the parent ``.surf`` this component holds. This is
      what `connected_component_to_surface` indexes with.
    * ``.nod``  -- positions into ``surf2vtx(surf)``, not parent vertex indices.
      The same function maps them back, so the indirection has to match.
    * ``.elem`` -- the triangles, reindexed to the component's own vertices.
    * ``.pts``  -- those vertices' coordinates.

    ``.eidx`` and ``.nod`` are binary, written at the width `read_nod_eidx`
    reads them.

    Args:
        component: The component to write, from `split_surface_components`.
            It carries its own handle, so there is no destination to pass: the
            name and the directory cannot disagree.
        points: Parent mesh coordinates.
        surf: The parent surface the component indexes into.
        tags: One region tag per row of `surf`, as `epi_endo.neubc` column 3
            carries them. They go into the `.elem` and onto the `.vtk` as
            `elemTag`, which is what `meshtool` writes and what makes the
            septum visible inside the RV endocardium. Omitted, every triangle
            is tagged 1 and that distinction is lost.
        write_vtk: Also write a `.vtk`, for looking at.

    Raises:
        ValueError: If `tags` is given but does not describe `surf`.
    """
    if tags is not None and len(tags) != surf.shape[0]:
        raise ValueError(
            f"Got {len(tags)} tags for a surface of {surf.shape[0]} triangles; "
            "they do not describe each other."
        )
    mesh = component.mesh

    subsurf = surf[component.triangles]

    # Local reindexing, the same lookup-table trick as `surf2vtk`.
    index_map = np.full(component.vertices.max() + 1, -1, dtype=int)
    index_map[component.vertices] = np.arange(len(component.vertices))
    local = index_map[subsurf]

    element_tags = (
        np.ones(local.shape[0], dtype=int)
        if tags is None
        else np.asarray(tags)[component.triangles]
    )

    write_pts(points[component.vertices], mesh.pts)
    write_elem(local, element_tags, mesh.elem, elem_type=ElemType.Tr)

    # `.nod` is relative to the whole surface's vertex list, not to the parent
    # mesh: `connected_component_to_surface` does `surf2vtx(surf)[nod]`.
    all_vertices = surf2vtx(surf)
    nod = np.searchsorted(all_vertices, component.vertices)
    nod.astype(np.int64).tofile(mesh.index.nod)
    component.triangles.astype(np.int64).tofile(mesh.index.eidx)

    if write_vtk:
        padding = np.full((local.shape[0], 1), 3)
        polydata = pv.PolyData(
            points[component.vertices],
            faces=np.hstack((padding, local)).flatten(),
        )
        # Named *and typed* to match what meshtool writes (`SCALARS elemTag
        # int`), so the same ParaView colouring works on either without a cast.
        polydata.cell_data["elemTag"] = element_tags.astype(np.int32)
        polydata.save(mesh.vtk, binary=False)

    logger.info(
        f"Wrote {component.name}: {component.n_triangles} triangles, "
        f"{len(component.vertices)} vertices"
    )
# FIBRE OPERATIONS
def correct_degenerate_fibres(
    fibres: NDArray[np.float64],
    element_centres: NDArray[np.float64],
    tolerance: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Replace degenerate fibre vectors with their nearest well-defined neighbour.

    `GlRuleFibers` leaves some elements without a usable fibre direction. The
    repair, ported from `cemrg-heartbuilder`'s `correct_fibres`, is to give
    each such element the whole fibre row of the nearest element that does
    have one, measured straight-line between element centres.

    Two properties of the reference implementation are deliberate and are
    kept, because changing them would change the fibres a mesh gets:

    - **The degeneracy test is on the z-component alone**, ``|f_z| < tolerance``.
      It is not a vector-norm test, not a NaN test, and not an element-quality
      test. An element whose fibre lies flat in the xy-plane is treated as
      degenerate whether or not its vector is a valid unit vector.
    - **The neighbour search is not restricted by tag.** A degenerate LV
      element can inherit its fibre from the nearest RV element. Upstream
      reads no tags at all; its `read_elem` call is commented out.

    One upstream defect is **not** ported. It resolves ties with
    ``good[np.where(d == d.min())[0]]``, which yields more than one index when
    two elements are equidistant, and the row assignment then raises. This
    takes the first such neighbour instead.

    Args:
        fibres: Fibre array of shape (n_elements, 3) or (n_elements, 6) --
            fibre direction alone, or fibre plus sheet.
        element_centres: Element centres, shape (n_elements, 3), as
            `GlElemCenters` writes them and in the same element order.
        tolerance: Magnitude below which a z-component counts as degenerate.

    Returns:
        A corrected copy. The input is not modified.

    Raises:
        ValueError: If the two arrays disagree on the element count, if the
            shapes are not two-dimensional as described, or if every element
            is degenerate, leaving no neighbour to copy from.
    """
    if fibres.ndim != 2 or fibres.shape[1] % 3 != 0:
        raise ValueError(
            f"Fibre array must have shape (n, 3k); got {fibres.shape}."
        )
    if element_centres.ndim != 2 or element_centres.shape[1] != 3:
        raise ValueError(
            f"Element centres must have shape (n, 3); got {element_centres.shape}."
        )
    if fibres.shape[0] != element_centres.shape[0]:
        raise ValueError(
            f"Fibres and element centres disagree on the element count: "
            f"{fibres.shape[0]} fibre rows against "
            f"{element_centres.shape[0]} centres. They must be in the same "
            f"element order, so a mismatch means one of them is stale."
        )

    degenerate = np.where(np.abs(fibres[:, 2]) < tolerance)[0]
    logger.info(f"Found {degenerate.shape[0]} elements to correct")

    if degenerate.shape[0] == 0:
        return fibres.copy()

    well_defined = np.setdiff1d(
        np.arange(element_centres.shape[0]), degenerate, assume_unique=True
    )
    if well_defined.shape[0] == 0:
        raise ValueError(
            "Every element has a degenerate fibre, so there is no "
            "well-defined neighbour to copy from. This usually means the "
            "fibre file is not the one GlRuleFibers wrote."
        )

    corrected = fibres.copy()
    for index in degenerate:
        distances = np.linalg.norm(
            element_centres[well_defined, :] - element_centres[index, :], axis=1
        )
        # argmin, not `where(d == d.min())`: the latter returns every tied
        # index and the row assignment below then raises. See the docstring.
        nearest = well_defined[np.argmin(distances)]
        corrected[index, :] = fibres[nearest, :]

    remaining = np.where(np.abs(corrected[:, 2]) < tolerance)[0]
    logger.info(f"Left to correct {remaining.shape[0]}")

    return corrected

# src/pycemrg_model_creation/utilities/topology.py

"""
Edge-adjacency topology of a triangulated surface.

`meshtool extract unreachable` separates a surface by *node* reachability: two
triangles belong to the same component if a chain of shared nodes joins them.
One shared vertex is therefore enough to fuse two otherwise separate sheets,
which is how an epi/endo surface that looks separable in ParaView comes back as
a single component.

This module answers the same question under **edge** adjacency -- two triangles
are connected only if they share an edge -- and names the defects that make the
two answers differ:

* a **pinch vertex** carries more than one triangle fan. Two sheets meet at a
  single point. This is what `meshtool` follows and edge adjacency does not.
* a **non-manifold edge** is shared by more than two triangles. Two sheets
  cross along a line.

The two are different defects and are detected separately. PyVista's
`extract_feature_edges(non_manifold_edges=True)` finds only the second, and
reports a surface full of pinch vertices as clean.

**Both matter, and the second is not optional.** On AJR5d03297 the epi/endo
surface carries 17 pinch vertices *and* one edge shared by four triangles.
Pinch vertices alone take `meshtool`'s single component to two; only treating
that one edge as a crossing rather than a connection yields the three the
pipeline expects. A surface can fail for either reason, so neither pass can be
dropped.

Stateless: every function takes arrays and returns arrays. Nothing here touches
the filesystem.
"""

import logging

import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class _UnionFind:
    """
    Disjoint-set forest over `size` items, with path compression.

    Deliberately small and local. The alternative is `scipy.sparse.csgraph`,
    which would be a new dependency for twenty lines, and the same structure
    answers both questions this module asks.
    """

    def __init__(self, size: int) -> None:
        self._parent = np.arange(size, dtype=np.int64)

    def find(self, item: int) -> int:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        # Path compression: point everything on the way up at the root.
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return int(root)

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def roots(self) -> NDArray[np.int_]:
        """The root of every item, as an array."""
        return np.array([self.find(i) for i in range(len(self._parent))])


@dataclass(frozen=True)
class SurfaceTopology:
    """
    What an edge-adjacency reading of a surface says about it.

    Frozen, like the path contracts: the arrays are computed together from one
    edge table and must not drift apart afterwards.

    Attributes:
        triangle_component: Per triangle, its edge-adjacency component id.
            Labelled largest-component-first, so ids are stable across runs.
        n_components: Number of edge-adjacency components.
        pinch_vertices: Vertices carrying more than one triangle fan.
        pinch_fan_counts: Fans on each entry of `pinch_vertices`, parallel to it.
        nonmanifold_edges: Shape (n, 2), vertex pairs shared by >2 triangles.
        n_triangles: Triangles the report was built from.
    """

    triangle_component: NDArray[np.int_]
    n_components: int
    pinch_vertices: NDArray[np.int_]
    pinch_fan_counts: NDArray[np.int_]
    nonmanifold_edges: NDArray[np.int_]
    n_triangles: int

    @property
    def component_sizes(self) -> NDArray[np.int_]:
        """Triangles in each component, largest first."""
        return np.bincount(self.triangle_component, minlength=self.n_components)

    @property
    def is_clean(self) -> bool:
        """True when no defect was found that could fuse two sheets."""
        return len(self.pinch_vertices) == 0 and len(self.nonmanifold_edges) == 0

    def pinched_triangles(self, surf: NDArray[np.int_]) -> NDArray[np.bool_]:
        """
        Per-triangle mask: True where a triangle touches a pinch vertex.

        A vertex cannot be shaded on its own, so this is what a viewer colours
        to show where the defects are.

        Args:
            surf: The same (n, 3) connectivity the report was built from.
        """
        if len(self.pinch_vertices) == 0:
            return np.zeros(surf.shape[0], dtype=bool)
        return np.isin(surf, self.pinch_vertices).any(axis=1)

    def summary(self) -> str:
        """One-paragraph account, for a log line or an exception message."""
        sizes = " / ".join(str(size) for size in self.component_sizes)
        return (
            f"Edge-adjacency finds {self.n_components} components ({sizes} triangles). "
            f"{len(self.pinch_vertices)} pinch vertices, "
            f"{len(self.nonmanifold_edges)} non-manifold edges."
        )


def _edge_table(
    surf: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Every triangle edge, keyed so that the two orientations collide.

    Triangle `t` with vertices `(v0, v1, v2)` contributes edges `(v0, v1)`,
    `(v1, v2)` and `(v2, v0)`. Each row records which triangle it came from and
    which two *local slots* of that triangle it joins, because the pinch pass
    needs to union corners, not triangles.

    Returns:
        `(key, triangle, slot_a, slot_b)`, each of length `3 * n_triangles`.
        `key` is the unordered vertex pair encoded as a single integer.
    """
    n_triangles = surf.shape[0]

    # Local slot j joins slot (j + 1) % 3, for j in 0, 1, 2.
    slot_a = np.tile(np.array([0, 1, 2]), n_triangles)
    slot_b = np.tile(np.array([1, 2, 0]), n_triangles)
    triangle = np.repeat(np.arange(n_triangles), 3)

    ends_a = surf[triangle, slot_a]
    ends_b = surf[triangle, slot_b]

    # Encode the unordered pair as one integer so grouping is a 1-D unique.
    # `stride` exceeds any vertex index, so (low, high) maps injectively.
    stride = np.int64(surf.max()) + 1
    low = np.minimum(ends_a, ends_b).astype(np.int64)
    high = np.maximum(ends_a, ends_b).astype(np.int64)
    key = low * stride + high

    return key, triangle, slot_a, slot_b


def analyse_surface_topology(surf: NDArray[np.int_]) -> SurfaceTopology:
    """
    Read a surface's edge adjacency and name the defects that break it.

    Args:
        surf: Triangle connectivity, shape (n_triangles, 3), as `read_surf`
            returns it. Vertex indices may be in any indexing -- parent mesh or
            submesh-local -- because only their equality is used.

    Returns:
        A `SurfaceTopology` report.

    Raises:
        ValueError: If `surf` is not an (n, 3) array of at least one triangle.
    """
    surf = np.asarray(surf)
    if surf.ndim != 2 or surf.shape[1] != 3:
        raise ValueError(f"Expected a surface of shape (n, 3), got {surf.shape}.")
    if surf.shape[0] == 0:
        raise ValueError("Cannot analyse a surface with no triangles.")

    n_triangles = surf.shape[0]
    key, triangle, slot_a, slot_b = _edge_table(surf)

    # Group the rows by edge. `inverse` maps each row onto its edge's id.
    unique_keys, inverse, counts = np.unique(
        key, return_inverse=True, return_counts=True
    )
    logger.debug(
        f"{n_triangles} triangles, {len(unique_keys)} distinct edges; "
        f"edge multiplicities {np.bincount(counts)[1:]}"
    )

    # Rows sorted so that every edge's rows sit contiguously.
    order = np.argsort(inverse, kind="stable")
    group_starts = np.zeros(len(unique_keys), dtype=np.int64)
    group_starts[1:] = np.cumsum(counts)[:-1]

    triangles = _UnionFind(n_triangles)
    # One corner per (triangle, local slot); corner id is t * 3 + slot, which
    # matches the row order of `surf.reshape(-1)`.
    corners = _UnionFind(n_triangles * 3)

    nonmanifold: list[tuple[int, int]] = []
    stride = np.int64(surf.max()) + 1

    for edge_id, (start, count) in enumerate(zip(group_starts, counts)):
        rows = order[start : start + count]

        if count > 2:
            # A non-manifold edge is *not* evidence of connection. Four
            # triangles on one edge are two sheets crossing, not one bundle of
            # four, and unioning them fuses surfaces that are separate
            # everywhere else. Leaving them unjoined is conservative: anything
            # genuinely one sheet stays connected through its other edges, so
            # this only separates what is joined *solely* through this edge.
            #
            # Measured on AJR5d03297, where one such edge is the whole
            # difference between finding 2 components and finding 3.
            edge_key = int(unique_keys[edge_id])
            nonmanifold.append((edge_key // int(stride), edge_key % int(stride)))
            continue
        if count != 2:
            # A boundary edge, carried by one triangle. Nothing to join.
            continue

        triangles.union(int(triangle[rows[0]]), int(triangle[rows[1]]))

        # A manifold edge joins the fans of both its endpoints. Only these
        # unions run, which is what makes a pinch vertex stay split: nothing
        # joins the two fans that merely touch at it.
        first, second = rows[0], rows[1]
        t_first, t_second = int(triangle[first]), int(triangle[second])
        for slot_first in (slot_a[first], slot_b[first]):
            vertex = surf[t_first, slot_first]
            slot_second = (
                slot_a[second] if surf[t_second, slot_a[second]] == vertex
                else slot_b[second]
            )
            corners.union(t_first * 3 + int(slot_first), t_second * 3 + int(slot_second))

    # -- Components, relabelled largest-first ---------------------------------
    raw_component = triangles.roots()
    labels, sizes = np.unique(raw_component, return_counts=True)
    # Descending size, ties broken by root id so the order is deterministic.
    ranking = np.lexsort((labels, -sizes))
    relabel = np.zeros(raw_component.max() + 1, dtype=np.int64)
    relabel[labels[ranking]] = np.arange(len(labels))
    triangle_component = relabel[raw_component]

    # -- Pinch vertices -------------------------------------------------------
    corner_vertex = surf.reshape(-1)
    corner_root = corners.roots()
    # One row per distinct (vertex, fan). A vertex appearing more than once here
    # carries more than one fan.
    pairs = np.unique(np.column_stack((corner_vertex, corner_root)), axis=0)
    vertices, fan_counts = np.unique(pairs[:, 0], return_counts=True)
    pinched = fan_counts > 1

    report = SurfaceTopology(
        triangle_component=triangle_component,
        n_components=len(labels),
        pinch_vertices=vertices[pinched],
        pinch_fan_counts=fan_counts[pinched],
        nonmanifold_edges=np.array(nonmanifold, dtype=np.int64).reshape(-1, 2),
        n_triangles=n_triangles,
    )
    logger.info(report.summary())
    return report


@dataclass(frozen=True)
class SplitComponent:
    """
    One edge-adjacency component, written out as `meshtool` would have written it.

    Attributes:
        name: Base name on disk, e.g. `epi_endo_cc.part0`.
        triangles: Rows of the parent surface this component holds, which is
            also the content of its `.eidx`.
        vertices: The parent-surface vertices it uses, ascending.
        shared_with: Component indices this one shares vertices with, mapped to
            the shared vertices. Empty on a clean surface.
    """

    name: str
    triangles: NDArray[np.int_]
    vertices: NDArray[np.int_]
    shared_with: dict

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.size)


def split_surface_components(
    surf: NDArray[np.int_], topology: SurfaceTopology, base_prefix: str
) -> list:
    """
    Describe each component as the files `find_numbered_parts` expects.

    Nothing is written here and nothing is dropped. The components are complete
    and mutually consistent: a vertex where two sheets pinch belongs to both,
    which is what the geometry says.

    That overlap only matters if two components that *both* become `mguvc`
    boundary inputs share a vertex, since the same node would then carry two
    Dirichlet values. `mguvc` reads `base`, `lvendo`, `rvendo` and `rvsept` and
    **derives** `epi`, so an epi/endo overlap is harmless while an endo/endo one
    is not. Which component is which is decided later, by
    `extract_ventricular_surfaces`, so this reports the overlap per pair and
    leaves the judgement to the caller.

    Args:
        surf: The parent surface, (n, 3).
        topology: Its report, from `analyse_surface_topology`.
        base_prefix: Stem the parts hang off, e.g. `epi_endo_cc`. The names
            produced are `{base_prefix}.part{N}`, largest component first.

    Returns:
        A list of `SplitComponent`, in component-label order.
    """
    vertex_sets = [
        np.unique(surf[topology.triangle_component == k])
        for k in range(topology.n_components)
    ]

    components = []
    for k, vertices in enumerate(vertex_sets):
        shared = {}
        for other in range(topology.n_components):
            if other == k:
                continue
            overlap = np.intersect1d(vertices, vertex_sets[other])
            if overlap.size:
                shared[other] = overlap

        components.append(
            SplitComponent(
                name=f"{base_prefix}.part{k}",
                triangles=np.where(topology.triangle_component == k)[0],
                vertices=vertices,
                shared_with=shared,
            )
        )

    logger.info(
        f"Split into {len(components)} components: "
        + ", ".join(f"{c.name} ({c.n_triangles})" for c in components)
    )
    return components

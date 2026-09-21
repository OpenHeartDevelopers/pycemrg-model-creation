"""
Edge-adjacency topology of a triangulated surface.

Every mesh here is small enough to check by eye, which is the point: the
production case has 120666 triangles and cannot be an oracle for anything.

The naming follows the defects rather than the functions. A *pinch vertex*
carries more than one triangle fan; a *non-manifold edge* is carried by more
than two triangles. They are different defects, they fail differently, and the
real mesh that prompted this module has both.
"""

import numpy as np
import pytest

from pycemrg_model_creation.utilities.topology import (
    SurfaceTopology,
    analyse_surface_topology,
)


# A closed tetrahedron surface: four triangles, every edge shared by exactly two.
TETRAHEDRON = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def shifted(surf: np.ndarray, by: int) -> np.ndarray:
    """The same surface on a disjoint set of vertex indices."""
    return surf + by


class TestComponents:
    """Edge adjacency: two triangles connect only if they share an edge."""

    def test_two_triangles_sharing_an_edge_are_one_component(self):
        report = analyse_surface_topology(np.array([[0, 1, 2], [0, 1, 3]]))
        assert report.n_components == 1

    def test_two_triangles_sharing_only_a_vertex_are_two_components(self):
        """
        The whole reason this module exists. `meshtool extract unreachable`
        follows nodes and calls this one component.
        """
        report = analyse_surface_topology(np.array([[0, 1, 2], [0, 3, 4]]))
        assert report.n_components == 2

    def test_fully_disjoint_triangles_are_two_components(self):
        report = analyse_surface_topology(np.array([[0, 1, 2], [3, 4, 5]]))
        assert report.n_components == 2

    def test_a_closed_surface_is_one_component(self):
        assert analyse_surface_topology(TETRAHEDRON).n_components == 1

    def test_components_are_labelled_largest_first(self):
        """
        The *small* component comes first in the triangle array, so its
        union-find root is the lower id. Ordering the labels by root -- the
        obvious thing to do, and what happens if the relabel is dropped --
        would put the one-triangle component at label 0.

        An earlier version of this test listed the large component first, where
        root order and size order agree. It passed with the relabel deleted and
        so proved nothing.
        """
        surf = np.vstack([shifted(np.array([[0, 1, 2]]), 10), TETRAHEDRON])
        report = analyse_surface_topology(surf)

        assert report.component_sizes.tolist() == [4, 1]
        assert report.triangle_component[0] == 1  # the lone triangle
        assert report.triangle_component[1:].tolist() == [0, 0, 0, 0]

    def test_component_sizes_account_for_every_triangle(self):
        surf = np.vstack([TETRAHEDRON, shifted(TETRAHEDRON, 10)])
        report = analyse_surface_topology(surf)
        assert report.component_sizes.sum() == surf.shape[0]


class TestPinchVertices:
    """A vertex carrying more than one fan, and nothing else."""

    def test_a_bowtie_vertex_is_found_with_two_fans(self):
        report = analyse_surface_topology(np.array([[0, 1, 2], [0, 3, 4]]))

        assert report.pinch_vertices.tolist() == [0]
        assert report.pinch_fan_counts.tolist() == [2]

    def test_three_sheets_at_one_vertex_report_three_fans(self):
        """Fan count is the number of sheets, not a flag. Two would be wrong."""
        surf = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]])
        report = analyse_surface_topology(surf)

        assert report.pinch_vertices.tolist() == [0]
        assert report.pinch_fan_counts.tolist() == [3]

    def test_two_closed_surfaces_welded_at_a_vertex(self):
        """The production shape: two sheets meeting at a single point."""
        second = shifted(TETRAHEDRON, 3)  # vertex 3 is shared, 0..3 and 3..6
        report = analyse_surface_topology(np.vstack([TETRAHEDRON, second]))

        assert report.n_components == 2
        assert report.pinch_vertices.tolist() == [3]
        assert report.pinch_fan_counts.tolist() == [2]

    def test_disconnection_alone_is_not_a_pinch(self):
        """
        Guards the likeliest wrong implementation: reporting every vertex of a
        multi-component surface, or confusing "more than one component" with
        "more than one fan at a vertex".
        """
        report = analyse_surface_topology(np.array([[0, 1, 2], [3, 4, 5]]))

        assert report.n_components == 2
        assert report.pinch_vertices.size == 0

    def test_a_closed_surface_has_no_pinch_vertices(self):
        assert analyse_surface_topology(TETRAHEDRON).pinch_vertices.size == 0

    def test_a_boundary_vertex_is_not_a_pinch(self):
        """
        An open sheet has vertices whose fan does not close. One fan is still
        one fan, so an implementation counting closure rather than connectivity
        would wrongly flag all of these.
        """
        report = analyse_surface_topology(np.array([[0, 1, 2], [0, 1, 3]]))
        assert report.pinch_vertices.size == 0


class TestNonManifoldEdges:
    """An edge carried by more than two triangles: two sheets crossing."""

    def test_three_triangles_on_one_edge_are_recorded(self):
        surf = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        report = analyse_surface_topology(surf)

        assert report.nonmanifold_edges.tolist() == [[0, 1]]

    def test_a_non_manifold_edge_does_not_join_its_triangles(self):
        """
        Corrected against AJR5d03297, where one four-triangle edge is the whole
        difference between finding 2 components and finding 3. Unioning across
        it — the rule this module first used — would make this 1.
        """
        surf = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        assert analyse_surface_topology(surf).n_components == 3

    def test_it_only_separates_what_is_joined_solely_through_that_edge(self):
        """
        The rule has to be conservative. Here the four triangles of a closed
        surface also touch along other edges, so declining to union across one
        non-manifold edge must not cut the surface up.
        """
        # A tetrahedron plus a flap sharing edge (0, 1) with two of its faces.
        surf = np.vstack([TETRAHEDRON, np.array([[0, 1, 9]])])
        report = analyse_surface_topology(surf)

        assert report.nonmanifold_edges.tolist() == [[0, 1]]
        assert report.n_components == 2  # the tetrahedron, and the loose flap

    def test_a_clean_surface_reports_no_non_manifold_edges(self):
        assert analyse_surface_topology(TETRAHEDRON).nonmanifold_edges.size == 0


class TestTheReport:
    """The contract itself."""

    def test_it_is_frozen(self):
        report = analyse_surface_topology(TETRAHEDRON)
        with pytest.raises(Exception):
            report.n_components = 99  # type: ignore[misc]

    def test_a_clean_surface_is_clean_and_a_pinched_one_is_not(self):
        assert analyse_surface_topology(TETRAHEDRON).is_clean
        assert not analyse_surface_topology(np.array([[0, 1, 2], [0, 3, 4]])).is_clean

    def test_pinched_triangles_marks_both_sides_of_a_bowtie(self):
        surf = np.array([[0, 1, 2], [0, 3, 4], [5, 6, 7]])
        report = analyse_surface_topology(surf)

        assert report.pinched_triangles(surf).tolist() == [True, True, False]

    def test_pinched_triangles_is_all_false_on_a_clean_surface(self):
        report = analyse_surface_topology(TETRAHEDRON)
        assert not report.pinched_triangles(TETRAHEDRON).any()

    def test_the_summary_names_the_counts_it_found(self):
        report = analyse_surface_topology(np.array([[0, 1, 2], [0, 3, 4]]))
        summary = report.summary()

        assert "2 components" in summary
        assert "1 pinch vertices" in summary
        assert "0 non-manifold edges" in summary

    def test_n_triangles_records_what_it_was_built_from(self):
        report = analyse_surface_topology(TETRAHEDRON)
        assert report.n_triangles == TETRAHEDRON.shape[0]


class TestRejectedInput:
    def test_a_non_triangular_surface_is_refused(self):
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            analyse_surface_topology(np.array([[0, 1, 2, 3]]))

    def test_an_empty_surface_is_refused(self):
        with pytest.raises(ValueError, match="no triangles"):
            analyse_surface_topology(np.zeros((0, 3), dtype=int))


class TestVertexIndexing:
    """
    Only equality of indices is used, so nothing may depend on their range.
    Both the parent mesh and a submesh reach this function.
    """

    def test_sparse_indices_give_the_same_answer_as_dense_ones(self):
        dense = np.array([[0, 1, 2], [0, 3, 4]])
        sparse = dense * 1000

        dense_report = analyse_surface_topology(dense)
        sparse_report = analyse_surface_topology(sparse)

        assert dense_report.n_components == sparse_report.n_components
        assert dense_report.pinch_fan_counts.tolist() == (
            sparse_report.pinch_fan_counts.tolist()
        )
        assert sparse_report.pinch_vertices.tolist() == [0]

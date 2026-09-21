"""
Writing an edge-adjacency split as the `.partN` families the pipeline reads.

The split has to be indistinguishable from `meshtool extract unreachable`'s
output, because nothing downstream is told which produced it. These tests pin
it against the functions that actually consume it -- `find_numbered_parts`,
`keep_largest_n_components`, `read_carp_mesh` and, above all,
`connected_component_to_surface`, which is where `.eidx` and `.nod` are proved
right or wrong.

Every surface here uses vertex indices **offset from zero**. That is deliberate:
`.nod` holds positions into the surface's vertex list, not parent vertex
indices, and on a surface numbered from zero the two coincide and a wrong
implementation passes.
"""

import numpy as np
import pytest

from pathlib import Path

import pycemrg_model_creation.utilities.mesh as mshu

from pycemrg_model_creation.utilities.mesh import (
    ElemType,
    write_pts,
    write_split_component,
    write_surf,
)
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.utilities.topology import (
    SplitComponent,
    analyse_surface_topology,
    split_surface_components,
)


# Two triangles pinched at vertex 10, numbered from 10 rather than 0.
PINCHED = np.array([[10, 11, 12], [10, 13, 14]])
POINTS = np.vstack([np.zeros((10, 3)), np.eye(3)[[0, 1, 2, 0, 1]] * 5.0])


@pytest.fixture
def split(tmp_path):
    report = analyse_surface_topology(PINCHED)
    return split_surface_components(PINCHED, report, tmp_path / "epi_endo_cc")


class TestTheSplitItself:
    def test_it_names_parts_the_way_meshtool_does(self, split):
        assert [c.name for c in split] == [
            "epi_endo_cc.part0",
            "epi_endo_cc.part1",
        ]

    def test_each_part_holds_its_own_triangle_rows(self, split):
        assert split[0].triangles.tolist() == [0]
        assert split[1].triangles.tolist() == [1]

    def test_each_part_holds_its_own_vertices(self, split):
        assert split[0].vertices.tolist() == [10, 11, 12]
        assert split[1].vertices.tolist() == [10, 13, 14]

    def test_the_pinch_vertex_is_reported_as_shared(self, split):
        """
        Nothing is dropped, so the pinch vertex belongs to both parts. The
        caller has to be told, because it matters if both parts become mguvc
        boundaries.
        """
        assert split[0].shared_with[1].tolist() == [10]
        assert split[1].shared_with[0].tolist() == [10]

    def test_a_clean_surface_reports_no_sharing(self, tmp_path):
        disjoint = np.array([[10, 11, 12], [13, 14, 15]])
        report = analyse_surface_topology(disjoint)
        parts = split_surface_components(disjoint, report, tmp_path / "cc")

        assert all(part.shared_with == {} for part in parts)

    def test_parts_are_ordered_largest_first(self, tmp_path):
        surf = np.array([[10, 11, 12], [20, 21, 22], [20, 21, 23]])
        report = analyse_surface_topology(surf)
        parts = split_surface_components(surf, report, tmp_path / "cc")

        assert [part.n_triangles for part in parts] == [2, 1]


class TestTheFilesWritten:
    def test_it_writes_the_family_find_numbered_parts_looks_for(
        self, split, tmp_path
    ):
        for component in split:
            write_split_component(component, POINTS, PINCHED)

        parts = mshu.find_numbered_parts(tmp_path, "epi_endo_cc")

        # `find_numbered_parts` returns handles, so compare the names it found
        # rather than the collection.
        assert [part.name for part in parts] == [
            "epi_endo_cc.part0",
            "epi_endo_cc.part1",
        ]

    def test_index_files_are_binary_int64(self, split, tmp_path):
        """
        `read_nod_eidx` defaults to a 64-bit width. Writing a narrower one
        produces indices that are wrong rather than absent.
        """
        write_split_component(split[0], POINTS, PINCHED)

        assert (tmp_path / "epi_endo_cc.part0.eidx").stat().st_size == 1 * 8
        assert (tmp_path / "epi_endo_cc.part0.nod").stat().st_size == 3 * 8

    def test_nod_holds_positions_into_the_surface_not_parent_indices(
        self, split, tmp_path
    ):
        """
        `connected_component_to_surface` does `surf2vtx(surf)[nod]`, so `.nod`
        is an index into that list. Here the parent indices are 10, 11, 12 and
        the positions are 0, 1, 2 -- which is why this surface is not numbered
        from zero.
        """
        write_split_component(split[0], POINTS, PINCHED)
        nod = mshu.read_nod_eidx(tmp_path / "epi_endo_cc.part0.nod")

        assert nod.tolist() == [0, 1, 2]

    def test_eidx_holds_rows_of_the_parent_surface(self, split, tmp_path):
        write_split_component(split[1], POINTS, PINCHED)
        eidx = mshu.read_nod_eidx(tmp_path / "epi_endo_cc.part1.eidx")

        assert eidx.tolist() == [1]

    def test_elem_is_reindexed_to_the_component(self, split, tmp_path):
        write_split_component(split[1], POINTS, PINCHED)
        points, elements = mshu.read_carp_mesh(
            tmp_path / "epi_endo_cc.part1", elem_type=ElemType.Tr, read_tags=False
        )

        assert elements.max() < points.shape[0]
        assert points.shape[0] == 3

    def test_the_files_land_where_the_handle_says(self, split, tmp_path):
        """
        The point of carrying a `CarpMesh` rather than a name: there is no
        destination argument to get wrong, so the files cannot land anywhere
        but where the part is named. Writing into a subdirectory that the
        fixture never mentions is what makes that visible.
        """
        nested = tmp_path / "elsewhere"
        nested.mkdir()
        component = SplitComponent(
            mesh=CarpMesh(nested / "cc.part0"),
            triangles=split[0].triangles,
            vertices=split[0].vertices,
            shared_with={},
        )

        write_split_component(component, POINTS, PINCHED)

        for extension in (".pts", ".elem", ".nod", ".eidx", ".vtk"):
            assert (nested / f"cc.part0{extension}").is_file()
        assert not (tmp_path / "cc.part0.pts").exists()

    def test_the_vtk_can_be_skipped(self, split, tmp_path):
        write_split_component(split[0], POINTS, PINCHED, write_vtk=False)
        assert not (tmp_path / "epi_endo_cc.part0.vtk").exists()


class TestElementTags:
    """
    `meshtool` writes the region tag of the element each triangle bounds, and
    it is load-bearing: inside the RV endocardium it is what separates the
    septum (LV tissue) from the free wall. Dropping it is a silent loss --
    the files are still valid, just less informative.
    """

    # Tag the two triangles differently, so a component picking the wrong row
    # is visible rather than lucky.
    TAGS = np.array([1, 2])

    def test_tags_are_carried_into_the_elem(self, split, tmp_path):
        write_split_component(split[1], POINTS, PINCHED, tags=self.TAGS)
        # `read_elem`, like `read_surf`, has no `ndmin=2`, so a one-element
        # file comes back 1-D. A property of the reader, not of what we wrote.
        elements = np.atleast_2d(
            mshu.read_elem(split[1].mesh.elem, elem_type=ElemType.Tr, read_tags=True)
        )

        assert elements[:, 3].tolist() == [2]

    def test_each_component_gets_its_own_rows(self, split, tmp_path):
        for component in split:
            write_split_component(component, POINTS, PINCHED, tags=self.TAGS)

        first = np.atleast_2d(
            mshu.read_elem(split[0].mesh.elem, elem_type=ElemType.Tr, read_tags=True)
        )
        second = np.atleast_2d(
            mshu.read_elem(split[1].mesh.elem, elem_type=ElemType.Tr, read_tags=True)
        )

        assert first[:, 3].tolist() == [1]
        assert second[:, 3].tolist() == [2]

    def test_the_vtk_carries_elemtag_under_meshtools_name(self, split, tmp_path):
        """
        Named `elemTag` so the same ParaView colouring works on our parts and
        on meshtool's.
        """
        import pyvista as pv

        write_split_component(split[1], POINTS, PINCHED, tags=self.TAGS)
        mesh = pv.read(split[1].mesh.vtk)

        assert mesh.cell_data["elemTag"].tolist() == [2]

    def test_omitting_tags_falls_back_to_one(self, split, tmp_path):
        write_split_component(split[1], POINTS, PINCHED)
        elements = np.atleast_2d(
            mshu.read_elem(split[1].mesh.elem, elem_type=ElemType.Tr, read_tags=True)
        )

        assert elements[:, 3].tolist() == [1]

    def test_tags_that_do_not_describe_the_surface_are_refused(
        self, split, tmp_path
    ):
        with pytest.raises(ValueError, match="do not describe each other"):
            write_split_component(
                split[0], POINTS, PINCHED, tags=np.array([1, 2, 3, 4])
            )


class TestTheRoundTrip:
    """
    The one that matters: what the pipeline gets back out.

    `connected_component_to_surface` uses `with_suffix`, which truncates a
    dotted stem, so the pipeline copies each part to an undotted name first.
    These tests write undotted names directly rather than re-stage that copy.
    """

    def test_the_surface_comes_back_exactly(self, tmp_path):
        report = analyse_surface_topology(PINCHED)
        parts = split_surface_components(PINCHED, report, tmp_path / "cc")
        write_surf(PINCHED, tmp_path / "parent.surf")

        for k, component in enumerate(parts):
            renamed = SplitComponent(
                mesh=CarpMesh(tmp_path / f"comp{k}"),
                triangles=component.triangles,
                vertices=component.vertices,
                shared_with=component.shared_with,
            )
            write_split_component(renamed, POINTS, PINCHED)

            mshu.connected_component_to_surface(
                eidx_path=tmp_path / f"comp{k}",
                input_surface_path=tmp_path / "parent.surf",
                output_surface_path=tmp_path / f"out{k}",
            )

            # `read_surf` has no `ndmin=2`, so a one-triangle file comes back
            # 1-D. That is a property of the reader, not of what was written,
            # and reshaping here keeps this test about the split.
            got = np.atleast_2d(mshu.read_surf(tmp_path / f"out{k}.surf"))
            assert got.tolist() == PINCHED[report.triangle_component == k].tolist()

    def test_the_vertex_set_comes_back_exactly(self, tmp_path):
        report = analyse_surface_topology(PINCHED)
        parts = split_surface_components(PINCHED, report, tmp_path / "cc")
        write_surf(PINCHED, tmp_path / "parent.surf")

        component = parts[0]
        renamed = SplitComponent(
            mesh=CarpMesh(tmp_path / "comp0"),
            triangles=component.triangles,
            vertices=component.vertices,
            shared_with=component.shared_with,
        )
        write_split_component(renamed, POINTS, PINCHED)

        mshu.connected_component_to_surface(
            eidx_path=tmp_path / "comp0",
            input_surface_path=tmp_path / "parent.surf",
            output_surface_path=tmp_path / "out0",
        )

        vtx = mshu.read_vtx(tmp_path / "out0.vtx")
        assert vtx.tolist() == [10, 11, 12]

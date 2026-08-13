"""Unit tests for the CARP readers and writers added to `utilities.mesh`.

Round-trip coverage only: each writer is checked against its matching reader.
Nothing here verifies the formats against meshtool or mguvc themselves.
"""

import numpy as np
import pytest

from pycemrg_model_creation.utilities.mesh import (
    ElemType,
    connected_component_to_surface,
    extract_points_by_vtx,
    read_dat,
    read_elem,
    read_lon,
    read_neubc,
    read_nod_eidx,
    read_surf,
    read_vtx,
    reindex_surf,
    reindex_vtx,
    write_dat,
    write_elem,
    write_lon,
    write_surf,
    write_vtx,
)


class TestVtxRoundTrip:
    def test_round_trip(self, tmp_path):
        vtx = np.array([3, 17, 42, 108])
        path = tmp_path / "nodes.vtx"

        write_vtx(vtx, path)
        np.testing.assert_array_equal(read_vtx(path), vtx)

    def test_header_is_count_then_group(self, tmp_path):
        path = tmp_path / "nodes.vtx"
        write_vtx(np.array([1, 2, 3]), path)

        lines = path.read_text().splitlines()
        assert lines[0] == "3"
        assert lines[1] == "intra"

    def test_single_index_stays_1d(self, tmp_path):
        path = tmp_path / "one.vtx"
        write_vtx(np.array([7]), path)

        result = read_vtx(path)
        assert result.ndim == 1
        assert result[0] == 7


class TestSurfRoundTrip:
    """
    These were missing, which is how the reader and writer came to disagree.

    `read_surf` used to run `with_suffix(".surf")` on its argument while
    `write_surf` took its path literally, so a caller handing a stem to both
    read one file and wrote another. Nothing caught it because neither helper
    had a test.
    """

    def test_round_trip(self, tmp_path):
        surf = np.array([[0, 1, 2], [2, 3, 4]])
        path = tmp_path / "endo.surf"

        write_surf(surf, path)
        np.testing.assert_array_equal(read_surf(path), surf)

    def test_header_is_the_triangle_count(self, tmp_path):
        path = tmp_path / "endo.surf"
        write_surf(np.array([[0, 1, 2], [2, 3, 4]]), path)

        assert path.read_text().splitlines()[0] == "2"

    def test_row_prefix_is_tr(self, tmp_path):
        path = tmp_path / "endo.surf"
        write_surf(np.array([[7, 8, 9]]), path)

        assert path.read_text().splitlines()[1] == "Tr 7 8 9"

    def test_both_sides_take_the_path_literally(self, tmp_path):
        # The pairing that C establishes: whatever name the writer is given is
        # the name the reader opens. A stem must fail on both sides, not
        # silently work on one.
        # Two triangles, because `read_surf` returns 1-D for a single row.
        surf = np.array([[0, 1, 2], [2, 3, 4]])
        stem = tmp_path / "endo"

        write_surf(surf, stem)

        assert stem.exists()
        assert not stem.with_suffix(".surf").exists()
        np.testing.assert_array_equal(read_surf(stem), surf)

    def test_a_dotted_stem_is_not_truncated(self, tmp_path):
        # The name must have no `.surf` on it, or this proves nothing:
        # `with_suffix(".surf")` is a no-op on a path that already ends in it.
        # A dotted stem is where the old reader broke -- it turned
        # `BiV.rvendo_nosept` into `BiV.surf` and read the wrong file.
        surf = np.array([[4, 5, 6], [6, 7, 8]])
        dotted_stem = tmp_path / "BiV.rvendo_nosept"

        write_surf(surf, dotted_stem)
        np.testing.assert_array_equal(read_surf(dotted_stem), surf)
        assert not (tmp_path / "BiV.surf").exists()

    def test_rejects_wrong_column_count(self, tmp_path):
        with pytest.raises(AssertionError):
            write_surf(np.array([[0, 1, 2, 3]]), tmp_path / "bad.surf")


class TestConnectedComponentToSurface:
    """
    Its three path arguments are not the same shape, and nothing covered it.

    `input_surface_path` arrives with `.surf` already on it; `eidx_path` and
    `output_surface_path` are stems the function extends itself. Wrapping the
    input in a `CarpSurface` looks right and raises, because the handle refuses
    an extension -- a real break that the unit suite let through once.
    """

    def _write_component(self, tmp_path, eidx, nod):
        stem = tmp_path / "component"
        np.array(eidx, dtype=int).tofile(stem.with_suffix(".eidx"))
        np.array(nod, dtype=int).tofile(stem.with_suffix(".nod"))
        return stem

    def test_carves_the_named_triangles_out(self, tmp_path):
        surf = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
        source = tmp_path / "epi_endo.surf"
        write_surf(surf, source)

        stem = self._write_component(tmp_path, eidx=[0, 2], nod=[0, 1])
        out = tmp_path / "biv_epi"

        connected_component_to_surface(
            eidx_path=stem, input_surface_path=source, output_surface_path=out
        )

        np.testing.assert_array_equal(
            read_surf(out.with_suffix(".surf")), surf[[0, 2], :]
        )

    def test_the_input_surface_is_taken_literally(self, tmp_path):
        # The regression: the input already carries `.surf`, so the function
        # must not convert it. Naming a file whose stem alone does not exist
        # is what makes this fail if anything re-derives the path.
        surf = np.array([[0, 1, 2], [2, 3, 4]])
        source = tmp_path / "epi_endo.surf"
        write_surf(surf, source)
        assert not (tmp_path / "epi_endo").exists()

        stem = self._write_component(tmp_path, eidx=[1], nod=[0])

        connected_component_to_surface(
            eidx_path=stem,
            input_surface_path=source,
            output_surface_path=tmp_path / "out",
        )

        assert (tmp_path / "out.surf").exists()
        assert (tmp_path / "out.vtx").exists()


class TestDatRoundTrip:
    def test_round_trip(self, tmp_path):
        data = np.array([0.0, 0.25, 0.5, 1.0])
        path = tmp_path / "uvc_z.dat"

        write_dat(data, path)
        np.testing.assert_allclose(read_dat(path), data)

    def test_no_header_row(self, tmp_path):
        path = tmp_path / "field.dat"
        write_dat(np.array([1.0, 2.0]), path)

        assert len(path.read_text().splitlines()) == 2

    def test_single_value_stays_1d(self, tmp_path):
        path = tmp_path / "one.dat"
        write_dat(np.array([2.5]), path)

        assert read_dat(path).ndim == 1


class TestElemRoundTrip:
    def test_tetrahedra_round_trip(self, tmp_path):
        elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        tags = np.array([1, 7])
        path = tmp_path / "mesh.elem"

        write_elem(elements, tags, path, elem_type=ElemType.Tt)
        result = read_elem(path, elem_type=ElemType.Tt, read_tags=True)

        np.testing.assert_array_equal(result[:, :4], elements)
        np.testing.assert_array_equal(result[:, 4], tags)

    def test_triangles_round_trip(self, tmp_path):
        elements = np.array([[0, 1, 2], [2, 3, 4]])
        tags = np.array([3, 3])
        path = tmp_path / "surface.elem"

        write_elem(elements, tags, path, elem_type=ElemType.Tr)
        result = read_elem(path, elem_type=ElemType.Tr, read_tags=True)

        np.testing.assert_array_equal(result[:, :3], elements)

    def test_row_prefix_is_the_element_type(self, tmp_path):
        path = tmp_path / "mesh.elem"
        write_elem(np.array([[0, 1, 2, 3]]), np.array([1]), path)

        lines = path.read_text().splitlines()
        assert lines[0] == "1"
        assert lines[1].startswith("Tt ")

    def test_rejects_wrong_node_count(self, tmp_path):
        with pytest.raises(ValueError, match="Tt elements must have shape"):
            write_elem(
                np.array([[0, 1, 2]]), np.array([1]), tmp_path / "bad.elem"
            )

    def test_rejects_mismatched_tag_count(self, tmp_path):
        with pytest.raises(ValueError, match="does not match element count"):
            write_elem(
                np.array([[0, 1, 2, 3]]),
                np.array([1, 2]),
                tmp_path / "bad.elem",
            )


class TestLonRoundTrip:
    def test_fibre_only_round_trip(self, tmp_path):
        fibres = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        path = tmp_path / "mesh.lon"

        write_lon(fibres, path)
        np.testing.assert_allclose(read_lon(path), fibres)

    def test_fibre_and_sheet_round_trip(self, tmp_path):
        fibres = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        path = tmp_path / "mesh.lon"

        write_lon(fibres, path)
        np.testing.assert_allclose(read_lon(path), fibres[0])

    def test_header_counts_directions_not_elements(self, tmp_path):
        path = tmp_path / "mesh.lon"
        write_lon(np.zeros((5, 6)), path)

        # Five elements, two direction vectors each -> header is 2, not 5.
        assert path.read_text().splitlines()[0] == "2"

    def test_rejects_non_multiple_of_three(self, tmp_path):
        with pytest.raises(ValueError, match=r"shape \(n, 3k\)"):
            write_lon(np.zeros((2, 4)), tmp_path / "bad.lon")


class TestReadNodEidx:
    def test_round_trip_against_numpy_tofile(self, tmp_path):
        indices = np.array([10, 20, 30, 40], dtype=np.int64)
        path = tmp_path / "submesh.nod"
        indices.tofile(path)

        np.testing.assert_array_equal(read_nod_eidx(path), indices)

    def test_dtype_is_overridable(self, tmp_path):
        indices = np.array([1, 2, 3], dtype=np.int32)
        path = tmp_path / "submesh.eidx"
        indices.tofile(path)

        np.testing.assert_array_equal(
            read_nod_eidx(path, dtype=np.int32), indices
        )


class TestReindexing:
    def test_reindex_surf_maps_into_local_numbering(self):
        vtx_surf = np.array([5, 9, 12, 20])
        surf = np.array([[5, 9, 12], [9, 12, 20]])

        result = reindex_surf(vtx_surf, surf)
        np.testing.assert_array_equal(result, [[0, 1, 2], [1, 2, 3]])

    def test_reindex_surf_round_trips_through_the_vertex_list(self):
        vtx_surf = np.array([2, 4, 8, 16])
        surf = np.array([[2, 8, 16], [4, 8, 2]])

        np.testing.assert_array_equal(vtx_surf[reindex_surf(vtx_surf, surf)], surf)

    def test_reindex_surf_rejects_unknown_vertex(self):
        with pytest.raises(ValueError, match="absent from vtx_surf"):
            reindex_surf(np.array([1, 2, 3]), np.array([[1, 2, 99]]))

    def test_reindex_vtx_maps_into_local_numbering(self):
        vtx_surf = np.array([5, 9, 12, 20])
        result = reindex_vtx(np.array([9, 20]), vtx_surf)

        np.testing.assert_array_equal(result, [1, 3])

    def test_reindex_vtx_drops_indices_outside_the_surface(self):
        vtx_surf = np.array([5, 9, 12])
        result = reindex_vtx(np.array([9, 99]), vtx_surf)

        np.testing.assert_array_equal(result, [1])


class TestExtractPointsByVtx:
    def test_selects_named_points_in_order(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        result = extract_points_by_vtx(points, np.array([2, 0]))

        np.testing.assert_allclose(result, [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])


class TestReadNeubc:
    def test_reads_first_six_columns_after_header(self, tmp_path):
        path = tmp_path / "mesh.neubc"
        path.write_text("2\n0 1 2 3 4 5 6.5\n7 8 9 10 11 12 13.5\n")

        result = read_neubc(path)
        np.testing.assert_array_equal(result, [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])

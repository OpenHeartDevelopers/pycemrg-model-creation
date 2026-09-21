"""
What `SurfaceLogic` says when a component split comes out wrong.

The failure this covers is real: on AJR5d03297, `meshtool extract unreachable`
returned one component where the pipeline needed three, and the message said
only "found 1". These tests pin what replaces it -- the counts, the defects, and
a mesh written out to look at.

`SurfaceLogic` is constructed with `None` collaborators throughout. Diagnosis
touches neither `meshtool` nor the label manager, and passing `None` says so.
"""

import numpy as np
import pytest

from pathlib import Path

from pycemrg_model_creation.logic.surfaces import SurfaceLogic
from pycemrg_model_creation.utilities.mesh import write_pts, write_surf


# Two triangles joined at vertex 0 only: the smallest mesh that reproduces the
# production failure, since meshtool would call this one component.
PINCHED = np.array([[0, 1, 2], [0, 3, 4]])
PINCHED_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # the pinch
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


@pytest.fixture
def pinched_surface(tmp_path: Path) -> Path:
    """A pinched `.surf` and its `.pts`, returning the surface's stem."""
    stem = tmp_path / "epi_endo"
    write_surf(PINCHED, Path(f"{stem}.surf"))
    write_pts(PINCHED_POINTS, tmp_path / "mesh.pts")
    return stem


@pytest.fixture
def logic() -> SurfaceLogic:
    return SurfaceLogic(None, None)


class TestTheMessage:
    def test_it_keeps_the_original_counts(self, logic, pinched_surface, tmp_path):
        message = logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )
        assert "Expected 3 connected components, found 1." in message

    def test_it_reports_what_edge_adjacency_found_instead(
        self, logic, pinched_surface, tmp_path
    ):
        """
        The point of the whole change: meshtool said 1, edge adjacency says 2,
        and the difference is a named defect rather than a mystery.
        """
        message = logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )

        assert "Edge-adjacency finds 2 components" in message
        assert "1 pinch vertices" in message

    def test_it_names_the_file_it_wrote_and_how_to_read_it(
        self, logic, pinched_surface, tmp_path
    ):
        message = logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )

        assert str(tmp_path / "epi_endo_topology.vtk") in message
        assert "colour by 'pinch'" in message
        assert "colour by 'component'" in message


class TestTheDiagnosticFile:
    def test_it_is_written_beside_the_surface(
        self, logic, pinched_surface, tmp_path
    ):
        logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )
        assert (tmp_path / "epi_endo_topology.vtk").is_file()

    def test_it_carries_both_cell_arrays(self, logic, pinched_surface, tmp_path):
        import pyvista as pv

        logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )
        mesh = pv.read(tmp_path / "epi_endo_topology.vtk")

        assert "component" in mesh.cell_data
        assert "pinch" in mesh.cell_data
        # Both triangles of a bowtie touch the pinch vertex.
        assert mesh.cell_data["pinch"].tolist() == [1, 1]
        assert sorted(mesh.cell_data["component"].tolist()) == [0, 1]


class TestItNeverMasksTheFailure:
    """
    Diagnosis runs inside an error path. If it throws, the caller loses the
    error it was already raising, which would be worse than no diagnosis.
    """

    def test_a_missing_surface_still_returns_the_counts(self, logic, tmp_path):
        message = logic._diagnose_component_failure(
            surface_stem=tmp_path / "absent",
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )
        assert message == "Expected 3 connected components, found 1."

    def test_a_missing_points_file_still_returns_the_counts(
        self, logic, pinched_surface, tmp_path
    ):
        message = logic._diagnose_component_failure(
            surface_stem=pinched_surface,
            points_path=tmp_path / "not_here.pts",
            expected=3,
            found=1,
        )
        assert message == "Expected 3 connected components, found 1."

    def test_it_warns_rather_than_raising(self, logic, tmp_path, caplog):
        logic._diagnose_component_failure(
            surface_stem=tmp_path / "absent",
            points_path=tmp_path / "mesh.pts",
            expected=2,
            found=1,
        )
        assert "Could not diagnose" in caplog.text


class TestDottedStems:
    """
    `with_suffix` is banned on mesh stems here, and the surfaces this reads are
    exactly the kind that carry a dot (`epi_endo.surfmesh`, `*.part0`).
    """

    def test_a_dotted_stem_is_extended_not_truncated(self, logic, tmp_path):
        stem = tmp_path / "epi_endo.surfmesh"
        write_surf(PINCHED, Path(f"{stem}.surf"))
        write_pts(PINCHED_POINTS, tmp_path / "mesh.pts")

        message = logic._diagnose_component_failure(
            surface_stem=stem,
            points_path=tmp_path / "mesh.pts",
            expected=3,
            found=1,
        )

        # with_suffix would have looked for epi_endo.surf and found nothing,
        # degrading to the bare message.
        assert "Edge-adjacency finds 2 components" in message
        assert (tmp_path / "epi_endo.surfmesh_topology.vtk").is_file()

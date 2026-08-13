"""
Unit tests for the rename hop in the boundary VTX chain.

`meshtool map` keeps each input's basename, so the remapped boundaries arrive
in the output directory still carrying the four-chamber stem. `mguvc` builds
its filenames from the model name, so it looks for the submesh stem. The rename
is what joins the two, and the chain had no such step at all — it wrote the
right names in the wrong generation and never renamed the right one.

`SurfaceLogic` only stores its collaborators, so these run with nulls injected.
No meshtool, no mesh data.
"""

from pathlib import Path

import pytest

from pycemrg_model_creation.logic.builders import ModelCreationPathBuilder
from pycemrg_model_creation.logic.surfaces import SurfaceExtractionError, SurfaceLogic


@pytest.fixture
def logic():
    return SurfaceLogic(None, None)


@pytest.fixture
def paths(tmp_path):
    builder = ModelCreationPathBuilder(tmp_path)
    ventricular = builder.build_ventricular_paths(Path("/in/heart"))
    return builder.build_biv_mesh_paths(Path("/in/heart"), ventricular)


def place_mapped_outputs(paths) -> None:
    """What `meshtool map` leaves behind: the input basenames, in -outdir."""
    paths.mapped_vtx_output_dir.mkdir(parents=True, exist_ok=True)
    for source in paths.vtx_files_to_map:
        (paths.mapped_vtx_output_dir / source.name).write_text(source.name)


class TestTheRename:
    def test_every_boundary_reaches_the_submesh_stem(self, logic, paths):
        place_mapped_outputs(paths)

        logic._rename_mapped_boundaries(paths)

        for target in paths.renamed_vtx_targets:
            assert target.is_file(), f"{target.name} was not produced"

    def test_the_four_chamber_names_do_not_survive(self, logic, paths):
        # Leaving them behind is what `cemrg-heartbuilder` does, and it makes
        # the output directory ambiguous to read. A rename, not a copy.
        place_mapped_outputs(paths)

        logic._rename_mapped_boundaries(paths)

        leftovers = sorted(
            p.name for p in paths.mapped_vtx_output_dir.glob("heart.*.vtx")
        )
        assert leftovers == []

    def test_content_travels_with_the_name(self, logic, paths):
        # A rename that moved the wrong file would still leave five files with
        # the right names, and this is the only check that would notice.
        place_mapped_outputs(paths)

        logic._rename_mapped_boundaries(paths)

        for source, target in zip(paths.vtx_files_to_map, paths.renamed_vtx_targets):
            assert target.read_text() == source.name

    def test_a_missing_mapped_file_is_an_error(self, logic, paths):
        # If `-outdir` is ever misread as a stem prefix again, the mapped files
        # land somewhere else and every rename silently finds nothing.
        place_mapped_outputs(paths)
        (paths.mapped_vtx_output_dir / paths.vtx_files_to_map[0].name).unlink()

        with pytest.raises(SurfaceExtractionError, match="did not produce"):
            logic._rename_mapped_boundaries(paths)

    def test_the_rename_reads_from_the_output_dir_not_from_tmp(self, logic, paths):
        # The mapping's inputs stay in tmp/. Renaming those instead would move
        # the four-chamber generation onto the submesh names — the exact
        # confusion the two stems exist to prevent.
        place_mapped_outputs(paths)
        for source in paths.vtx_files_to_map:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("four-chamber indices")

        logic._rename_mapped_boundaries(paths)

        for source in paths.vtx_files_to_map:
            assert source.is_file(), "the tmp/ generation must be left alone"
        for target in paths.renamed_vtx_targets:
            assert target.read_text() != "four-chamber indices"

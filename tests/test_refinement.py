"""Unit tests for `logic.refinement` — no meshtool binary, no test data.

The refinement flow had exactly one test before this file, an integration test
that hardcodes ``simplify=True`` (`tests/integration/test_refinement_workflow.py:78`).
Everything reached only when simplification is skipped was therefore unexercised,
including the companion-copy branch these tests are mostly about.

`meshtool` is faked. What is *not* faked is the path builder: the names asserted
here travel from `RefinementPathBuilder` through the contract into the logic, so a
disagreement between the three shows up as a failure rather than as a missing file
much later.
"""

import logging
from pathlib import Path

import pytest

from pycemrg_model_creation.logic import refinement
from pycemrg_model_creation.logic.builders import RefinementPathBuilder
from pycemrg_model_creation.logic.contracts import MeshPostprocessingPaths
from pycemrg_model_creation.logic.refinement import RefinementLogic
from pycemrg_model_creation.meshpaths import CarpMesh

# Markers written into the fake files, so a test can say which tool produced a
# given file rather than merely that it exists.
FROM_EXTRACT = "written-by-extract\n"
FROM_SIMPLIFY = "written-by-simplify\n"
FROM_RELABEL = "written-by-relabel\n"

# meshtool fabricates these when the input mesh carries no fibres. Five rows of
# [1 0 0] is the shape a user is being asked to recognise as discardable.
DEFAULT_FIBRES = "1 0 0\n" * 5
REAL_FIBRES = "0.71 0.20 0.67\n" * 3


class FakeMeshtool:
    """
    Stands in for MeshtoolWrapper, recording calls and writing plausible files.

    Deliberately builds its filenames by *appending* to the argument it was
    given. The real wrapper takes a stem and does the same, so a handle leaking
    through instead of a stem shows up here as an AttributeError, which is what
    would happen in production.
    """

    def __init__(self, simplify_available: bool = True):
        self.is_simplify_topology_available = simplify_available
        self.calls = []

    @staticmethod
    def _write_family(base, marker, extensions=(".pts", ".elem", ".lon")):
        for extension in extensions:
            body = DEFAULT_FIBRES if extension == ".lon" else marker
            Path(str(base) + extension).write_text(body)

    def extract_mesh(self, input_mesh_path, output_submesh_path, tags, ifmt):
        self.calls.append(("extract_mesh", input_mesh_path, output_submesh_path))
        self._write_family(output_submesh_path, FROM_EXTRACT)

    def simplify_topology(self, input_mesh_path, output_mesh_path, ifmt, ofmt):
        self.calls.append(("simplify_topology", input_mesh_path, output_mesh_path))
        self._write_family(output_mesh_path, FROM_SIMPLIFY)

    def convert(self, input_mesh_path, output_mesh_path, ifmt, ofmt):
        self.calls.append(("convert", input_mesh_path, output_mesh_path))
        self._write_family(output_mesh_path, "vtk\n", extensions=(".vtk",))

    def output_args(self):
        """Every path this wrapper was asked to write to."""
        return [call[2] for call in self.calls]


@pytest.fixture
def relabel_calls(monkeypatch):
    """Replace the relabeller, recording where it read from and wrote to."""
    calls = []

    def fake_relabel(input_elem_path, output_elem_path, tag_mapping):
        calls.append((Path(input_elem_path), Path(output_elem_path)))
        Path(output_elem_path).write_text(FROM_RELABEL)

    monkeypatch.setattr(refinement, "relabel_carp_elem_file", fake_relabel)
    return calls


@pytest.fixture
def raw_mesh(tmp_path):
    """A raw input mesh, as produced upstream of this library."""
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    mesh = CarpMesh(source_dir / "heart_mesh")
    mesh.pts.write_text("raw\n")
    mesh.elem.write_text("raw\n")
    return mesh


@pytest.fixture
def paths(tmp_path, raw_mesh):
    builder = RefinementPathBuilder(output_dir=tmp_path / "output")
    return builder.build_postprocessing_paths(input_mesh_base=raw_mesh)


def run(paths, meshtool, simplify):
    RefinementLogic(meshtool_wrapper=meshtool).run_myocardium_postprocessing(
        paths=paths,
        myocardium_tags=[2, 103],
        tag_mapping={2: 1, 103: 2},
        simplify=simplify,
    )


class TestBuilderPurity:
    def test_building_a_contract_touches_no_disk(self, tmp_path):
        root = tmp_path / "output"
        builder = RefinementPathBuilder(output_dir=root)
        builder.build_postprocessing_paths(input_mesh_base=tmp_path / "heart_mesh")

        # The mkdir used to live in __init__, so merely inspecting a contract
        # created directories. Naming and disk mutation are now separate.
        assert not root.exists()

    def test_refined_subdir_is_overridable(self, tmp_path):
        builder = RefinementPathBuilder(
            output_dir=tmp_path / "output", refined_subdir="02_refined"
        )
        paths = builder.build_postprocessing_paths(
            input_mesh_base=tmp_path / "heart_mesh"
        )

        assert paths.output_dir.name == "02_refined"
        assert paths.output_mesh_base.elem.parent.name == "02_refined"

    def test_default_subdir_carries_no_stage_number(self, paths):
        assert paths.output_dir.name == "refined"

    def test_directories_are_the_meshes_own_parents(self, paths):
        # They are properties now, not stored fields, so they cannot drift
        # away from the handles the way two separate fields could.
        assert paths.output_dir == paths.output_mesh_base.directory
        assert paths.tmp_dir == paths.intermediate_myocardium_mesh.directory


class TestDirectoryPreparation:
    def test_run_creates_both_directories(self, paths, relabel_calls):
        assert not paths.output_dir.exists()
        assert not paths.tmp_dir.exists()

        run(paths, FakeMeshtool(), simplify=True)

        assert paths.output_dir.is_dir()
        assert paths.tmp_dir.is_dir()

    def test_existing_directories_are_not_an_error(self, paths, relabel_calls):
        paths.output_dir.mkdir(parents=True)
        paths.tmp_dir.mkdir(parents=True)

        # Unlike UvcLogic, which must clear its output dir because mguvc
        # prompts interactively when one exists, meshtool does not care.
        run(paths, FakeMeshtool(), simplify=True)

        assert paths.output_mesh_base.elem.exists()


class TestWrapperBoundary:
    """The wrappers take plain stems and build their own expected-output names."""

    def test_every_output_argument_is_a_stem(self, paths, relabel_calls):
        meshtool = FakeMeshtool()

        run(paths, meshtool, simplify=True)

        for argument in meshtool.output_args():
            assert not isinstance(argument, CarpMesh), (
                "a handle reached the wrapper; the real wrapper calls "
                "with_suffix on this and would raise AttributeError"
            )
            assert Path(argument).suffix == ""

    def test_convert_is_given_the_stem_not_the_vtk_name(self, paths, relabel_calls):
        meshtool = FakeMeshtool()

        run(paths, meshtool, simplify=True)

        convert = [c for c in meshtool.calls if c[0] == "convert"][0]
        # meshtool appends .vtk itself. Passing the .vtk name would produce
        # myocardium_clean.vtk.vtk.
        assert Path(convert[2]) == paths.output_mesh_base.stem
        assert paths.output_mesh_base.vtk.exists()


class TestRelabelRouting:
    def test_simplify_relabels_the_output_mesh_in_place(self, paths, relabel_calls):
        run(paths, FakeMeshtool(), simplify=True)

        source, destination = relabel_calls[0]
        assert source == paths.output_mesh_base.elem
        assert destination == paths.output_mesh_base.elem

    def test_without_simplify_relabels_from_the_intermediate(
        self, paths, relabel_calls
    ):
        run(paths, FakeMeshtool(), simplify=False)

        source, destination = relabel_calls[0]
        assert source == paths.intermediate_myocardium_mesh.elem
        assert destination == paths.output_mesh_base.elem

    def test_unavailable_tool_falls_back_to_the_intermediate(
        self, paths, relabel_calls
    ):
        # simplify was asked for but the standalone is missing: the run must
        # behave as though it had not been asked for at all.
        run(paths, FakeMeshtool(simplify_available=False), simplify=True)

        source, _ = relabel_calls[0]
        assert source == paths.intermediate_myocardium_mesh.elem
        assert paths.output_mesh_base.pts.read_text() == FROM_EXTRACT

    def test_unavailable_tool_warns(self, paths, relabel_calls, caplog):
        with caplog.at_level(logging.WARNING):
            run(paths, FakeMeshtool(simplify_available=False), simplify=True)

        assert "not available" in caplog.text


class TestCompanionCopy:
    def test_skipping_simplify_still_produces_the_full_family(
        self, paths, relabel_calls
    ):
        run(paths, FakeMeshtool(), simplify=False)

        for companion in (
            paths.output_mesh_base.pts,
            paths.output_mesh_base.elem,
            paths.output_mesh_base.lon,
        ):
            assert companion.exists(), f"{companion.name} missing"

    def test_elem_comes_from_relabel_not_from_the_copy(self, paths, relabel_calls):
        run(paths, FakeMeshtool(), simplify=False)

        # Copying .elem would overwrite the relabelled tags with the raw ones.
        assert paths.output_mesh_base.elem.read_text() == FROM_RELABEL

    def test_pts_and_lon_come_from_the_intermediate(self, paths, relabel_calls):
        run(paths, FakeMeshtool(), simplify=False)

        assert paths.output_mesh_base.pts.read_text() == FROM_EXTRACT
        assert paths.output_mesh_base.lon.read_text() == DEFAULT_FIBRES

    def test_no_copy_happens_when_simplify_ran(self, paths, relabel_calls):
        run(paths, FakeMeshtool(), simplify=True)

        # simplify_topology wrote these directly; a copy would replace them
        # with the intermediate's.
        assert paths.output_mesh_base.pts.read_text() == FROM_SIMPLIFY

    def test_missing_source_pts_warns_rather_than_raising(
        self, paths, relabel_calls, caplog
    ):
        class NoPts(FakeMeshtool):
            def extract_mesh(self, input_mesh_path, output_submesh_path, tags, ifmt):
                self.calls.append(
                    ("extract_mesh", input_mesh_path, output_submesh_path)
                )
                self._write_family(
                    output_submesh_path, FROM_EXTRACT, extensions=(".elem",)
                )

        with caplog.at_level(logging.WARNING):
            run(paths, NoPts(), simplify=False)

        assert "No .pts" in caplog.text


class TestLonBackup:
    @pytest.fixture
    def with_existing_fibres(self, paths):
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.output_mesh_base.lon.write_text(REAL_FIBRES)
        return paths

    def test_existing_lon_is_moved_aside(self, with_existing_fibres, relabel_calls):
        paths = with_existing_fibres

        run(paths, FakeMeshtool(), simplify=False)

        backups = [
            f
            for f in paths.output_dir.iterdir()
            if f.name.startswith("myocardium_clean.lon.") and f.suffix == ".bak"
        ]
        assert len(backups) == 1
        assert backups[0].read_text() == REAL_FIBRES

    def test_the_new_lon_replaces_it(self, with_existing_fibres, relabel_calls):
        paths = with_existing_fibres

        run(paths, FakeMeshtool(), simplify=False)

        assert paths.output_mesh_base.lon.read_text() == DEFAULT_FIBRES

    def test_warning_shows_enough_to_judge_the_backup(
        self, with_existing_fibres, relabel_calls, caplog
    ):
        with caplog.at_level(logging.WARNING):
            run(with_existing_fibres, FakeMeshtool(), simplify=False)

        # The point of the preview is that a user can see these are not the
        # 1 0 0 defaults and so the backup is worth keeping.
        assert "0.71 0.20 0.67" in caplog.text
        assert "backed up" in caplog.text

    def test_no_backup_when_nothing_would_be_overwritten(self, paths, relabel_calls):
        run(paths, FakeMeshtool(), simplify=False)

        assert not any(f.suffix == ".bak" for f in paths.output_dir.iterdir())


class TestDottedMeshStems:
    """The whole reason the contract holds handles instead of bare paths."""

    @pytest.fixture
    def dotted_paths(self, tmp_path, raw_mesh):
        output = tmp_path / "output"
        return MeshPostprocessingPaths(
            input_mesh_base=raw_mesh,
            intermediate_myocardium_mesh=CarpMesh(tmp_path / "tmp" / "myo.v2.raw"),
            output_mesh_base=CarpMesh(output / "myocardium.clean.v2"),
        )

    def test_dotted_stem_keeps_all_its_segments(self, dotted_paths, relabel_calls):
        run(dotted_paths, FakeMeshtool(), simplify=False)

        # with_suffix would have truncated these to myocardium.clean.elem and
        # myo.v2.elem. This is the regression the slice exists to prevent.
        assert dotted_paths.output_mesh_base.elem.name == "myocardium.clean.v2.elem"
        assert dotted_paths.output_mesh_base.elem.exists()
        assert dotted_paths.output_mesh_base.pts.exists()

    def test_backup_name_does_not_eat_a_dot_segment(self, dotted_paths, relabel_calls):
        dotted_paths.output_dir.mkdir(parents=True, exist_ok=True)
        dotted_paths.output_mesh_base.lon.write_text(REAL_FIBRES)

        run(dotted_paths, FakeMeshtool(), simplify=False)

        backup = [f for f in dotted_paths.output_dir.iterdir() if f.suffix == ".bak"][0]
        assert backup.name.startswith("myocardium.clean.v2.lon.")

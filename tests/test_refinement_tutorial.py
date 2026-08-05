"""Pins the claims made by `docs/refinement_tutorial.md`.

The tutorial tells an external orchestrator how to drive the refinement flow. Every
API call, filename and behaviour it advertises is exercised here, so the guide fails
loudly rather than rotting quietly when the library moves underneath it.

Only the `meshtool` binary is faked. `LabelManager`, `LabelMapper`, the builder, the
contract, the logic and `relabel_carp_elem_file` are all the real thing — the label
YAML format below is exactly what the tutorial tells users to write.
"""

from pathlib import Path

import pytest

from pycemrg.data import LabelManager, LabelMapper
from pycemrg_model_creation.logic import RefinementLogic, RefinementPathBuilder
from pycemrg_model_creation.meshpaths import CarpMesh

TUTORIAL = Path(__file__).parent.parent / "docs" / "refinement_tutorial.md"

# Reproduced verbatim from the tutorial's "Two label YAML files" section.
SOURCE_LABELS = """\
labels:
  LV_myo: 103
  RV_myo: 104
  LA_myo: 105

groups:
  MYOCARDIUM: [LV_myo, RV_myo, LA_myo]
"""
TARGET_LABELS = SOURCE_LABELS.replace("103", "1").replace("104", "2").replace("105", "3")


def documented_output_tree() -> str:
    """The fenced block in the tutorial showing what lands on disk."""
    blocks = TUTORIAL.read_text().split("```")
    trees = [b for b in blocks if "refined/" in b and "tmp/" in b]
    assert len(trees) == 1, (
        f"expected exactly one output-tree block in {TUTORIAL.name}, found "
        f"{len(trees)}"
    )
    return trees[0]


class TutorialMeshtool:
    """
    Stands in for the binary. Writes structurally valid CARP files, because the
    real relabeller is in play and rejects anything else.
    """

    is_simplify_topology_available = True

    BODIES = {
        ".pts": "2\n0.0 0.0 0.0\n1.0 0.0 0.0\n",
        ".elem": "2\nTt 0 1 2 3 103\nTt 0 1 2 4 104\n",
        ".lon": "1 0 0\n" * 2,
        ".vtk": "# vtk DataFile Version 3.0\n",
    }

    @classmethod
    def _family(cls, base, extensions=(".pts", ".elem", ".lon")):
        for extension in extensions:
            Path(str(base) + extension).write_text(cls.BODIES[extension])

    def extract_mesh(self, input_mesh_path, output_submesh_path, tags, ifmt):
        # The tutorial tells users to hand over stems, never files.
        assert Path(input_mesh_path).suffix == ""
        self._family(output_submesh_path)

    def simplify_topology(self, input_mesh_path, output_mesh_path, ifmt, ofmt):
        self._family(output_mesh_path)

    def convert(self, input_mesh_path, output_mesh_path, ifmt, ofmt):
        self._family(output_mesh_path, extensions=(".vtk",))


@pytest.fixture
def labels(tmp_path):
    (tmp_path / "source_labels.yaml").write_text(SOURCE_LABELS)
    (tmp_path / "target_labels.yaml").write_text(TARGET_LABELS)
    return (
        LabelManager(tmp_path / "source_labels.yaml"),
        LabelManager(tmp_path / "target_labels.yaml"),
    )


@pytest.fixture
def input_mesh(tmp_path):
    """A raw mesh with no .lon, which the tutorial says is normal."""
    case = tmp_path / "case01"
    case.mkdir()
    (case / "heart_mesh.pts").write_text("1\n0.0 0.0 0.0\n")
    (case / "heart_mesh.elem").write_text("1\nTt 0 1 2 3 103\n")
    return CarpMesh(case / "heart_mesh")


@pytest.fixture
def refined(tmp_path, input_mesh, labels):
    """The tutorial's section 2, run end to end."""
    source, target = labels
    builder = RefinementPathBuilder(output_dir=tmp_path / "work")
    paths = builder.build_postprocessing_paths(input_mesh_base=input_mesh)

    RefinementLogic(meshtool_wrapper=TutorialMeshtool()).run_myocardium_postprocessing(
        paths=paths,
        myocardium_tags=source.get_values_from_names(["MYOCARDIUM"]),
        tag_mapping=LabelMapper(
            source=source, target=target
        ).get_source_to_target_mapping(),
        simplify=True,
    )
    return paths


class TestTutorialExists:
    def test_the_document_these_tests_pin_is_present(self):
        # If the tutorial is deleted or moved, these tests are pinning nothing.
        assert TUTORIAL.is_file()


class TestLabelFilesAsDocumented:
    def test_a_group_resolves_to_its_members_tags(self, labels):
        source, _ = labels

        assert source.get_values_from_names(["MYOCARDIUM"]) == [103, 104, 105]

    def test_mapping_is_built_by_matching_label_names(self, labels):
        source, target = labels

        mapping = LabelMapper(
            source=source, target=target
        ).get_source_to_target_mapping()

        assert mapping == {103: 1, 104: 2, 105: 3}

    def test_a_name_missing_from_target_is_ignored(self, tmp_path, labels):
        source, _ = labels
        partial = tmp_path / "partial.yaml"
        partial.write_text("labels:\n  LV_myo: 1\n")

        mapping = LabelMapper(
            source=source, target=LabelManager(partial)
        ).get_source_to_target_mapping()

        # The tutorial warns that this is silent, so it is worth pinning.
        assert mapping == {103: 1}


class TestOrchestrationAsDocumented:
    def test_building_a_contract_touches_no_disk(self, tmp_path, input_mesh):
        builder = RefinementPathBuilder(output_dir=tmp_path / "work")
        builder.build_postprocessing_paths(input_mesh_base=input_mesh)

        assert not (tmp_path / "work").exists()

    @pytest.mark.parametrize("member", ["pts", "elem", "vtk"])
    def test_the_advertised_outputs_exist(self, refined, member):
        assert getattr(refined.output_mesh_base, member).exists()

    def test_the_output_tree_matches_the_documented_one(self, refined):
        produced = {f.name for f in refined.output_dir.iterdir()}

        assert produced == {
            "myocardium_clean.pts",
            "myocardium_clean.elem",
            "myocardium_clean.lon",
            "myocardium_clean.vtk",
        }
        assert refined.output_dir.name == "refined"
        assert refined.tmp_dir.name == "tmp"

    def test_the_documented_filenames_appear_in_the_tutorial(self, refined):
        # Scoped to the fenced tree block on purpose. A bare substring search over
        # the whole document passes on filenames that only appear in prose
        # elsewhere -- the .lon is also named in the backup gotcha, so a search
        # over the full text does not notice it vanishing from the tree.
        tree = documented_output_tree()
        for produced in refined.output_dir.iterdir():
            assert produced.name in tree, (
                f"{produced.name} is produced but is missing from the tutorial's "
                f"directory tree"
            )


class TestCarpMeshAsDocumented:
    MESH = CarpMesh("/data/case01/heart_mesh")

    def test_the_family_members_shown_in_the_guide(self):
        assert str(self.MESH.pts) == "/data/case01/heart_mesh.pts"
        assert str(self.MESH.elem) == "/data/case01/heart_mesh.elem"
        assert str(self.MESH.lon) == "/data/case01/heart_mesh.lon"
        assert str(self.MESH.vtk) == "/data/case01/heart_mesh.vtk"

    def test_identity_helpers(self):
        assert str(self.MESH.directory) == "/data/case01"
        assert self.MESH.name == "heart_mesh"
        assert str(self.MESH) == "/data/case01/heart_mesh"

    def test_constructing_from_a_file_is_refused(self):
        with pytest.raises(ValueError):
            CarpMesh("/data/case01/heart_mesh.pts")

    @pytest.mark.parametrize("form", ["str", "path", "handle"])
    def test_the_builder_accepts_all_three_documented_forms(self, tmp_path, form):
        stem = tmp_path / "case01" / "heart_mesh"
        argument = {
            "str": str(stem),
            "path": stem,
            "handle": CarpMesh(stem),
        }[form]

        paths = RefinementPathBuilder(
            output_dir=tmp_path / "work"
        ).build_postprocessing_paths(input_mesh_base=argument)

        assert paths.input_mesh_base == CarpMesh(stem)


class TestRenamingAsDocumented:
    def test_both_documented_overrides(self, tmp_path, input_mesh):
        paths = RefinementPathBuilder(
            output_dir=tmp_path / "work", refined_subdir="02_refined"
        ).build_postprocessing_paths(
            input_mesh_base=input_mesh, refined_mesh_basename="myo"
        )

        assert paths.output_dir.name == "02_refined"
        assert paths.output_mesh_base.name == "myo"
        assert paths.output_mesh_base.elem.name == "myo.elem"

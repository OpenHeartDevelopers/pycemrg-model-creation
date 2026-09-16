# tests/test_surface_mesh_extraction.py
"""
The submesh-extraction methods of `SurfaceLogic`.

`logic/surfaces.py` has had no unit coverage at all, so the fold that removed
the `tags: TagsConfig` parameter from these five methods had nothing but the
integration suite behind it. These pin the part that fold changed: which
vocabulary the tags come from, and what reaches `meshtool extract mesh`.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pycemrg_model_creation.config import label_manager_from_mapping
from pycemrg_model_creation.logic.contracts import AtrialMeshPaths, BiVMeshPaths
from pycemrg_model_creation.logic.surfaces import (
    SurfaceExtractionError,
    SurfaceLogic,
)
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.types import Chamber


FOUR_CHAMBER_TAGS = {
    "LV": 1, "RV": 2, "LA": 3, "RA": 4,
    "MV": 7, "TV": 8, "AV": 9, "PV": 10, "PArt": 6,
}


@pytest.fixture
def logic():
    return SurfaceLogic(
        meshtool=MagicMock(),
        label_manager=label_manager_from_mapping(FOUR_CHAMBER_TAGS),
    )


def biv_paths(tmp_path: Path) -> BiVMeshPaths:
    return BiVMeshPaths(
        source_mesh=tmp_path / "heart",
        output_mesh=CarpMesh(tmp_path / "BiV" / "BiV"),
        output_dir=tmp_path / "BiV",
        vtx_files_to_map=[],
        renamed_vtx_targets=[],
    )


def atrial_paths(tmp_path: Path) -> AtrialMeshPaths:
    return AtrialMeshPaths(
        source_mesh=tmp_path / "heart",
        output_mesh=tmp_path / "LA" / "LA",
        output_dir=tmp_path / "LA",
        vtx_files_to_map=[],
        mapped_vtx_output_dir=tmp_path / "LA",
        apex_template=tmp_path / "apex.vtx",
        rv_septum_template=tmp_path / "rvsept.vtx",
        apex_output=tmp_path / "LA" / "apex.vtx",
        rv_septum_output=tmp_path / "LA" / "rvsept.vtx",
    )


class TestTagsComeFromTheLabelManager:
    """
    The fold's substance: one vocabulary, injected, not a second contract
    passed per call.
    """

    def test_biv_submesh_asks_the_label_manager_for_lv_and_rv(self, logic, tmp_path):
        logic.extract_biv_submesh(biv_paths(tmp_path))

        tags = logic.meshtool.extract_mesh.call_args.kwargs["tags"]
        assert tags == [1, 2]

    @pytest.mark.parametrize(
        "chamber,expected", [(Chamber.LA, [3]), (Chamber.RA, [4])]
    )
    def test_atrial_submesh_asks_for_its_own_chamber(
        self, logic, tmp_path, chamber, expected
    ):
        logic.extract_atrial_submesh(atrial_paths(tmp_path), chamber)

        tags = logic.meshtool.extract_mesh.call_args.kwargs["tags"]
        assert tags == expected

    def test_a_label_manager_missing_a_name_raises(self, tmp_path):
        """
        Strictness is the point of the fold.

        `TagsConfig` skipped names it did not recognise, so a mesh missing RV
        would have been extracted as LV alone, silently.
        """
        logic = SurfaceLogic(
            meshtool=MagicMock(),
            label_manager=label_manager_from_mapping({"LV": 1}),
        )

        with pytest.raises(SurfaceExtractionError, match="'RV' not found"):
            logic.extract_biv_submesh(biv_paths(tmp_path))

        logic.meshtool.extract_mesh.assert_not_called()


class TestTheTagsParameterIsGone:
    """
    Pinned by signature, because these five are public and an orchestrator
    still passing `tags` positionally would otherwise bind it to the wrong
    argument rather than fail.
    """

    @pytest.mark.parametrize(
        "method_name",
        [
            "extract_biv_submesh",
            "extract_atrial_submesh",
            "run_biv_mesh_extraction",
            "run_atrial_mesh_extraction",
            "run_all",
        ],
    )
    def test_no_extraction_method_takes_tags(self, method_name):
        signature = inspect.signature(getattr(SurfaceLogic, method_name))
        assert "tags" not in signature.parameters

    def test_chamber_is_still_the_second_argument_of_the_atrial_methods(self):
        """
        Removing `tags` shifted `chamber` left. A caller passing it
        positionally must still land on `chamber`.
        """
        for method_name in ("extract_atrial_submesh", "run_atrial_mesh_extraction"):
            parameters = list(
                inspect.signature(getattr(SurfaceLogic, method_name)).parameters
            )
            assert parameters[:3] == ["self", "paths", "chamber"], method_name

"""Unit tests for `meshpaths` — pure path arithmetic, no test data required.

Names used here are taken from real runs recorded in
`.claude/tree_output_of_tests.md` and the step 0 native mguvc log, so the
dotted-stem cases are the ones the library actually encounters.
"""

import os
from pathlib import Path

import pytest

from pycemrg_model_creation.meshpaths import CarpMesh


class TestConstruction:
    def test_accepts_a_string(self):
        assert CarpMesh("/data/BiV/BiV").stem == Path("/data/BiV/BiV")

    def test_is_frozen(self):
        mesh = CarpMesh("/data/BiV/BiV")
        with pytest.raises(Exception):
            mesh.stem = Path("/other")

    def test_equal_stems_compare_equal(self):
        assert CarpMesh("/data/BiV/BiV") == CarpMesh(Path("/data/BiV/BiV"))

    @pytest.mark.parametrize(
        "extension", [".pts", ".elem", ".lon", ".vtk", ".bpts", ".belem"]
    )
    def test_rejects_a_stem_that_is_already_a_file(self, extension):
        with pytest.raises(ValueError, match="without an extension"):
            CarpMesh(f"/data/BiV/BiV{extension}")

    def test_allows_a_dotted_stem(self):
        # .surfmesh and .partN stems are legitimate, not mistakes
        assert CarpMesh("/tmp/epi_endo.surfmesh").name == "epi_endo.surfmesh"


class TestFamily:
    def test_the_carp_triple_and_vtk(self):
        mesh = CarpMesh("/data/BiV/BiV")
        assert mesh.pts == Path("/data/BiV/BiV.pts")
        assert mesh.elem == Path("/data/BiV/BiV.elem")
        assert mesh.lon == Path("/data/BiV/BiV.lon")
        assert mesh.vtk == Path("/data/BiV/BiV.vtk")

    def test_binary_variants(self):
        mesh = CarpMesh("/data/BiV/BiV")
        assert mesh.bpts == Path("/data/BiV/BiV.bpts")
        assert mesh.belem == Path("/data/BiV/BiV.belem")

    def test_name_and_directory(self):
        mesh = CarpMesh("/data/BiV/BiV")
        assert mesh.name == "BiV"
        assert mesh.directory == Path("/data/BiV")


class TestDottedStemsAreNotTruncated:
    """The reason this class exists instead of `Path.with_suffix`."""

    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("/tmp/epi_endo_cc.part0", "epi_endo_cc.part0.pts"),
            ("/tmp/septum_cc.part1", "septum_cc.part1.pts"),
            ("/tmp/epi_endo.surfmesh", "epi_endo.surfmesh.pts"),
            ("/data/BiV/biv.base", "biv.base.pts"),
        ],
    )
    def test_extension_is_appended_not_substituted(self, stem, expected):
        assert CarpMesh(stem).pts.name == expected

    def test_with_suffix_would_have_truncated(self):
        # Guards the intent: if this ever stops differing, the class is moot.
        stem = Path("/tmp/epi_endo_cc.part0")
        assert Path(stem).with_suffix(".pts").name == "epi_endo_cc.pts"
        assert CarpMesh(stem).pts.name == "epi_endo_cc.part0.pts"


class TestRole:
    @pytest.mark.parametrize(
        "role, expected",
        [
            ("base", "BiV.base.vtx"),
            ("lvendo", "BiV.lvendo.vtx"),
            ("rvendo", "BiV.rvendo.vtx"),
            ("rvsept", "BiV.rvsept.vtx"),
        ],
    )
    def test_the_four_boundary_sets_mguvc_reads(self, role, expected):
        assert CarpMesh("/data/BiV/BiV").role(role, ".vtx").name == expected

    @pytest.mark.parametrize(
        "role, extension, expected",
        [
            ("lvfree_face", ".surf", "BiV.lvfree_face.surf"),
            ("rvjuncant_face", ".surf", "BiV.rvjuncant_face.surf"),
            ("epi", ".surf.vtx", "BiV.epi.surf.vtx"),
            ("rvendo_nosept", ".surf.vtx", "BiV.rvendo_nosept.surf.vtx"),
            ("uvc_z", ".dat", "BiV.uvc_z.dat"),
            ("sol_apba_lap", ".dat", "BiV.sol_apba_lap.dat"),
            ("etags", ".sh", "BiV.etags.sh"),
        ],
    )
    def test_names_mguvc_actually_emits(self, role, extension, expected):
        assert CarpMesh("/data/BiV/BiV").role(role, extension).name == expected

    def test_role_on_a_dotted_stem(self):
        mesh = CarpMesh("/tmp/epi_endo.surfmesh")
        assert mesh.role("base", ".vtx").name == "epi_endo.surfmesh.base.vtx"

    def test_role_lands_in_the_mesh_directory(self):
        mesh = CarpMesh("/data/BiV/BiV")
        assert mesh.role("base", ".vtx").parent == Path("/data/BiV")

    @pytest.mark.parametrize("bad_role", ["", ".base"])
    def test_rejects_a_bad_role(self, bad_role):
        with pytest.raises(ValueError, match="Role name"):
            CarpMesh("/data/BiV/BiV").role(bad_role, ".vtx")

    def test_rejects_an_extension_without_a_dot(self):
        with pytest.raises(ValueError, match="must start with a dot"):
            CarpMesh("/data/BiV/BiV").role("base", "vtx")


class TestInterop:
    def test_str_and_fspath_yield_the_stem(self):
        # meshtool and the CARPentry binaries take the stem on the command line
        mesh = CarpMesh("/data/BiV/BiV")
        assert str(mesh) == "/data/BiV/BiV"
        assert os.fspath(mesh) == "/data/BiV/BiV"

    def test_relative_stems_keep_their_directory(self):
        assert CarpMesh("sub/BiV").pts == Path("sub/BiV.pts")


class TestPurity:
    def test_touches_no_filesystem(self, tmp_path):
        mesh = CarpMesh(tmp_path / "absent")
        assert not mesh.pts.exists()
        assert mesh.pts == tmp_path / "absent.pts"
        assert list(tmp_path.iterdir()) == []

"""Unit tests for `meshpaths` — pure path arithmetic, no test data required.

Names used here are taken from real runs recorded in
`.claude/tree_output_of_tests.md` and the step 0 native mguvc log, so the
dotted-stem cases are the ones the library actually encounters.
"""

import os
from pathlib import Path

import pytest

from pycemrg_model_creation.meshpaths import (
    CarpMesh,
    CarpSurface,
    SubmeshIndex,
    SurfaceMesh,
)


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


# --- Surfaces -------------------------------------------------------------
#
# Every name below was observed in `.claude/tree_output_of_tests.md`, from the
# recorded ventricular surface-extraction run.

SURFACE = CarpSurface("/data/BiV/tmp/septum")


class TestSurfaceConstruction:
    def test_accepts_a_string_and_a_path(self):
        assert CarpSurface("/t/septum").stem == Path("/t/septum")
        assert CarpSurface(Path("/t/septum")).stem == Path("/t/septum")

    def test_is_frozen(self):
        with pytest.raises(Exception):
            SURFACE.stem = Path("/elsewhere")

    @pytest.mark.parametrize("extension", [".surf", ".surfmesh", ".neubc", ".vtx"])
    def test_a_concrete_file_is_refused(self, extension):
        # `.vtx` is refused too: the companion is <stem>.surf.vtx, so a caller
        # holding a .vtx is one level too deep.
        with pytest.raises(ValueError, match="without an extension"):
            CarpSurface(f"/t/septum{extension}")


class TestSurfaceFamily:
    def test_the_names_meshtool_writes(self):
        assert SURFACE.surf == Path("/data/BiV/tmp/septum.surf")
        assert SURFACE.neubc == Path("/data/BiV/tmp/septum.neubc")

    def test_the_vtx_companion_carries_the_doubled_extension(self):
        # septum.surf.vtx, never septum.vtx. This is the case with_suffix
        # silently gets wrong, and the reason this handle exists.
        assert SURFACE.surf_vtx == Path("/data/BiV/tmp/septum.surf.vtx")
        assert SURFACE.surf_vtx.name != "septum.vtx"

    def test_identity_helpers(self):
        assert SURFACE.name == "septum"
        assert SURFACE.directory == Path("/data/BiV/tmp")


class TestSurfaceMeshCompanion:
    def test_is_reached_through_the_surface(self):
        assert isinstance(SURFACE.surfmesh, SurfaceMesh)
        assert str(SURFACE.surfmesh) == "/data/BiV/tmp/septum.surfmesh"

    def test_the_three_extensions_real_runs_produce(self):
        assert SURFACE.surfmesh.vtk == Path("/data/BiV/tmp/septum.surfmesh.vtk")
        assert SURFACE.surfmesh.nod == Path("/data/BiV/tmp/septum.surfmesh.nod")
        assert SURFACE.surfmesh.fcon == Path("/data/BiV/tmp/septum.surfmesh.fcon")

    @pytest.mark.parametrize("absent", ["pts", "elem", "lon", "bpts", "belem"])
    def test_does_not_advertise_carp_mesh_members(self, absent):
        # A .surfmesh is not a CARP mesh. No recorded run has ever produced
        # septum.surfmesh.pts, so exposing one would be a contract that lies.
        assert not hasattr(SURFACE.surfmesh, absent)

    def test_is_not_a_carp_mesh(self):
        assert not isinstance(SURFACE.surfmesh, CarpMesh)


class TestSurfaceDottedStemsAreNotTruncated:
    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("septum_cc.part0", "septum_cc.part0.surf"),
            ("lv_epi_intermediate", "lv_epi_intermediate.surf"),
            ("myocardium.clean.v2", "myocardium.clean.v2.surf"),
        ],
    )
    def test_every_dot_segment_survives(self, stem, expected):
        assert CarpSurface(f"/t/{stem}").surf.name == expected

    def test_a_surfmesh_stem_is_not_a_surface_stem(self):
        # In the recorded runs `epi_endo` is the surface — epi_endo.surf,
        # epi_endo.neubc — and epi_endo.surfmesh is its companion. Reaching the
        # companion goes through the surface, never by constructing it directly.
        with pytest.raises(ValueError, match="without an extension"):
            CarpSurface("/t/epi_endo.surfmesh")

        assert CarpSurface("/t/epi_endo").surfmesh.vtk.name == "epi_endo.surfmesh.vtk"

    def test_with_suffix_would_have_truncated(self):
        # Pins the bug rather than merely the fix.
        assert Path("/t/septum_cc.part0").with_suffix(".surf").name == "septum_cc.surf"
        assert CarpSurface("/t/septum_cc.part0").surf.name == "septum_cc.part0.surf"


class TestSurfaceInterop:
    def test_is_os_pathlike_and_yields_the_stem(self):
        assert os.fspath(SURFACE) == "/data/BiV/tmp/septum"
        assert str(SURFACE) == "/data/BiV/tmp/septum"

    def test_touches_no_filesystem(self, tmp_path):
        surface = CarpSurface(tmp_path / "absent")
        assert not surface.surf.exists()
        assert not surface.surfmesh.vtk.exists()
        assert list(tmp_path.iterdir()) == []


# --- Submesh index --------------------------------------------------------
#
# `.nod` maps a submesh's nodes back to its parent, `.eidx` its elements. Every
# stem below was observed in `.claude/tree_output_of_tests.md`: the `.partN`
# components in tmp/, and the renamed BiV surfaces they become.


class TestSubmeshIndexConstruction:
    def test_accepts_a_string_and_a_path(self):
        assert SubmeshIndex("/t/biv_epi").stem == Path("/t/biv_epi")
        assert SubmeshIndex(Path("/t/biv_epi")).stem == Path("/t/biv_epi")

    def test_is_frozen(self):
        index = SubmeshIndex("/t/biv_epi")
        with pytest.raises(Exception):
            index.stem = Path("/elsewhere")

    def test_equal_stems_compare_equal(self):
        assert SubmeshIndex("/t/biv_epi") == SubmeshIndex(Path("/t/biv_epi"))

    @pytest.mark.parametrize("extension", [".nod", ".eidx"])
    def test_a_concrete_file_is_refused(self, extension):
        with pytest.raises(ValueError, match="without an extension"):
            SubmeshIndex(f"/t/biv_epi{extension}")

    def test_allows_a_partn_stem(self):
        # `.part0` is a real stem segment, not an extension to be stripped.
        assert SubmeshIndex("/t/epi_endo_cc.part0").stem.name == "epi_endo_cc.part0"


class TestSubmeshIndexPair:
    @pytest.mark.parametrize(
        "stem, nod, eidx",
        [
            ("/t/biv_epi", "biv_epi.nod", "biv_epi.eidx"),
            ("/t/biv_lvendo", "biv_lvendo.nod", "biv_lvendo.eidx"),
            ("/t/biv_septum", "biv_septum.nod", "biv_septum.eidx"),
            (
                "/t/lv_epi_intermediate",
                "lv_epi_intermediate.nod",
                "lv_epi_intermediate.eidx",
            ),
        ],
    )
    def test_the_names_real_runs_produce(self, stem, nod, eidx):
        index = SubmeshIndex(stem)
        assert index.nod.name == nod
        assert index.eidx.name == eidx

    def test_the_pair_lands_beside_the_stem(self):
        index = SubmeshIndex("/data/BiV/biv_epi")
        assert index.nod.parent == Path("/data/BiV")
        assert index.eidx.parent == Path("/data/BiV")


class TestSubmeshIndexDottedStemsAreNotTruncated:
    @pytest.mark.parametrize(
        "stem, expected_eidx",
        [
            ("epi_endo_cc.part0", "epi_endo_cc.part0.eidx"),
            ("epi_endo_cc.part2", "epi_endo_cc.part2.eidx"),
            ("septum_cc.part1", "septum_cc.part1.eidx"),
        ],
    )
    def test_every_dot_segment_survives(self, stem, expected_eidx):
        assert SubmeshIndex(f"/t/{stem}").eidx.name == expected_eidx

    def test_with_suffix_would_have_truncated(self):
        # Pins the bug, not merely the fix. This is the shape of the latent
        # defect at utilities/mesh.py:225-226.
        stem = Path("/t/epi_endo_cc.part0")
        assert stem.with_suffix(".eidx").name == "epi_endo_cc.eidx"
        assert SubmeshIndex(stem).eidx.name == "epi_endo_cc.part0.eidx"


class TestSubmeshIndexIsReachedThroughCarpMesh:
    def test_a_mesh_offers_its_index(self):
        mesh = CarpMesh("/data/BiV/biv_epi")
        assert isinstance(mesh.index, SubmeshIndex)
        assert mesh.index.nod == Path("/data/BiV/biv_epi.nod")
        assert mesh.index.eidx == Path("/data/BiV/biv_epi.eidx")

    def test_the_index_shares_the_meshs_stem(self):
        mesh = CarpMesh("/t/septum_cc.part1")
        assert mesh.index.stem == mesh.stem
        assert mesh.index.eidx.name == "septum_cc.part1.eidx"

    @pytest.mark.parametrize("absent", ["nod", "eidx"])
    def test_the_mesh_does_not_advertise_them_directly(self, absent):
        # A trunk mesh was never carved out of anything and has no index pair.
        # Reaching them goes through `.index`, which names the precondition.
        assert not hasattr(CarpMesh("/data/BiV/BiV"), absent)


class TestSurfaceMeshShareTheNodOwner:
    def test_nod_delegates_rather_than_restating(self, monkeypatch):
        # The constraint this handle exists to satisfy: one definition of
        # where `.nod` sits, not two that can drift apart.
        #
        # Comparing values cannot tell delegation from a second, identical
        # implementation — both spell the same path. So redefine the owner and
        # require the surfmesh to follow it. A restated `_sibling(".nod")` here
        # would ignore this and keep returning the real name.
        monkeypatch.setattr(
            SubmeshIndex, "nod", property(lambda self: Path(f"{self.stem}.SENTINEL"))
        )
        assert SURFACE.surfmesh.nod == Path("/data/BiV/tmp/septum.surfmesh.SENTINEL")

    def test_nod_still_spells_the_recorded_name(self):
        assert SURFACE.surfmesh.nod.name == "septum.surfmesh.nod"

    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("/t/base", "base.surfmesh.nod"),
            ("/t/epi_endo", "epi_endo.surfmesh.nod"),
            ("/t/septum", "septum.surfmesh.nod"),
        ],
    )
    def test_the_three_surfmesh_nods_real_runs_produce(self, stem, expected):
        assert CarpSurface(stem).surfmesh.nod.name == expected

    def test_a_surfmesh_has_no_eidx(self):
        # No recorded run produced one. Exposing the pair here would advertise
        # a file that is never written.
        assert not hasattr(SURFACE.surfmesh, "eidx")
        assert not hasattr(SURFACE.surfmesh, "index")


class TestSubmeshIndexInterop:
    def test_is_os_pathlike_and_yields_the_stem(self):
        index = SubmeshIndex("/data/BiV/biv_epi")
        assert os.fspath(index) == "/data/BiV/biv_epi"
        assert str(index) == "/data/BiV/biv_epi"

    def test_touches_no_filesystem(self, tmp_path):
        index = SubmeshIndex(tmp_path / "absent")
        assert not index.nod.exists()
        assert not index.eidx.exists()
        assert list(tmp_path.iterdir()) == []

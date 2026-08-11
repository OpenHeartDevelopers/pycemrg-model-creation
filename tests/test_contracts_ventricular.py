"""Unit tests for `VentricularSurfacePaths` — pure path arithmetic, no data.

`logic/` had no unit coverage at all, which is why reshaping this contract
broke `build_ventricular_paths` without a single test noticing. These tests
close that specific hole: they pin the names the contract derives, and they
pin that the builder and the contract still agree about them.

The expected strings below are the literals `builders.py` used to carry before
the nine derived names moved into the contract. They are what real runs have
on disk (see `.claude/tree_output_of_tests.md`), so a change here is a change
to the output tree, not a refactor.
"""

from pathlib import Path

import pytest

from pycemrg_model_creation.logic.builders import ModelCreationPathBuilder
from pycemrg_model_creation.logic.contracts import VentricularSurfacePaths
from pycemrg_model_creation.meshpaths import CarpMesh

ROOT = Path("/out")
BIV = ROOT / "BiV"
TMP = BIV / "tmp"
MESH_STEM = Path("/in/heart")


def make_paths(output_dir: Path = BIV) -> VentricularSurfacePaths:
    """A contract with only its real fields supplied."""
    return VentricularSurfacePaths(
        mesh=CarpMesh(MESH_STEM),
        output_dir=output_dir,
        base_vtx=output_dir / "biv.base.vtx",
        epi_vtx=output_dir / "biv.epi.vtx",
        lv_endo_vtx=output_dir / "biv.lvendo.vtx",
        rv_endo_vtx=output_dir / "biv.rvendo.vtx",
        septum_vtx=output_dir / "biv.septum.vtx",
        rv_septum_point_vtx=output_dir / "biv.rvsept_pt.vtx",
    )


class TestDerivedNames:
    """The nine names that became properties. Values are the on-disk truth."""

    @pytest.mark.parametrize(
        "attribute, expected",
        [
            ("tmp_dir", TMP),
            # Intermediate surfaces, in tmp/
            ("base_surface", TMP / "base"),
            ("epi_endo_combined", TMP / "epi_endo"),
            ("septum_raw", TMP / "septum"),
            ("lv_epi_intermediate", TMP / "lv_epi_intermediate"),
            # Final surfaces, in the chamber directory
            ("epi_surface", BIV / "biv_epi"),
            ("lv_endo_surface", BIV / "biv_lvendo"),
            ("rv_endo_surface", BIV / "biv_rvendo"),
            ("septum_surface", BIV / "biv_septum"),
        ],
    )
    def test_the_name_real_runs_produce(self, attribute, expected):
        assert getattr(make_paths(), attribute) == expected

    def test_surfaces_are_stems_not_files(self):
        # These are handed to CarpSurface/CarpMesh, which refuse a stem that
        # already carries an extension. A suffix here would break them.
        paths = make_paths()
        for attribute in (
            "epi_surface",
            "lv_endo_surface",
            "rv_endo_surface",
            "septum_surface",
            "base_surface",
            "epi_endo_combined",
            "septum_raw",
            "lv_epi_intermediate",
        ):
            assert getattr(paths, attribute).suffix == "", attribute


class TestNamesAreDerivedNotCoincidental:
    """
    Equality against a constant cannot tell a derived name from a hardcoded
    one — both spell the same path. Move `output_dir` and require every
    derived name to follow it.
    """

    @pytest.mark.parametrize(
        "attribute",
        [
            "tmp_dir",
            "base_surface",
            "epi_endo_combined",
            "septum_raw",
            "lv_epi_intermediate",
            "epi_surface",
            "lv_endo_surface",
            "rv_endo_surface",
            "septum_surface",
        ],
    )
    def test_every_derived_name_follows_output_dir(self, attribute):
        elsewhere = Path("/somewhere/else/BiV")
        moved = getattr(make_paths(output_dir=elsewhere), attribute)
        assert elsewhere in moved.parents or moved == elsewhere

    def test_tmp_dir_sits_under_the_output_dir(self):
        paths = make_paths()
        assert paths.tmp_dir.parent == paths.output_dir
        assert paths.tmp_dir.name == "tmp"


class TestContractDiscipline:
    def test_is_frozen(self):
        # A derived name that could be reassigned defeats the point.
        with pytest.raises(Exception):
            make_paths().output_dir = Path("/elsewhere")

    @pytest.mark.parametrize(
        "removed_name",
        [
            "tmp_dir",  # removed in slice 3a step 5
            "base_surface",
            "epi_endo_combined",
            "septum_raw",
            "lv_epi_intermediate",
            "epi_surface",
            "lv_endo_surface",
            "rv_endo_surface",
            "septum_surface",
            "epi_endo_cc_base",  # became properties after step 5
            "septum_cc_base",
        ],
    )
    def test_a_derived_name_is_not_a_constructor_argument(self, removed_name):
        # Each derived name must be rejected *individually*. Passing them all
        # in one call would raise on the first and prove nothing about the
        # rest — which is how this test read before `epi_endo_cc_base` and
        # `septum_cc_base` joined the list.
        with pytest.raises(TypeError):
            VentricularSurfacePaths(
                mesh=CarpMesh(MESH_STEM),
                output_dir=BIV,
                base_vtx=BIV / "biv.base.vtx",
                epi_vtx=BIV / "biv.epi.vtx",
                lv_endo_vtx=BIV / "biv.lvendo.vtx",
                rv_endo_vtx=BIV / "biv.rvendo.vtx",
                septum_vtx=BIV / "biv.septum.vtx",
                rv_septum_point_vtx=BIV / "biv.rvsept_pt.vtx",
                **{removed_name: TMP},
            )

    def test_constructing_touches_no_disk(self, tmp_path):
        paths = make_paths(output_dir=tmp_path / "BiV")
        assert not paths.output_dir.exists()
        assert not paths.tmp_dir.exists()
        assert list(tmp_path.iterdir()) == []


class TestBuilderAndContractAgree:
    """
    The builder no longer restates the derived names, so nothing but a test
    checks that what it *does* supply still yields the same tree.
    """

    def test_every_path_matches_the_pre_refactor_layout(self, tmp_path):
        builder = ModelCreationPathBuilder(tmp_path)
        paths = builder.build_ventricular_paths(Path("/in/heart"))

        biv, tmp = tmp_path / "BiV", tmp_path / "BiV" / "tmp"
        expected = {
            "output_dir": biv,
            "tmp_dir": tmp,
            "base_surface": tmp / "base",
            "epi_endo_combined": tmp / "epi_endo",
            "septum_raw": tmp / "septum",
            "lv_epi_intermediate": tmp / "lv_epi_intermediate",
            "epi_endo_cc_base": tmp / "epi_endo_cc",
            "septum_cc_base": tmp / "septum_cc",
            "epi_surface": biv / "biv_epi",
            "lv_endo_surface": biv / "biv_lvendo",
            "rv_endo_surface": biv / "biv_rvendo",
            "septum_surface": biv / "biv_septum",
            "base_vtx": biv / "biv.base.vtx",
            "epi_vtx": biv / "biv.epi.vtx",
            "lv_endo_vtx": biv / "biv.lvendo.vtx",
            "rv_endo_vtx": biv / "biv.rvendo.vtx",
            "septum_vtx": biv / "biv.septum.vtx",
            "rv_septum_point_vtx": biv / "biv.rvsept_pt.vtx",
        }
        actual = {name: getattr(paths, name) for name in expected}
        assert actual == expected

    def test_building_the_contract_does_not_create_the_biv_tree(self, tmp_path):
        # Naming must not touch disk. `SurfaceLogic._prepare_directories`
        # creates these when the workflow runs.
        builder = ModelCreationPathBuilder(tmp_path)
        builder.build_ventricular_paths(Path("/in/heart"))

        assert not (tmp_path / "BiV").exists()
        assert not (tmp_path / "BiV" / "tmp").exists()

    def test_la_and_ra_are_still_created_eagerly(self, tmp_path):
        # Deliberate, not an oversight: the atrial flow has nowhere else to
        # do this until it is refactored. Pinned so the asymmetry is visible
        # rather than surprising, and so removing it is a conscious edit.
        ModelCreationPathBuilder(tmp_path)

        for chamber in ("LA", "RA"):
            assert (tmp_path / chamber).is_dir()
            assert (tmp_path / chamber / "tmp").is_dir()


class TestTheMeshIsAHandle:
    """
    `mesh` became a `CarpMesh`, and the layout test above deliberately does
    not cover it — every name it checks is derived from `output_dir`, not
    from the mesh. Without these, the builder could hand the contract a bare
    `Path` and nothing would notice: dataclasses do no runtime type checking.
    """

    def test_the_builder_yields_a_handle(self, tmp_path):
        paths = ModelCreationPathBuilder(tmp_path).build_ventricular_paths(MESH_STEM)

        assert isinstance(paths.mesh, CarpMesh)

    def test_the_stem_keeps_its_directory(self, tmp_path):
        # The regression that matters. `_StemHandle.stem` is the *full* path
        # without extensions; `Path.stem` is the bare basename. The six
        # wrapper sites in `surfaces.py` pass `paths.mesh.stem`, so anyone
        # "simplifying" that to `Path.stem` semantics would send `heart`
        # where `/in/heart` belongs — and meshtool would read nothing.
        paths = ModelCreationPathBuilder(tmp_path).build_ventricular_paths(MESH_STEM)

        assert paths.mesh.stem == MESH_STEM
        assert paths.mesh.stem != Path(MESH_STEM.name)

    def test_a_handle_survives_being_passed_in(self, tmp_path):
        # `build_all` passes a `Path`, but a library user holding a handle
        # should not have to unwrap it. Re-wrapping must be idempotent.
        builder = ModelCreationPathBuilder(tmp_path)

        assert builder.build_ventricular_paths(
            CarpMesh(MESH_STEM)
        ).mesh.stem == builder.build_ventricular_paths(MESH_STEM).mesh.stem

    def test_a_stem_carrying_an_extension_is_refused(self, tmp_path):
        # New reachable behaviour: before the conversion this sailed through
        # and produced `heart.pts.pts` downstream.
        builder = ModelCreationPathBuilder(tmp_path)

        with pytest.raises(ValueError, match="without an extension"):
            builder.build_ventricular_paths(Path("/in/heart.pts"))

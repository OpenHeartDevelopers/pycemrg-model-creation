"""Unit tests for `tools.mguvc` — pure name prediction, no test data required.

Every name asserted here was observed in the native mguvc run recorded by
`scripts/integration/run_mguvc_native.sh` (exit 0). If mguvc's naming changes,
these fail rather than the pipeline failing later and further away.
"""

from pathlib import Path

import pytest

from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools.mguvc import (
    MguvcOutputs,
    UvcBoundary,
    boundary_inputs,
    outputs_for,
)

MESH = CarpMesh("/data/surfaces_uvc/BiV/BiV")
UVC_DIR = Path("/data/surfaces_uvc/BiV/uvc")


class TestUvcBoundary:
    def test_has_exactly_the_four_roles_mguvc_reads(self):
        assert {b.value for b in UvcBoundary} == {
            "base",
            "lvendo",
            "rvendo",
            "rvsept",
        }

    @pytest.mark.parametrize(
        "derived", ["epi", "lvepi", "rvendo_nosept", "rvjunc", "tissue"]
    )
    def test_surfaces_mguvc_derives_are_not_inputs(self, derived):
        # These exist in the test data and in the output, but mguvc never
        # reads them. Adding one here would resurrect the over-validation bug.
        assert derived not in {b.value for b in UvcBoundary}


class TestBoundaryInputs:
    def test_names_are_stem_prefixed_and_beside_the_mesh(self):
        paths = boundary_inputs(MESH)

        assert paths[UvcBoundary.BASE] == MESH.directory / "BiV.base.vtx"
        assert paths[UvcBoundary.LV_ENDO] == MESH.directory / "BiV.lvendo.vtx"
        assert paths[UvcBoundary.RV_ENDO] == MESH.directory / "BiV.rvendo.vtx"
        assert paths[UvcBoundary.RV_SEPT] == MESH.directory / "BiV.rvsept.vtx"

    def test_covers_every_role(self):
        assert set(boundary_inputs(MESH)) == set(UvcBoundary)

    def test_is_not_the_bare_name_form(self):
        # mguvc wants BiV.base.vtx, not base.vtx. This was wrong in the docs
        # for a long time; pin it.
        assert boundary_inputs(MESH)[UvcBoundary.BASE].name != "base.vtx"


class TestOutputNames:
    @pytest.mark.parametrize(
        "attribute, expected",
        [
            ("uvc_z", "BiV.uvc_z.dat"),
            ("uvc_rho", "BiV.uvc_rho.dat"),
            ("uvc_phi", "BiV.uvc_phi.dat"),
            ("uvc_ven", "BiV.uvc_ven.dat"),
            ("sol_apba", "BiV.sol_apba_lap.dat"),
            ("sol_endoepi", "BiV.sol_endoepi_lap.dat"),
            ("sol_lvendo", "BiV.sol_lvendo_lap.dat"),
            ("sol_rvendo", "BiV.sol_rvendo_lap.dat"),
            ("aff_dat", "BiV.aff.dat"),
            ("m2s_dat", "BiV.m2s.dat"),
        ],
    )
    def test_matches_the_recorded_run(self, attribute, expected):
        assert getattr(outputs_for(MESH, UVC_DIR), attribute).name == expected

    def test_outputs_land_in_the_output_dir_not_beside_the_mesh(self):
        outputs = outputs_for(MESH, UVC_DIR)

        assert outputs.uvc_z.parent == UVC_DIR
        assert outputs.uvc_z.parent != MESH.directory

    def test_stem_follows_the_mesh_not_the_output_dir(self):
        # Output dir is "uvc"; the files are still called BiV.*
        assert outputs_for(MESH, UVC_DIR).uvc_z.name.startswith("BiV.")

    def test_accepts_a_string_output_dir(self):
        assert outputs_for(MESH, str(UVC_DIR)).uvc_z.parent == UVC_DIR


class TestGroupings:
    def test_coordinates_are_the_four_uvc_fields(self):
        outputs = outputs_for(MESH, UVC_DIR)
        assert outputs.coordinates == (
            outputs.uvc_z,
            outputs.uvc_rho,
            outputs.uvc_phi,
            outputs.uvc_ven,
        )

    def test_laplace_solutions_are_the_four_sol_files(self):
        outputs = outputs_for(MESH, UVC_DIR)
        assert outputs.laplace_solutions == (
            outputs.sol_apba,
            outputs.sol_endoepi,
            outputs.sol_lvendo,
            outputs.sol_rvendo,
        )

    def test_mappings_are_aff_and_m2s(self):
        outputs = outputs_for(MESH, UVC_DIR)
        assert outputs.mappings == (outputs.aff_dat, outputs.m2s_dat)

    def test_the_ten_modelled_outputs_are_all_distinct(self):
        outputs = outputs_for(MESH, UVC_DIR)
        modelled = outputs.coordinates + outputs.laplace_solutions + outputs.mappings
        assert len(set(modelled)) == 10


class TestPurity:
    def test_predicts_without_touching_the_filesystem(self, tmp_path):
        outputs = outputs_for(CarpMesh(tmp_path / "BiV"), tmp_path / "uvc")

        assert outputs.uvc_z == tmp_path / "uvc" / "BiV.uvc_z.dat"
        assert not outputs.uvc_z.exists()
        assert list(tmp_path.iterdir()) == []

    def test_is_frozen(self):
        outputs = outputs_for(MESH, UVC_DIR)
        with pytest.raises(Exception):
            outputs.output_dir = Path("/elsewhere")

    def test_equal_inputs_compare_equal(self):
        assert outputs_for(MESH, UVC_DIR) == MguvcOutputs(MESH, UVC_DIR)


class TestDottedMeshStems:
    def test_a_dotted_mesh_stem_survives_into_the_output_names(self):
        outputs = outputs_for(CarpMesh("/data/heart.v2"), UVC_DIR)
        assert outputs.uvc_z.name == "heart.v2.uvc_z.dat"

"""Unit tests for the CARPentry command wrappers in `tools.wrappers`.

The wrappers had no coverage at all until this file, which is how an argv
defect survived in `gl_elem_centers`: it emitted `"-f 0"` as a *single* argv
element, so the flag and its value reached the binary glued into one
unrecognised token. Nothing here runs a real binary -- `CarpRunner` is a
`MagicMock` and these tests pin the command list handed to it.

The general pin is `test_no_argv_element_packs_a_flag_with_its_value`: an
argv element containing a space is almost always this same defect, and it is
cheap to check for every command rather than one at a time.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pycemrg_model_creation.tools.wrappers import CarpWrapper


@pytest.fixture
def runner():
    return MagicMock()


@pytest.fixture
def carp(runner):
    return CarpWrapper(carp_runner=runner)


def command_from(runner) -> list:
    """The argv list the wrapper handed the runner, as strings."""
    runner.run.assert_called_once()
    return [str(token) for token in runner.run.call_args.args[0]]


MESH = Path("/data/surfaces_uvc/BiV/BiV")
CENTRES = Path("/data/surfaces_uvc/BiV/BiV_elem_centres.pts")


class TestGlElemCenters:
    def test_invokes_the_american_spelling(self, carp, runner):
        # The binary is GlElemCenters. We were already bitten once by
        # GlRuleFibres/GlRuleFibers, so the literal is pinned.
        carp.gl_elem_centers(meshname=MESH, output=CENTRES)

        assert command_from(runner)[0] == "GlElemCenters"

    def test_each_flag_and_value_is_its_own_argv_element(self, carp, runner):
        # The defect this test exists for. Before the fix the list held
        # "-f 0" and "-p 0" as single elements.
        carp.gl_elem_centers(meshname=MESH, output=CENTRES)

        assert command_from(runner) == [
            "GlElemCenters",
            "-m",
            str(MESH),
            "-f",
            "0",
            "-p",
            "0",
            "-o",
            str(CENTRES),
        ]

    def test_non_default_format_and_strategy_travel_as_values(self, carp, runner):
        carp.gl_elem_centers(meshname=MESH, output=CENTRES, fmt="1", pstrat="2")

        command = command_from(runner)
        assert command[command.index("-f") + 1] == "1"
        assert command[command.index("-p") + 1] == "2"

    def test_the_centres_file_is_declared_as_an_expected_output(self, carp, runner):
        # The runner checks it appeared; correct_fibres reads it next.
        carp.gl_elem_centers(meshname=MESH, output=CENTRES)

        assert runner.run.call_args.kwargs["expected_outputs"] == [CENTRES]


class TestGlRuleFibres:
    def test_invokes_the_american_spelling(self, carp, runner):
        carp.gl_rule_fibres(
            mesh_name=MESH,
            uvc_apba=Path("/uvc/BiV.sol_apba_lap.dat"),
            uvc_epi=Path("/uvc/BiV.sol_endoepi_lap.dat"),
            uvc_lv=Path("/uvc/BiV.sol_lvendo_lap.dat"),
            uvc_rv=Path("/uvc/BiV.sol_rvendo_lap.dat"),
            output_name=Path("/data/BiV.lon"),
        )

        assert command_from(runner)[0] == "GlRuleFibers"

    def test_the_four_laplace_solutions_reach_their_own_flags(self, carp, runner):
        # `-e` is the endo-epi Laplace *solution*, not an epicardial surface.
        # Reading the letter as "epi" is what put a false claim in
        # contracts.py once already.
        carp.gl_rule_fibres(
            mesh_name=MESH,
            uvc_apba=Path("/uvc/apba.dat"),
            uvc_epi=Path("/uvc/endoepi.dat"),
            uvc_lv=Path("/uvc/lvendo.dat"),
            uvc_rv=Path("/uvc/rvendo.dat"),
            output_name=Path("/data/BiV.lon"),
        )

        command = command_from(runner)
        for flag, value in (
            ("-a", "/uvc/apba.dat"),
            ("-e", "/uvc/endoepi.dat"),
            ("-l", "/uvc/lvendo.dat"),
            ("-r", "/uvc/rvendo.dat"),
        ):
            assert command[command.index(flag) + 1] == value


class TestArgvIsNeverPacked:
    """One flag, one value, one element each -- across every wrapper here."""

    def _invoke(self, carp, name):
        if name == "gl_elem_centers":
            carp.gl_elem_centers(meshname=MESH, output=CENTRES)
        elif name == "gl_rule_fibres":
            carp.gl_rule_fibres(
                mesh_name=MESH,
                uvc_apba=Path("/uvc/a.dat"),
                uvc_epi=Path("/uvc/e.dat"),
                uvc_lv=Path("/uvc/l.dat"),
                uvc_rv=Path("/uvc/r.dat"),
                output_name=Path("/data/BiV.lon"),
            )

    @pytest.mark.parametrize("command_name", ["gl_elem_centers", "gl_rule_fibres"])
    def test_no_argv_element_packs_a_flag_with_its_value(
        self, carp, runner, command_name
    ):
        self._invoke(carp, command_name)

        packed = [token for token in command_from(runner) if " " in token]
        assert packed == [], f"{command_name} packed flag and value: {packed}"

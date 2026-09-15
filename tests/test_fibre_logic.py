"""Unit tests for `FibreLogic` — the first `logic/` class covered by unit tests.

`logic/uvc.py` and `logic/surfaces.py` have no unit coverage; only the
integration suite exercises them, which is why a green unit run has twice
failed to catch a broken `logic/` change (see `CLAUDE.md`). `CarpWrapper`
here is a plain `unittest.mock.MagicMock`, so these tests never invoke a real
tool — they pin what `FibreLogic` asks `CarpWrapper` to do, and that
validation runs before it asks.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pycemrg_model_creation.logic.contracts import VentricularFibrePaths
from pycemrg_model_creation.logic.fibres import FibreLogic
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools.mguvc import outputs_for


def make_biv_mesh(root: Path) -> CarpMesh:
    mesh_dir = root / "BiV"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh = CarpMesh(mesh_dir / "BiV")
    mesh.pts.write_text("points")
    mesh.elem.write_text("elements")
    return mesh


def write_laplace_solutions(mesh: CarpMesh, uvc_output_dir: Path) -> None:
    uvc_output_dir.mkdir(parents=True, exist_ok=True)
    for solution in outputs_for(mesh, uvc_output_dir).laplace_solutions:
        solution.write_text("laplace")


class TestRunVentricularFibreGeneration:
    """What `FibreLogic` asks `CarpWrapper.gl_rule_fibres` to do."""

    def test_calls_gl_rule_fibres_with_the_four_laplace_solutions(self, tmp_path):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        write_laplace_solutions(mesh, uvc_output_dir)
        outputs = outputs_for(mesh, uvc_output_dir)
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)

        wrapper = MagicMock()
        FibreLogic(wrapper).run_ventricular_fibre_generation(paths)

        wrapper.gl_rule_fibres.assert_called_once_with(
            mesh_name=mesh.stem,
            uvc_apba=outputs.sol_apba,
            uvc_epi=outputs.sol_endoepi,
            uvc_lv=outputs.sol_lvendo,
            uvc_rv=outputs.sol_rvendo,
            output_name=mesh.lon,
            angles=None,
            fibre_type="biv",
        )

    def test_passes_through_angles_and_fibre_type(self, tmp_path):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        write_laplace_solutions(mesh, uvc_output_dir)
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)
        angles = {"alpha_endo": 50, "alpha_epi": -50, "beta_endo": -60, "beta_epi": 20}

        wrapper = MagicMock()
        FibreLogic(wrapper).run_ventricular_fibre_generation(
            paths, angles=angles, fibre_type="lv"
        )

        call = wrapper.gl_rule_fibres.call_args
        assert call.kwargs["angles"] == angles
        assert call.kwargs["fibre_type"] == "lv"


class TestValidateInputs:
    """Validation must fail before `CarpWrapper` is ever called."""

    def test_missing_mesh_points_raises_before_the_wrapper_runs(self, tmp_path):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        write_laplace_solutions(mesh, uvc_output_dir)
        mesh.pts.unlink()
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)

        wrapper = MagicMock()
        with pytest.raises(FileNotFoundError, match="points"):
            FibreLogic(wrapper).run_ventricular_fibre_generation(paths)

        wrapper.gl_rule_fibres.assert_not_called()

    def test_missing_mesh_elements_raises_before_the_wrapper_runs(self, tmp_path):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        write_laplace_solutions(mesh, uvc_output_dir)
        mesh.elem.unlink()
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)

        wrapper = MagicMock()
        with pytest.raises(FileNotFoundError, match="elements"):
            FibreLogic(wrapper).run_ventricular_fibre_generation(paths)

        wrapper.gl_rule_fibres.assert_not_called()

    @pytest.mark.parametrize(
        "missing_role",
        ["sol_apba", "sol_endoepi", "sol_lvendo", "sol_rvendo"],
    )
    def test_missing_laplace_solution_raises_before_the_wrapper_runs(
        self, tmp_path, missing_role
    ):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        write_laplace_solutions(mesh, uvc_output_dir)
        getattr(outputs_for(mesh, uvc_output_dir), missing_role).unlink()
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)

        wrapper = MagicMock()
        with pytest.raises(FileNotFoundError, match="Laplace solution"):
            FibreLogic(wrapper).run_ventricular_fibre_generation(paths)

        wrapper.gl_rule_fibres.assert_not_called()

    def test_all_four_missing_solutions_are_named_in_one_error(self, tmp_path):
        mesh = make_biv_mesh(tmp_path)
        uvc_output_dir = tmp_path / "BiV" / "uvc"
        # Laplace solutions deliberately not written.
        paths = VentricularFibrePaths(biv_mesh=mesh, uvc_output_dir=uvc_output_dir)

        wrapper = MagicMock()
        with pytest.raises(FileNotFoundError) as excinfo:
            FibreLogic(wrapper).run_ventricular_fibre_generation(paths)

        message = str(excinfo.value)
        for name in ("apico-basal", "endo-epi", "LV endo", "RV endo"):
            assert name in message

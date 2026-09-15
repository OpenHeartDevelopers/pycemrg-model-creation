# src/pycemrg_model_creation/logic/fibres.py
"""
Rule-based ventricular fibre generation logic.

This module orchestrates GlRuleFibres to generate biventricular fibre
orientations from a completed UVC (Universal Ventricular Coordinate) run.
"""

import logging
from typing import Dict, Optional

from pycemrg_model_creation.logic.contracts import VentricularFibrePaths
from pycemrg_model_creation.tools.mguvc import outputs_for
from pycemrg_model_creation.tools.wrappers import CarpWrapper


class FibreLogic:
    """
    Stateless logic for rule-based ventricular fibre generation.

    Orchestrates CARPentry's GlRuleFibres tool to assign fibre orientations
    to a biventricular mesh from the four Laplace solutions a prior UVC run
    already produced. `VentricularFibrePaths` carries the two choices the
    caller actually makes — which mesh, and where the UVC run wrote its
    Laplace solutions. The Laplace inputs and the `.lon` output are asked of
    `tools.mguvc` and the mesh handle rather than restated here.
    """

    def __init__(self, carp_wrapper: CarpWrapper):
        """
        Initialize FibreLogic with dependencies.

        Args:
            carp_wrapper: Initialized CarpWrapper for running GlRuleFibres
        """
        self.carp = carp_wrapper
        self.logger = logging.getLogger(__name__)

    def run_ventricular_fibre_generation(
        self,
        paths: VentricularFibrePaths,
        angles: Optional[Dict[str, float]] = None,
        fibre_type: str = "biv",
    ) -> None:
        """
        Complete ventricular fibre generation workflow.

        Executes the full pipeline:
        1. Validate the mesh and the four Laplace solutions exist
        2. Run GlRuleFibres to assign fibre orientations

        Args:
            paths: Path contract for the fibre generation workflow
            angles: Fibre rotation angles (alpha_endo, alpha_epi, beta_endo,
                    beta_epi). Defaults to `CarpWrapper`'s own module default.
            fibre_type: Fibre type passed to GlRuleFibres (default: "biv")

        Raises:
            FileNotFoundError: If the mesh or a Laplace solution is missing

        Example:
            >>> fibre_logic.run_ventricular_fibre_generation(paths=fibre_paths)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ventricular fibre generation")
        self.logger.info("=" * 60)

        self._validate_inputs(paths)

        outputs = outputs_for(paths.biv_mesh, paths.uvc_output_dir)
        self.carp.gl_rule_fibres(
            mesh_name=paths.biv_mesh.stem,
            uvc_apba=outputs.sol_apba,
            uvc_epi=outputs.sol_endoepi,
            uvc_lv=outputs.sol_lvendo,
            uvc_rv=outputs.sol_rvendo,
            output_name=paths.biv_mesh.lon,
            angles=angles,
            fibre_type=fibre_type,
        )

        self.logger.info("=" * 60)
        self.logger.info("Fibre generation completed successfully")
        self.logger.info(f"Fibres written to: {paths.biv_mesh.lon}")
        self.logger.info("=" * 60)

    def _validate_inputs(self, paths: VentricularFibrePaths) -> None:
        """
        Validate the files GlRuleFibres will read.

        That is the mesh (.pts, .elem) and the four Laplace solutions a
        prior UVC run wrote under `paths.uvc_output_dir`.

        Args:
            paths: VentricularFibrePaths contract

        Raises:
            FileNotFoundError: If any required file is missing
        """
        self.logger.debug("Validating input files")

        mesh = paths.biv_mesh
        for description, mesh_file in (("points", mesh.pts), ("elements", mesh.elem)):
            if not mesh_file.exists():
                raise FileNotFoundError(
                    f"Mesh {description} file not found: {mesh_file}"
                )

        self.logger.debug(f"✓ Mesh files validated: {mesh}")

        outputs = outputs_for(paths.biv_mesh, paths.uvc_output_dir)
        missing_files = []
        for name, solution_path in (
            ("apico-basal", outputs.sol_apba),
            ("endo-epi", outputs.sol_endoepi),
            ("LV endo", outputs.sol_lvendo),
            ("RV endo", outputs.sol_rvendo),
        ):
            if solution_path.exists():
                self.logger.debug(f"✓ {name} Laplace solution: {solution_path.name}")
            else:
                missing_files.append(f"  - {name}: {solution_path}")

        if missing_files:
            raise FileNotFoundError(
                "Required Laplace solution files not found:\n"
                + "\n".join(missing_files)
            )

        self.logger.debug("All input files validated successfully")

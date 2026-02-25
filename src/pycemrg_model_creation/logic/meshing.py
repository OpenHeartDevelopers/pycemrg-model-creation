# src/pycemrg_model_creation/logic/meshing.py
"""
Logic layer for executing volumetric meshing workflows.
"""
import logging
from pathlib import Path
from typing import Dict, Any

from pycemrg_model_creation.logic.contracts import MeshingPaths
from pycemrg_model_creation.tools.wrappers import Meshtools3DWrapper
from pycemrg_model_creation.utilities.config import Meshtools3DParameters
from pycemrg_model_creation.utilities.image import convert_image_to_inr


class MeshingLogic:
    """
    Stateless logic for running the meshtools3d workflow.

    This class orchestrates the steps required to generate a mesh from a NIfTI
    segmentation file, including pre-processing, parameter file generation,
    and execution of the core tool.
    """

    def __init__(self, meshtools3d_wrapper: Meshtools3DWrapper):
        """
        Initializes the MeshingLogic with a meshtools3d wrapper.

        Args:
            meshtools3d_wrapper: An initialized wrapper for the meshtools3d binary.
        """
        self.wrapper = meshtools3d_wrapper
        self.logger = logging.getLogger(__name__)

    def run_meshing(
        self,
        paths: MeshingPaths,
        meshing_params: Dict[str, Any] = None,
        cleanup: bool = True,
    ) -> None:
        """
        Executes the full meshtools3d workflow.

        Args:
            paths: A MeshingPaths contract defining all I/O paths.
            meshing_params: Optional dictionary of parameters to override
                            the meshtools3d defaults. Example:
                            {'meshing': {'facet_size': '0.7'}}
            cleanup: If True, deletes intermediate .inr and .par files.
        """
        self.logger.info("--- Starting meshtools3d meshing workflow ---")

        # 1. Convert NIfTI segmentation to INR format
        convert_image_to_inr(
            nifti_path=paths.input_segmentation_nifti,
            inr_path=paths.intermediate_inr,
        )

        # 2. Create the .par file for meshtools3d
        self._create_par_file(paths, meshing_params)

        # 3. Execute the meshtools3d binary via the wrapper
        expected_outputs = [
            paths.output_mesh_base.with_suffix(".pts"),
            paths.output_mesh_base.with_suffix(".elem"),
        ]
        self.wrapper.run(
            parameter_file=paths.intermediate_parameter_file,
            expected_outputs=expected_outputs,
        )

        # 4. Clean up intermediate files
        if cleanup:
            self.logger.info("Cleaning up intermediate meshing files...")
            try:
                paths.intermediate_inr.unlink()
                paths.intermediate_parameter_file.unlink()
            except FileNotFoundError:
                self.logger.warning("Could not find intermediate files to clean up.")

        self.logger.info("--- Meshtools3D meshing workflow FINISHED ---")

    def _create_par_file(
        self, paths: MeshingPaths, custom_params: Dict[str, Any] = None
    ) -> None:
        """Generates the .par file required by meshtools3d."""
        self.logger.info(
            f"Generating parameter file: {paths.intermediate_parameter_file.name}"
        )
        params = Meshtools3DParameters()

        # Update paths from the contract
        params.update("segmentation", "seg_dir", str(paths.intermediate_inr.parent))
        params.update("segmentation", "seg_name", paths.intermediate_inr.name)
        params.update("output", "outdir", str(paths.output_mesh_base.parent))
        params.update("output", "name", paths.output_mesh_base.name)

        # Apply any user-provided overrides
        if custom_params:
            for section, options in custom_params.items():
                if section in params.config:
                    for option, value in options.items():
                        self.logger.debug(
                            f"Overriding meshing param [{section}]: {option} = {value}"
                        )
                        params.update(section, option, str(value))

        params.save(paths.intermediate_parameter_file)
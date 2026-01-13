# src/pycemrg_model_creation/logic/refinement.py
"""
Logic layer for mesh post-processing and refinement workflows.
"""
import logging
from pathlib import Path
from typing import Dict, List

from pycemrg_model_creation.logic.contracts import MeshPostprocessingPaths
from pycemrg_model_creation.tools.wrappers import MeshtoolWrapper
from pycemrg_model_creation.utilities.mesh import relabel_carp_elem_file


class RefinementLogic:
    """
    Stateless logic for running mesh post-processing workflows.

    This class orchestrates steps like extracting submeshes, simplifying
    topology, and relabeling element tags.
    """

    def __init__(self, meshtool_wrapper: MeshtoolWrapper):
        """
        Initializes the RefinementLogic with a MeshtoolWrapper.

        Args:
            meshtool_wrapper: An initialized wrapper for the meshtool binary.
        """
        self.meshtool = meshtool_wrapper
        self.logger = logging.getLogger(__name__)

    def run_myocardium_postprocessing(
        self,
        paths: MeshPostprocessingPaths,
        myocardium_tags: List[int],
        tag_mapping: Dict[int, int],
        simplify: bool = False,
    ) -> None:
        """
        Executes a post-processing workflow for a raw heart mesh.

        The workflow consists of:
        1. Extracting the myocardium based on a list of element tags.
        2. Optionally, simplifying the mesh topology.
        3. Relabeling the element tags to a new, standardized mapping.

        Args:
            paths: A MeshPostprocessingPaths contract defining all I/O paths.
            myocardium_tags: A list of integer tags to extract for the myocardium.
            tag_mapping: A dictionary mapping {old_tag: new_tag} for relabeling.
            simplify: If True, attempts to run the topology simplification step.
        """
        self.logger.info("--- Starting myocardium post-processing workflow ---")

        # Step 1: Extract the myocardium from the raw mesh
        self.meshtool.extract_mesh(
            input_mesh_path=paths.input_mesh_base,
            output_submesh_path=paths.intermediate_myocardium_mesh,
            tags=myocardium_tags,
            ifmt="carp_txt",
        )

        # Step 2: Optionally, simplify the mesh topology
        # The input for this step is the mesh we just created.
        mesh_to_relabel = paths.intermediate_myocardium_mesh
        if simplify:
            if self.meshtool.is_simplify_topology_available:
                self.logger.info("Simplifying mesh topology.")
                # The output becomes the new input for the relabeling step
                mesh_to_relabel = paths.output_mesh_base
                self.meshtool.simplify_topology(
                    input_mesh_path=paths.intermediate_myocardium_mesh,
                    output_mesh_path=mesh_to_relabel,
                    ifmt="carp_txt",
                    ofmt="carp_txt",
                )
            else:
                self.logger.warning(
                    "Topology simplification requested but tool is not available. Skipping."
                )

        # Step 3: Relabel the element tags to the new standard
        self.logger.info("Relabeling element tags.")
        input_elem_file = mesh_to_relabel.with_suffix(".elem")
        output_elem_file = paths.output_mesh_base.with_suffix(".elem")

        relabel_carp_elem_file(
            input_elem_path=input_elem_file,
            output_elem_path=output_elem_file,
            tag_mapping=tag_mapping,
        )

        # Step 4: If simplification was skipped, we need to copy the other
        # mesh files (.pts, etc.) to the final output location.
        if not (simplify and self.meshtool.is_simplify_topology_available):
            self.logger.info("Copying non-element files to final output location.")
            input_pts_file = paths.intermediate_myocardium_mesh.with_suffix(".pts")
            output_pts_file = paths.output_mesh_base.with_suffix(".pts")
            # This is a simple copy; a more robust implementation could handle other
            # file types like .lon if they exist.
            if input_pts_file.exists():
                import shutil
                shutil.copy(input_pts_file, output_pts_file)
        
        # Step 5: Create a final VTK for visualization
        self.meshtool.convert(
            input_mesh_path=paths.output_mesh_base,
            output_mesh_path=paths.output_mesh_base, # convert will add .vtk suffix
            ifmt="carp_txt",
            ofmt="vtk"
        )

        self.logger.info("--- Myocardium post-processing workflow FINISHED ---")
# src/pycemrg_model_creation/logic/refinement.py
"""
Logic layer for mesh post-processing and refinement workflows.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.logic.contracts import MeshPostprocessingPaths
from pycemrg_model_creation.tools.wrappers import MeshtoolWrapper
from pycemrg_model_creation.utilities.mesh import relabel_carp_elem_file

# How much of an about-to-be-overwritten .lon to show the user. Enough to tell
# meshtool's default [1 0 0] fibres from real ones, without printing a mesh.
_LON_PREVIEW_LINES = 5


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

        self._prepare_directories(paths)

        # Step 1: Extract the myocardium from the raw mesh
        self.meshtool.extract_mesh(
            input_mesh_path=paths.input_mesh_base.stem,
            output_submesh_path=paths.intermediate_myocardium_mesh.stem,
            tags=myocardium_tags,
            ifmt="carp_txt",
        )

        # Step 2: Optionally, simplify the mesh topology
        # The input for this step is the mesh we just created.
        simplified = False
        mesh_to_relabel = paths.intermediate_myocardium_mesh
        if simplify:
            if self.meshtool.is_simplify_topology_available:
                self.logger.info("Simplifying mesh topology.")
                # The output becomes the new input for the relabeling step
                simplified = True
                mesh_to_relabel = paths.output_mesh_base
                self.meshtool.simplify_topology(
                    input_mesh_path=paths.intermediate_myocardium_mesh.stem,
                    output_mesh_path=mesh_to_relabel.stem,
                    ifmt="carp_txt",
                    ofmt="carp_txt",
                )
            else:
                self.logger.warning(
                    "Topology simplification requested but tool is not available. Skipping."
                )

        # Step 3: Relabel the element tags to the new standard.
        # When simplification ran, mesh_to_relabel *is* output_mesh_base, so this
        # rewrites that .elem in place. That is intended, not an oversight.
        self.logger.info("Relabeling element tags.")
        relabel_carp_elem_file(
            input_elem_path=mesh_to_relabel.elem,
            output_elem_path=paths.output_mesh_base.elem,
            tag_mapping=tag_mapping,
        )

        # Step 4: If simplification was skipped, nothing has written the
        # companions of the output mesh, so they are carried over from the
        # intermediate. The .elem is deliberately not copied -- relabelling
        # above already produced it, and a copy would only be overwritten.
        if not simplified:
            self._copy_companions(
                source=paths.intermediate_myocardium_mesh,
                target=paths.output_mesh_base,
            )

        # Step 5: Create a final VTK for visualization.
        # meshtool takes the stem and appends .vtk itself, so the output is
        # named as a stem here, not as paths.output_mesh_base.vtk.
        self.meshtool.convert(
            input_mesh_path=paths.output_mesh_base.stem,
            output_mesh_path=paths.output_mesh_base.stem,
            ifmt="carp_txt",
            ofmt="vtk"
        )

        self.logger.info("--- Myocardium post-processing workflow FINISHED ---")

    # -- Internals ------------------------------------------------------

    def _prepare_directories(self, paths: MeshPostprocessingPaths) -> None:
        """
        Create the directories the workflow writes into.

        This lives here rather than in the path builder so that constructing a
        contract in order to inspect it has no effect on disk. Note the
        contrast with ``UvcLogic._prepare_output_dir``, which *clears* its
        directory because mguvc refuses to write into one that exists;
        meshtool has no such objection, so these are simply created.
        """
        for directory in (paths.tmp_dir, paths.output_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _copy_companions(self, source: CarpMesh, target: CarpMesh) -> None:
        """
        Carry a mesh's companion files across when no tool has written them.

        Copies ``.pts`` and ``.lon``. An existing target ``.lon`` is backed up
        first: it is nearly always meshtool's fabricated ``1 0 0`` default, but
        it may hold real fibre data, and that is not recoverable once
        overwritten.

        Args:
            source: CarpMesh to copy from.
            target: CarpMesh to copy to.
        """
        self.logger.info("Copying companion files to final output location.")

        if source.pts.exists():
            shutil.copy(source.pts, target.pts)
        else:
            self.logger.warning(
                f"No .pts beside {source}; the output mesh will be incomplete."
            )

        if not source.lon.exists():
            return

        if target.lon.exists():
            self._backup_lon(target.lon)
        shutil.copy(source.lon, target.lon)

    def _backup_lon(self, lon_file: Path) -> None:
        """
        Move an existing .lon aside under a timestamped name, and say so loudly.

        The preview matters: five rows of ``1 0 0`` mean meshtool invented the
        fibres and nothing of value was displaced. Anything else means real
        fibre data was in the way and the backup is worth keeping.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # with_name, not with_suffix -- the stem may carry meaningful dots.
        backup = lon_file.with_name(f"{lon_file.name}.{timestamp}.bak")
        shutil.move(str(lon_file), str(backup))

        preview = self._preview(backup)
        self.logger.warning(
            f"An existing fibre file {lon_file.name} was about to be overwritten. "
            f"It has been backed up to {backup.name}. "
            f"First {_LON_PREVIEW_LINES} lines:\n{preview}\n"
            f"If those are rows of '1 0 0' they are meshtool's default fibres and "
            f"the backup can be discarded; otherwise it holds real fibre data."
        )

    def _preview(self, file_path: Path) -> str:
        """Read the first few lines of a file without loading all of it."""
        try:
            with open(file_path) as handle:
                lines = [
                    line.rstrip("\n")
                    for _, line in zip(range(_LON_PREVIEW_LINES), handle)
                ]
        except OSError as error:
            return f"    <could not be read: {error}>"
        return "\n".join(f"    {line}" for line in lines)
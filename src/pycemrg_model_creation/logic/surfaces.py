# src/pycemrg_model_creation/logic/surfaces.py;

"""
Surface extraction logic for UVC (Universal Ventricular Coordinate) generation.

This module implements the scientific workflow for extracting cardiac surfaces
from four-chamber heart meshes. It is stateless and path-agnostic - all paths
are provided explicitly through dataclass contracts.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from pycemrg.data.labels import LabelManager

from pycemrg_model_creation.config import TagsConfig
from pycemrg_model_creation.logic.contracts import (
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths,
)
from pycemrg_model_creation.meshpaths import CarpMesh, CarpSurface
from pycemrg_model_creation.tools import CarpWrapper, MeshtoolWrapper
from pycemrg_model_creation.types import Chamber, SurfaceType
from pycemrg_model_creation.utilities.mesh import ElemType

import pycemrg_model_creation.utilities.mesh as mshu
import pycemrg_model_creation.utilities.geometry as geom


class SurfaceExtractionError(Exception):
    """Base exception for surface extraction errors"""

    pass


class SurfaceIdentificationError(SurfaceExtractionError):
    """Raised when surfaces cannot be automatically identified"""

    pass


class SurfaceLogic:
    """
    Stateless logic for UVC surface extraction workflows.

    This class orchestrates meshtool operations to extract cardiac surfaces
    from four-chamber heart meshes. All file paths are provided explicitly
    through path contract dataclasses.

    Usage:
        meshtool = MeshtoolWrapper.from_system_path()
        logic = SurfaceLogic(meshtool, label_manager)
        logic.run_ventricular_extraction(paths)
    """

    def __init__(self, meshtool: MeshtoolWrapper, label_manager: LabelManager):
        """
        Initialize SurfaceLogic with its collaborators.

        Args:
            meshtool: MeshtoolWrapper instance for executing meshtool commands
            label_manager: Resolves anatomical names ("LV", "MV") to the tag
                values used in the mesh. Reached as `self.labels`.
        """
        self.meshtool = meshtool
        self.labels = label_manager
        self.logger = logging.getLogger(__name__)

    # VENTRICULAR SURFACE EXTRACTION
    def extract_ventricular_base(self, paths: VentricularSurfacePaths) -> None:
        """
        Extract the base surface for ventricles.

        The base is defined as the interface between ventricular myocardium
        and valve planes.

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting ventricular base surface")

            # Get tags for ventricles (LV, RV) and valve planes (MV, TV, AV, PV)
            ventricle_tags = self.labels.get_tags_string(["LV", "RV"])
            valve_tags = self.labels.get_tags_string(["MV", "TV", "AV", "PV"])

            # Extract surface where ventricles meet valves
            op_tag = f"{ventricle_tags}:{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh.stem,
                output_surface_path=paths.base_surface,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            self.logger.info(f"Base surface extracted to {paths.base_surface}")

        except Exception as e:
            msg = f"Failed to extract ventricular base: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_ventricular_surfaces(self, paths: VentricularSurfacePaths) -> None:
        """
        Extract and identify epicardium, LV endocardium, and RV endocardium.

        This method:
        1. Extracts the combined epi/endo surface
        2. Identifies connected components
        3. Uses geometric analysis to identify which is epi vs endo
        4. Uses distance from LV center to distinguish LV endo from RV endo

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceIdentificationError: If surfaces cannot be identified
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting ventricular surfaces (epi, LV endo, RV endo)")

            # Step 1: Extract combined epi/endo surface
            ventricle_tags = self.labels.get_tags_string(["LV", "RV"])
            valve_tags = self.labels.get_tags_string(["MV", "TV", "AV", "PV"])

            # Surface excluding valve openings
            op_tag = f"{ventricle_tags}-{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh.stem,
                output_surface_path=paths.epi_endo_combined,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Step 2: Extract connected components (should be 3: epi, LV endo, RV endo)
            self.logger.info("Extracting connected components")

            for output_ext in ["vtk", "carp_txt"]:
                self.meshtool.extract_unreachable(
                    # The .surfmesh companion of the extracted surface. Asked of
                    # the handle: with_suffix would truncate any dotted stem.
                    input_mesh_path=CarpSurface(paths.epi_endo_combined).surfmesh.stem,
                    submsh_path=paths.epi_endo_cc_base,
                    ofmt=output_ext,
                    ifmt="vtk",
                )

            cc_parts = mshu.find_numbered_parts(
                directory=paths.tmp_dir, base_prefix=paths.epi_endo_cc_base.name
            )

            EXPECTED_CC_COUNT = 3
            if len(cc_parts) < EXPECTED_CC_COUNT:
                raise SurfaceIdentificationError(
                    f"Expected {EXPECTED_CC_COUNT} connected components, found {len(cc_parts)}"
                )

            cc_parts = mshu.keep_largest_n_components(
                component_names=cc_parts,
                directory=paths.tmp_dir,
                keep_n=EXPECTED_CC_COUNT,
                delete_smaller=True,
            )

            # Step 4: Identify which surface is which
            surfaces_data = []
            for cc_name in cc_parts:
                cc_path = paths.tmp_dir / cc_name
                pts, surf = mshu.read_carp_mesh(
                    cc_path, elem_type=ElemType.Tr, read_tags=False
                )
                cog = geom.compute_surface_center_of_gravity(pts)
                surfaces_data.append(
                    {"name": cc_name, "pts": pts, "surf": surf, "cog": cog}
                )

            # Step 5: Get LV blood pool center for distance calculations
            mesh_pts, mesh_elem = mshu.read_carp_mesh(
                paths.mesh.stem, elem_type=ElemType.Tt, read_tags=True
            )

            lv_tag = self.labels.get_values_from_names(["LV"])[0]
            lv_cog = geom.compute_mesh_region_cog(mesh_pts, mesh_elem, lv_tag)

            # Calculate distances from LV center
            for surf_data in surfaces_data:
                surf_data["dist_to_lv"] = np.linalg.norm(surf_data["cog"] - lv_cog)

            # Step 6: Identify epicardium by outward-pointing normals
            epi_found = False
            for i, surf_data in enumerate(surfaces_data):
                outward_fraction = geom.outward_normal_fraction(
                    surf_data["pts"], surf_data["surf"], lv_cog
                )

                if outward_fraction > 0.7:  # Majority point outward
                    epi_found = True
                    epi_idx = i

                    # The other two are endocardia
                    endo_indices = [j for j in range(3) if j != i]

                    # Closer to LV center = LV endo, farther = RV endo
                    dist0 = surfaces_data[endo_indices[0]]["dist_to_lv"]
                    dist1 = surfaces_data[endo_indices[1]]["dist_to_lv"]

                    if dist0 < dist1:
                        lv_endo_idx = endo_indices[0]
                        rv_endo_idx = endo_indices[1]
                    else:
                        lv_endo_idx = endo_indices[1]
                        rv_endo_idx = endo_indices[0]

                    self.logger.info(
                        f"Identified: EPI={cc_parts[epi_idx]}, "
                        f"LV_ENDO={cc_parts[lv_endo_idx]}, "
                        f"RV_ENDO={cc_parts[rv_endo_idx]}"
                    )
                    break

            if not epi_found:
                raise SurfaceIdentificationError(
                    "Could not identify epicardium surface"
                )

            # Step 7: Rename files to final output paths
            self._copy_mesh_family(
                CarpMesh(paths.tmp_dir / cc_parts[epi_idx]),  # epicardium
                CarpMesh(paths.epi_surface),
                with_vtk=True,
            )
            self._copy_mesh_family(
                CarpMesh(paths.tmp_dir / cc_parts[lv_endo_idx]),  # lv_endocardium
                CarpMesh(paths.lv_endo_surface),
                with_vtk=True,
            )
            self._copy_mesh_family(
                CarpMesh(paths.tmp_dir / cc_parts[rv_endo_idx]),  # rv_endocardium
                CarpMesh(paths.rv_endo_surface),
                with_vtk=True,
            )

            self.logger.info("Successfully extracted ventricular surfaces")

        except SurfaceIdentificationError:
            raise
        except Exception as e:
            msg = f"Failed to extract ventricular surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_septum(self, paths: VentricularSurfacePaths) -> None:
        """
        Extract the interventricular septum surface.

        The septum is extracted as the surface of the LV that faces the RV,
        excluding regions that face the atria.

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting septum surface")

            # Extract LV surface, excluding RV, RA, and pulmonary artery
            lv_tags = self.labels.get_tags_string(["LV"])
            exclude_tags = self.labels.get_tags_string(["RV", "RA"])

            op_tag = f"{lv_tags}-{exclude_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh.stem,
                output_surface_path=paths.septum_raw,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Extract connected components
            self.logger.info("Extracting connected components")
            septum_surfmesh_input = Path(f"{paths.septum_raw}.surfmesh")

            for output_ext in ["vtk", "carp_txt"]:
                self.meshtool.extract_unreachable(
                    input_mesh_path=septum_surfmesh_input,
                    submsh_path=paths.septum_cc_base,  # output from VentricularPathsContract
                    ofmt=output_ext,
                    ifmt="vtk",
                )

            # Find connected components
            cc_parts = mshu.find_numbered_parts(
                directory=paths.tmp_dir,
                base_prefix=paths.septum_cc_base.name,
            )

            # Keep 2 largest (LV epi and septum)
            cc_parts = mshu.keep_largest_n_components(cc_parts, paths.tmp_dir, keep_n=2)

            self.logger.info("Renaming septum connected components")

            # First is LV epi (intermediate), second is actual septum
            self._copy_mesh_family(
                CarpMesh(paths.tmp_dir / cc_parts[0]),
                CarpMesh(paths.lv_epi_intermediate),
            )
            self._copy_mesh_family(
                CarpMesh(paths.tmp_dir / cc_parts[1]),
                CarpMesh(paths.septum_surface),
            )

            self.logger.info(f"Septum extracted to {paths.septum_surface}")

        except Exception as e:
            msg = f"Failed to extract septum: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def map_ventricular_surfaces(self, paths: VentricularSurfacePaths) -> None:
        """
        Map connected component indices to create final surface files.

        This takes the connected component `.eidx` and `.nod` files and uses them
        to extract subsurfaces from the original combined surfaces, creating both
        .surf and .vtx files for each surface.

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If mapping fails
        """
        try:
            self.logger.info("Mapping connected components to final surfaces")

            # Define the mapping operations
            original_surface_epi_endo = (
                paths.epi_endo_combined.parent / f"{paths.epi_endo_combined.name}.surf"
            )
            original_surface_septum_raw = (
                paths.septum_raw.parent / f"{paths.septum_raw.name}.surf"
            )

            # Each pair is: (component_stem, surface_it_was_carved_from).
            # The component stem is used three ways — its `.nod`/`.eidx` are
            # read, and the `.surf`/`.vtx` results are written back to the same
            # stem — so it appears once, not once per role.
            mapping_operations = [
                # Epi, LV endo, RV endo all come from epi_endo combined surface
                (
                    paths.epi_surface,  # biv_epi.eidx / .nod
                    original_surface_epi_endo,
                ),
                (
                    paths.lv_endo_surface,  # biv_lvendo.eidx / .nod
                    original_surface_epi_endo,
                ),
                (
                    paths.rv_endo_surface,  # biv_rvendo.eidx / .nod
                    original_surface_epi_endo,
                ),
                # LV epi intermediate and septum come from septum extraction
                (
                    paths.lv_epi_intermediate,  # lv_epi_intermediate.eidx / .nod
                    original_surface_septum_raw,
                ),
                (
                    paths.septum_surface,  # biv_septum.eidx / .nod
                    original_surface_septum_raw,
                ),
            ]

            # Execute each mapping operation
            for io_path, orig_surf in mapping_operations:
                eidx_path = io_path
                output_path = io_path

                original_surface = paths.tmp_dir / orig_surf.name
                temp_output_path = paths.tmp_dir / output_path.name

                self.logger.debug(
                    f"Mapping {eidx_path.name} from {orig_surf.name} -> {temp_output_path.name}"
                )

                mshu.connected_component_to_surface(
                    eidx_path=eidx_path,
                    input_surface_path=original_surface,
                    output_surface_path=output_path,
                )

            mshu.surf2vtk(
                mesh_base_path=paths.mesh.stem,
                surface_path=paths.epi_surface,
                output_vtk_path=CarpMesh(paths.epi_surface).vtk,
            )
            self.logger.info("Surface mapping completed")

        except Exception as e:
            msg = f"Failed to map ventricular surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def remove_septum_from_rv_endo(self, paths: VentricularSurfacePaths) -> None:
        """
        Removes the septum from the RV endocardial surface to isolate the free wall.

        This process overwrites the existing RV endo surface file with the new
        free wall surface and generates the corresponding .vtx and .vtk files.

        Args:
            paths: The data contract with all required ventricular surface paths.

        Raises:
            SurfaceExtractionError: If the removal process fails.
        """
        try:
            self.logger.info("Removing septum from RV endocardium.")

            # Step 1: Create the new free wall surface by removing septal triangles.
            # The output_path overwrites the original rv_endo_surface.
            mshu.remove_septum_from_endo(
                endo_surface_path=paths.rv_endo_surface,
                septum_surface_path=paths.septum_surface,
                output_path=paths.rv_endo_surface,
            )
            self.logger.info(
                f"Generated new free wall surface: {paths.rv_endo_surface.name}"
            )

            # Step 2: Generate the corresponding .vtx file for the new surface.
            # Read the surface we just created to get its vertex list.
            freewall_surf = mshu.read_surf(paths.rv_endo_surface)
            freewall_vtx = mshu.surf2vtx(freewall_surf)
            mshu.write_vtx(freewall_vtx, paths.rv_endo_vtx)
            self.logger.info(
                f"Generated corresponding vtx file: {paths.rv_endo_vtx.name}"
            )

            # Step 3: Generate a .vtk file for visualization/debugging.
            # This is not a final artifact, so it goes in the temporary directory.
            debug_vtk_path = paths.tmp_dir / CarpMesh(paths.rv_endo_surface).vtk.name
            mshu.surf2vtk(
                mesh_base_path=paths.mesh.stem,
                surface_path=paths.rv_endo_surface,
                output_vtk_path=debug_vtk_path,
            )
            self.logger.info(f"Generated debug VTK file: {debug_vtk_path.name}")

        except FileNotFoundError as e:
            msg = f"A required surface file was not found during septum removal: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e
        except Exception as e:
            msg = f"An unexpected error occurred during septum removal: {e}"
            self.logger.error(msg, exc_info=True)  # exc_info=True logs the traceback
            raise SurfaceExtractionError(msg) from e

    def prepare_ventricular_vtx_files(self, paths: VentricularSurfacePaths) -> None:
        """
        Generates .vtx files from their corresponding .surf files for UVC.

        Args:
            paths: The data contract containing all required ventricular surface paths.

        Raises:
            SurfaceExtractionError: If any VTX file generation fails.
        """
        self.logger.info("Preparing VTX files from surfaces for UVC.")

        # A list of (source_surface, destination_vtx) pairs.
        # This is explicit and easy to modify.
        surface_to_vtx_map = [
            (paths.epi_surface, paths.epi_vtx),
            (paths.lv_endo_surface, paths.lv_endo_vtx),
            (paths.rv_endo_surface, paths.rv_endo_vtx),
            (paths.septum_surface, paths.rvsept_vtx),
            # The 'base' surface is also required per the old contract.
            (paths.base_surface, paths.base_vtx),
        ]

        try:
            for surf_path, vtx_path in surface_to_vtx_map:
                surf_file_with_ext = CarpSurface(surf_path).surf

                if not surf_file_with_ext.is_file():
                    # Raise an error if an input is missing, as this is a logic error.
                    raise FileNotFoundError(
                        f"Required input surface does not exist: {surf_file_with_ext}"
                    )

                mshu.generate_vtx_from_surf(
                    input_surf_path=surf_file_with_ext, output_vtx_path=vtx_path
                )

            self.logger.info("Successfully generated all required VTX files.")

        except Exception as e:
            msg = f"Failed to prepare VTX files: {e}"
            self.logger.error(msg, exc_info=True)
            raise SurfaceExtractionError(msg) from e

    # ATRIAL SURFACE EXTRACTION
    def _get_atrial_valve_tags_string(self, chamber: Chamber) -> str:
        """
        Gets a comma-separated string of valve tags associated with an atrium.

        Args:
            chamber: The atrial chamber (LA or RA).

        Returns:
            A string of tag numbers for the relevant valves (e.g., "7,9").
        """
        match chamber:
            case Chamber.LA:
                return self.labels.get_tags_string(["MV", "AV"])
            case Chamber.RA:
                return self.labels.get_tags_string(["TV", "PV"])
            case _:
                # This helps catch logic errors if a non-atrial chamber is passed
                raise ValueError(
                    f"No valve association defined for chamber: {chamber.name}"
                )

    def extract_atrial_base(self, paths: AtrialSurfacePaths, chamber: Chamber) -> None:
        """
        Extract the base surface for an atrium (LA or RA).

        Args:
            paths: Atrial surface paths contract
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} base surface")

            # Get atrial tag and valve tags
            atrial_tags = self.labels.get_tags_string([chamber.value])
            valve_tags = self._get_atrial_valve_tags_string(chamber)

            op_tag = f"{atrial_tags}:{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh,
                output_surface_path=paths.base_surface,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            self.logger.info(f"{chamber.value} base extracted")

        except Exception as e:
            msg = f"Failed to extract {chamber.value} base: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_atrial_surfaces(
        self, paths: AtrialSurfacePaths, chamber: Chamber
    ) -> None:
        """
        Extract epicardium and endocardium for an atrium.

        Args:
            paths: Atrial surface paths contract
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} epi and endo surfaces")

            # Get atrial tag
            atrial_tags = self.labels.get_tags_string([chamber.value])
            valve_tags = self._get_atrial_valve_tags_string(chamber)

            op_tag = f"{atrial_tags}-{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh,
                output_surface_path=paths.epi_endo_combined,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # TODO: Extract connected components and identify epi vs endo
            # For atria, this is simpler than ventricles - typically just 2 surfaces

            self.logger.info(f"{chamber.value} surfaces extracted")

        except Exception as e:
            msg = f"Failed to extract {chamber.value} surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def map_atrial_surfaces(
        self, paths: AtrialSurfacePaths, files_to_map: List[Path], chamber: Chamber
    ) -> None:
        """
        Map data files onto atrial surfaces.

        Args:
            paths: Atrial surface paths contract
            files_to_map: List of data files to map
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If mapping fails
        """
        try:
            self.logger.info(
                f"Mapping {len(files_to_map)} files to {chamber.value} surfaces"
            )

            files_str = [str(f) for f in files_to_map]

            self.meshtool.map(
                submesh_path=paths.mesh,
                files_list=files_str,
                output_folder=paths.output_dir,
                mode="m2s",
            )

            self.logger.info(f"{chamber.value} surface mapping completed")

        except Exception as e:
            msg = f"Failed to map {chamber.value} surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    # SUBMESH EXTRACTION AND MAPPING

    def extract_biv_submesh(self, paths: BiVMeshPaths, tags: TagsConfig) -> None:
        """
        Extract biventricular (BiV) submesh from four-chamber mesh.

        Args:
            paths: BiV mesh paths contract
            tags: Tag configuration

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting BiV submesh")

            biv_tags = tags.get_tags_list(["LV", "RV"])

            self.meshtool.extract_mesh(
                input_mesh_path=paths.source_mesh,
                output_submesh_path=paths.output_mesh.stem,
                tags=biv_tags,
                ifmt="carp_txt",
                normalise=False,
            )

            self.logger.info(f"BiV submesh extracted to {paths.output_mesh}")

        except Exception as e:
            msg = f"Failed to extract BiV submesh: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def map_vtx_to_submesh(self, paths: BiVMeshPaths) -> None:
        """
        Map VTX files from four-chamber mesh to BiV submesh.

        Args:
            paths: BiV mesh paths contract

        Raises:
            SurfaceExtractionError: If mapping fails
        """
        try:
            self.logger.info("Mapping VTX files to BiV submesh")

            files_str = [str(f) for f in paths.vtx_files_to_map]

            self.meshtool.map(
                submesh_path=paths.output_mesh.stem,
                files_list=files_str,
                output_folder=paths.mapped_vtx_output_dir,
                mode="m2s",
            )

            self.logger.info("VTX mapping to BiV completed")

        except Exception as e:
            msg = f"Failed to map VTX to BiV: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_atrial_submesh(
        self, paths: AtrialMeshPaths, tags: TagsConfig, chamber: Chamber
    ) -> None:
        """
        Extract atrial submesh (LA or RA) from four-chamber mesh.

        Args:
            paths: Atrial mesh paths contract
            tags: Tag configuration
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} submesh")

            atrial_tags = tags.get_tags_list([chamber.value])

            self.meshtool.extract_mesh(
                input_mesh_path=paths.source_mesh,
                output_submesh_path=paths.output_mesh,
                tags=atrial_tags,
                ifmt="carp_txt",
                normalise=False,
            )

            self.logger.info(
                f"{chamber.value} submesh extracted to {paths.output_mesh}"
            )

        except Exception as e:
            msg = f"Failed to extract {chamber.value} submesh: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def map_vtx_to_atrial_submesh(
        self, paths: AtrialMeshPaths, chamber: Chamber
    ) -> None:
        """
        Map VTX files from four-chamber mesh to atrial submesh.

        Also copies blank template files for apex and septum (not applicable
        to atria but required for UVC interface compatibility).

        Args:
            paths: Atrial mesh paths contract
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If mapping fails
        """
        try:
            self.logger.info(f"Mapping VTX files to {chamber.value} submesh")

            files_str = [str(f) for f in paths.vtx_files_to_map]

            self.meshtool.map(
                submesh_path=paths.output_mesh,
                files_list=files_str,
                output_folder=paths.mapped_vtx_output_dir,
                mode="m2s",
            )

            # Copy blank template files
            import shutil

            shutil.copy(paths.apex_template, paths.apex_output)
            shutil.copy(paths.rv_septum_template, paths.rv_septum_output)

            self.logger.info(f"VTX mapping to {chamber.value} completed")

        except Exception as e:
            msg = f"Failed to map VTX to {chamber.value}: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    # CONVENIENCE WORKFLOWS

    def run_ventricular_extraction(
        self,
        paths: VentricularSurfacePaths,
    ) -> None:
        """
        Run complete ventricular surface extraction workflow.

        This executes all steps required to extract ventricular surfaces
        for UVC coordinate generation:
        1. Extract base surface
        2. Extract and identify epi, LV endo, RV endo
        3. Extract septum
        4. Map connected components back onto the original surfaces
        5. Remove septum from RV endo
        6. Prepare VTX files

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING VENTRICULAR SURFACE EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self._prepare_directories(paths)

        self.extract_ventricular_base(paths)
        self.extract_ventricular_surfaces(paths)
        self.extract_septum(paths)

        self.map_ventricular_surfaces(paths)

        self.remove_septum_from_rv_endo(paths)
        self.prepare_ventricular_vtx_files(paths)

        self.logger.info("=" * 60)
        self.logger.info("VENTRICULAR EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_atrial_extraction(
        self,
        paths: AtrialSurfacePaths,
        chamber: Chamber,
        files_to_map: Optional[List[Path]] = None,
    ) -> None:
        """
        Run complete atrial surface extraction workflow.

        Args:
            paths: Atrial surface paths contract
            chamber: Chamber.LA or Chamber.RA
            files_to_map: Optional list of data files to map onto surfaces

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING {chamber.value} SURFACE EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_atrial_base(paths, chamber)
        self.extract_atrial_surfaces(paths, chamber)

        if files_to_map:
            self.map_atrial_surfaces(paths, files_to_map, chamber)

        self.logger.info("=" * 60)
        self.logger.info(f"{chamber.value} EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_biv_mesh_extraction(self, paths: BiVMeshPaths, tags: TagsConfig) -> None:
        """
        Run BiV submesh extraction and VTX mapping workflow.

        Args:
            paths: BiV mesh paths contract
            tags: Tag configuration

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING BIV MESH EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_biv_submesh(paths, tags)
        self.map_vtx_to_submesh(paths)

        self.logger.info("=" * 60)
        self.logger.info("BIV MESH EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_atrial_mesh_extraction(
        self, paths: AtrialMeshPaths, tags: TagsConfig, chamber: Chamber
    ) -> None:
        """
        Run atrial submesh extraction and VTX mapping workflow.

        Args:
            paths: Atrial mesh paths contract
            tags: Tag configuration
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING {chamber.value} MESH EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_atrial_submesh(paths, tags, chamber)
        self.map_vtx_to_atrial_submesh(paths, chamber)

        self.logger.info("=" * 60)
        self.logger.info(f"{chamber.value} MESH EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_all(
        self,
        paths: UVCSurfaceExtractionPaths,
        tags: TagsConfig,
        ventricular_files_to_map: Optional[List[Path]] = None,
        la_files_to_map: Optional[List[Path]] = None,
        ra_files_to_map: Optional[List[Path]] = None,
    ) -> None:
        """
        Not implemented. Drive the per-phase methods directly.

        This orchestrated the six extraction phases end to end, but its calls
        drifted out of step with the methods they invoke:
        `run_ventricular_extraction` takes `(self, paths)` alone while this
        passed three arguments, and `run_atrial_extraction` likewise. Every
        call raised `TypeError`, so nothing depended on it.

        It is left as a stub rather than deleted because the phase methods are
        being reshaped by the contract refactor, and the right signature for a
        whole-workflow entry point is not knowable until they settle. Rebuild
        it then, against whatever the phases actually take.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SurfaceLogic.run_all is not implemented. Its calls disagreed with "
            "the phase methods' signatures, so it never ran. Call the phase "
            "methods directly: run_ventricular_extraction, run_atrial_extraction, "
            "run_biv_mesh_extraction, run_atrial_mesh_extraction."
        )

    # HELPER METHODS
    def _prepare_directories(self, paths: VentricularSurfacePaths) -> None:
        """
        Create the directories the ventricular workflow writes into.

        This lives here rather than in the path builder so that constructing a
        contract in order to inspect it has no effect on disk (see also
        `RefinementLogic._prepare_directories`).

        Only the BiV directories are created here. LA/RA are still made
        eagerly by `ModelCreationPathBuilder`, and move when the atrial flow
        is refactored.
        """
        for directory in (paths.output_dir, paths.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _copy_mesh_family(
        self, source: CarpMesh, target: CarpMesh, *, with_vtk: bool = False
    ) -> None:
        """
        Copy a CARP submesh family from one stem to another.

        `meshtool extract unreachable` writes each connected component under a
        `.partN` stem; this moves one of those components to the name the rest
        of the pipeline expects. Despite the surrounding "surface" vocabulary
        these are *mesh* families — the CARP triple plus the submesh index
        pair, because a component is meaningless without the `.nod`/`.eidx`
        that map it back to the mesh it was carved from.

        Every member listed is required. A missing one means the extraction did
        not produce what was expected, which is worth failing on rather than
        carrying a partial family forward.

        Args:
            source: The component to copy from.
            target: The stem to copy to.
            with_vtk: Also copy the `.vtk` visualisation companion.

        Raises:
            FileNotFoundError: If any expected member is absent.
        """
        members = [
            (source.pts, target.pts),
            (source.elem, target.elem),
            (source.lon, target.lon),
            (source.index.nod, target.index.nod),
            (source.index.eidx, target.index.eidx),
        ]
        if with_vtk:
            members.append((source.vtk, target.vtk))

        for source_file, target_file in members:
            if not source_file.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")

            shutil.copy(str(source_file), str(target_file))
            self.logger.debug(f"Copied {source_file.name} -> {target_file.name}")

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
import pycemrg_model_creation.utilities.topology as topo


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
            # The pulmonary artery is excluded alongside the valve planes;
            # cemrg-heartbuilder does the same (meshtools_utils.py:651).
            exclude_tags = self.labels.get_tags_string(
                ["MV", "TV", "AV", "PV", "PArt"]
            )

            # Surface excluding valve openings and the pulmonary artery
            op_tag = f"{ventricle_tags}-{exclude_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh.stem,
                output_surface_path=paths.epi_endo_combined,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Step 2: Extract connected components (should be 3: epi, LV endo, RV endo)
            self.logger.info("Extracting connected components")

            EXPECTED_CC_COUNT = 3
            split_base = paths.epi_endo_cc_base
            cc_parts, roles = self._connected_components(
                surface_stem=paths.epi_endo_combined,
                split_base=split_base,
                points_path=paths.mesh.pts,
                expected=EXPECTED_CC_COUNT,
            )

            cc_parts = mshu.keep_largest_n_components(
                cc_parts,
                keep_n=EXPECTED_CC_COUNT,
                delete_smaller=True,
            )

            if roles is not None:
                # Hand-made roles replace Steps 4-6. See `SurfaceRoles` for
                # why the geometric test can name the wrong part.
                epi_idx, lv_endo_idx, rv_endo_idx = self._parts_from_roles(
                    roles, cc_parts, split_base
                )
                self.logger.info(
                    f"Surface roles from {mshu.split_sidecar_path(split_base).name}: "
                    f"EPI={cc_parts[epi_idx].name}, "
                    f"LV_ENDO={cc_parts[lv_endo_idx].name}, "
                    f"RV_ENDO={cc_parts[rv_endo_idx].name}"
                )
            else:
                # Step 4: Identify which surface is which
                surfaces_data = []
                for component in cc_parts:
                    # Stems at the utility boundary: read_carp_mesh takes a Path.
                    pts, surf = mshu.read_carp_mesh(
                        component.stem, elem_type=ElemType.Tr, read_tags=False
                    )
                    cog = geom.compute_surface_center_of_gravity(pts)
                    surfaces_data.append(
                        {"name": component.name, "pts": pts, "surf": surf, "cog": cog}
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
                            f"Identified: EPI={cc_parts[epi_idx].name}, "
                            f"LV_ENDO={cc_parts[lv_endo_idx].name}, "
                            f"RV_ENDO={cc_parts[rv_endo_idx].name}"
                        )
                        break

                if not epi_found:
                    raise SurfaceIdentificationError(
                        "Could not identify epicardium surface"
                    )

            # Step 7: Rename files to final output paths
            self._copy_mesh_family(
                cc_parts[epi_idx],  # epicardium
                CarpMesh(paths.epi_surface),
                with_vtk=True,
            )
            self._copy_mesh_family(
                cc_parts[lv_endo_idx],  # lv_endocardium
                CarpMesh(paths.lv_endo_surface),
                with_vtk=True,
            )
            self._copy_mesh_family(
                cc_parts[rv_endo_idx],  # rv_endocardium
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
            exclude_tags = self.labels.get_tags_string(["RV", "RA", "PArt"])

            op_tag = f"{lv_tags}-{exclude_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh.stem,
                output_surface_path=paths.septum_raw,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Same exposure as the epi/endo split above: pinched sheets fuse
            # under meshtool, so the septum gets the same accept path. Roles
            # do not apply; the two parts are told apart by size below.
            self.logger.info("Extracting connected components")
            EXPECTED_SEPTUM_CC_COUNT = 2
            cc_parts, _ = self._connected_components(
                surface_stem=paths.septum_raw,
                split_base=paths.septum_cc_base,
                points_path=paths.mesh.pts,
                expected=EXPECTED_SEPTUM_CC_COUNT,
            )

            # Keep 2 largest (LV epi and septum)
            cc_parts = mshu.keep_largest_n_components(
                cc_parts, keep_n=EXPECTED_SEPTUM_CC_COUNT
            )

            self.logger.info("Renaming septum connected components")

            # First is LV epi (intermediate), second is actual septum
            self._copy_mesh_family(
                cc_parts[0],
                CarpMesh(paths.lv_epi_intermediate),
            )
            self._copy_mesh_family(
                cc_parts[1],
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
                surface_path=CarpSurface(paths.epi_surface).surf,
                output_vtk_path=CarpMesh(paths.epi_surface).vtk,
            )
            self.logger.info("Surface mapping completed")

        except Exception as e:
            msg = f"Failed to map ventricular surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def remove_septum_from_rv_endo(self, paths: VentricularSurfacePaths) -> None:
        """
        Writes the RV free wall as its own surface, beside the endocardium.

        The two are separate files. `rv_endo_surface` keeps the complete RV
        endocardium, septal surface included, because that is what mguvc
        reads; `rv_endo_nosept_surface` is the free wall alone, for the fibre
        step. `cemrg-heartbuilder` keeps the same pair.

        Args:
            paths: The data contract with all required ventricular surface paths.

        Raises:
            SurfaceExtractionError: If the removal process fails.
        """
        try:
            self.logger.info("Removing septum from RV endocardium.")

            # The leaf readers and writers take literal paths, so the extension
            # is named here. The contract holds stems; `CarpSurface` is what
            # turns one into the file a reader will actually open.
            free_wall_surface = CarpSurface(paths.rv_endo_nosept_surface).surf

            # Step 1: Create the new free wall surface by removing septal triangles.
            mshu.remove_septum_from_endo(
                endo_surface_path=CarpSurface(paths.rv_endo_surface).surf,
                septum_surface_path=CarpSurface(paths.septum_surface).surf,
                output_path=free_wall_surface,
            )
            self.logger.info(
                f"Generated new free wall surface: {free_wall_surface.name}"
            )

            # There was a VTX write here. It read the surface back and wrote
            # `rv_endo_vtx`, which `prepare_ventricular_vtx_files` then wrote
            # again from the same surface one step later. Two writers, one
            # file, same content. The later one is where every boundary is
            # written, so this one is gone.
            #
            # Step 2: Generate a .vtk file for visualization/debugging.
            # This is not a final artifact, so it goes in the temporary directory.
            # It shows the free wall, which is what this method produces; it
            # used to name `rv_endo_surface` because the two were one file.
            debug_vtk_path = (
                paths.tmp_dir / CarpMesh(paths.rv_endo_nosept_surface).vtk.name
            )
            mshu.surf2vtk(
                mesh_base_path=paths.mesh.stem,
                surface_path=free_wall_surface,
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

        # (source_surface, mguvc role). The surfaces carry four-chamber node
        # indices, so the VTX files derived from them do too, and they are
        # written to the four-chamber generation in tmp/ — not beside the
        # submesh, where a reader would take them for remapped ones.
        surface_to_role = [
            (paths.epi_surface, "epi"),
            (paths.lv_endo_surface, "lvendo"),
            (paths.rv_endo_surface, "rvendo"),
            (paths.rv_endo_nosept_surface, "rvendo_nosept"),
            (paths.septum_surface, "rvsept"),
            (paths.base_surface, "base"),
        ]

        try:
            for surf_path, role in surface_to_role:
                vtx_path = paths.source_boundaries.role(role, ".vtx")
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

    def extract_biv_submesh(self, paths: BiVMeshPaths) -> None:
        """
        Extract biventricular (BiV) submesh from four-chamber mesh.

        Tags come from the injected `LabelManager`, as they do everywhere else
        in this class.

        Args:
            paths: BiV mesh paths contract

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting BiV submesh")

            biv_tags = self.labels.get_values_from_names(["LV", "RV"])

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

            self.meshtool.map(
                submesh_path=paths.output_mesh.stem,
                files_list=paths.vtx_files_to_map,
                output_folder=paths.mapped_vtx_output_dir,
                mode="m2s",
            )

            self._rename_mapped_boundaries(paths)

            self.logger.info("VTX mapping to BiV completed")

        except Exception as e:
            msg = f"Failed to map VTX to BiV: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def _rename_mapped_boundaries(self, paths: BiVMeshPaths) -> None:
        """
        Give the mapped boundaries the submesh stem.

        `meshtool map` keeps each input's basename, so the remapped files
        arrive in the output directory still carrying the four-chamber stem —
        `BiV/heart.base.vtx`. `mguvc` builds its filenames from the model name
        it is given, so it looks for `BiV/BiV.base.vtx` and would find nothing.
        This is the hop that was missing: the chain wrote the right names in
        the wrong generation and then never renamed the right generation.

        Args:
            paths: BiV mesh paths contract. `vtx_files_to_map` and
                   `renamed_vtx_targets` are in the same order by construction.
        """
        for source, target in zip(paths.vtx_files_to_map, paths.renamed_vtx_targets):
            mapped = paths.mapped_vtx_output_dir / source.name
            if not mapped.is_file():
                raise SurfaceExtractionError(
                    f"meshtool map did not produce {mapped}. Expected it beside "
                    f"the submesh, because -outdir names a directory and the "
                    f"input basename is kept."
                )
            mapped.replace(target)
            self.logger.debug(f"Renamed {mapped.name} -> {target.name}")

    def extract_atrial_submesh(
        self, paths: AtrialMeshPaths, chamber: Chamber
    ) -> None:
        """
        Extract atrial submesh (LA or RA) from four-chamber mesh.

        Tags come from the injected `LabelManager`, as they do everywhere else
        in this class.

        Args:
            paths: Atrial mesh paths contract
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} submesh")

            atrial_tags = self.labels.get_values_from_names([chamber.value])

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

    def run_biv_mesh_extraction(self, paths: BiVMeshPaths) -> None:
        """
        Run BiV submesh extraction and VTX mapping workflow.

        Args:
            paths: BiV mesh paths contract

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING BIV MESH EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_biv_submesh(paths)
        self.map_vtx_to_submesh(paths)

        self.logger.info("=" * 60)
        self.logger.info("BIV MESH EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_atrial_mesh_extraction(
        self, paths: AtrialMeshPaths, chamber: Chamber
    ) -> None:
        """
        Run atrial submesh extraction and VTX mapping workflow.

        Args:
            paths: Atrial mesh paths contract
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING {chamber.value} MESH EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_atrial_submesh(paths, chamber)
        self.map_vtx_to_atrial_submesh(paths, chamber)

        self.logger.info("=" * 60)
        self.logger.info(f"{chamber.value} MESH EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_all(
        self,
        paths: UVCSurfaceExtractionPaths,
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
    def _diagnose_component_failure(
        self,
        surface_stem: Path,
        points_path: Path,
        *,
        expected: int,
        found: int,
        split_base: Optional[Path] = None,
    ) -> str:
        """
        Explain a wrong component count, and leave a mesh behind to look at.

        `meshtool extract unreachable` separates by *node* reachability, so two
        sheets touching at a single vertex come back as one component. This
        re-reads the same surface under edge adjacency, names the defects that
        make the two answers differ, and writes a VTK carrying both as cell
        data.

        Diagnosis is best-effort and must never mask the failure it describes:
        anything going wrong here is logged and the caller still raises.

        Args:
            surface_stem: Stem of the surface that was split, e.g.
                `paths.epi_endo_combined`. The `.surf` read and the `_topology.vtk`
                written are both derived from it, by appending rather than by
                `with_suffix`, which would truncate a dotted stem.
            points_path: The parent mesh `.pts`, whose indexing `.surf` uses.
            expected: Components the caller needed.
            found: Components `meshtool` returned.
            split_base: Stem the stage reads parts from, e.g.
                `paths.epi_endo_cc_base`. Used to write the accept command
                with the right `--prefix`; omitted, the command shows a
                placeholder.

        Returns:
            A message for `SurfaceIdentificationError`, whatever happens.
        """
        message = f"Expected {expected} connected components, found {found}."

        surface_path = Path(f"{surface_stem}.surf")
        diagnostic_path = Path(f"{surface_stem}_topology.vtk")

        try:
            surf = mshu.read_surf(surface_path)
            report = topo.analyse_surface_topology(surf)

            mshu.write_topology_vtk(
                points=mshu.read_pts(points_path),
                surf=surf,
                topology=report,
                output_path=diagnostic_path,
            )

            if split_base is None:
                prefix_args = "--prefix <name>"
            else:
                prefix_args = (
                    f"--prefix {split_base.name} --output-dir {split_base.parent}"
                )
            accept_command = (
                f"python scripts/utilities/mesh_cli.py split "
                f"--input {surface_path} --pts {points_path} {prefix_args} "
                f"--neubc {surface_stem}.neubc --expect {expected}"
            )

            return (
                f"{message}\n\n"
                f"{report.summary()}\n\n"
                f"Inspect: {diagnostic_path}\n"
                f"  colour by 'pinch'     -> the defects\n"
                f"  colour by 'component' -> the proposed split\n"
                f"Accept with (from the pycemrg-model-creation repo):\n"
                f"  {accept_command}\n"
                f"then re-run the surface stage."
            )
        except Exception as exc:  # noqa: BLE001 - the caller's error is the one that matters
            self.logger.warning(
                f"Could not diagnose the component failure on {surface_path}: {exc}"
            )
            return message

    def _connected_components(
        self,
        surface_stem: Path,
        split_base: Path,
        points_path: Path,
        expected: int,
    ) -> tuple:
        """
        Split an extracted surface into its connected components.

        `meshtool extract unreachable` is the default. An accepted split (see
        `mshu.write_accepted_split`) replaces it when meshtool fuses pinched
        sheets. It applies only while the surface just extracted has the same
        triangles as the one it was accepted for, and the parts are then
        written again from the new surface, never reused: meshtool changes
        the row order on each run, and `.eidx` holds row positions.

        Args:
            surface_stem: Stem of the extracted surface, e.g.
                `paths.epi_endo_combined`; its `.surf`, `.neubc` and
                `.surfmesh` are derived by appending.
            split_base: Stem the parts are written under, e.g.
                `paths.epi_endo_cc_base`.
            points_path: The parent mesh `.pts`, whose indexing `.surf` uses.
            expected: Components the caller needs, at least.

        Returns:
            `(cc_parts, roles)`: the parts found, and the hand-made
            `SurfaceRoles` from the accepted split, or None.

        Raises:
            SurfaceIdentificationError: If fewer than `expected` parts result.
        """
        surface_path = Path(f"{surface_stem}.surf")
        surf = mshu.read_surf(surface_path)
        roles = None

        if mshu.accepted_split_matches(split_base, surf):
            # Read before splitting again: the split rewrites the sidecar,
            # and the hand-made roles must survive that.
            roles = mshu.read_split_roles(split_base)
            self.logger.info(
                f"Split accepted in {mshu.split_sidecar_path(split_base).name}; "
                "splitting the new surface by edge adjacency, meshtool "
                "extract unreachable skipped"
            )
            mshu.write_accepted_split(
                surface_path=surface_path,
                pts_path=points_path,
                base=split_base,
                neubc_path=Path(f"{surface_stem}.neubc"),
                expect=expected,
                roles=roles,
            )
        else:
            if mshu.split_sidecar_path(split_base).exists():
                self.logger.warning(
                    f"The accepted split for {surface_path.name} is stale (the "
                    "surface changed since it was made) and is ignored. "
                    "Using meshtool."
                )
            # meshtool may write fewer parts than the last run; old
            # higher-numbered parts must not be counted with the new ones.
            mshu.remove_split(split_base)

            for output_ext in ["vtk", "carp_txt"]:
                self.meshtool.extract_unreachable(
                    # The .surfmesh companion of the extracted surface. Asked of
                    # the handle: with_suffix would truncate any dotted stem.
                    input_mesh_path=CarpSurface(surface_stem).surfmesh.stem,
                    submsh_path=split_base,
                    ofmt=output_ext,
                    ifmt="vtk",
                )

        cc_parts = mshu.find_numbered_parts(
            directory=split_base.parent, base_prefix=split_base.name
        )

        if len(cc_parts) < expected:
            raise SurfaceIdentificationError(
                self._diagnose_component_failure(
                    surface_stem=surface_stem,
                    points_path=points_path,
                    expected=expected,
                    found=len(cc_parts),
                    split_base=split_base,
                )
            )
        return cc_parts, roles

    def _parts_from_roles(
        self, roles: topo.SurfaceRoles, cc_parts, split_base: Path
    ) -> tuple:
        """
        Positions in `cc_parts` of the parts that `roles` names.

        Found by name, not by list position: `keep_largest_n_components` may
        drop or reorder parts.

        Raises:
            SurfaceIdentificationError: If a role names a part that is not in
                `cc_parts`, e.g. a sliver that was dropped as too small.
        """
        positions = {part.name: i for i, part in enumerate(cc_parts)}
        found = []
        for role, number in (
            ("epi", roles.epi),
            ("lv_endo", roles.lv_endo),
            ("rv_endo", roles.rv_endo),
        ):
            name = f"{split_base.name}.part{number}"
            if name not in positions:
                raise SurfaceIdentificationError(
                    f"Role {role} names {name}, which is not among the kept "
                    f"parts {sorted(positions)}. Fix `roles` in "
                    f"{mshu.split_sidecar_path(split_base)}."
                )
            found.append(positions[name])
        return tuple(found)

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

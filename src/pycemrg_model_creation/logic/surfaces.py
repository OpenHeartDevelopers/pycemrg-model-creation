# src/pycemrg_model_creation/logic/surfaces.py;

"""
Surface extraction logic for UVC (Universal Ventricular Coordinate) generation.

This module implements the scientific workflow for extracting cardiac surfaces
from four-chamber heart meshes. It is stateless and path-agnostic - all paths
are provided explicitly through dataclass contracts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from pycemrg_model_creation.config import TagsConfig 
from pycemrg_model_creation.logic.contracts import VentricularSurfacePaths, AtrialSurfacePaths, BiVMeshPaths, AtrialMeshPaths, UVCSurfaceExtractionPaths
from pycemrg_model_creation.tools import CarpWrapper, MeshtoolWrapper, 
from pycemrg_model_creation.types import Chamber, SurfaceType 
import pycemrg_model_creation.utilities.mesh as mshu

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
        logic = SurfaceLogic(meshtool)
        logic.run_ventricular_extraction(paths, tags)
    """

    def __init__(self, meshtool):
        """
        Initialize SurfaceLogic with a meshtool wrapper.

        Args:
            meshtool: MeshtoolWrapper instance for executing meshtool commands
        """
        self.meshtool = meshtool
        self.logger = logging.getLogger(__name__)

    # VENTRICULAR SURFACE EXTRACTION
    def extract_ventricular_base(
        self, paths: VentricularSurfacePaths, tags: TagsConfig
    ) -> None:
        """
        Extract the base surface for ventricles.

        The base is defined as the interface between ventricular myocardium
        and valve planes.

        Args:
            paths: Ventricular surface paths contract
            tags: Tag configuration

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting ventricular base surface")

            # Get tags for ventricles (LV, RV) and valve planes (MV, TV, AV, PV)
            vent_tags = tags.get_tags_string(["LV", "RV"])
            valve_tags_keys = ["MV", "TV", "AV", "PV"]
            if tags.PArt is not None:
                valve_tags_keys.append("PArt")
            valve_tags = tags.get_tags_string(valve_tags_keys)

            # Extract surface where ventricles meet valves
            op_tag = f"{vent_tags}:{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh,
                output_surface_path=paths.base_surface,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            self.logger.info(f"Base surface extracted to {paths.base_surface}")

        except Exception as e:
            msg = f"Failed to extract ventricular base: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_ventricular_surfaces(
        self, paths: VentricularSurfacePaths, tags: TagsConfig
    ) -> None:
        """
        Extract and identify epicardium, LV endocardium, and RV endocardium.

        This method:
        1. Extracts the combined epi/endo surface
        2. Identifies connected components
        3. Uses geometric analysis to identify which is epi vs endo
        4. Uses distance from LV center to distinguish LV endo from RV endo

        Args:
            paths: Ventricular surface paths contract
            tags: Tag configuration

        Raises:
            SurfaceIdentificationError: If surfaces cannot be identified
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting ventricular surfaces (epi, LV endo, RV endo)")

            # Step 1: Extract combined epi/endo surface
            vent_tags = tags.get_tags_string(["LV", "RV"])
            valve_tags_keys = ["MV", "TV", "AV", "PV"]
            if tags.PArt is not None:
                valve_tags_keys.append("PArt")
            valve_tags = tags.get_tags_string(valve_tags_keys)

            # Surface excluding valve openings
            op_tag = f"{vent_tags}-{valve_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh,
                output_surface_path=paths.epi_endo_combined,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Step 2: Extract connected components (should be 3: epi, LV endo, RV endo)
            self.logger.info("Extracting connected components")
            # NOTE: extract_unreachable not in provided MeshtoolWrapper
            # This should be added to MeshtoolWrapper or implemented as utility
            # For now, assume it creates files with pattern: {base_name}_CC.part_*.{ext}

            # TODO: Call meshtool extract unreachable
            # self.meshtool.extract_unreachable(
            #     input_surface=paths.epi_endo_combined.with_suffix(".surfmesh.vtk"),
            #     output_base=paths.epi_endo_cc_base,
            #     ofmt=["vtk", "carp_txt"],
            #     ifmt="vtk"
            # )

            # Step 3: Find connected components
            cc_names = find_connected_components(paths.epi_endo_cc_base, paths.tmp_dir)

            if len(cc_names) < 3:
                raise SurfaceIdentificationError(
                    f"Expected 3 connected components, found {len(cc_names)}"
                )

            # Keep only 3 largest components
            cc_names = keep_largest_n_components(cc_names, paths.tmp_dir, n=3)

            # Step 4: Identify which surface is which
            surfaces_data = []
            for cc_name in cc_names:
                cc_path = paths.tmp_dir / cc_name
                pts, surf = read_carp_mesh(cc_path, elem_type="Tr", read_tags=False)
                cog = compute_surface_center_of_gravity(pts)
                surfaces_data.append(
                    {"name": cc_name, "pts": pts, "surf": surf, "cog": cog}
                )

            # Step 5: Get LV blood pool center for distance calculations
            mesh_pts, mesh_elem = read_carp_mesh(
                paths.mesh, elem_type="Tt", read_tags=True
            )

            lv_tag = tags.get_tags_list(["LV"])[0]
            lv_cog = compute_mesh_region_cog(mesh_pts, mesh_elem, lv_tag)

            # Calculate distances from LV center
            for surf_data in surfaces_data:
                surf_data["dist_to_lv"] = np.linalg.norm(surf_data["cog"] - lv_cog)

            # Step 6: Identify epicardium by outward-pointing normals
            epi_found = False
            for i, surf_data in enumerate(surfaces_data):
                outward_fraction = identify_surface_orientation(
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
                        f"Identified: EPI={cc_names[epi_idx]}, "
                        f"LV_ENDO={cc_names[lv_endo_idx]}, "
                        f"RV_ENDO={cc_names[rv_endo_idx]}"
                    )
                    break

            if not epi_found:
                raise SurfaceIdentificationError(
                    "Could not identify epicardium surface"
                )

            # Step 7: Rename files to final output paths
            self._rename_surface_files(
                paths.tmp_dir / cc_names[epi_idx],
                paths.epi_surface,
                formats=["pts", "elem", "lon", "nod", "eidx"],
            )
            self._rename_surface_files(
                paths.tmp_dir / cc_names[lv_endo_idx],
                paths.lv_endo_surface,
                formats=["pts", "elem", "lon", "nod", "eidx"],
            )
            self._rename_surface_files(
                paths.tmp_dir / cc_names[rv_endo_idx],
                paths.rv_endo_surface,
                formats=["pts", "elem", "lon", "nod", "eidx"],
            )

            self.logger.info("Successfully extracted ventricular surfaces")

        except SurfaceIdentificationError:
            raise
        except Exception as e:
            msg = f"Failed to extract ventricular surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def extract_septum(self, paths: VentricularSurfacePaths, tags: TagsConfig) -> None:
        """
        Extract the interventricular septum surface.

        The septum is extracted as the surface of the LV that faces the RV,
        excluding regions that face the atria.

        Args:
            paths: Ventricular surface paths contract
            tags: Tag configuration

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting septum surface")

            # Extract LV surface, excluding RV, RA, and pulmonary artery
            lv_tags = tags.get_tags_string(["LV"])
            exclude_tags_keys = ["RV", "RA"]
            if tags.PArt is not None:
                exclude_tags_keys.append("PArt")
            exclude_tags = tags.get_tags_string(exclude_tags_keys)

            op_tag = f"{lv_tags}-{exclude_tags}"

            self.meshtool.extract_surface(
                input_mesh_path=paths.mesh,
                output_surface_path=paths.septum_raw,
                ofmt="vtk",
                op_tag_base=op_tag,
            )

            # Extract connected components
            # TODO: Add extract_unreachable to MeshtoolWrapper
            # self.meshtool.extract_unreachable(...)

            # Find connected components
            cc_names = find_connected_components(paths.septum_cc_base, paths.tmp_dir)

            # Keep 2 largest (LV epi and septum)
            cc_names = keep_largest_n_components(cc_names, paths.tmp_dir, n=2)

            self.logger.info("Renaming septum connected components")

            # First is LV epi (intermediate), second is actual septum
            self._rename_surface_files(
                paths.tmp_dir / cc_names[0],
                paths.lv_epi_intermediate,
                formats=["pts", "elem", "lon", "nod", "eidx"],
            )
            self._rename_surface_files(
                paths.tmp_dir / cc_names[1],
                paths.septum_surface,
                formats=["pts", "elem", "lon", "nod", "eidx"],
            )

            self.logger.info(f"Septum extracted to {paths.septum_surface}")

        except Exception as e:
            msg = f"Failed to extract septum: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def map_ventricular_surfaces(
        self, paths: VentricularSurfacePaths, files_to_map: List[Path]
    ) -> None:
        """
        Map data files onto ventricular surfaces.

        Args:
            paths: Ventricular surface paths contract
            files_to_map: List of data files to map

        Raises:
            SurfaceExtractionError: If mapping fails
        """
        try:
            self.logger.info(f"Mapping {len(files_to_map)} files to surfaces")

            files_str = [str(f) for f in files_to_map]

            self.meshtool.map(
                submesh_path=paths.mesh,
                files_list=files_str,
                output_folder=paths.output_dir,
                mode="m2s",
            )

            self.logger.info("Surface mapping completed")

        except Exception as e:
            msg = f"Failed to map surfaces: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def remove_septum_from_lv_endo(self, paths: VentricularSurfacePaths) -> None:
        """
        Remove septum region from LV endocardium surface.

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If removal fails
        """
        try:
            self.logger.info("Removing septum from LV endocardium")

            remove_septum_from_endo(
                lv_endo_path=paths.lv_endo_surface,
                septum_path=paths.septum_surface,
                output_path=paths.lv_endo_surface,  # Overwrite
            )

            self.logger.info("Septum removed from LV endo")

        except Exception as e:
            msg = f"Failed to remove septum: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    def prepare_ventricular_vtx_files(self, paths: VentricularSurfacePaths) -> None:
        """
        Prepare VTX files required for UVC computation on ventricles.

        Args:
            paths: Ventricular surface paths contract

        Raises:
            SurfaceExtractionError: If preparation fails
        """
        try:
            self.logger.info("Preparing VTX files for UVC")

            vtx_mapping = {
                "base": paths.base_vtx,
                "epi": paths.epi_vtx,
                "lv_endo": paths.lv_endo_vtx,
                "rv_endo": paths.rv_endo_vtx,
                "septum": paths.septum_vtx,
                "apex": paths.apex_vtx,
                "rv_sept_pt": paths.rv_septum_point_vtx,
            }

            prepare_vtx_files_for_uvc(
                surface_dir=paths.output_dir, output_vtx_paths=vtx_mapping
            )

            self.logger.info("VTX files prepared")

        except Exception as e:
            msg = f"Failed to prepare VTX files: {e}"
            self.logger.error(msg)
            raise SurfaceExtractionError(msg) from e

    # ATRIAL SURFACE EXTRACTION

    def extract_atrial_base(
        self, paths: AtrialSurfacePaths, tags: TagsConfig, chamber: Chamber
    ) -> None:
        """
        Extract the base surface for an atrium (LA or RA).

        Args:
            paths: Atrial surface paths contract
            tags: Tag configuration
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} base surface")

            # Get atrial tag and valve tags
            atrial_tags = tags.get_tags_string([chamber.value])

            if chamber == Chamber.LA:
                valve_tags = tags.get_tags_string(["MV", "AV"])
            else:  # RA
                valve_tags_keys = ["TV", "PV"]
                if tags.PArt is not None:
                    valve_tags_keys.append("PArt")
                valve_tags = tags.get_tags_string(valve_tags_keys)

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
        self, paths: AtrialSurfacePaths, tags: TagsConfig, chamber: Chamber
    ) -> None:
        """
        Extract epicardium and endocardium for an atrium.

        Args:
            paths: Atrial surface paths contract
            tags: Tag configuration
            chamber: Chamber.LA or Chamber.RA

        Raises:
            SurfaceExtractionError: If extraction fails
        """
        try:
            self.logger.info(f"Extracting {chamber.value} epi and endo surfaces")

            # Get atrial tag
            atrial_tags = tags.get_tags_string([chamber.value])

            # Determine valve tags to exclude
            if chamber == Chamber.LA:
                valve_tags = tags.get_tags_string(["MV", "AV"])
            else:  # RA
                valve_tags_keys = ["TV", "PV"]
                if tags.PArt is not None:
                    valve_tags_keys.append("PArt")
                valve_tags = tags.get_tags_string(valve_tags_keys)

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
                output_submesh_path=paths.output_mesh,
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
                submesh_path=paths.output_mesh,
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
        tags: TagsConfig,
        files_to_map: Optional[List[Path]] = None,
    ) -> None:
        """
        Run complete ventricular surface extraction workflow.

        This executes all steps required to extract ventricular surfaces
        for UVC coordinate generation:
        1. Extract base surface
        2. Extract and identify epi, LV endo, RV endo
        3. Extract septum
        4. Map surfaces (if files provided)
        5. Remove septum from LV endo
        6. Prepare VTX files

        Args:
            paths: Ventricular surface paths contract
            tags: Tag configuration
            files_to_map: Optional list of data files to map onto surfaces

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING VENTRICULAR SURFACE EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_ventricular_base(paths, tags)
        self.extract_ventricular_surfaces(paths, tags)
        self.extract_septum(paths, tags)

        if files_to_map:
            self.map_ventricular_surfaces(paths, files_to_map)

        self.remove_septum_from_lv_endo(paths)
        self.prepare_ventricular_vtx_files(paths)

        self.logger.info("=" * 60)
        self.logger.info("VENTRICULAR EXTRACTION COMPLETE")
        self.logger.info("=" * 60)

    def run_atrial_extraction(
        self,
        paths: AtrialSurfacePaths,
        tags: TagsConfig,
        chamber: Chamber,
        files_to_map: Optional[List[Path]] = None,
    ) -> None:
        """
        Run complete atrial surface extraction workflow.

        Args:
            paths: Atrial surface paths contract
            tags: Tag configuration
            chamber: Chamber.LA or Chamber.RA
            files_to_map: Optional list of data files to map onto surfaces

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING {chamber.value} SURFACE EXTRACTION WORKFLOW")
        self.logger.info("=" * 60)

        self.extract_atrial_base(paths, tags, chamber)
        self.extract_atrial_surfaces(paths, tags, chamber)

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
        Run the complete UVC surface extraction workflow.

        This executes all extraction steps in the correct order:
        1. Ventricular surfaces
        2. LA surfaces
        3. RA surfaces
        4. BiV submesh and mapping
        5. LA submesh and mapping
        6. RA submesh and mapping

        Args:
            paths: Master paths contract containing all sub-paths
            tags: Tag configuration
            ventricular_files_to_map: Optional files to map for ventricles
            la_files_to_map: Optional files to map for LA
            ra_files_to_map: Optional files to map for RA

        Raises:
            SurfaceExtractionError: If any step fails
        """
        self.logger.info("#" * 60)
        self.logger.info("STARTING COMPLETE UVC SURFACE EXTRACTION")
        self.logger.info("#" * 60)

        # Phase 1: Extract ventricular surfaces
        self.run_ventricular_extraction(
            paths.ventricular, tags, ventricular_files_to_map
        )

        # Phase 2: Extract LA surfaces
        self.run_atrial_extraction(paths.left_atrial, tags, Chamber.LA, la_files_to_map)

        # Phase 3: Extract RA surfaces
        self.run_atrial_extraction(
            paths.right_atrial, tags, Chamber.RA, ra_files_to_map
        )

        # Phase 4: Extract BiV submesh
        self.run_biv_mesh_extraction(paths.biv_mesh, tags)

        # Phase 5: Extract LA submesh
        self.run_atrial_mesh_extraction(paths.la_mesh, tags, Chamber.LA)

        # Phase 6: Extract RA submesh
        self.run_atrial_mesh_extraction(paths.ra_mesh, tags, Chamber.RA)

        self.logger.info("#" * 60)
        self.logger.info("COMPLETE UVC SURFACE EXTRACTION FINISHED")
        self.logger.info("#" * 60)

    # HELPER METHODS

    def _rename_surface_files(
        self, source_base: Path, target_base: Path, formats: List[str]
    ) -> None:
        """
        Rename surface files from source to target base name.

        Args:
            source_base: Source base path (without extension)
            target_base: Target base path (without extension)
            formats: List of file extensions to rename
        """
        import shutil

        for fmt in formats:
            source = source_base.with_suffix(f".{fmt}")
            target = target_base.with_suffix(f".{fmt}")

            if source.exists():
                shutil.move(str(source), str(target))
                self.logger.debug(f"Renamed {source.name} -> {target.name}")
            else:
                self.logger.warning(f"Source file not found: {source}")

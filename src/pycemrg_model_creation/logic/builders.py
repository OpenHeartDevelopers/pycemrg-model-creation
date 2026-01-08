# src/pycemrg_model_creation/logic/builders.py

from pathlib import Path
from typing import List, Union

from .contracts import (
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths,
)


class PathContractBuilder:
    """
    A helper class to simplify the creation of path contract dataclasses.

    This builder takes a few root directories and a mesh name, then uses
    standard naming conventions to populate the detailed path contracts,
    reducing boilerplate for the user.
    """

    def __init__(self, output_dir: Union[Path, str]):
        """
        Initializes the builder with a main output directory.

        Args:
            output_dir (Union[Path, str]): The root directory where all generated
                                           subdirectories and files will be placed.
        """
        self.root_output_dir = Path(output_dir)

        # Define and create the primary structured directories
        self.biv_dir = self.root_output_dir / "BiV"
        self.la_dir = self.root_output_dir / "LA"
        self.ra_dir = self.root_output_dir / "RA"

        # Temporary directories for intermediate files
        self.biv_tmp_dir = self.biv_dir / "tmp"
        self.la_tmp_dir = self.la_dir / "tmp"
        self.ra_tmp_dir = self.ra_dir / "tmp"

        # Ensure all necessary directories exist upon initialization
        for d in [
            self.biv_dir,
            self.la_dir,
            self.ra_dir,
            self.biv_tmp_dir,
            self.la_tmp_dir,
            self.ra_tmp_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def build_ventricular_paths(self, mesh_base_path: Path) -> VentricularSurfacePaths:
        """Constructs the VentricularSurfacePaths contract."""
        return VentricularSurfacePaths(
            # Input
            mesh=mesh_base_path,
            # Directories
            output_dir=self.biv_dir,
            tmp_dir=self.biv_tmp_dir,
            # Intermediate surfaces
            base_surface=self.biv_tmp_dir / "base",
            epi_endo_combined=self.biv_tmp_dir / "epi_endo",
            epi_endo_cc_base=self.biv_tmp_dir / "epi_endo_cc",
            septum_raw=self.biv_tmp_dir / "septum",
            septum_cc_base=self.biv_tmp_dir / "septum_cc",
            lv_epi_intermediate=self.biv_tmp_dir / "lv_epi_intermediate",
            # Final surfaces
            epi_surface=self.biv_dir / "biv_epi",
            lv_endo_surface=self.biv_dir / "biv_lvendo",
            rv_endo_surface=self.biv_dir / "biv_rvendo",
            septum_surface=self.biv_dir / "biv_septum",
            # VTX files
            base_vtx=self.biv_dir / "biv.base.vtx",
            epi_vtx=self.biv_dir / "biv.epi.vtx",
            lv_endo_vtx=self.biv_dir / "biv.lvendo.vtx",
            rv_endo_vtx=self.biv_dir / "biv.rvendo.vtx",
            septum_vtx=self.biv_dir / "biv.septum.vtx",
            apex_vtx=self.biv_dir / "biv.lvapex.vtx",
            rv_septum_point_vtx=self.biv_dir / "biv.rvsept_pt.vtx",
        )

    def build_atrial_paths(
        self, mesh_base_path: Path, chamber_prefix: str
    ) -> AtrialSurfacePaths:
        """
        Constructs the AtrialSurfacePaths contract for either LA or RA.

        Args:
            mesh_base_path (Path): Base path of the four-chamber mesh.
            chamber_prefix (str): 'la' for Left Atrium, 'ra' for Right Atrium.
        """
        if chamber_prefix.lower() not in ["la", "ra"]:
            raise ValueError("Chamber prefix must be 'la' or 'ra'.")

        output_dir = self.la_dir if chamber_prefix.lower() == "la" else self.ra_dir
        tmp_dir = self.la_tmp_dir if chamber_prefix.lower() == "la" else self.ra_tmp_dir

        return AtrialSurfacePaths(
            mesh=mesh_base_path,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            base_surface=tmp_dir / f"{chamber_prefix}_base",
            epi_endo_combined=tmp_dir / f"{chamber_prefix}_epi_endo",
            epi_surface=output_dir / f"{chamber_prefix}_epi.vtk",
            endo_surface=output_dir / f"{chamber_prefix}_endo.vtk",
            base_vtx=output_dir / f"{chamber_prefix}.base.vtx",
            epi_vtx=output_dir / f"{chamber_prefix}.epi.vtx",
            endo_vtx=output_dir / f"{chamber_prefix}.endo.vtx",
            apex_vtx=output_dir / f"{chamber_prefix}.lvapex.vtx",
            rv_septum_point_vtx=output_dir / f"{chamber_prefix}.rvsept_pt.vtx",
        )

    def build_biv_mesh_paths(
        self, mesh_base_path: Path, ventricular_paths: VentricularSurfacePaths
    ) -> BiVMeshPaths:
        """Constructs the BiVMeshPaths contract."""
        return BiVMeshPaths(
            source_mesh=mesh_base_path,
            output_mesh=self.biv_dir / "myocardium_biv",
            output_dir=self.biv_dir,
            vtx_files_to_map=[
                ventricular_paths.base_vtx,
                ventricular_paths.epi_vtx,
                ventricular_paths.lv_endo_vtx,
                ventricular_paths.rv_endo_vtx,
                ventricular_paths.apex_vtx,
                ventricular_paths.rv_septum_point_vtx,
            ],
            mapped_vtx_output_dir=self.biv_dir / "biv",
        )

    def build_atrial_mesh_paths(
        self,
        mesh_base_path: Path,
        atrial_paths: AtrialSurfacePaths,
        blank_files_dir: Path,
        chamber_prefix: str,
    ) -> AtrialMeshPaths:
        """Constructs the AtrialMeshPaths contract."""
        output_dir = self.la_dir if chamber_prefix.lower() == "la" else self.ra_dir

        return AtrialMeshPaths(
            source_mesh=mesh_base_path,
            output_mesh=output_dir / f"myocardium_{chamber_prefix}",
            output_dir=output_dir,
            vtx_files_to_map=[
                atrial_paths.base_vtx,
                atrial_paths.epi_vtx,
                atrial_paths.endo_vtx,
            ],
            mapped_vtx_output_dir=output_dir / chamber_prefix,
            apex_template=blank_files_dir / f"{chamber_prefix}.lvapex.vtx",
            rv_septum_template=blank_files_dir / f"{chamber_prefix}.rvsept_pt.vtx",
            apex_output=atrial_paths.apex_vtx,
            rv_septum_output=atrial_paths.rv_septum_point_vtx,
        )

    def build_all(
        self, mesh_base_path: Path, blank_files_dir: Path
    ) -> UVCSurfaceExtractionPaths:
        """
        Constructs the master path contract for the entire workflow.
        This is a convenience method that calls all other builders.

        Args:
            mesh_base_path (Path): Base path of the four-chamber mesh.
            blank_files_dir (Path): Directory containing template/blank files for
                                    atrial apex/septum points.

        Returns:
            A fully populated UVCSurfaceExtractionPaths dataclass instance.
        """
        # Build contracts in logical order, as some depend on others
        vent_paths = self.build_ventricular_paths(mesh_base_path)
        la_paths = self.build_atrial_paths(mesh_base_path, "la")
        ra_paths = self.build_atrial_paths(mesh_base_path, "ra")

        biv_mesh_paths = self.build_biv_mesh_paths(mesh_base_path, vent_paths)
        la_mesh_paths = self.build_atrial_mesh_paths(
            mesh_base_path, la_paths, blank_files_dir, "la"
        )
        ra_mesh_paths = self.build_atrial_mesh_paths(
            mesh_base_path, ra_paths, blank_files_dir, "ra"
        )

        return UVCSurfaceExtractionPaths(
            ventricular=vent_paths,
            left_atrial=la_paths,
            right_atrial=ra_paths,
            biv_mesh=biv_mesh_paths,
            la_mesh=la_mesh_paths,
            ra_mesh=ra_mesh_paths,
        )

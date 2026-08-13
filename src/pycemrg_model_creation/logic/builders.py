# src/pycemrg_model_creation/logic/builders.py

import logging
from pathlib import Path
from typing import List, Union

from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.logic.contracts import (
    MeshPostprocessingPaths,
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths,
    VentricularUVCPaths,
)

logger = logging.getLogger(__name__)

class RefinementPathBuilder:
    """
    Builds path contracts for the mesh refinement workflow.

    This builder is initialized with a single root output directory and
    names the subdirectories and files needed for post-processing/refinement
    (RefinementLogic).

    Naming only: nothing here touches the file system. The directories are
    created by ``RefinementLogic`` at the start of a run, so that building a
    contract in order to inspect it has no side effect on disk.

    The raw volumetric mesh is an input, produced upstream of this library,
    so the builder does not own or name a directory for it.
    """

    def __init__(
        self,
        output_dir: Union[Path, str],
        refined_subdir: str = "refined",
    ):
        """
        Initializes the builder with a main output directory for all
        refinement-related activities.

        Args:
            output_dir: The root directory for refinement outputs.
            refined_subdir: Name of the subdirectory holding the refined mesh.
                            Earlier workflows used "02_refined", pass it here
                            to preserve that behaviour.
        """
        self.root_output_dir = Path(output_dir)

        # Define the structured subdirectories. Not created here.
        self.refined_mesh_dir = self.root_output_dir / refined_subdir
        self.tmp_dir = self.root_output_dir / "tmp"

    def build_postprocessing_paths(
        self,
        input_mesh_base: Union[CarpMesh, Path, str],
        refined_mesh_basename: str = "myocardium_clean"
    ) -> MeshPostprocessingPaths:
        """
        Constructs the MeshPostprocessingPaths contract for the refinement workflow.

        Args:
            input_mesh_base: The raw volumetric mesh to be processed (produced
                             upstream of this library), as a CarpMesh or as the
                             stem to build one from. Passing a path that still
                             carries a mesh extension is an error, and CarpMesh
                             says so.
            refined_mesh_basename: The base name for the final, refined mesh.

        Returns:
            A fully populated MeshPostprocessingPaths dataclass instance.
        """
        # CarpMesh normalises str, Path and CarpMesh alike -- it is os.PathLike,
        # so re-wrapping one that is already a handle is a no-op.
        return MeshPostprocessingPaths(
            input_mesh_base=CarpMesh(input_mesh_base),
            intermediate_myocardium_mesh=CarpMesh(
                self.tmp_dir / "myocardium_intermediate"
            ),
            output_mesh_base=CarpMesh(self.refined_mesh_dir / refined_mesh_basename),
        )

class ModelCreationPathBuilder:
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

        # The BiV submesh, constructed once. Two contracts refer to this same
        # mesh: `BiVMeshPaths` cuts it, and `VentricularSurfacePaths` names the
        # boundary VTX files against it. Building it here means the stem `BiV`
        # is spelled in one place, so the two contracts cannot disagree.
        self.biv_mesh = CarpMesh(self.biv_dir / "BiV")

        # Temporary directories for intermediate files. There is no BiV
        # equivalent: `VentricularSurfacePaths.tmp_dir` is a property, so the
        # BiV scratch name is owned by the contract and restating it here is
        # exactly the drift this refactor removes. LA/RA remain because their
        # contracts still carry `tmp_dir` as a field, and because the mkdir
        # below needs them.
        self.la_tmp_dir = self.la_dir / "tmp"
        self.ra_tmp_dir = self.ra_dir / "tmp"

        # Naming must not touch disk: constructing a contract to inspect it
        # should have no effect. `SurfaceLogic._prepare_directories` creates
        # the BiV tree when the workflow actually runs.
        #
        # LA/RA are still created here because the atrial flow has not been
        # refactored and has nowhere else to do it. They move with it.
        for d in [self.la_dir, self.ra_dir, self.la_tmp_dir, self.ra_tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def build_ventricular_paths(
        self, mesh_base_path: Union[CarpMesh, Path, str]
    ) -> VentricularSurfacePaths:
        """
        Constructs the VentricularSurfacePaths contract.

        Args:
            mesh_base_path: The four-chamber mesh stem, without extensions. A
                            stem carrying a mesh extension is an error, and
                            CarpMesh raises rather than silently truncating it.
        """
        return VentricularSurfacePaths(
            # Input
            mesh=CarpMesh(mesh_base_path),
            # Output root. `tmp_dir` and every intermediate and final surface
            # name is now derived from this by the contract itself, so the
            # builder no longer restates them — there was one spelling of
            # `biv_epi` here and another in the logic, and they could drift.
            output_dir=self.biv_dir,
            # The BiV submesh the boundary VTX files are named against. The
            # five names are now properties on the contract, asked of this
            # handle, so the builder no longer restates them.
            #
            # `apex_vtx` stays retired: for BiV, mguvc *writes* `BiV.lvapex.vtx`.
            # `rv_septum_point_vtx` is gone from here too — mguvc asks for
            # `rvsept_pt` only under `--output-model lv`, which BiV never uses.
            submesh=self.biv_mesh,
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
            # Stem `BiV`, not `myocardium_biv`. This is mguvc's vocabulary, and
            # it is what the UVC stage already expects beside the mesh
            # (`BiV.pts`, `BiV.base.vtx`). This renames files on disk.
            output_mesh=self.biv_mesh,
            output_dir=self.biv_dir,
            # Every boundary the surface stage writes must be remapped, or the
            # UVC stage reads four-chamber indices against a BiV mesh.
            #
            # This list used to name `rv_septum_point_vtx`, which nothing in
            # this flow writes, and to omit `rvsept`, which mguvc requires. So
            # it asked meshtool to map a file that did not exist and skipped one
            # that did.
            #
            # It also named the *submesh* generation, which is the output of the
            # mapping rather than its input. It now names the four-chamber
            # generation in tmp/, and carries the rename targets alongside.
            vtx_files_to_map=[
                source for source, _ in ventricular_paths.boundary_vtx_pairs()
            ],
            renamed_vtx_targets=[
                target for _, target in ventricular_paths.boundary_vtx_pairs()
            ],
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
    
    def build_ventricular_uvc_paths(
        self,
        biv_mesh: Union[CarpMesh, Path, str],
        output_subdir: str = "uvc",
    ) -> VentricularUVCPaths:
        """
        Build the path contract for ventricular UVC calculation.

        Encodes the two naming conventions this library chooses, and nothing
        else: the UVC outputs go in a subdirectory beside the mesh, and the
        etags script sits next to the mesh rather than inside that
        subdirectory.

        The etags location is deliberate. mguvc prompts interactively if its
        --output-dir already exists, and that prompt cannot reach a user of
        this library, so nothing may create that directory ahead of the run.

        Boundary inputs and outputs are not stored. Ask for them:
        `tools.mguvc.boundary_inputs(paths.biv_mesh)` and
        `tools.mguvc.outputs_for(paths.biv_mesh, paths.output_dir)`.

        Args:
            biv_mesh: BiV submesh stem, without extension, e.g.
                      /data/surfaces_uvc/BiV/BiV. Its directory must also hold
                      the four boundary VTX files mguvc reads.
            output_subdir: Name of the UVC output directory, created beside
                           the mesh by mguvc itself.

        Returns:
            A VentricularUVCPaths contract.

        Note:
            Pure construction — no directory is created, moved or removed.
            Clearing a stale output directory before a run belongs to
            UvcLogic, which knows when the run actually happens.

        Example:
            >>> builder = ModelCreationPathBuilder(output_dir=Path("/data/surfaces"))
            >>> paths = builder.build_ventricular_uvc_paths(
            ...     biv_mesh=Path("/data/surfaces/BiV/BiV")
            ... )
            >>> paths.output_dir
            PosixPath('/data/surfaces/BiV/uvc')
        """
        mesh = biv_mesh if isinstance(biv_mesh, CarpMesh) else CarpMesh(biv_mesh)

        return VentricularUVCPaths(
            biv_mesh=mesh,
            output_dir=mesh.directory / output_subdir,
            etags_file=mesh.role("etags", ".sh"),
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

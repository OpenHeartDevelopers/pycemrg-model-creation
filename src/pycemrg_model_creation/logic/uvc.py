# src/pycemrg_model_creation/logic/uvc.py
"""
UVC (Universal Ventricular Coordinate) calculation logic.

This module orchestrates the generation of UVC coordinate systems
for biventricular cardiac meshes using CARPentry tools (mguvc).
"""

import logging
import shutil
from datetime import datetime

from pycemrg_model_creation.logic.contracts import VentricularUVCPaths
from pycemrg_model_creation.tools.mguvc import boundary_inputs, outputs_for
from pycemrg_model_creation.tools.wrappers import CarpWrapper
from pycemrg_model_creation.utilities.uvc import write_etags_file


class UvcLogic:
    """
    Stateless logic for UVC coordinate generation workflows.

    Orchestrates CARPentry's mguvc tool to calculate Universal Ventricular
    Coordinates on biventricular meshes. The `VentricularUVCPaths` contract
    carries the three choices the caller actually makes — which mesh, where
    outputs go, where the etags script is written. Boundary inputs and output
    filenames are asked of `tools.mguvc`, which owns mguvc's naming, rather
    than being restated here.

    The UVC coordinate system consists of four fields:
    - Z (uvc_z): Apico-basal coordinate (apex=0, base=1)
    - Rho (uvc_rho): Transmural coordinate (endo=0, epi=1)
    - Phi (uvc_phi): Rotational/circumferential coordinate
    - Ven (uvc_ven): Ventricular identifier (LV vs RV)
    """

    def __init__(self, carp_wrapper: CarpWrapper):
        """
        Initialize UvcLogic with dependencies.

        Args:
            carp_wrapper: Initialized CarpWrapper for running mguvc
        """
        self.carp = carp_wrapper
        self.logger = logging.getLogger(__name__)

    def run_ventricular_uvc_calculation(
        self,
        paths: VentricularUVCPaths,
        lv_tag: int,
        rv_tag: int,
        np: int = 1,
        backup_existing: bool = True,
        overwrite_existing: bool = True,
    ) -> None:
        """
        Complete ventricular UVC calculation workflow.

        Executes the full pipeline:
        1. Validate input files exist (mesh + boundary VTX files)
        2. Generate etags script for element region mapping
        3. Clear any stale output directory
        4. Run mguvc to solve Laplace equations

        Args:
            paths: Path contract for the UVC workflow
            lv_tag: Element tag value for LV myocardium in the mesh
            rv_tag: Element tag value for RV myocardium in the mesh
            np: Number of processors for mguvc (default: 1)
            backup_existing: Move an existing output directory aside rather
                             than deleting it (default: True)
            overwrite_existing: Delete an existing output directory when not
                                backing it up (default: True)

        Raises:
            FileNotFoundError: If required input files are missing
            FileExistsError: If the output directory exists and neither
                             backup_existing nor overwrite_existing is set
            RuntimeError: If mguvc execution fails or outputs are invalid

        Example:
            >>> uvc_logic.run_ventricular_uvc_calculation(
            ...     paths=uvc_paths,
            ...     lv_tag=10,
            ...     rv_tag=20,
            ...     np=4
            ... )
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ventricular UVC calculation")
        self.logger.info(f"LV tag: {lv_tag}, RV tag: {rv_tag}, Processors: {np}")
        self.logger.info("=" * 60)

        self._validate_inputs(paths)
        self._generate_etags(paths, lv_tag, rv_tag)
        self._prepare_output_dir(paths, backup_existing, overwrite_existing)
        self._run_mguvc(paths, np)

        self.logger.info("=" * 60)
        self.logger.info("UVC calculation completed successfully")
        self.logger.info(f"UVC coordinates written to: {paths.output_dir}")
        self.logger.info("=" * 60)

    def _validate_inputs(self, paths: VentricularUVCPaths) -> None:
        """
        Validate the files mguvc will read.

        That is the mesh (.pts, .elem) and the four boundary VTX files named
        by `tools.mguvc.boundary_inputs`.

        Four, not six: mguvc derives epi, lvepi and rvendo_nosept itself and
        never opens files of those names. Requiring them here rejected valid
        input directories.

        Args:
            paths: VentricularUVCPaths contract

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

        missing_files = []
        for boundary, vtx_path in boundary_inputs(mesh).items():
            if vtx_path.exists():
                self.logger.debug(f"✓ {boundary.value} VTX: {vtx_path.name}")
            else:
                missing_files.append(f"  - {boundary.value}: {vtx_path}")

        if missing_files:
            raise FileNotFoundError(
                "Required VTX boundary files not found:\n" + "\n".join(missing_files)
            )

        self.logger.debug("All input files validated successfully")

    def _generate_etags(
        self,
        paths: VentricularUVCPaths,
        lv_tag: int,
        rv_tag: int
    ) -> None:
        """
        Generate etags bash script for element region mapping.

        The etags script maps mesh element tags to anatomical regions
        for the mguvc solver using bash variable definitions.

        Args:
            paths: VentricularUVCPaths contract
            lv_tag: Element tag for LV myocardium
            rv_tag: Element tag for RV myocardium
        """
        self.logger.info(f"Generating etags script: LV={lv_tag}, RV={rv_tag}")

        write_etags_file(
            output_path=paths.etags_file,
            lv_tag=lv_tag,
            rv_tag=rv_tag,
            mode='base'
        )

        self.logger.debug(f"Etags script written: {paths.etags_file}")

    def _prepare_output_dir(
        self,
        paths: VentricularUVCPaths,
        backup_existing: bool,
        overwrite_existing: bool,
    ) -> None:
        """
        Clear a stale output directory, immediately before the run.

        mguvc prompts interactively when its --output-dir already exists, and
        that prompt cannot reach a user of this library, so the directory must
        be gone before mguvc starts. It is emphatically not created here —
        mguvc creates it.

        Done at run time rather than when the contract is built: constructing
        a path contract must not destroy a previous run's results, least of
        all for a run that may never happen.

        Args:
            paths: VentricularUVCPaths contract
            backup_existing: Move the directory aside instead of deleting it
            overwrite_existing: Delete the directory when not backing it up

        Raises:
            FileExistsError: If the directory exists and neither flag is set
        """
        uvc_dir = paths.output_dir

        if not uvc_dir.exists():
            return

        if backup_existing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = uvc_dir.parent / f"{uvc_dir.name}_backup_{timestamp}"
            self.logger.info(
                f"Backing up existing directory: {uvc_dir} → {backup_dir}"
            )
            shutil.move(str(uvc_dir), str(backup_dir))
        elif overwrite_existing:
            self.logger.warning(f"Removing existing directory: {uvc_dir}")
            shutil.rmtree(uvc_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {uvc_dir}\n"
                f"Use overwrite_existing=True to remove or "
                f"backup_existing=True to backup"
            )

    def _run_mguvc(self, paths: VentricularUVCPaths, np: int) -> None:
        """
        Execute mguvc tool via CarpWrapper.

        Args:
            paths: VentricularUVCPaths contract
            np: Number of processors
        """
        self.logger.info("Running mguvc to solve Laplace equations")

        outputs = outputs_for(paths.biv_mesh, paths.output_dir)

        self.carp.run_mguvc(
            model_name=paths.biv_mesh.stem,   # mguvc takes the stem
            input_model_type="biv",           # Input type: biventricular
            output_model_type="biv",          # Output type: biventricular
            tags_file=paths.etags_file,
            output_dir=paths.output_dir,
            np=np,
            laplace_solution=True,
            custom_apex=False,
            expected_outputs=list(outputs.coordinates),
        )

        # Optional outputs, logged when present. mguvc writes many more files
        # than these; outputs_for names the ones this library acts on.
        for group_name, group in (
            ("Laplace solutions", outputs.laplace_solutions),
            ("Mapping files", outputs.mappings),
        ):
            existing = [f.name for f in group if f.exists()]
            if existing:
                self.logger.debug(f"✓ {group_name}: {', '.join(existing)}")

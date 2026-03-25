# src/pycemrg_model_creation/logic/uvc.py
# CORRECTED VERSION - matches actual mguvc interface
"""
UVC (Universal Ventricular Coordinate) calculation logic.

This module orchestrates the generation of UVC coordinate systems
for biventricular cardiac meshes using CARPentry tools (mguvc).
"""

import logging
from pathlib import Path

from pycemrg_model_creation.logic.contracts import VentricularUVCPaths
from pycemrg_model_creation.tools.wrappers import CarpWrapper
from pycemrg_model_creation.utilities.uvc import write_etags_file


class UvcLogic:
    """
    Stateless logic for UVC coordinate generation workflows.
    
    Orchestrates CARPentry's mguvc tool to calculate Universal Ventricular
    Coordinates on biventricular meshes. All paths are provided explicitly
    through the VentricularUVCPaths contract.
    
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
        np: int = 1
    ) -> None:
        """
        Complete ventricular UVC calculation workflow.

        Executes the full pipeline:
        1. Validate input files exist (mesh + boundary VTX files)
        2. Generate etags script for element region mapping
        3. Run mguvc to solve Laplace equations
        4. Validate output coordinate files were created

        Args:
            paths: Complete path contract for UVC workflow
            lv_tag: Element tag value for LV myocardium in the mesh
            rv_tag: Element tag value for RV myocardium in the mesh
            np: Number of processors for mguvc (default: 1)

        Raises:
            FileNotFoundError: If required input files are missing
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

        # Step 1: Validate inputs
        self._validate_inputs(paths)

        # Step 2: Generate etags script
        self._generate_etags(paths, lv_tag, rv_tag)

        # Step 3: Run mguvc
        self._run_mguvc(paths, np)

        self.logger.info("=" * 60)
        self.logger.info("UVC calculation completed successfully")
        self.logger.info(f"UVC coordinates written to: {paths.output_dir}")
        self.logger.info("=" * 60)

    def _validate_inputs(self, paths: VentricularUVCPaths) -> None:
        """
        Validate all required input files exist.

        Checks:
        - Mesh files (.pts, .elem)
        - All 6 boundary VTX files

        Args:
            paths: VentricularUVCPaths contract

        Raises:
            FileNotFoundError: If any required file is missing
        """
        self.logger.debug("Validating input files")

        # Check mesh files
        mesh_pts = paths.biv_mesh.with_suffix(".pts")
        mesh_elem = paths.biv_mesh.with_suffix(".elem")

        if not mesh_pts.exists():
            raise FileNotFoundError(f"Mesh points file not found: {mesh_pts}")
        if not mesh_elem.exists():
            raise FileNotFoundError(f"Mesh elements file not found: {mesh_elem}")

        self.logger.debug(f"✓ Mesh files validated: {paths.biv_mesh}")

        # Check VTX boundary files (6 files)
        vtx_files = [
            ("base", paths.base_vtx),
            ("epi", paths.epi_vtx),
            ("lv_endo", paths.lv_endo_vtx),
            ("rv_endo", paths.rv_endo_vtx),
            ("septum (rvsept)", paths.septum_vtx),
            ("rv_endo_nosept", paths.rvendo_nosept_vtx)
        ]

        missing_files = []
        for name, vtx_path in vtx_files:
            if not vtx_path.exists():
                missing_files.append(f"  - {name}: {vtx_path}")
            else:
                self.logger.debug(f"✓ {name} VTX: {vtx_path.name}")

        if missing_files:
            raise FileNotFoundError(
                f"Required VTX boundary files not found:\n" +
                "\n".join(missing_files)
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
    
    def _run_mguvc(self, paths: VentricularUVCPaths, np: int) -> None:
        """
        Execute mguvc tool via CarpWrapper.

        Args:
            paths: VentricularUVCPaths contract
            np: Number of processors
        """
        self.logger.info("Running mguvc to solve Laplace equations")

        expected_outputs = [
            paths.uvc_z,
            paths.uvc_rho,
            paths.uvc_phi,
            paths.uvc_ven
        ]


        self.carp.run_mguvc(
            model_name=paths.biv_mesh,        # Full path to BiV mesh
            input_model_type="biv",           # Input type: biventricular
            output_model_type="biv",          # Output type: biventricular
            tags_file=paths.etags_file,
            output_dir=paths.output_dir,
            np=np,
            laplace_solution=True,
            custom_apex=False, 
            expected_outputs=expected_outputs,
        )

        # Log optional outputs if present
        self._log_optional_outputs(paths)

    
    def _log_optional_outputs(self, paths: VentricularUVCPaths) -> None:
        """Log which optional output files were created."""
        optional_files = [
            ("Laplace solutions", [paths.sol_apba, paths.sol_endoepi, 
                                   paths.sol_lvendo, paths.sol_rvendo]),
            ("Mapping files", [paths.aff_dat, paths.m2s_dat])
        ]
        
        for group_name, file_list in optional_files:
            existing = [f.name for f in file_list if f.exists()]
            if existing:
                self.logger.debug(f"✓ {group_name}: {', '.join(existing)}")

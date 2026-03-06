# src/pycemrg_model_creation/logic/contracts.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class MeshingPaths:
    """Path contract for the meshtools3d meshing workflow."""

    # Input
    input_segmentation_nifti: Path

    # Directories
    output_dir: Path
    tmp_dir: Path

    # Intermediate files (in tmp_dir)
    intermediate_inr: Path
    intermediate_parameter_file: Path

    # Final output (in output_dir)
    # This is a base path; extensions (.vtk, .pts, .elem) will be added.
    output_mesh_base: Path

@dataclass
class MeshPostprocessingPaths:
    """Path contract for the mesh post-processing workflow."""

    # Input
    # Base path to the raw mesh from meshtools3d (e.g., .../heart_mesh)
    input_mesh_base: Path

    # Directories
    output_dir: Path
    tmp_dir: Path

    # Intermediate mesh (in tmp_dir)
    # The mesh after extracting myocardium tags, before optional simplification
    intermediate_myocardium_mesh: Path

    # Final output mesh (in output_dir)
    # The final, cleaned, and relabeled mesh base name
    output_mesh_base: Path

@dataclass
class VentricularSurfacePaths:
    """
    Path contract for ventricular (BiV) surface extraction.

    All paths must be provided by the orchestrator. The logic layer
    never derives or constructs paths.
    """

    # Input
    mesh: Path  # Full four-chamber mesh base name (without extensions)

    # Directories
    output_dir: Path
    tmp_dir: Path

    # Intermediate surfaces (in tmp_dir)
    base_surface: Path
    epi_endo_combined: Path  # Before separation
    epi_endo_cc_base: Path  # Base name for connected components
    septum_raw: Path
    septum_cc_base: Path  # Base name for septum connected components
    lv_epi_intermediate: Path  # Intermediate septum extraction result

    # Final surfaces (in output_dir)
    epi_surface: Path
    lv_endo_surface: Path
    rv_endo_surface: Path
    septum_surface: Path

    # VTX files for UVC (in output_dir)
    base_vtx: Path
    epi_vtx: Path
    lv_endo_vtx: Path
    rv_endo_vtx: Path
    septum_vtx: Path
    #    apex_vtx: Path
    rv_septum_point_vtx: Path


@dataclass
class AtrialSurfacePaths:
    """
    Path contract for atrial surface extraction (LA or RA).
    """

    # Input
    mesh: Path  # Full four-chamber mesh base name

    # Directories
    output_dir: Path
    tmp_dir: Path

    # Intermediate surfaces (in tmp_dir)
    base_surface: Path
    epi_endo_combined: Path

    # Final surfaces (in output_dir)
    epi_surface: Path
    endo_surface: Path

    # VTX files (in output_dir)
    base_vtx: Path
    epi_vtx: Path
    endo_vtx: Path

    # Blank files for compatibility
    apex_vtx: Path
    rv_septum_point_vtx: Path


@dataclass
class BiVMeshPaths:
    """
    Path contract for BiV mesh extraction and mapping.
    """

    source_mesh: Path  # Full four-chamber mesh
    output_mesh: Path  # BiV submesh
    output_dir: Path

    # VTX files to map from four-chamber to BiV
    vtx_files_to_map: List[Path]
    mapped_vtx_output_dir: Path


@dataclass
class AtrialMeshPaths:
    """
    Path contract for LA/RA mesh extraction and mapping.
    """

    source_mesh: Path
    output_mesh: Path
    output_dir: Path

    # VTX files to map
    vtx_files_to_map: List[Path]
    mapped_vtx_output_dir: Path

    # Template files for apex/septum (blank files)
    apex_template: Path
    rv_septum_template: Path
    apex_output: Path
    rv_septum_output: Path


@dataclass
class UVCSurfaceExtractionPaths:
    """
    Master path contract for complete UVC surface extraction workflow.
    """

    ventricular: VentricularSurfacePaths
    left_atrial: AtrialSurfacePaths
    right_atrial: AtrialSurfacePaths
    biv_mesh: BiVMeshPaths
    la_mesh: AtrialMeshPaths
    ra_mesh: AtrialMeshPaths

@dataclass(frozen=True)
class VentricularUVCPaths:
    """
    Path contract for ventricular UVC (Universal Ventricular Coordinate) calculation.
    
    All paths are explicit. The logic layer never derives paths.
    
    CRITICAL: mguvc expects the BiV mesh and all VTX boundary files to be in 
    the SAME directory with these exact standard names:
    - base.vtx, epi.vtx, lvendo.vtx, rvendo.vtx, rvsept.vtx, rvendo_nosept.vtx
    
    The workflow:
    1. BiV mesh is extracted from four-chamber mesh
    2. VTX files are mapped from four-chamber to BiV space
    3. Both mesh and VTX files are placed in the same directory
    4. mguvc reads from that directory and writes outputs to output_dir
    """
    # Input: BiV submesh (extracted from four-chamber mesh)
    # Path WITHOUT extension (e.g., /data/surfaces_uvc/BiV/BiV)
    # The directory must contain: BiV.pts, BiV.elem, and all VTX files
    biv_mesh: Path
    
    # Input: Boundary condition VTX files (for validation only)
    # These MUST exist in biv_mesh.parent with standard names
    # Listed here for explicit validation in logic layer
    base_vtx: Path          # biv_mesh.parent / "base.vtx"
    epi_vtx: Path           # biv_mesh.parent / "epi.vtx"
    lv_endo_vtx: Path       # biv_mesh.parent / "lvendo.vtx"
    rv_endo_vtx: Path       # biv_mesh.parent / "rvendo.vtx"
    septum_vtx: Path        # biv_mesh.parent / "rvsept.vtx"
    rvendo_nosept_vtx: Path # biv_mesh.parent / "rvendo_nosept.vtx"
    
    # Input: Element tags configuration (bash script with T_LV, T_RV definitions)
    etags_file: Path
    
    # Output directory for UVC coordinate files
    # Typically: biv_mesh.parent / "uvc"
    output_dir: Path
    
    # Output: Primary UVC coordinate files (CARP .dat format)
    # All use biv_mesh.name as basename (e.g., BiV.uvc_z.dat)
    uvc_z: Path           # Apico-basal coordinate (apex=0, base=1)
    uvc_rho: Path         # Transmural coordinate (endo=0, epi=1)
    uvc_phi: Path         # Rotational coordinate
    uvc_ven: Path         # Ventricular identifier (LV vs RV)
    
    # Output: Intermediate Laplace solutions (for debugging/validation)
    sol_apba: Path        # Apico-basal Laplace solution
    sol_endoepi: Path     # Endo-epi Laplace solution
    sol_lvendo: Path      # LV endo Laplace solution
    sol_rvendo: Path      # RV endo Laplace solution
    
    # Output: Mapping files
    aff_dat: Path         # Affine transformation data
    m2s_dat: Path         # Mesh-to-surface mapping
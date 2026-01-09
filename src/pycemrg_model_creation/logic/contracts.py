# src/pycemrg_model_creation/logic/contracts.py

from dataclasses import dataclass
from pathlib import Path
from typing import List


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

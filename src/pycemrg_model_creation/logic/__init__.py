# src/pycemrg_model_creation/logic/__init__.py

"""
This module provides high-level, multi-step workflows and scientific
logic built on top of the low-level tool wrappers.
"""

from pycemrg_model_creation.logic.contracts import (
    MeshingPaths,
    MeshPostprocessingPaths,
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths, 
    VentricularUVCPaths,
)

from pycemrg_model_creation.logic.builders import (
    MeshingPathBuilder, 
    ModelCreationPathBuilder,
)

from pycemrg_model_creation.logic.surfaces import SurfaceLogic
from pycemrg_model_creation.logic.meshing import MeshingLogic
from pycemrg_model_creation.logic.refinement import RefinementLogic
from pycemrg_model_creation.logic.uvc import UvcLogic

__all__ = [
    # Logic classes
    "SurfaceLogic",
    "MeshingLogic",
    "RefinementLogic",
    "UvcLogic",
    # Path contracts
    "MeshingPaths",
    "MeshPostprocessingPaths",
    "VentricularSurfacePaths",
    "AtrialSurfacePaths",
    "BiVMeshPaths",
    "AtrialMeshPaths",
    "UVCSurfaceExtractionPaths",
    "VentricularUVCPaths",
    # Path builders
    "MeshingPathBuilder",
    "ModelCreationPathBuilder",
]

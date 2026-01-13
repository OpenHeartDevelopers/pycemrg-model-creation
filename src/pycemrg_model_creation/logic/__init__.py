# src/pycemrg_model_creation/logic/__init__.py

"""
This module provides high-level, multi-step workflows and scientific
logic built on top of the low-level tool wrappers.
"""

from .contracts import (
    MeshingPaths,
    MeshPostprocessingPaths,
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths  
)

from .builders import (
    MeshingPathBuilder, 
    ModelCreationPathBuilder
)

from .surfaces import SurfaceLogic
from .meshing import MeshingLogic
from .refinement import RefinementLogic

__all__ = [
    "SurfaceLogic",
    "MeshingLogic",
    "RefinementLogic",
    "MeshingPaths",
    "MeshPostprocessingPaths",
    "ModelCreationPathBuilder",
    "VentricularSurfacePaths",
    "AtrialSurfacePaths",
    "BiVMeshPaths",
    "AtrialMeshPaths",
    "UVCSurfaceExtractionPaths"
]

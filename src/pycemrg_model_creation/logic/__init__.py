# src/pycemrg_model_creation/logic/__init__.py

"""
This module provides high-level, multi-step workflows and scientific
logic built on top of the low-level tool wrappers.
"""

from pycemrg_model_creation.logic.contracts import (
    MeshPostprocessingPaths,
    VentricularSurfacePaths,
    AtrialSurfacePaths,
    BiVMeshPaths,
    AtrialMeshPaths,
    UVCSurfaceExtractionPaths,
    VentricularUVCPaths,
    VentricularFibrePaths,
)

from pycemrg_model_creation.logic.builders import (
    RefinementPathBuilder,
    ModelCreationPathBuilder,
)

from pycemrg_model_creation.logic.surfaces import SurfaceLogic
from pycemrg_model_creation.logic.refinement import RefinementLogic
from pycemrg_model_creation.logic.uvc import UvcLogic
from pycemrg_model_creation.logic.fibres import FibreLogic

__all__ = [
    # Logic classes
    "SurfaceLogic",
    "RefinementLogic",
    "UvcLogic",
    "FibreLogic",
    # Path contracts
    "MeshPostprocessingPaths",
    "VentricularSurfacePaths",
    "AtrialSurfacePaths",
    "BiVMeshPaths",
    "AtrialMeshPaths",
    "UVCSurfaceExtractionPaths",
    "VentricularUVCPaths",
    "VentricularFibrePaths",
    # Path builders
    "RefinementPathBuilder",
    "ModelCreationPathBuilder",
]

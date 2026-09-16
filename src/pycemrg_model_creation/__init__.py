# src/pycemrg_model_creation/__init__.py

"""
pycemrg-carp-tools: A Pythonic SDK for the CARPentry/openCARP ecosystem.
"""

from pycemrg_model_creation import tools
from pycemrg_model_creation import logic
from pycemrg_model_creation import utilities

# Elevate the most important user-facing classes for convenience
from pycemrg_model_creation.config import label_manager_from_mapping
from pycemrg_model_creation.meshpaths import (
    CarpMesh,
    CarpSurface,
    SurfaceMesh,
    SubmeshIndex,
)
from pycemrg_model_creation.logic import SurfaceLogic
from pycemrg_model_creation.tools import CarpWrapper, MeshtoolWrapper

__all__ = [
    "CarpMesh",
    "CarpSurface",
    "SurfaceMesh",
    "SubmeshIndex",
    "label_manager_from_mapping",
    "SurfaceLogic",
    "CarpWrapper",
    "MeshtoolWrapper",
    "tools",
    "logic",
    "utilities",
]


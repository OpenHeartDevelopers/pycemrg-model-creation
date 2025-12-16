# src/pycemrg_carp_tools/__init__.py

"""
pycemrg-carp-tools: A Pythonic SDK for the CARPentry/openCARP ecosystem.
"""

from . import tools
from . import logic
from . import utilities

# Elevate the most important user-facing classes for convenience
from .config import TagsConfig
from .logic import SurfaceLogic
from .tools import CarpWrapper, MeshtoolWrapper

__all__ = [
    "TagsConfig",
    "SurfaceLogic",
    "CarpWrapper",
    "MeshtoolWrapper",
    "tools",
    "logic",
    "utilities",
]


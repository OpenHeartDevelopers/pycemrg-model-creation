# src/pycemrg_carp_tools/__init__.py

"""
pycemrg-carp-tools: A Pythonic SDK for the CARPentry/openCARP ecosystem.
"""

# Import the submodules themselves to make them available
from . import tools
from . import logic

# For user convenience, elevate the most commonly used classes to the top level
from .tools import CarpWrapper, MeshtoolWrapper

# --- Versioning ---
# # This is a standard way to make your package version accessible
# try:
#     from ._version import version as __version__
# except ImportError:
#     # This will be the case when the package is not installed
#     __version__ = "0.0.0.dev0"


__all__ = [
    # Expose the most important classes directly
    "CarpWrapper",
    "MeshtoolWrapper",
    # Expose the submodules for users who want to be explicit
    "tools",
    "logic",
]
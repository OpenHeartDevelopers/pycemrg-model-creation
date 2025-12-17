# src/pycemrg_model_creation/types.py

from enum import Enum


class Chamber(str, Enum):
    """Cardiac chamber identifiers"""

    LV = "LV"
    RV = "RV"
    LA = "LA"
    RA = "RA"


class SurfaceType(str, Enum):
    """Surface type identifiers"""

    EPI = "epi"
    ENDO = "endo"
    BASE = "base"
    SEPTUM = "septum"

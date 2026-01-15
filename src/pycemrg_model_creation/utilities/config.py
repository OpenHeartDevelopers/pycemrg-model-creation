# src/pycemrg_model_creation/utilities/config.py
"""
Utilities for generating configuration files for external tools.
"""
import configparser
from pathlib import Path


class Meshtools3DParameters:
    """
    A helper class to generate .par configuration files for meshtools3d.
    """
    DEFAULT_VALUES = {
        'segmentation': {
            'seg_dir': './', 'seg_name': 'seg.inr',
            'mesh_from_segmentation': 1, 'boundary_relabeling': 0,
        },
        'meshing': {
            'facet_angle': 30, 'facet_size': 0.8, 'facet_distance': 4,
            'cell_rad_edge_ratio': 2.0, 'cell_size': 0.8, 'rescaleFactor': 1000
        },
        'laplacesolver': {
            'abs_toll': 1e-6, 'rel_toll': 1e-6, 'itr_max': 700,
            'dimKrilovSp': 500, 'verbose': 1,
        },
        'others': {'eval_thickness': 0},
        'output': {
            'outdir': './out', 'name': 'mesh', 'out_medit': 0, 'out_carp': 1,
            'out_carp_binary': 0, 'out_vtk': 0, 'out_vtk_binary': 0,
            'out_potential': 0,
        }
    }

    def __init__(self):
        """Initializes the parameter builder with default values."""
        self.config = configparser.ConfigParser()
        # Preserve case sensitivity of keys
        self.config.optionxform = str
        self.config.read_dict(self.DEFAULT_VALUES)

    def update(self, section: str, option: str, value: str):
        """
        Updates a specific parameter in the configuration.

        Args:
            section: The section name (e.g., 'meshing').
            option: The parameter name (e.g., 'facet_size').
            value: The new value for the parameter.
        """
        self.config[section][option] = str(value)

    def save(self, filename: Path):
        """
        Saves the current configuration to a .par file.

        Args:
            filename: The path to the output file.
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as configfile:
            self.config.write(configfile)
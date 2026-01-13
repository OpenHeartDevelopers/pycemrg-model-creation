# src/pycemrg_model_creation/utilities/image.py
"""
Utilities for medical image format conversion.
"""
import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def convert_image_to_inr(nifti_path: Path, inr_path: Path) -> None:
    """
    Converts a NIfTI file to the INRIMAGE-4 format required by some legacy tools.

    This function is stateless and operates purely on the provided paths.

    Args:
        nifti_path: The path to the input NIfTI file (.nii or .nii.gz).
        inr_path: The path for the output .inr file.

    Raises:
        FileNotFoundError: If the input nifti_path does not exist.
        ValueError: If the pixel data type is not supported.
    """
    if not nifti_path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {nifti_path}")

    logger.info(f"Converting {nifti_path.name} to {inr_path.name}")
    image = sitk.ReadImage(str(nifti_path))

    data = sitk.GetArrayViewFromImage(image)
    spacing = image.GetSpacing()
    dtype = data.dtype
    bitlen = data.dtype.itemsize * 8

    btype_map = {
        np.bool_: "unsigned fixed", np.uint8: "unsigned fixed",
        np.uint16: "unsigned fixed", np.int16: "signed fixed",
        np.float32: "float", np.float64: "float"
    }
    btype = btype_map.get(dtype.type)
    if btype is None:
        raise ValueError(f"Volume format not supported for INR conversion: {dtype}")

    zdim, ydim, xdim = data.shape
    header = (
        f"#INRIMAGE-4#{{\n"
        f"XDIM={xdim}\nYDIM={ydim}\nZDIM={zdim}\nVDIM=1\n"
        f"VX={spacing[0]:.4f}\nVY={spacing[1]:.4f}\nVZ={spacing[2]:.4f}\n"
    )
    if 'fixed' in btype:
        header += "SCALE=2**0\n"
    header += f"TYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm"
    
    # Pad header to the required 256 bytes
    header += "\n" * (252 - len(header)) + "##}\n"

    inr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inr_path, "wb") as file:
        file.write(header.encode('utf-8'))
        file.write(data.tobytes())
    
    logger.info("INR conversion successful.")
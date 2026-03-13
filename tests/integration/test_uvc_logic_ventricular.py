# tests/integration/test_uvc_logic.py
"""
Integration tests for UVC coordinate generation workflows.
"""

import os
import shutil
import logging
import pytest
import numpy as np
from pathlib import Path

from pycemrg.system import CarpRunner, CommandRunner
from pycemrg_model_creation.logic import UvcLogic, ModelCreationPathBuilder
from pycemrg_model_creation.tools import CarpWrapper
from pycemrg.data import LabelManager


@pytest.fixture(scope="module")
def carp_wrapper(test_data_root):
    """
    Initialize CarpWrapper for UVC tests.
    
    Requires PYCEMRG_CARP_CONFIG environment variable.
    """
    carp_config_str = os.environ.get("PYCEMRG_CARP_CONFIG")
    if not carp_config_str:
        pytest.skip("PYCEMRG_CARP_CONFIG environment variable not set")
    
    carp_config = Path(carp_config_str)
    if not carp_config.exists():
        pytest.fail(f"CARP config not found: {carp_config}")
    
    runner = CommandRunner()
    carp_runner = CarpRunner(runner=runner, carp_config_path=carp_config)
    
    return CarpWrapper(carp_runner)


def test_ventricular_uvc_calculation(test_data_root, carp_wrapper, tmp_path):
    """
    End-to-end test for ventricular UVC generation.
    
    Test data structure:
        test_data_root/
        └── ventricular_uvc/
            ├── config/
            │   └── labels.yaml
            ├── input_mesh/
            │   ├── BiV.pts              # BiV submesh
            │   ├── BiV.elem
            │   ├── base.vtx             # Standard named VTX files
            │   ├── epi.vtx
            │   ├── lvendo.vtx
            │   ├── rvendo.vtx
            │   ├── rvsept.vtx
            │   └── rvendo_nosept.vtx
            └── outputs/                 # Optional expected outputs
                ├── BiV.uvc_z.dat
                ├── BiV.uvc_rho.dat
                ├── BiV.uvc_phi.dat
                └── BiV.uvc_ven.dat
    """
    # Setup test directories
    test_case_dir = test_data_root / "ventricular_uvc"
    
    if not test_case_dir.exists():
        pytest.skip(f"Test data not found: {test_case_dir}")
    
    config_dir = test_case_dir / "config"
    input_mesh_dir = test_case_dir / "input_mesh"
    expected_dir = test_case_dir / "outputs"
    
    # Load labels
    labels_path = config_dir / "labels.yaml"
    assert labels_path.exists(), f"Labels config not found: {labels_path}"
    
    label_manager = LabelManager(config_path=labels_path)
    lv_tag = label_manager.get_value("LV")
    rv_tag = label_manager.get_value("RV")
    
    logging.info(f"Loaded tags: LV={lv_tag}, RV={rv_tag}")
    
    # Copy mesh and VTX files into tmp_path so mguvc runs in an isolated,
    # clean directory and never touches the source test data.
    work_dir = tmp_path / "BiV"
    work_dir.mkdir()
    for suffix in (".pts", ".elem"):
        src = input_mesh_dir / f"BiV{suffix}"
        assert src.exists(), f"BiV mesh not found: {src}"
        shutil.copy(src, work_dir / src.name)
    for vtx_file in input_mesh_dir.glob("*.vtx"):
        shutil.copy(vtx_file, work_dir / vtx_file.name)

    biv_mesh = work_dir / "BiV"

    # Build UVC paths contract
    builder = ModelCreationPathBuilder(output_dir=tmp_path)

    uvc_paths = builder.build_ventricular_uvc_paths(biv_mesh=biv_mesh)
    
    # Initialize UVC logic
    uvc_logic = UvcLogic(carp_wrapper)
    
    # Run UVC calculation
    logging.info("Running UVC calculation...")
    uvc_logic.run_ventricular_uvc_calculation(
        paths=uvc_paths,
        lv_tag=lv_tag,
        rv_tag=rv_tag,
        np=1
    )
    
    # Validate outputs exist
    logging.info("Validating outputs...")
    assert uvc_paths.uvc_z.exists(), f"UVC Z not generated: {uvc_paths.uvc_z}"
    assert uvc_paths.uvc_rho.exists(), f"UVC Rho not generated: {uvc_paths.uvc_rho}"
    assert uvc_paths.uvc_phi.exists(), f"UVC Phi not generated: {uvc_paths.uvc_phi}"
    assert uvc_paths.uvc_ven.exists(), f"UVC Ven not generated: {uvc_paths.uvc_ven}"
    
    # Validate files are non-empty
    assert uvc_paths.uvc_z.stat().st_size > 0, "UVC Z is empty"
    assert uvc_paths.uvc_rho.stat().st_size > 0, "UVC Rho is empty"
    assert uvc_paths.uvc_phi.stat().st_size > 0, "UVC Phi is empty"
    assert uvc_paths.uvc_ven.stat().st_size > 0, "UVC Ven is empty"
    
    # Validate coordinate ranges
    def read_dat_file(path):
        """Read CARP .dat file as numpy array."""
        return np.loadtxt(path)
    
    uvc_z_data = read_dat_file(uvc_paths.uvc_z)
    uvc_rho_data = read_dat_file(uvc_paths.uvc_rho)
    
    logging.info(f"UVC Z range: [{uvc_z_data.min():.3f}, {uvc_z_data.max():.3f}]")
    logging.info(f"UVC Rho range: [{uvc_rho_data.min():.3f}, {uvc_rho_data.max():.3f}]")
    
    # UVC coordinates should be in [0, 1] range (with small tolerance)
    tolerance = 0.01
    assert uvc_z_data.min() >= -tolerance, f"UVC Z has values < 0: {uvc_z_data.min()}"
    assert uvc_z_data.max() <= 1.0 + tolerance, f"UVC Z has values > 1: {uvc_z_data.max()}"
    assert uvc_rho_data.min() >= -tolerance, f"UVC Rho has values < 0: {uvc_rho_data.min()}"
    assert uvc_rho_data.max() <= 1.0 + tolerance, f"UVC Rho has values > 1: {uvc_rho_data.max()}"
    
    # Optional: Compare with expected outputs if they exist
    if expected_dir.exists():
        expected_z = expected_dir / "BiV.uvc_z.dat"
        if expected_z.exists():
            expected_z_data = read_dat_file(expected_z)
            
            # Check correlation (mguvc may have minor variability)
            correlation = np.corrcoef(uvc_z_data.flatten(), expected_z_data.flatten())[0, 1]
            logging.info(f"UVC Z correlation with expected: {correlation:.4f}")
            assert correlation > 0.95, f"UVC Z differs from expected (corr={correlation:.4f})"
    
    logging.info("✓ UVC calculation test passed")
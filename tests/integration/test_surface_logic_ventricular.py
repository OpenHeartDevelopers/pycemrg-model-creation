import logging
import pytest
from pathlib import Path

# Import from our library
from pycemrg.data import LabelManager
from pycemrg.core import setup_logging
from pycemrg_model_creation.logic import PathContractBuilder, SurfaceLogic
from pycemrg_model_creation.tools import MeshtoolWrapper


# --- Test Configuration ---
# Mark this test as 'integration' so we can run it selectively
# e.g., pytest -m integration
# Also mark it as slow.
@pytest.mark.integration
@pytest.mark.slow
def test_run_ventricular_extraction_on_sample_01(tmp_path, test_data_root):
    """
    An integration test for the full ventricular surface extraction workflow.

    This test uses a known-good input dataset and validates that the
    SurfaceLogic engine produces the expected output files.
    """

    # --- 0. Setup Detailed Logging --- # <-- ADD THIS SECTION
    # Create a log file inside the test's temporary directory
    log_file = tmp_path / "test_run.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logging.info(f"Detailed logs for this test run are in: {log_file}")

    # --- 1. Define Paths to the Centralized Test Data ---
    # The test expects the data to be in the location we just set up.
    sample_case_dir = test_data_root / "ventricular_extraction"
    mesh_base_path = sample_case_dir / "input_mesh/myocardium"
    labels_config_path = sample_case_dir / "config/labels.yaml"

    # Use pytest's tmp_path fixture for clean, temporary output directories
    # This ensures the test doesn't leave junk files behind.
    output_dir = tmp_path / "test_output"

    # --- Pre-condition Checks ---
    assert mesh_base_path.with_suffix(".pts").exists(), (
        f"Test data mesh not found at {mesh_base_path}.pts"
    )
    assert labels_config_path.exists(), (
        f"Test data labels config not found at {labels_config_path}"
    )

    # --- 2. Initialize Library Dependencies ---
    meshtool = MeshtoolWrapper.from_system_path()
    label_manager = LabelManager(config_path=labels_config_path)

    # --- 3. Generate the Path Contract ---
    path_builder = PathContractBuilder(output_dir=output_dir)
    all_paths = path_builder.build_all(mesh_base_path, blank_files_dir=Path("."))
    ventricular_paths = all_paths.ventricular

    # --- 4. Execute the Logic Engine ---
    surface_logic = SurfaceLogic(meshtool, label_manager)
    surface_logic.run_ventricular_extraction(paths=ventricular_paths)

    # --- 5. Validation: Assert that key output files were created ---
    # This is the most important part of the test.
    logging.info("Validating key output files...")
    assert ventricular_paths.epi_surface.with_suffix(".vtk").exists()
    assert ventricular_paths.lv_endo_surface.with_suffix(".vtk").exists()
    assert ventricular_paths.rv_endo_surface.with_suffix(".vtk").exists()
    assert ventricular_paths.septum_surface.with_suffix(".vtk").exists()
    assert ventricular_paths.base_vtx.exists()
    assert ventricular_paths.apex_vtx.exists()  # Assuming this is created

    logging.info("Integration test passed: All key ventricular files were generated.")

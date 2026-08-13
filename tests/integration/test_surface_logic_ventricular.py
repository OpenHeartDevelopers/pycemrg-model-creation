import logging
import pytest
from pathlib import Path

# Import from our library
from pycemrg.data import LabelManager
from pycemrg.core import setup_logging
from pycemrg_model_creation.logic import ModelCreationPathBuilder, SurfaceLogic
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
    log_file = tmp_path / "test_run_extract_surfaces.log"
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
    path_builder = ModelCreationPathBuilder(output_dir=output_dir)
    blank_files_dir = sample_case_dir / "input_blank"
    all_paths = path_builder.build_all(mesh_base_path, blank_files_dir=blank_files_dir)
    ventricular_paths = all_paths.ventricular

    # --- 4. Execute the Logic Engine ---
    surface_logic = SurfaceLogic(meshtool, label_manager)
    surface_logic.run_ventricular_extraction(paths=ventricular_paths)

    # --- 5. Validation: Assert that key output files were created ---
    # This is the most important part of the test.
    logging.info("Validating key functional output files...")
    #
    # These are this stage's outputs, not the files mguvc eventually opens.
    # Every boundary exists twice: once in four-chamber node indices, once in
    # BiV ones. This stage produces only the first generation, in tmp/ under
    # the source mesh's own stem. `meshtool map` and the rename that turn
    # `tmp/heart.base.vtx` into `BiV/BiV.base.vtx` live in
    # `run_biv_mesh_extraction`, which this test never calls -- so asserting
    # `ventricular_paths.base_vtx` here would be asserting another stage's work.
    files_to_check = [
        # Surface geometry, beside the submesh under underscore names.
        ventricular_paths.epi_surface.with_suffix(".surf"),
        ventricular_paths.lv_endo_surface.with_suffix(".surf"),
        # The full RV endocardium, septal surface included. The septum removal
        # writes its result elsewhere and does not overwrite this file.
        ventricular_paths.rv_endo_surface.with_suffix(".surf"),
        ventricular_paths.septum_surface.with_suffix(".surf"),
    ]

    # The four-chamber generation of the boundary VTX files. Taking the roles
    # from the contract keeps this test from spelling mguvc's vocabulary a
    # third time and drifting from it.
    files_to_check += [
        ventricular_paths.source_boundaries.role(role, ".vtx")
        for role in ventricular_paths.BOUNDARY_ROLES
    ]

    for file_path in files_to_check:
        assert file_path.exists(), (
            f"Expected functional output file not found: {file_path}"
        )

    logging.info("Integration test passed: All key ventricular files were generated.")

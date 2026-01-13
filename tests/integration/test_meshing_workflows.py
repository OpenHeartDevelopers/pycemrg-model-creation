# tests/integration/test_meshing_workflows.py
import logging
import pytest
from pathlib import Path

# Import from our library
from pycemrg.data import LabelManager
from pycemrg.core import setup_logging
from pycemrg.system import CommandRunner
from pycemrg_model_creation.logic import (
    MeshingLogic, 
    RefinementLogic, 
    MeshingPathBuilder
)
from pycemrg_model_creation.tools import MeshtoolWrapper, Meshtools3DWrapper

@pytest.mark.integration
@pytest.mark.slow
def test_full_meshing_and_refinement_workflow(tmp_path, test_data_root, test_meshtool_root, test_m3d_root): 
    """
    Tests the end-to-end workflow from a segmentation image to a clean,
    refined, and relabeled mesh.
    """
    # --- 1. Setup Logging and Paths ---
    log_file = tmp_path / "test_run_meshing.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logging.info(f"Detailed logs for this test run are in: {log_file}")

    sample_case_dir = test_data_root / "meshing_and_refinement"
    segmentation_path = sample_case_dir / "input_segmentation/whole_heart_segmentation_smooth.nrrd"
    labels_config_path = sample_case_dir / "config/labels.yaml"
    output_dir = tmp_path / "test_output"

    # --- Pre-condition Checks ---
    assert segmentation_path.exists(), f"Test data not found at {segmentation_path}"
    assert labels_config_path.exists(), f"Labels config not found at {labels_config_path}"

    # --- 2. Run Meshing Workflow ---
    logging.info("--- STAGE 1: Running Meshing Workflow ---")
    meshing_builder = MeshingPathBuilder(output_dir=output_dir)
    meshing_paths = meshing_builder.build_meshing_paths(input_image=segmentation_path)

    runner = CommandRunner(logger=logging.getLogger("TestRunner"))
    m3d_wrapper = Meshtools3DWrapper(
        runner=runner, 
        meshtools3d_path=test_m3d_root / "meshtools3d"
    )
    meshing_logic = MeshingLogic(meshtools3d_wrapper=m3d_wrapper)
    meshing_logic.run_meshing(paths=meshing_paths)

    # --- 3. First-pass Validation ---
    logging.info("Meshing workflow completed. Raw mesh created.")

    # --- 4. Run Refinement Workflow ---
    logging.info("--- STAGE 2: Running Refinement Workflow ---")
    refinement_paths = meshing_builder.build_postprocessing_paths(
        input_mesh_base=meshing_paths.output_mesh_base
    )
    
    label_manager = LabelManager(labels_config_path)
    myocardium_tags = label_manager.get_source_tags(["LV", "RV"])
    tag_mapping = label_manager.get_source_to_target_mapping()

    meshtool_wrapper = MeshtoolWrapper(
        runner=runner, 
        meshtool_install_dir=test_meshtool_root
    )
    refinement_logic = RefinementLogic(meshtool_wrapper=meshtool_wrapper)
    refinement_logic.run_myocardium_postprocessing(
        paths=refinement_paths,
        myocardium_tags=myocardium_tags,
        tag_mapping=tag_mapping,
        simplify=True
    )

    # --- 5. Final Validation ---
    final_mesh_path = refinement_paths.output_mesh_base
    assert final_mesh_path.with_suffix(".vtk").exists()
    assert final_mesh_path.with_suffix(".pts").exists()
    assert final_mesh_path.with_suffix(".elem").exists()
    
    logging.info("Refinement workflow completed. Final mesh created.")
    logging.info("--- Integration test PASSED ---")
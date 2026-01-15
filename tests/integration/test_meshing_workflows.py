# tests/integration/test_meshing_workflows.py
import os
import logging
import pytest
from pathlib import Path

# Import from our library
from pycemrg.data import LabelManager, LabelMapper
from pycemrg.core import setup_logging
from pycemrg.system import CommandRunner
from pycemrg_model_creation.logic import (
    MeshingLogic,
    RefinementLogic,
    MeshingPathBuilder,
)
from pycemrg_model_creation.tools import MeshtoolWrapper, Meshtools3DWrapper


@pytest.mark.integration
@pytest.mark.slow
def test_full_meshing_and_refinement_workflow(
    tmp_path, test_data_root, test_meshtool_root, test_m3d_root
):
    """
    Tests the end-to-end workflow from a segmentation image to a clean,
    refined, and relabeled mesh.
    """
    # --- 1. Setup Logging and Paths ---
    log_file = tmp_path / "test_run_meshing.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logging.info(f"Detailed logs for this test run are in: {log_file}")

    sample_case_dir = test_data_root / "meshing_and_refinement"
    segmentation_path = (
        sample_case_dir / "input_segmentation/whole_heart_segmentation_smooth.nrrd"
    )

    source_labels_path = sample_case_dir / "config/source_labels.yaml"
    target_labels_path = sample_case_dir / "config/target_labels.yaml"

    output_dir = tmp_path / "test_output"

    # --- Pre-condition Checks ---
    assert segmentation_path.exists(), f"Test data not found at {segmentation_path}"
    assert source_labels_path.exists(), (
        f"Labels config not found at {source_labels_path}"
    )
    assert target_labels_path.exists(), (
        f"Labels config not found at {target_labels_path}"
    )

    skip_meshing = os.environ.get("PYCEMRG_SKIP_MESHING")

    meshing_builder = MeshingPathBuilder(output_dir=output_dir)

    if skip_meshing:
        logging.warning("PYCEMRG_SKIP_MESHING is set. Skipping meshing step.")
        logging.warning("Using pre-generated 'golden' mesh as input for refinement.")

        # Point to the pre-generated raw mesh instead of running the logic
        golden_mesh_dir = test_data_root / "meshing_and_refinement/golden_mesh_input"
        raw_mesh_base = golden_mesh_dir / "heart_mesh"  # The basename we used

        # Fail fast if the developer hasn't generated the golden files
        assert raw_mesh_base.with_suffix(".pts").exists(), (
            "Skip failed: Golden mesh output not found. "
            "Run the test once without PYCEMRG_SKIP_MESHING to generate it."
        )

        # This is the "output" of the skipped stage
        meshing_output_base = raw_mesh_base

    else:
        # --- 2. Run Meshing Workflow (as normal) ---
        logging.info("--- STAGE 1: Running Meshing Workflow ---")
        meshing_paths = meshing_builder.build_meshing_paths(
            input_image=segmentation_path
        )

        runner = CommandRunner(logger=logging.getLogger("TestRunner"))
        m3d_wrapper = Meshtools3DWrapper(
            runner=runner, meshtools3d_path=test_m3d_root / "meshtools3d"
        )
        meshing_logic = MeshingLogic(meshtools3d_wrapper=m3d_wrapper)
        meshing_logic.run_meshing(paths=meshing_paths)

        # --- First-pass Validation ---
        assert meshing_paths.output_mesh_base.with_suffix(".pts").exists()
        logging.info("Meshing workflow completed. Raw mesh created.")

        # This is the output of the completed stage
        meshing_output_base = meshing_paths.output_mesh_base

    logging.info("--- STAGE 2: Running Refinement Workflow ---")
    refinement_paths = meshing_builder.build_postprocessing_paths(
        input_mesh_base=meshing_output_base
    )

    meshtool_wrapper = MeshtoolWrapper.from_system_path(
        meshtool_install_dir=test_meshtool_root
    )
    refinement_logic = RefinementLogic(meshtool_wrapper=meshtool_wrapper)

    source_label_manager = LabelManager(source_labels_path)
    target_label_manager = LabelManager(target_labels_path)
    myocardium_tags = source_label_manager.get_values_from_names(["MYOCARDIUM"])

    label_mapper = LabelMapper(source=source_label_manager, target=target_label_manager)
    tag_mapping = label_mapper.get_source_to_target_mapping()

    refinement_logic.run_myocardium_postprocessing(
        paths=refinement_paths,
        myocardium_tags=myocardium_tags,
        tag_mapping=tag_mapping,
        simplify=True,
    )

    # --- 5. Final Validation ---
    final_mesh_path = refinement_paths.output_mesh_base
    assert final_mesh_path.with_suffix(".pts").exists()
    assert final_mesh_path.with_suffix(".elem").exists()

    logging.info("Refinement workflow completed. Final mesh created.")
    logging.info("--- Integration test PASSED ---")

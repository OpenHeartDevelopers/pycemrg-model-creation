# tests/integration/test_refinement_workflow.py
import logging
import pytest
from pathlib import Path

# Import from our library
from pycemrg.data import LabelManager, LabelMapper
from pycemrg.core import setup_logging
from pycemrg_model_creation.logic import (
    RefinementLogic,
    RefinementPathBuilder,
)
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools import MeshtoolWrapper


@pytest.mark.integration
@pytest.mark.slow
def test_myocardium_refinement_workflow(
    tmp_path, test_data_root, test_meshtool_root
):
    """
    Tests the refinement workflow: from a raw volumetric mesh to a clean,
    refined, and relabeled myocardium mesh.

    The raw volumetric mesh is an input, produced upstream of this library
    (see pycemrg-meshing).
    """
    # --- 1. Setup Logging and Paths ---
    log_file = tmp_path / "test_run_refinement.log"
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logging.info(f"Detailed logs for this test run are in: {log_file}")

    sample_case_dir = test_data_root / "meshing_and_refinement"
    raw_mesh_base = CarpMesh(sample_case_dir / "golden_mesh_input/heart_mesh")

    source_labels_path = sample_case_dir / "config/source_labels.yaml"
    target_labels_path = sample_case_dir / "config/target_labels.yaml"

    output_dir = tmp_path / "test_output"

    # --- Pre-condition Checks ---
    assert raw_mesh_base.pts.exists(), (
        f"Raw input mesh not found at {raw_mesh_base.pts}"
    )
    assert source_labels_path.exists(), (
        f"Labels config not found at {source_labels_path}"
    )
    assert target_labels_path.exists(), (
        f"Labels config not found at {target_labels_path}"
    )

    # --- 2. Build the Path Contract ---
    refinement_builder = RefinementPathBuilder(output_dir=output_dir)
    refinement_paths = refinement_builder.build_postprocessing_paths(
        input_mesh_base=raw_mesh_base
    )

    # --- 3. Initialize Library Dependencies ---
    meshtool_wrapper = MeshtoolWrapper.from_system_path(
        meshtool_install_dir=test_meshtool_root
    )
    refinement_logic = RefinementLogic(meshtool_wrapper=meshtool_wrapper)

    # --- 4. Resolve Tags via LabelManager / LabelMapper ---
    source_label_manager = LabelManager(source_labels_path)
    target_label_manager = LabelManager(target_labels_path)
    myocardium_tags = source_label_manager.get_values_from_names(["MYOCARDIUM"])

    label_mapper = LabelMapper(source=source_label_manager, target=target_label_manager)
    tag_mapping = label_mapper.get_source_to_target_mapping()

    # --- 5. Execute the Logic Engine ---
    logging.info("--- Running Refinement Workflow ---")
    refinement_logic.run_myocardium_postprocessing(
        paths=refinement_paths,
        myocardium_tags=myocardium_tags,
        tag_mapping=tag_mapping,
        simplify=True,
    )

    # --- 6. Final Validation ---
    # Ask the handle for the family; with_suffix is banned here because it
    # replaces the final dot-segment rather than appending to it.
    final_mesh = refinement_paths.output_mesh_base
    assert final_mesh.pts.exists()
    assert final_mesh.elem.exists()
    assert final_mesh.vtk.exists()

    logging.info("Refinement workflow completed. Final mesh created.")
    logging.info("--- Integration test PASSED ---")

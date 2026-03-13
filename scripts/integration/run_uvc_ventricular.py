#!/usr/bin/env python3
"""
Debug script for ventricular UVC calculation.

Mirrors test_uvc_logic_ventricular.py but runs outside pytest so that
mguvc stdout/stderr is visible directly in the terminal.

Required environment variables:
    PYCEMRG_TEST_DATA_ROOT  - path to test data root
    PYCEMRG_CARP_CONFIG     - path to CARPentry config file

Expected test data layout:
    $PYCEMRG_TEST_DATA_ROOT/ventricular_uvc/
        config/labels.yaml
        input_mesh/BiV.pts
        input_mesh/BiV.elem
        input_mesh/base.vtx
        input_mesh/epi.vtx
        input_mesh/lvendo.vtx
        input_mesh/rvendo.vtx
        input_mesh/rvsept.vtx
        input_mesh/rvendo_nosept.vtx
"""

import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

from pycemrg.system import CarpRunner, CommandRunner
from pycemrg.data import LabelManager
from pycemrg_model_creation.logic import UvcLogic, ModelCreationPathBuilder
from pycemrg_model_creation.tools import CarpWrapper


def main() -> None:
    test_data_root_str = os.environ.get("PYCEMRG_TEST_DATA_ROOT")
    carp_config_str = os.environ.get("PYCEMRG_CARP_CONFIG")

    if not test_data_root_str:
        sys.exit("ERROR: PYCEMRG_TEST_DATA_ROOT not set")
    if not carp_config_str:
        sys.exit("ERROR: PYCEMRG_CARP_CONFIG not set")

    test_data_root = Path(test_data_root_str)
    carp_config = Path(carp_config_str)

    if not carp_config.exists():
        sys.exit(f"ERROR: CARP config not found: {carp_config}")

    test_case_dir = test_data_root / "ventricular_uvc"
    if not test_case_dir.exists():
        sys.exit(f"ERROR: Test data not found: {test_case_dir}")

    # Load tags
    labels_path = test_case_dir / "config" / "labels.yaml"
    label_manager = LabelManager(config_path=labels_path)
    lv_tag = label_manager.get_value("LV")
    rv_tag = label_manager.get_value("RV")
    logging.info(f"Loaded tags: LV={lv_tag}, RV={rv_tag}")

    # Copy mesh and VTX files into the working dir so mguvc runs in an
    # isolated, clean directory and never touches the source test data.
    input_mesh_dir = test_case_dir / "input_mesh"
    output_dir = Path("/tmp/uvc_debug")
    work_dir = output_dir / "BiV"
    work_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pts", ".elem"):
        src = input_mesh_dir / f"BiV{suffix}"
        shutil.copy(src, work_dir / src.name)
    for vtx_file in input_mesh_dir.glob("*.vtx"):
        shutil.copy(vtx_file, work_dir / vtx_file.name)

    biv_mesh = work_dir / "BiV"
    builder = ModelCreationPathBuilder(output_dir=output_dir)
    uvc_paths = builder.build_ventricular_uvc_paths(biv_mesh=biv_mesh)

    # Initialize and run
    runner = CommandRunner()
    carp_runner = CarpRunner(runner=runner, carp_config_path=carp_config)
    carp_wrapper = CarpWrapper(carp_runner)
    uvc_logic = UvcLogic(carp_wrapper)

    logging.info("Starting UVC calculation...")
    uvc_logic.run_ventricular_uvc_calculation(
        paths=uvc_paths,
        lv_tag=lv_tag,
        rv_tag=rv_tag,
        np=1,
    )

    logging.info("Done. Output directory: %s", uvc_paths.output_dir)


if __name__ == "__main__":
    main()

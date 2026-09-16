# tests/integration/test_fibre_logic_ventricular.py
"""
Integration test for ventricular rule-based fibre generation.

Self-contained: runs the UVC step first, in the same tmp_path, to produce
real Laplace solutions, then generates fibres from them. Reuses the
`ventricular_uvc` fixture data — no fibre-specific fixture exists yet, and
none is needed, since the Laplace solutions FibreLogic consumes are a live
side effect of `run_ventricular_uvc_calculation` (`laplace_solution=True` is
hardcoded there).
"""

import os
import shutil
import logging
import pytest
import numpy as np
from pathlib import Path

from pycemrg.system import CarpRunner, CommandRunner
from pycemrg_model_creation.logic import FibreLogic, UvcLogic, ModelCreationPathBuilder
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools import CarpWrapper
from pycemrg_model_creation.tools.mguvc import boundary_inputs, outputs_for
from pycemrg_model_creation.utilities.mesh import read_elem, read_lon
from pycemrg.data import LabelManager


@pytest.fixture(scope="module")
def carp_wrapper(test_data_root):
    """
    Initialize CarpWrapper for fibre generation tests.

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


@pytest.mark.integration
@pytest.mark.slow
def test_ventricular_fibre_generation(test_data_root, carp_wrapper, tmp_path):
    """
    End-to-end test: UVC calculation, then rule-based fibre generation.

    Test data structure (shared with test_uvc_logic_ventricular.py):
        test_data_root/
        └── ventricular_uvc/
            ├── config/
            │   └── labels.yaml
            └── input_mesh/
                ├── BiV.pts              # BiV submesh
                ├── BiV.elem
                ├── BiV.lon              # [1 0 0] placeholder; GlRuleFibers
                │                        # reads it, then overwrites it
                ├── BiV.base.vtx         # the four boundary sets mguvc reads,
                ├── BiV.lvendo.vtx       # stem-prefixed, in the mesh's own
                ├── BiV.rvendo.vtx       # directory
                └── BiV.rvsept.vtx

    UVC runs first so its Laplace solutions exist for the fibre step; this
    is the first test where the two logic classes run back to back.
    """
    test_case_dir = test_data_root / "ventricular_uvc"

    if not test_case_dir.exists():
        pytest.skip(f"Test data not found: {test_case_dir}")

    config_dir = test_case_dir / "config"
    input_mesh_dir = test_case_dir / "input_mesh"

    labels_path = config_dir / "labels.yaml"
    assert labels_path.exists(), f"Labels config not found: {labels_path}"

    label_manager = LabelManager(config_path=labels_path)
    lv_tag = label_manager.get_value("LV")
    rv_tag = label_manager.get_value("RV")

    logging.info(f"Loaded tags: LV={lv_tag}, RV={rv_tag}")

    # Copy the mesh and the four boundary VTX files into tmp_path so mguvc
    # runs in an isolated, clean directory and never touches the source data.
    work_dir = tmp_path / "BiV"
    work_dir.mkdir()

    source_mesh = CarpMesh(input_mesh_dir / "BiV")
    # .lon travels with .pts/.elem: GlRuleFibers loads the whole family behind
    # -m and opens the .lon before it writes one. In a real chain the submesh
    # inherits its parent's fibres from `meshtool extract mesh`; the fixture
    # carries a [1 0 0] placeholder standing in for that.
    for mesh_file in (source_mesh.pts, source_mesh.elem, source_mesh.lon):
        assert mesh_file.exists(), f"BiV mesh not found: {mesh_file}"
        shutil.copy(mesh_file, work_dir / mesh_file.name)

    for boundary, vtx_file in boundary_inputs(source_mesh).items():
        assert vtx_file.exists(), f"Missing {boundary.value} VTX: {vtx_file}"
        shutil.copy(vtx_file, work_dir / vtx_file.name)

    biv_mesh = work_dir / "BiV"

    builder = ModelCreationPathBuilder(output_dir=tmp_path)
    uvc_paths = builder.build_ventricular_uvc_paths(biv_mesh=biv_mesh)

    # -- UVC stage: produces the Laplace solutions fibre generation needs ---
    logging.info("Running UVC calculation...")
    UvcLogic(carp_wrapper).run_ventricular_uvc_calculation(
        paths=uvc_paths, lv_tag=lv_tag, rv_tag=rv_tag, np=1
    )

    uvc_outputs = outputs_for(uvc_paths.biv_mesh, uvc_paths.output_dir)
    for solution in uvc_outputs.laplace_solutions:
        assert solution.exists(), f"Laplace solution not generated: {solution}"

    # -- Fibre stage ----------------------------------------------------
    fibre_paths = builder.build_ventricular_fibre_paths(uvc_paths)

    logging.info("Running fibre generation...")
    FibreLogic(carp_wrapper).run_ventricular_fibre_generation(fibre_paths)

    lon_path = fibre_paths.biv_mesh.lon
    assert lon_path.exists(), f"Fibre file not generated: {lon_path}"
    assert lon_path.stat().st_size > 0, f"{lon_path.name} is empty"

    fibres = read_lon(lon_path)
    elements = read_elem(fibre_paths.biv_mesh.elem)

    logging.info(f"Fibre array shape: {fibres.shape}, element count: {len(elements)}")

    assert fibres.shape[0] == len(elements), (
        f"Fibre vector count ({fibres.shape[0]}) does not match element "
        f"count ({len(elements)})"
    )
    assert fibres.shape[1] in (3, 6), f"Unexpected fibre vector width: {fibres.shape[1]}"

    # Fibre vectors should be unit length (within tolerance), not the [1 0 0]
    # placeholder mguvc fabricates when a mesh has no .lon.
    norms = np.linalg.norm(fibres[:, :3], axis=1)
    assert np.allclose(norms, 1.0, atol=0.05), (
        f"Fibre vectors are not unit length: min={norms.min():.3f}, "
        f"max={norms.max():.3f}"
    )
    not_axis_aligned = ~np.all(np.isclose(fibres[:, :3], [1.0, 0.0, 0.0]), axis=1)
    assert not_axis_aligned.any(), "All fibres are the [1 0 0] default — GlRuleFibres did not run"

    logging.info("✓ Fibre generation test passed")

    # -- Correction stage -----------------------------------------------
    # A separate public step, as cemrg-heartbuilder sequences it: generate,
    # then element centres, then repair. It runs here rather than in its own
    # test so the slow UVC and generation stages are paid for once.
    tolerance = 1e-6
    degenerate_before = int(np.count_nonzero(np.abs(fibres[:, 2]) < tolerance))
    logging.info(f"Degenerate fibres before correction: {degenerate_before}")

    logging.info("Running fibre correction...")
    FibreLogic(carp_wrapper).correct_ventricular_fibres(
        fibre_paths, tolerance=tolerance
    )

    centres_path = fibre_paths.biv_mesh.directory / (
        f"{fibre_paths.biv_mesh.name}_elem_centres.pts"
    )
    assert centres_path.exists(), (
        f"GlElemCenters output not found: {centres_path}"
    )

    corrected = read_lon(lon_path)
    assert corrected.shape == fibres.shape, (
        f"Correction changed the fibre array shape: {fibres.shape} -> "
        f"{corrected.shape}"
    )

    # The post-condition of the repair: every degenerate row took the whole
    # fibre row of a well-defined neighbour, so none can be left behind.
    degenerate_after = int(np.count_nonzero(np.abs(corrected[:, 2]) < tolerance))
    assert degenerate_after == 0, (
        f"{degenerate_after} degenerate fibres survived correction "
        f"(was {degenerate_before})"
    )

    if degenerate_before == 0:
        logging.warning(
            "No degenerate fibres were present, so correction was a no-op on "
            "this mesh. The assertions above did not exercise the repair."
        )
    else:
        changed = ~np.all(np.isclose(corrected, fibres), axis=1)
        assert changed.any(), (
            f"{degenerate_before} fibres were degenerate but nothing changed"
        )
        logging.info(
            f"Corrected {degenerate_before} degenerate fibres; "
            f"{int(changed.sum())} rows differ"
        )

    logging.info("✓ Fibre correction test passed")

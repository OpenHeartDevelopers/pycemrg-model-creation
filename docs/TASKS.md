# `pycemrg-model-creation`: Development Tasks

This document outlines the planned development tasks for the `pycemrg-model-creation` library.

### Immediate Next Steps

-   [ ] Port mesh smoothing from legacy code.
-   [ ] Create an integration test for the `MeshingLogic` workflow to validate the newly refactored `meshtools3d` functionality.
-   [ ] Implement the Ventricular UVC Calculation Workflow:
    -   [ ] Define the `VentricularUVCPaths` dataclass in `logic/contracts.py`.
    -   [ ] Create the `UvcLogic` class in a new `logic/uvc.py` file, initialized with a `CarpWrapper`.
    -   [ ] Implement the `run_ventricular_uvc_calculation` method within `UvcLogic`.
    -   [ ] Create a new integration test, `test_uvc_logic.py`, to validate the full UVC generation process.

### Core Feature Development

-   [ ] **Implement Ventricular Fibre Generation Workflow:**
    -   [ ] Define a `VentricularFibrePaths` contract.
    -   [ ] Create a `FibreLogic` class.
    -   [ ] Implement a `run_fibre_generation` method that utilizes the existing `CarpWrapper.gl_rule_fibres` method.
    -   [ ] Create an integration test to validate fibre file (`.lon`) creation.

-   [ ] **Implement Atrial Surface Extraction Workflow:**
    -   [ ] Flesh out the logic for `run_atrial_extraction` in `SurfaceLogic`, including connected component identification for atria.
    -   [ ] Create a dedicated integration test for atrial surface extraction.

-   [ ] **Implement Atrial Submesh & UVC Workflow:**
    -   [ ] Implement and test the `run_atrial_mesh_extraction` workflow.
    -   [ ] Implement and test the atrial UVC calculation workflow, noting the differences from the ventricular process (e.g., `custom_apex` flag).

### Testing & Validation

-   [ ] Create a dedicated integration test for the `run_biv_mesh_extraction` workflow to validate the four-chamber to BiV submeshing use case.

### Refinement & Maintenance

-   [ ] Refine `MeshtoolWrapper.extract_unreachable` to specify and validate its expected outputs, removing the current `TODO`.
-   [ ] Update `API_reference.md` to include the `MeshingLogic`, `UvcLogic`, and `FibreLogic` classes as they are completed.
-   [ ] Review all `ModelCreationPathBuilder` methods to ensure they handle file copying for templates (like `apex_vtx` for atria) consistently.

---

This task list is recorded. Ready to resume when you are.
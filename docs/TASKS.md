# `pycemrg-model-creation`: Development Tasks

This document outlines the planned development tasks for the `pycemrg-model-creation` library.

### Immediate Next Steps

-   [X] Port mesh smoothing from legacy code.
-   [X] Create an integration test for the `MeshingLogic` workflow to validate the newly refactored `meshtools3d` functionality.
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

-   [ ] **Implement Open-Source FEM Laplace Solver for Ventricular UVCs:**

    > **Context:** The current `UvcLogic` class depends on openCARP/CARPentry (proprietary) to
    > solve the four Laplace problems that yield ventricular UVC fields (`psi_ab`, `phi_epi`,
    > `phi_lv`, `phi_rv`). This task tracks a planned open-source alternative using a custom
    > FEM solver, intended as a drop-in backend behind the `UvcSolver` protocol.
    >
    > **Reference code:** A collaborator's UAC (Universal Atrial Coordinate) implementation
    > (`vtkfunctions.py`, `my_uac.py`) contains a working surface FEM Laplace solver
    > (`assemble_laplace_problem`, `solve_linear_system`) built on `scipy.sparse`. The
    > assembly and solve pipeline is anatomy-agnostic and has been assessed as architecturally
    > sound. It currently operates on surface triangulations (`vtkPolyData`) and must be
    > extended to volumetric tetrahedral meshes (`vtkUnstructuredGrid`) for ventricular use.

    -   [ ] Define a `UvcSolver` protocol in `logic/uvc.py` with a
            `solve(paths: VentricularUVCPaths) -> dict` interface, so both the openCARP
            backend and the FEM backend are interchangeable.
    -   [ ] Create `logic/fem/` subpackage with two modules:
        -   `mesh_io.py`: Extract points, tetrahedra, and connectivity from
                `vtkUnstructuredGrid`. Analogous to collaborator's
                `extract_mesh_dict_object_from_polydata`.
        -   `laplace.py`: Assemble and solve the Laplace stiffness system for
                tetrahedral elements. Extend collaborator's `compute_contravariant_basis`
                and `assemble_laplace_problem` from 2D surface (3x3 local stiffness) to
                3D volumetric (4x4 local stiffness). Retain `scipy.sparse` / `splu`
                solver approach.
    -   [ ] Implement `FemUvcSolver(UvcSolver)` in `logic/fem/uvc_solver.py` that
            orchestrates the four ventricular Laplace solves with correct boundary
            conditions:

            | Field   | BC = 0       | BC = 1       |
            | ------- | ------------ | ------------ |
            | psi_ab  | Apex surface | Base surface |
            | phi_epi | LV + RV endo | Epi surface  |
            | phi_lv  | RV endo      | LV endo      |
            | phi_rv  | LV endo      | RV endo      |

    -   [ ] Output solved scalar fields to VTK/VTU format via existing
            `CarpWrapper.gl_vtk_convert` or direct PyVista write.
    -   [ ] Add `scipy` as an optional dependency in `pyproject.toml`
            (e.g., `scipy>=1.9; extra == "fem"`).
    -   [ ] Create integration test `test_fem_uvc_solver.py` validating:
        -   Field range is `[0, 1]` for all four coordinates.
        -   Boundary nodes satisfy their prescribed Dirichlet conditions.
        -   Output VTK file contains the expected point data arrays.

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

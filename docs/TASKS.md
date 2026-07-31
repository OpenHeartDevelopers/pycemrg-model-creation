# `pycemrg-model-creation`: Development Tasks

This document outlines the planned development tasks for the `pycemrg-model-creation` library.

The pipeline starts at refinement (myocardium extraction from a volumetric mesh). Volumetric
meshing from a segmentation is owned by `pycemrg-meshing`; the `MeshingLogic` /
`Meshtools3DWrapper` / `Meshtools3DParameters` / `convert_image_to_inr` layer was removed from
this repo.

### Immediate Next Steps

-   [X] Port mesh smoothing from legacy code.
-   [X] Port the leaf-function primitive layer from `cemrg-heartbuilder` (`14c4527`): 20
        functions across CARP I/O, submesh reindexing, linear algebra and geodesic paths,
        with 65 unit tests. See `.claude/plan_leaf_functions.md` and
        `.claude/leaf_function_status.md`. This is the substrate for fibre generation and
        atrial landmark detection below — neither is greenfield any more.
-   [X] Implement the Ventricular UVC Calculation Workflow:
    -   [X] Define the `VentricularUVCPaths` dataclass in `logic/contracts.py`.
    -   [X] Create the `UvcLogic` class in a new `logic/uvc.py` file, initialized with a `CarpWrapper`.
    -   [X] Implement the `run_ventricular_uvc_calculation` method within `UvcLogic`.
    -   [X] Create a new integration test, `test_uvc_logic_ventricular.py`, to validate the full UVC generation process.
-   [ ] Resolve the `mguvc` VTX naming conflict: `build_ventricular_uvc_paths` emits
        `{basename}.base.vtx` and `UvcLogic._validate_inputs` requires it, but the UVC test and
        debug script copy the plain `base.vtx` names that `mguvc` itself reads.
-   [ ] Fix `SurfaceLogic.run_all`: it still passes `tags` to `run_ventricular_extraction` and
        `run_atrial_extraction`, which dropped that parameter when tag lookup moved to the
        injected `LabelManager`. Any call raises `TypeError`.
-   [ ] Fix `MeshtoolWrapper.map`: its debug log calls `f.name` on each entry, but callers pass
        `str`. Also `insert_meshdata`, annotated `MeshDataOperation` but membership-tested against
        string keys, so passing the enum raises `ValueError`.

### Core Feature Development

-   [ ] **Implement Ventricular Fibre Generation Workflow:**

    > **Unblocked.** `utilities/linalg.py` supplies `rotation_matrix` (sheet
    > orthogonalisation), `create_csys` and `normalise_vectors`; `utilities/mesh.py`
    > supplies `write_lon` and `write_elem`. No new primitives needed.

    -   [ ] Define a `VentricularFibrePaths` contract.
    -   [ ] Create a `FibreLogic` class.
    -   [ ] Implement a `run_fibre_generation` method that utilizes the existing `CarpWrapper.gl_rule_fibres` method.
    -   [ ] Create an integration test to validate fibre file (`.lon`) creation.
    -   [ ] Before building on `gl_elem_centers`, fix `CarpWrapper`'s argv packing —
            `wrappers.py:221` does `cmd += f" --tagfile {tagfile}"` on a **list**, extending
            it character by character.

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

    > **Unblocked.** `utilities/geodesic.py` supplies the four path primitives
    > (`compute_path_length`, `find_midway_point`, `find_closest_point`,
    > `find_point_along_direction`) that carry ~25 call sites in `cemrg-heartbuilder`'s
    > `distance_utils.py`. Landmark detection ports on top of these rather than being
    > rewritten. `read_clipper` and the submesh reindexing functions are also in place.

    -   [ ] Flesh out the logic for `run_atrial_extraction` in `SurfaceLogic`, including connected component identification for atria.
    -   [ ] Create a dedicated integration test for atrial surface extraction.

-   [ ] **Implement Atrial Submesh & UVC Workflow:**
    -   [ ] Implement and test the `run_atrial_mesh_extraction` workflow.
    -   [ ] Implement and test the atrial UVC calculation workflow, noting the differences from the ventricular process (e.g., `custom_apex` flag).

### Testing & Validation

-   [ ] Create a dedicated integration test for the `run_biv_mesh_extraction` workflow to validate the four-chamber to BiV submeshing use case.
-   [ ] Verify the CARP writers against the binaries, not just our own readers: a file from
        `write_elem`, `write_lon` and `write_vtx` read back without complaint by `meshtool`
        and `mguvc`. The port assumes exact format reproduction and nothing has tested that
        assumption. *Blocked on test data.*
-   [ ] Verify `read_nod_eidx` against a `.nod`/`.eidx` pair emitted by `meshtool`. It
        currently round-trips our own writer only; endianness and integer width are inferred
        and are the likely failure points. *Blocked on a real submesh pair.*
-   [ ] Register the `integration` and `slow` markers in `pyproject.toml` — they are used but
        undeclared, so runs emit `PytestUnknownMarkWarning`, and
        `test_uvc_logic_ventricular.py` carries no marker so `-m integration` silently
        deselects the UVC test.

### Refinement & Maintenance

-   [ ] Refine `MeshtoolWrapper.extract_unreachable` to specify and validate its expected outputs, removing the current `TODO`.
-   [ ] Update `API_reference.md` to include the `UvcLogic` and `FibreLogic` classes as they are completed, and the `utilities/linalg.py` and `utilities/geodesic.py` modules.
-   [ ] Name the primitive layer in the roadmap's `B1` entry
        (`~/dev/python/cemrg-heartbuilder/.claude/pycemrg_model_creation_roadmap.md:91`),
        which scopes `B1` to CARP I/O only and treats linalg and geodesics as internal
        detail of the A2 and A5 stages.
-   [ ] Consolidate the three epi/endo classifiers, which disagree on the reference point:
        `utilities/mesh.py::identify_epi_from_endo` (own centroid, `>0.5`),
        `utilities/geometry.py::identify_surface_orientation` (injected reference), and a
        third inlined in `logic/surfaces.py::extract_ventricular_surfaces`. Establish which
        rule is correct against real three-component test data first — there is no single
        current behaviour to preserve.
-   [ ] Review all `ModelCreationPathBuilder` methods to ensure they handle file copying for templates (like `apex_vtx` for atria) consistently.

---

This task list is recorded. Ready to resume when you are.

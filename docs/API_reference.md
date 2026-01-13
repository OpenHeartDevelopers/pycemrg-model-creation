## **`pycemrg-model-creation` API Reference**

#### **1. Overview**

The `pycemrg-model-creation` library provides a high-level, Pythonic framework for executing complex cardiac modeling workflows. It encapsulates logic for volumetric meshing from segmentations, mesh refinement, and the extraction of surfaces for coordinate system generation.

The library is designed with a layered architecture, separating stateless scientific logic from orchestration helpers and low-level tool wrappers.

**A typical multi-stage user workflow is:**
1.  Initialize a `MeshingPathBuilder` for the meshing stage. Use it to generate a `MeshingPaths` contract.
2.  Instantiate and run `MeshingLogic` to create a raw volumetric mesh from a segmentation.
3.  Use the same `MeshingPathBuilder` to generate a `MeshPostprocessingPaths` contract.
4.  Instantiate and run `RefinementLogic` to clean, extract, and relabel the raw mesh.
5.  Initialize a `ModelCreationPathBuilder` for the model annotation stage. Use it to generate a `VentricularSurfacePaths` contract.
6.  Instantiate and run `SurfaceLogic` to extract geometric surfaces and boundary conditions (`.vtx` files).

---

#### **2. Orchestration Helpers: Path Builders**

Path builders are the recommended starting point for orchestrator scripts. They translate high-level directory paths into the detailed, explicit contracts required by the logic layer, enforcing consistent file and directory naming.

##### **`MeshingPathBuilder`**
**Entry Point:** `pycemrg_model_creation.logic.MeshingPathBuilder`
*   **Scope:** All workflows related to the creation and refinement of the primary volumetric mesh.
*   **Initialization:**
    ```python
    from pycemrg_model_creation.logic import MeshingPathBuilder
    meshing_output_dir = Path("/path/to/patient_01/04_meshing")
    meshing_builder = MeshingPathBuilder(output_dir=meshing_output_dir)
    ```
*   **Methods:**
    *   `build_meshing_paths(...) -> MeshingPaths`: Builds the contract for the initial volumetric meshing from a segmentation.
    *   `build_postprocessing_paths(...) -> MeshPostprocessingPaths`: Builds the contract for cleaning, submeshing, and relabeling a raw mesh.

##### **`ModelCreationPathBuilder`**
**Entry Point:** `pycemrg_model_creation.logic.ModelCreationPathBuilder`
*   **Scope:** All workflows related to annotating a mesh with scientific properties (surfaces, UVCs, fibres).
*   **Initialization:**
    ```python
    from pycemrg_model_creation.logic import ModelCreationPathBuilder
    model_output_dir = Path("/path/to/patient_01/05_model_creation")
    model_builder = ModelCreationPathBuilder(output_dir=model_output_dir)
    ```*   **Methods:**
    *   `build_ventricular_paths(...) -> VentricularSurfacePaths`: Builds the contract for extracting ventricular surfaces.

---

#### **3. The Logic Layer (The "Engines")**

These classes are the stateless orchestrators for the scientific workflows. They contain the "how-to" logic for each stage of the pipeline.

##### **`MeshingLogic` & `RefinementLogic`**
*   **Entry Points:** `pycemrg_model_creation.logic.MeshingLogic`, `pycemrg_model_creation.logic.RefinementLogic`
*   **Purpose:**
    *   `MeshingLogic`: Converts a segmentation image (`.nii`, `.nrrd`) into a raw volumetric mesh using `meshtools3d`.
    *   `RefinementLogic`: Takes a raw mesh, extracts a sub-region (e.g., myocardium), optionally simplifies its topology, and relabels its element tags to a consistent standard.
*   **Primary Methods:**
    *   `MeshingLogic.run_meshing(paths: MeshingPaths, ...)`
    *   `RefinementLogic.run_myocardium_postprocessing(paths: MeshPostprocessingPaths, ...)`

##### **`SurfaceLogic`**
*   **Entry Point:** `pycemrg_model_creation.logic.SurfaceLogic`
*   **Purpose:** Extracts geometric surfaces (e.g., epicardium, endocardium) and boundary condition files (`.vtx`) from a clean, volumetric mesh.
*   **Primary Method:**
    *   `run_ventricular_extraction(paths: VentricularSurfacePaths)`

---

#### **4. The Tool Layer (The "Power Tools")**

These classes provide low-level, direct Python wrappers for command-line utilities. They are used internally by the logic layer but are also available for advanced, one-off tasks.

*   `pycemrg_model_creation.tools.Meshtools3DWrapper`: Wraps the `meshtools3d` binary for volumetric meshing.
*   `pycemrg_model_creation.tools.MeshtoolWrapper`: Wraps the `meshtool` binary and its standalones (like `simplify_tag_topology`) for mesh manipulation.
*   `pycemrg_model_creation.tools.CarpWrapper`: Wraps `CARP` tools like `mguvc` and `GlRuleFibres`.

---

#### **5. Full Usage Example (Meshing & Surface Extraction)**

This example ties the first two major stages together.

```python
from pathlib import Path
from pycemrg.data import LabelManager
from pycemrg.system import CommandRunner
from pycemrg_model_creation.logic import (
    MeshingPathBuilder, ModelCreationPathBuilder,
    MeshingLogic, RefinementLogic, SurfaceLogic
)
from pycemrg_model_creation.tools import (
    Meshtools3DWrapper, MeshtoolWrapper
)

# --- 1. Define High-Level Paths and Dependencies ---
project_dir = Path("/data/patient_01")
segmentation_image = project_dir / "02_anatomy/labels/whole_heart.nii.gz"
labels_config = project_dir / "config/labels.yaml"
meshtools3d_binary = Path("/path/to/meshtools3d")
meshtool_install_dir = Path("/path/to/meshtool/installation")

runner = CommandRunner()
label_manager = LabelManager(labels_config)

# --- STAGE 1: MESHING & REFINEMENT ---
meshing_output_dir = project_dir / "04_meshing"
meshing_builder = MeshingPathBuilder(output_dir=meshing_output_dir)

# Run initial meshing
m3d_wrapper = Meshtools3DWrapper(runner, meshtools3d_binary)
meshing_logic = MeshingLogic(m3d_wrapper)
meshing_paths = meshing_builder.build_meshing_paths(segmentation_image)
meshing_logic.run_meshing(paths=meshing_paths)

# Run refinement
meshtool_wrapper = MeshtoolWrapper(runner, meshtool_install_dir)
refinement_logic = RefinementLogic(meshtool_wrapper)
refinement_paths = meshing_builder.build_postprocessing_paths(
    input_mesh_base=meshing_paths.output_mesh_base
)
refinement_logic.run_myocardium_postprocessing(
    paths=refinement_paths,
    myocardium_tags=label_manager.get_source_tags(["LV", "RV"]),
    tag_mapping=label_manager.get_source_to_target_mapping(),
    simplify=True
)
print("Meshing and refinement complete.")

# --- STAGE 2: SURFACE EXTRACTION ---
model_output_dir = project_dir / "05_model_creation"
model_builder = ModelCreationPathBuilder(output_dir=model_output_dir)

# The input to this stage is the output of the previous one
clean_mesh_base = refinement_paths.output_mesh_base 
surface_paths = model_builder.build_ventricular_paths(clean_mesh_base)

surface_logic = SurfaceLogic(meshtool_wrapper, label_manager)
surface_logic.run_ventricular_extraction(paths=surface_paths)

print("Ventricular surface extraction complete.")
```
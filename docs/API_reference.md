# pyCEMRG Model Creation - API Reference

**Version:** 1.0.0  
**Purpose:** A Pythonic SDK for cardiac electromechanical modeling using the CARPentry/openCARP ecosystem

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Modules](#core-modules)
   - [Configuration](#configuration)
   - [Logic Layer](#logic-layer)
   - [Tools & Wrappers](#tools--wrappers)
   - [Path Contracts](#path-contracts)
4. [Workflows](#workflows)
5. [API Reference](#api-reference)

---

## Overview

`pycemrg_model_creation` provides high-level abstractions for creating patient-specific cardiac computational models. The library follows a layered architecture:

- **Configuration Layer**: Define anatomical tags and meshing parameters
- **Path Contracts**: Explicit, type-safe path specifications for all I/O operations
- **Logic Layer**: Stateless, scientific workflows for meshing, refinement, and surface extraction
- **Tools Layer**: Low-level wrappers for CARPentry command-line tools

**Key Design Principles:**
- **Path-agnostic logic**: All file paths are provided explicitly through dataclass contracts
- **Stateless operations**: Logic classes don't maintain state between operations
- **Explicit over implicit**: No hidden path derivations or magic defaults

---

## Quick Start

### Basic Workflow Example

```python
from pathlib import Path
from pycemrg.data import LabelManager
from pycemrg_model_creation import (
    SurfaceLogic,
    MeshtoolWrapper,
    ModelCreationPathBuilder
)

# 1. Initialize dependencies
label_manager = LabelManager(config_path="labels.yaml")
meshtool = MeshtoolWrapper.from_system_path()

# 2. Set up paths for your patient/subject
output_dir = Path("/data/patient_001/model_outputs")
mesh_base = Path("/data/patient_001/mesh/four_chamber_mesh")
blank_files = Path("/templates/blank_files")

# 3. Build path contracts
builder = ModelCreationPathBuilder(output_dir=output_dir)
paths = builder.build_all(mesh_base, blank_files_dir=blank_files)

# 4. Execute workflows
surface_logic = SurfaceLogic(meshtool, label_manager)
surface_logic.run_ventricular_extraction(paths=paths.ventricular)
```

---

## Core Modules

### Configuration

#### `TagsConfig`

Configuration for mesh element tags representing anatomical regions.

```python
from pycemrg_model_creation import TagsConfig

tags = TagsConfig(
    LV=1,           # Left ventricle
    RV=2,           # Right ventricle
    LA=3,           # Left atrium
    RA=4,           # Right atrium
    MV=5,           # Mitral valve
    TV=6,           # Tricuspid valve
    AV=7,           # Aortic valve
    PV=8,           # Pulmonary valve
    PArt=9          # Pulmonary artery
)
```

**Methods:**

- `from_dict(tags_dict: Dict) -> TagsConfig`: Create from dictionary
- `to_dict() -> Dict`: Convert to dictionary
- `get_tags_string(keys: List[str]) -> str`: Get comma-separated tag string
- `get_tags_list(keys: List[str]) -> List[int]`: Get flat list of tags

**Example:**
```python
# Get tags for ventricles
ventricle_tags = tags.get_tags_string(["LV", "RV"])  # "1,2"
```

---

### Logic Layer

The logic layer provides high-level, multi-step scientific workflows.

#### `SurfaceLogic`

Orchestrates surface extraction for Universal Ventricular Coordinate (UVC) generation.

```python
class SurfaceLogic:
    def __init__(
        self, 
        meshtool: MeshtoolWrapper,
        label_manager: LabelManager
    )
```

**Ventricular Extraction Methods:**

##### `run_ventricular_extraction(paths: VentricularSurfacePaths) -> None`

Complete ventricular surface extraction workflow. Extracts:
- Epicardial surface
- LV endocardium
- RV endocardium (free wall, septum removed)
- Interventricular septum
- Base surface
- VTX boundary condition files

```python
surface_logic.run_ventricular_extraction(paths=ventricular_paths)
```

##### `extract_ventricular_base(paths: VentricularSurfacePaths) -> None`

Extract base surface (ventricle-valve interface).

##### `extract_ventricular_surfaces(paths: VentricularSurfacePaths) -> None`

Extract and identify epi/LV endo/RV endo using geometric analysis.

##### `extract_septum(paths: VentricularSurfacePaths) -> None`

Extract interventricular septum surface.

**Atrial Extraction Methods:**

##### `run_atrial_extraction(paths: AtrialSurfacePaths, chamber: Chamber) -> None`

Complete atrial surface extraction workflow.

```python
from pycemrg_model_creation.types import Chamber

surface_logic.run_atrial_extraction(
    paths=la_paths,
    chamber=Chamber.LA
)
```

**Submesh Extraction Methods:**

##### `run_biv_mesh_extraction(paths: BiVMeshPaths, tags: TagsConfig) -> None`

Extract biventricular submesh and map VTX files.

##### `run_atrial_mesh_extraction(paths: AtrialMeshPaths, tags: TagsConfig, chamber: Chamber) -> None`

Extract atrial submesh and map VTX files.

**Complete Workflow:**

##### `run_all(paths: UVCSurfaceExtractionPaths, tags: TagsConfig, ...) -> None`

Execute complete UVC surface extraction for all chambers.

---

#### `MeshingLogic`

Orchestrates volumetric meshing from segmentation images.

```python
class MeshingLogic:
    def __init__(self, meshtools3d_wrapper: Meshtools3DWrapper)
```

##### `run_meshing(paths: MeshingPaths, meshing_params: Dict = None, cleanup: bool = True) -> None`

Execute full meshtools3d workflow:
1. Convert NIfTI to INR format
2. Generate parameter file
3. Run meshtools3d
4. Clean up intermediate files

```python
from pycemrg_model_creation.logic import MeshingLogic

meshing_logic = MeshingLogic(meshtools3d_wrapper)
meshing_logic.run_meshing(
    paths=meshing_paths,
    meshing_params={'meshing': {'facet_size': '0.7'}},
    cleanup=True
)
```

---

#### `RefinementLogic`

Post-processes and refines raw meshes.

```python
class RefinementLogic:
    def __init__(self, meshtool_wrapper: MeshtoolWrapper)
```

##### `run_myocardium_postprocessing(paths: MeshPostprocessingPaths, myocardium_tags: List[int], tag_mapping: Dict[int, int], simplify: bool = False) -> None`

Post-processing workflow:
1. Extract myocardium by tags
2. Optionally simplify topology
3. Relabel element tags
4. Generate VTK for visualization

```python
refinement_logic = RefinementLogic(meshtool_wrapper)
refinement_logic.run_myocardium_postprocessing(
    paths=refinement_paths,
    myocardium_tags=[1, 2],  # LV, RV
    tag_mapping={1: 10, 2: 20},  # Old -> New tags
    simplify=True
)
```

---

### Tools & Wrappers

Low-level wrappers for CARPentry command-line tools.

#### `MeshtoolWrapper`

Wrapper for the `meshtool` utility.

```python
class MeshtoolWrapper:
    @classmethod
    def from_system_path(
        cls,
        logger: Optional[logging.Logger] = None,
        meshtool_install_dir: Optional[Path] = None
    ) -> "MeshtoolWrapper"
```

**Key Methods:**

##### `extract_mesh(input_mesh_path: Path, output_submesh_path: Path, tags: Sequence[int], ifmt: str = "carp_txt", normalise: bool = False) -> None`

Extract submesh by element tags.

##### `extract_surface(input_mesh_path: Path, output_surface_path: Path, ofmt: str = "vtk", op_tag_base: Optional[str] = None) -> None`

Extract surface from volume mesh.

##### `convert(input_mesh_path: Path, output_mesh_path: Path, ofmt: str = "vtk", ifmt: Optional[str] = None) -> None`

Convert mesh between formats.

##### `map(submesh_path: Path, files_list: List[str], output_folder: Path, mode: str = "m2s") -> None`

Map data files between mesh and submesh.

**Example:**
```python
meshtool = MeshtoolWrapper.from_system_path()
meshtool.extract_mesh(
    input_mesh_path=Path("full_mesh"),
    output_submesh_path=Path("biv_mesh"),
    tags=[1, 2],  # LV, RV tags
    ifmt="carp_txt"
)
```

---

#### `CarpWrapper`

Wrapper for CARP ecosystem tools (GlRuleFibres, GlVTKConvert, etc.).

```python
class CarpWrapper:
    def __init__(self, carp_runner: CarpRunner)
```

##### `gl_rule_fibres(mesh_name: Path, uvc_apba: Path, uvc_epi: Path, uvc_lv: Path, uvc_rv: Path, output_name: Path, angles: Dict[str, float] = None, fibre_type: str = "biv") -> None`

Generate rule-based fiber orientations.

**Default Fiber Angles:**
```python
DEFAULT_FIBRE_ANGLES = {
    "alpha_endo": 60,
    "alpha_epi": -60,
    "beta_endo": -65,
    "beta_epi": 25,
}
```

---

#### `Meshtools3DWrapper`

Low-level wrapper for the `meshtools3d` binary.

```python
class Meshtools3DWrapper:
    def __init__(self, runner: CommandRunner, meshtools3d_path: Path)
```

##### `run(parameter_file: Path, expected_outputs: List[Path]) -> None`

Execute meshtools3d with parameter file.

---

### Path Contracts

Path contracts are dataclasses that explicitly define all input/output paths for workflows. This eliminates ambiguity and makes orchestration transparent.

#### `VentricularSurfacePaths`

Defines all paths for ventricular surface extraction.

**Key Fields:**
- `mesh: Path` - Input four-chamber mesh
- `output_dir: Path` - Final outputs directory
- `tmp_dir: Path` - Intermediate files directory
- `epi_surface: Path` - Epicardial surface output
- `lv_endo_surface: Path` - LV endocardium output
- `rv_endo_surface: Path` - RV endocardium output
- `septum_surface: Path` - Septum surface output
- `base_vtx: Path` - Base boundary condition
- `epi_vtx: Path` - Epi boundary condition
- (and more...)

#### `AtrialSurfacePaths`

Defines paths for atrial (LA or RA) surface extraction.

#### `BiVMeshPaths`

Defines paths for biventricular submesh extraction and VTX mapping.

**Key Fields:**
- `source_mesh: Path` - Four-chamber mesh
- `output_mesh: Path` - BiV submesh output
- `vtx_files_to_map: List[Path]` - VTX files to map from full mesh
- `mapped_vtx_output_dir: Path` - Directory for mapped VTX files

#### `MeshingPaths`

Defines paths for volumetric meshing workflow.

**Key Fields:**
- `input_segmentation_nifti: Path` - Input segmentation
- `output_dir: Path` - Mesh outputs
- `tmp_dir: Path` - Temporary files
- `output_mesh_base: Path` - Base name for output mesh

#### `MeshPostprocessingPaths`

Defines paths for mesh refinement workflow.
---

### Path Builders

Builders simplify path contract creation using standard naming conventions.

#### `ModelCreationPathBuilder`

Constructs path contracts for UVC surface extraction workflows.

```python
class ModelCreationPathBuilder:
    def __init__(self, output_dir: Union[Path, str])
```

**Directory Structure Created:**
```
output_dir/
├── BiV/
│   ├── tmp/
│   └── biv/
├── LA/
│   ├── tmp/
│   └── la/
└── RA/
    ├── tmp/
    └── ra/
```

##### `build_ventricular_paths(mesh_base_path: Path) -> VentricularSurfacePaths`

Build ventricular paths contract.

##### `build_atrial_paths(mesh_base_path: Path, chamber_prefix: str) -> AtrialSurfacePaths`

Build atrial paths contract (`chamber_prefix` is "la" or "ra").

##### `build_biv_mesh_paths(mesh_base_path: Path, ventricular_paths: VentricularSurfacePaths) -> BiVMeshPaths`

Build BiV mesh extraction paths.

##### `build_all(mesh_base_path: Path, blank_files_dir: Path) -> UVCSurfaceExtractionPaths`

Build complete path contract for entire workflow.

**Example:**
```python
builder = ModelCreationPathBuilder(output_dir="/data/patient_001/outputs")
all_paths = builder.build_all(
    mesh_base_path=Path("/data/patient_001/mesh/heart"),
    blank_files_dir=Path("/templates/blank_vtx")
)

# Access specific paths
ventricular_paths = all_paths.ventricular
la_paths = all_paths.left_atrial
biv_mesh_paths = all_paths.biv_mesh
```

---

#### `MeshingPathBuilder`

Constructs path contracts for meshing and refinement workflows.

```python
class MeshingPathBuilder:
    def __init__(self, output_dir: Union[Path, str])
```

**Directory Structure Created:**
```
output_dir/
├── 01_raw/          # Raw mesh outputs
├── 02_refined/      # Refined mesh outputs
└── tmp/             # Intermediate files
```

##### `build_meshing_paths(input_image: Path, raw_mesh_basename: str = "heart_mesh") -> MeshingPaths`

Build meshing workflow paths.

##### `build_postprocessing_paths(input_mesh_base: Path, refined_mesh_basename: str = "myocardium_clean") -> MeshPostprocessingPaths`

Build refinement workflow paths.

**Example:**
```python
builder = MeshingPathBuilder(output_dir="/data/patient_001/meshing")

# Meshing paths
meshing_paths = builder.build_meshing_paths(
    input_image=Path("/data/patient_001/segmentation.nii.gz")
)

# Refinement paths
refinement_paths = builder.build_postprocessing_paths(
    input_mesh_base=meshing_paths.output_mesh_base
)
```

---

## Workflows

### Complete Model Creation Pipeline

```python
from pathlib import Path
from pycemrg.data import LabelManager
from pycemrg.system import CommandRunner
from pycemrg_model_creation import (
    MeshingLogic,
    RefinementLogic,
    SurfaceLogic,
    MeshingPathBuilder,
    ModelCreationPathBuilder,
    MeshtoolWrapper,
    Meshtools3DWrapper
)

# --- Configuration ---
patient_dir = Path("/data/patient_001")
segmentation = patient_dir / "segmentation.nii.gz"
label_config = patient_dir / "labels.yaml"
output_dir = patient_dir / "model_outputs"

# --- Initialize ---
label_manager = LabelManager(config_path=label_config)
runner = CommandRunner()

meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool")
)
meshtools3d = Meshtools3DWrapper(
    runner=runner,
    meshtools3d_path=Path("/opt/meshtools3d/meshtools3d")
)

# --- Step 1: Generate Volumetric Mesh ---
meshing_builder = MeshingPathBuilder(output_dir=output_dir / "meshing")
meshing_paths = meshing_builder.build_meshing_paths(input_image=segmentation)

meshing_logic = MeshingLogic(meshtools3d_wrapper=meshtools3d)
meshing_logic.run_meshing(paths=meshing_paths)

# --- Step 2: Refine and Relabel Mesh ---
refinement_paths = meshing_builder.build_postprocessing_paths(
    input_mesh_base=meshing_paths.output_mesh_base
)

myocardium_tags = label_manager.get_source_tags(["LV", "RV"])
tag_mapping = label_manager.get_source_to_target_mapping()

refinement_logic = RefinementLogic(meshtool_wrapper=meshtool)
refinement_logic.run_myocardium_postprocessing(
    paths=refinement_paths,
    myocardium_tags=myocardium_tags,
    tag_mapping=tag_mapping,
    simplify=True
)

# --- Step 3: Extract UVC Surfaces ---
surface_builder = ModelCreationPathBuilder(output_dir=output_dir / "surfaces")
surface_paths = surface_builder.build_all(
    mesh_base_path=refinement_paths.output_mesh_base,
    blank_files_dir=Path("/templates/blank_files")
)

surface_logic = SurfaceLogic(meshtool, label_manager)

# Extract ventricular surfaces
surface_logic.run_ventricular_extraction(paths=surface_paths.ventricular)

# Extract BiV submesh
surface_logic.run_biv_mesh_extraction(
    paths=surface_paths.biv_mesh,
    tags=label_manager.tags
)

# Extract atrial surfaces (optional)
from pycemrg_model_creation.types import Chamber

surface_logic.run_atrial_extraction(
    paths=surface_paths.left_atrial,
    chamber=Chamber.LA
)

print(f"✓ Model creation complete! Outputs in: {output_dir}")
```

---

## Utility Types

### `Chamber`

Enum for cardiac chamber identifiers.

```python
from pycemrg_model_creation.types import Chamber

chamber = Chamber.LV  # "LV"
chamber = Chamber.RV  # "RV"
chamber = Chamber.LA  # "LA"
chamber = Chamber.RA  # "RA"
```

### `SurfaceType`

Enum for surface type identifiers.

```python
from pycemrg_model_creation.types import SurfaceType

surface = SurfaceType.EPI      # "epi"
surface = SurfaceType.ENDO     # "endo"
surface = SurfaceType.BASE     # "base"
surface = SurfaceType.SEPTUM   # "septum"
```

---

## Error Handling

### `SurfaceExtractionError`

Base exception for surface extraction errors.

### `SurfaceIdentificationError`

Raised when surfaces cannot be automatically identified (e.g., connected components don't match expected count).

**Example:**
```python
from pycemrg_model_creation.logic.surfaces import (
    SurfaceExtractionError,
    SurfaceIdentificationError
)

try:
    surface_logic.run_ventricular_extraction(paths=paths)
except SurfaceIdentificationError as e:
    print(f"Could not identify surfaces: {e}")
    # Handle ambiguous geometry
except SurfaceExtractionError as e:
    print(f"Extraction failed: {e}")
    # Handle general failures
```

---

## Best Practices

### 1. Use Path Builders

Always use path builders rather than manually constructing path contracts. This ensures consistency and reduces errors.

```python
# ✓ Good
builder = ModelCreationPathBuilder(output_dir)
paths = builder.build_all(mesh_base, blank_files_dir)

# ✗ Avoid
paths = VentricularSurfacePaths(
    mesh=mesh,
    output_dir=output_dir,
    tmp_dir=tmp_dir,
    # ... 20+ more fields to fill manually
)
```

### 2. Validate Inputs Early

Check that input files exist before starting long-running workflows.

```python
assert segmentation_path.exists(), f"Segmentation not found: {segmentation_path}"
assert mesh_base.with_suffix(".pts").exists(), f"Mesh not found: {mesh_base}"
```

### 3. Enable Detailed Logging

Use pyCEMRG's logging utilities for debugging.

```python
from pycemrg.core import setup_logging
import logging

setup_logging(log_level=logging.DEBUG, log_file="workflow.log")
```

### 4. Organize Patient Data

Recommended directory structure:

```
/data/
└── patient_001/
    ├── input/
    │   ├── segmentation.nii.gz
    │   └── labels.yaml
    ├── meshing/
    │   ├── 01_raw/
    │   ├── 02_refined/
    │   └── tmp/
    └── surfaces/
        ├── BiV/
        ├── LA/
        └── RA/
```

### 5. Reuse Path Contracts

Path contracts can be saved and reloaded for reproducibility.

```python
import json
from dataclasses import asdict

# Save paths
with open("paths.json", "w") as f:
    json.dump(asdict(paths), f, default=str)

# Load paths (manual reconstruction)
with open("paths.json", "r") as f:
    paths_dict = json.load(f)
# Reconstruct dataclass from dict
```


## API Version History

### v1.0.0
- Initial public API
- Core workflows: meshing, refinement, surface extraction
- Path contract system
- Support for BiV and atrial geometries

---

## Further Reading

- **pyCEMRG Core Documentation**: For `LabelManager`, `CarpRunner`, and system utilities
<!-- - **CARPentry Documentation**: For underlying tool specifications
- **openCARP**: For cardiac simulation capabilities -->


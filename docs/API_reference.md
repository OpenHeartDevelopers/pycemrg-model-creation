# pycemrg Model Creation - API Reference

**Version:** 1.0.0  
**Purpose:** A Pythonic SDK for cardiac electromechanical modeling using the CARPentry/openCARP ecosystem

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Modules](#core-modules)
   - [Configuration](#configuration)
   - [Logic Layer](#logic-layer) — `SurfaceLogic`, `UvcLogic`, `MeshingLogic`, `RefinementLogic`
   - [Tools & Wrappers](#tools--wrappers)
   - [Path Contracts](#path-contracts) — including `VentricularUVCPaths`
   - [Path Builders](#path-builders)
   - [Utilities](#utilities)
4. [Complete Workflows](#complete-workflows)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

---

## Overview

`pycemrg_model_creation` provides high-level abstractions for creating patient-specific cardiac computational models. The library follows a layered architecture:

- **Configuration Layer**: Define anatomical tags and meshing parameters
- **Path Contracts**: Explicit, type-safe path specifications for all I/O operations
- **Logic Layer**: Stateless, scientific workflows for meshing, refinement, and surface extraction
- **Tools Layer**: Low-level wrappers for CARPentry command-line tools
- **Utilities Layer**: Pure functions for mesh/geometry operations

**Key Design Principles:**
- **Path-agnostic logic**: All file paths are provided explicitly through dataclass contracts
- **Stateless operations**: Logic classes don't maintain state between operations
- **Explicit over implicit**: No hidden path derivations or magic defaults
- **Separation of concerns**: Logic orchestrates, tools execute, utilities transform

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

**Constructor:**
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

##### `from_dict(tags_dict: Dict[str, Union[int, List[int]]]) -> TagsConfig`

Create TagsConfig from dictionary.

```python
tags = TagsConfig.from_dict({
    "LV": 1,
    "RV": 2,
    "LA": [3, 4]  # Multiple tags supported
})
```

##### `to_dict() -> Dict[str, Union[int, List[int]]]`

Convert to dictionary.

##### `get_tags_string(keys: List[str]) -> str`

Get comma-separated string of tags for specified anatomical regions.

```python
# Get tags for ventricles
ventricle_tags = tags.get_tags_string(["LV", "RV"])  # "1,2"

# Multiple tags per region are flattened
tags = TagsConfig(LV=[1, 2, 3], RV=4)
all_tags = tags.get_tags_string(["LV", "RV"])  # "1,2,3,4"
```

##### `get_tags_list(keys: List[str]) -> List[int]`

Get flat list of tag integers.

```python
tag_list = tags.get_tags_list(["LV", "RV"])  # [1, 2]
```

---

### Logic Layer

The logic layer provides high-level, multi-step scientific workflows.

#### `SurfaceLogic`

Orchestrates surface extraction for Universal Ventricular Coordinate (UVC) generation.

**Constructor:**
```python
class SurfaceLogic:
    def __init__(
        self, 
        meshtool: MeshtoolWrapper,
        label_manager: LabelManager
    )
```

**Parameters:**
- `meshtool`: Initialized MeshtoolWrapper instance
- `label_manager`: LabelManager for anatomical label translation

---

#### Ventricular Extraction Methods

##### `run_ventricular_extraction(paths: VentricularSurfacePaths) -> None`

Complete ventricular surface extraction workflow. Executes all steps required for UVC coordinate generation:

1. Extract base surface (ventricle-valve interface)
2. Extract and identify epicardium, LV endocardium, RV endocardium
3. Extract interventricular septum
4. Map surfaces to connected components
5. Remove septum from RV endocardium (create free wall)
6. Generate VTX boundary condition files

**Output files created:**
- `biv_epi.surf/.vtx` - Epicardial surface
- `biv_lvendo.surf/.vtx` - LV endocardium
- `biv_rvendo.surf/.vtx` - RV free wall (septum removed)
- `biv_septum.surf/.vtx` - Interventricular septum
- `biv.base.vtx` - Base boundary
- `biv.rvsept_pt.vtx` - RV septum reference point

```python
surface_logic.run_ventricular_extraction(paths=ventricular_paths)
```

##### `extract_ventricular_base(paths: VentricularSurfacePaths) -> None`

Extract base surface where ventricles meet valve planes.

**Technical details:**
- Uses meshtool's `extract surface` with operation `"{LV,RV}:{MV,TV,AV,PV}"`
- Creates interface surface between myocardium and valve labels

##### `extract_ventricular_surfaces(paths: VentricularSurfacePaths) -> None`

Extract and automatically identify epicardium and endocardia.

**Algorithm:**
1. Extract combined epi/endo surface (excludes valve openings)
2. Identify 3 connected components
3. Compute surface normals and center of gravity
4. Identify epicardium by outward-pointing normals
5. Distinguish LV/RV endo by distance from LV center

**Raises:**
- `SurfaceIdentificationError`: If component count ≠ 3 or identification fails

##### `extract_septum(paths: VentricularSurfacePaths) -> None`

Extract interventricular septum as LV surface facing RV.

**Technical details:**
- Extracts LV surface excluding RV, RA, and pulmonary artery
- Identifies 2 largest connected components (LV epi + septum)

##### `map_ventricular_surfaces(paths: VentricularSurfacePaths) -> None`

Map connected component indices to create final .surf and .vtx files.

##### `remove_septum_from_rv_endo(paths: VentricularSurfacePaths) -> None`

Remove septum triangles from RV endocardium to isolate free wall.

**Algorithm:**
- Identifies triangles where all 3 vertices are part of septum
- Removes these triangles from RV endo surface
- Updates .surf and .vtx files

##### `prepare_ventricular_vtx_files(paths: VentricularSurfacePaths) -> None`

Generate .vtx boundary condition files from .surf files.

---

#### Atrial Extraction Methods

##### `run_atrial_extraction(paths: AtrialSurfacePaths, chamber: Chamber) -> None`

Complete atrial surface extraction workflow.

**Parameters:**
- `paths`: Atrial path contract
- `chamber`: `Chamber.LA` or `Chamber.RA`

```python
from pycemrg_model_creation.types import Chamber

surface_logic.run_atrial_extraction(
    paths=la_paths,
    chamber=Chamber.LA
)
```

##### `extract_atrial_base(paths: AtrialSurfacePaths, chamber: Chamber) -> None`

Extract atrial base (atrium-valve interface).

##### `extract_atrial_surfaces(paths: AtrialSurfacePaths, chamber: Chamber) -> None`

Extract atrial epicardium and endocardium.

---

#### Submesh Extraction Methods

##### `run_biv_mesh_extraction(paths: BiVMeshPaths, tags: TagsConfig) -> None`

Extract biventricular submesh from four-chamber mesh and map VTX files.

**Workflow:**
1. Extract submesh containing only LV and RV tags
2. Map VTX boundary condition files from full mesh to BiV submesh

```python
surface_logic.run_biv_mesh_extraction(
    paths=biv_mesh_paths,
    tags=tags
)
```

##### `run_atrial_mesh_extraction(paths: AtrialMeshPaths, tags: TagsConfig, chamber: Chamber) -> None`

Extract atrial submesh and map VTX files.

**Special handling:**
- Copies blank template files for apex/septum (not applicable to atria but required for UVC interface compatibility)

---

#### Complete Workflow

##### `run_all(paths: UVCSurfaceExtractionPaths, tags: TagsConfig, ventricular_files_to_map: Optional[List[Path]] = None, la_files_to_map: Optional[List[Path]] = None, ra_files_to_map: Optional[List[Path]] = None) -> None`

Execute complete UVC surface extraction for all chambers.

**Workflow phases:**
1. Ventricular surfaces
2. LA surfaces
3. RA surfaces
4. BiV submesh extraction
5. LA submesh extraction
6. RA submesh extraction

---

#### `UvcLogic`

Orchestrates Universal Ventricular Coordinate (UVC) calculation using CARPentry's `mguvc` tool.

**Constructor:**
```python
class UvcLogic:
    def __init__(self, carp_wrapper: CarpWrapper)
```

**Parameters:**
- `carp_wrapper`: Initialized `CarpWrapper` instance

**Initialization:**
```python
from pycemrg.system import CarpRunner, CommandRunner
from pycemrg_model_creation.tools.wrappers import CarpWrapper
from pycemrg_model_creation.logic.uvc import UvcLogic

carp_runner = CarpRunner(
    runner=CommandRunner(),
    carp_config_path=Path("/opt/carpentry/config.sh")
)
carp_wrapper = CarpWrapper(carp_runner)
uvc_logic = UvcLogic(carp_wrapper)
```

---

##### `run_ventricular_uvc_calculation(paths: VentricularUVCPaths, lv_tag: int, rv_tag: int, np: int = 1) -> None`

Complete ventricular UVC calculation workflow.

**Steps:**
1. Validate input files (mesh `.pts`/`.elem` + 6 VTX boundary files)
2. Generate etags script mapping element tags to anatomical regions
3. Run `mguvc` to solve Laplace equations
4. (Output validation is handled by `CarpRunner` via `expected_outputs`)

**Parameters:**
- `paths`: Populated `VentricularUVCPaths` contract
- `lv_tag`: Element tag for LV myocardium in the mesh
- `rv_tag`: Element tag for RV myocardium in the mesh
- `np`: Number of processors for `mguvc` (default: 1)

**Raises:**
- `FileNotFoundError`: If required input files (mesh or VTX) are missing
- `RuntimeError`: If `mguvc` exits with an error or expected outputs are not created

**Output files created in `paths.output_dir`:**
- `{basename}.uvc_z.dat` — Apico-basal coordinate (apex=0, base=1)
- `{basename}.uvc_rho.dat` — Transmural coordinate (endo=0, epi=1)
- `{basename}.uvc_phi.dat` — Rotational/circumferential coordinate
- `{basename}.uvc_ven.dat` — Ventricular identifier (LV vs RV)
- `{basename}.sol_*_lap.dat` — Laplace solutions (4 files, when `--laplace-solution` is set)
- `{basename}.aff.dat`, `{basename}.m2s.dat` — Mapping files

**Example:**
```python
uvc_logic.run_ventricular_uvc_calculation(
    paths=uvc_paths,
    lv_tag=1,
    rv_tag=2,
    np=4
)
```

---

#### `MeshingLogic`

Orchestrates volumetric meshing from segmentation images using meshtools3d.

**Constructor:**
```python
class MeshingLogic:
    def __init__(self, meshtools3d_wrapper: Meshtools3DWrapper)
```

##### `run_meshing(paths: MeshingPaths, meshing_params: Dict[str, Any] = None, cleanup: bool = True) -> None`

Execute full meshtools3d workflow.

**Steps:**
1. Convert NIfTI segmentation to INR format
2. Generate meshtools3d parameter file (.par)
3. Execute meshtools3d binary
4. Clean up intermediate files (optional)

**Parameters:**
- `paths`: MeshingPaths contract
- `meshing_params`: Optional parameter overrides (nested dict by section)
- `cleanup`: If True, deletes .inr and .par intermediate files

**Parameter override example:**
```python
meshing_logic.run_meshing(
    paths=meshing_paths,
    meshing_params={
        'meshing': {
            'facet_size': '0.7',
            'cell_size': '0.8'
        },
        'output': {
            'out_vtk': 1
        }
    },
    cleanup=True
)
```

**Output files:**
- `{output_mesh_base}.pts` - Mesh vertices
- `{output_mesh_base}.elem` - Mesh elements with tags
- `{output_mesh_base}.vtk` - VTK visualization file

---

#### `RefinementLogic`

Post-processes and refines raw meshes.

**Constructor:**
```python
class RefinementLogic:
    def __init__(self, meshtool_wrapper: MeshtoolWrapper)
```

##### `run_myocardium_postprocessing(paths: MeshPostprocessingPaths, myocardium_tags: List[int], tag_mapping: Dict[int, int], simplify: bool = False) -> None`

Complete mesh refinement workflow.

**Steps:**
1. Extract myocardium submesh by element tags
2. Optionally simplify topology (requires `simplify_tag_topology` standalone tool)
3. Relabel element tags to standardized scheme
4. Generate VTK for visualization

**Parameters:**
- `paths`: MeshPostprocessingPaths contract
- `myocardium_tags`: List of tags to extract (e.g., [1, 2] for LV, RV)
- `tag_mapping`: Dictionary mapping old tags to new tags (e.g., {1: 10, 2: 20})
- `simplify`: Enable topology simplification (removes duplicate vertices, improves quality)

**Example:**
```python
from pycemrg.data import LabelManager, LabelMapper

source_labels = LabelManager("source_labels.yaml")
target_labels = LabelManager("target_labels.yaml")

myocardium_tags = source_labels.get_values_from_names(["LV", "RV"])

mapper = LabelMapper(source=source_labels, target=target_labels)
tag_mapping = mapper.get_source_to_target_mapping()

refinement_logic = RefinementLogic(meshtool_wrapper)
refinement_logic.run_myocardium_postprocessing(
    paths=refinement_paths,
    myocardium_tags=myocardium_tags,
    tag_mapping=tag_mapping,
    simplify=True
)
```

**Output files:**
- `{output_mesh_base}.pts` - Refined vertices
- `{output_mesh_base}.elem` - Elements with relabeled tags
- `{output_mesh_base}.vtk` - Visualization file

---

### Tools & Wrappers

Low-level wrappers for CARPentry command-line tools. These provide 1-to-1 mappings to command-line arguments.

#### `MeshtoolWrapper`

Wrapper for the `meshtool` utility.

**Constructors:**

##### `from_system_path(logger: Optional[logging.Logger] = None, meshtool_install_dir: Optional[Path] = None) -> MeshtoolWrapper`

Create wrapper using system-wide meshtool (must be in PATH).

```python
meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool")  # Required for standalone tools
)
```

**Note:** `meshtool_install_dir` is required to locate standalone tools like `simplify_tag_topology`.

##### `from_carp_runner(carp_runner: CarpRunner) -> MeshtoolWrapper`

Create wrapper using meshtool from CARPentry environment.

```python
from pycemrg.system import CarpRunner

carp_runner = CarpRunner(
    runner=CommandRunner(),
    carp_config_path=Path("/opt/carpentry/config.sh")
)
meshtool = MeshtoolWrapper.from_carp_runner(carp_runner)
```

---

#### Key Methods

##### `extract_mesh(input_mesh_path: Path, output_submesh_path: Path, tags: Sequence[Union[str, int]], ifmt: str = "carp_txt", normalise: bool = False) -> None`

Extract submesh containing only specified element tags.

**Command:** `meshtool extract mesh`

```python
meshtool.extract_mesh(
    input_mesh_path=Path("full_heart"),
    output_submesh_path=Path("biv_mesh"),
    tags=[1, 2],  # LV, RV
    ifmt="carp_txt",
    normalise=False
)
```

##### `extract_surface(input_mesh_path: Path, output_surface_path: Path, ofmt: str = "vtk", op_tag_base: Optional[str] = None) -> None`

Extract surface from volume mesh.

**Command:** `meshtool extract surface`

**Parameters:**
- `op_tag_base`: Operation string defining surface (e.g., "1:2" = interface between tags 1 and 2, "1-2" = tag 1 excluding tag 2)

```python
meshtool.extract_surface(
    input_mesh_path=Path("mesh"),
    output_surface_path=Path("base"),
    ofmt="vtk",
    op_tag_base="1,2:5,6"  # LV,RV meeting MV,TV
)
```

##### `extract_unreachable(input_mesh_path: Path, submsh_path: Path, ofmt: str = 'vtk', ifmt: str = '') -> None`

Extract connected components (unreachable regions).

**Command:** `meshtool extract unreachable`

**Output:** Creates numbered part files: `{submsh_path}.part0`, `{submsh_path}.part1`, etc.

```python
meshtool.extract_unreachable(
    input_mesh_path=Path("surface.surfmesh"),
    submsh_path=Path("components"),
    ofmt="vtk",
    ifmt="vtk"
)
```

##### `convert(input_mesh_path: Path, output_mesh_path: Path, ofmt: str = "vtk", ifmt: Optional[str] = None) -> None`

Convert mesh between formats.

**Command:** `meshtool convert`

**Supported formats:** `carp_txt`, `carp_bin`, `vtk`, `vtk_polydata`

```python
meshtool.convert(
    input_mesh_path=Path("mesh"),
    output_mesh_path=Path("mesh_vtk"),
    ifmt="carp_txt",
    ofmt="vtk"
)
```

##### `smooth(input_mesh_path: Path, output_mesh_path: Path, smoothing_params: str, tags: Optional[Sequence[Union[str, int]]] = None, ifmt: str = "carp_txt", ofmt: str = "carp_txt") -> None`

Smooth mesh surfaces.

**Command:** `meshtool smooth mesh`

```python
meshtool.smooth(
    input_mesh_path=Path("mesh"),
    output_mesh_path=Path("mesh_smooth"),
    smoothing_params="1:0.5:3",  # iterations:lambda:passes
    tags=[1, 2]  # Only smooth these tags
)
```

##### `map(submesh_path: Path, files_list: List[str], output_folder: Path, mode: str = "m2s") -> None`

Map data files between mesh and submesh.

**Command:** `meshtool map`

**Modes:**
- `m2s`: Mesh to submesh
- `s2m`: Submesh to mesh

```python
meshtool.map(
    submesh_path=Path("biv_mesh"),
    files_list=["base.vtx", "epi.vtx"],
    output_folder=Path("biv_mapped"),
    mode="m2s"
)
```

##### `interpolate(mesh_path: Path, input_data_path: Path, output_data_path: Path, mode: str) -> None`

Interpolate data between nodes and elements.

**Command:** `meshtool interpolate`

**Modes:** `node2elem`, `elem2node`

##### `insert_data(target_mesh_path: Path, source_submesh_path: Path, source_data_path: Path, output_data_path: Path, mode: str) -> None`

Insert data from submesh into larger mesh.

**Command:** `meshtool insert data`

##### `simplify_topology(input_mesh_path: Path, output_mesh_path: Path, neighbors: int = 50, ifmt: str = "carp_txt", ofmt: str = "carp_txt") -> None`

Simplify mesh topology (remove duplicate vertices, improve quality).

**Command:** `simplify_tag_topology` (standalone tool)

**Requirements:**
- Must initialize wrapper with `meshtool_install_dir` pointing to meshtool installation root
- Tool located at `{meshtool_install_dir}/standalones/simplify_tag_topology`

```python
meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool")
)

if meshtool.is_simplify_topology_available:
    meshtool.simplify_topology(
        input_mesh_path=Path("mesh"),
        output_mesh_path=Path("mesh_simplified"),
        neighbors=50
    )
```

**Property:** `is_simplify_topology_available: bool` - Check if tool is available

---

#### `CarpWrapper`

Wrapper for CARP ecosystem tools (GlRuleFibres, GlVTKConvert, igbextract, etc.).

**Constructor:**
```python
class CarpWrapper:
    def __init__(self, carp_runner: CarpRunner)
```

**Initialization:**
```python
from pycemrg.system import CarpRunner, CommandRunner

carp_runner = CarpRunner(
    runner=CommandRunner(),
    carp_config_path=Path("/opt/carpentry/config.sh")
)
carp_wrapper = CarpWrapper(carp_runner)
```

---

##### `gl_rule_fibres(mesh_name: Path, uvc_apba: Path, uvc_epi: Path, uvc_lv: Path, uvc_rv: Path, output_name: Path, angles: Dict[str, float] = None, fibre_type: str = "biv") -> None`

Generate rule-based fiber orientations using UVC coordinates.

**Command:** `GlRuleFibres`

**Default fiber angles:**
```python
DEFAULT_FIBRE_ANGLES = {
    "alpha_endo": 60,    # Endocardial helix angle
    "alpha_epi": -60,    # Epicardial helix angle
    "beta_endo": -65,    # Endocardial transverse angle
    "beta_epi": 25,      # Epicardial transverse angle
}
```

**Example:**
```python
from pycemrg_model_creation.tools import DEFAULT_FIBRE_ANGLES

carp_wrapper.gl_rule_fibres(
    mesh_name=Path("biv_mesh"),
    uvc_apba=Path("biv_COORDS_Z.dat"),
    uvc_epi=Path("biv_COORDS_RHO.dat"),
    uvc_lv=Path("biv_COORDS_PHI_LV.dat"),
    uvc_rv=Path("biv_COORDS_PHI_RV.dat"),
    output_name=Path("biv_fibres"),
    angles=DEFAULT_FIBRE_ANGLES,
    fibre_type="biv"
)
```

##### `gl_vtk_convert(mesh_name: Path, output_name: Path, node_data: Sequence[str] = (), elem_data: Sequence[str] = (), trim_names: bool = True) -> None`

Convert CARP mesh to VTK with data overlays.

**Command:** `GlVTKConvert`

```python
carp_wrapper.gl_vtk_convert(
    mesh_name=Path("biv_mesh"),
    output_name=Path("biv_with_data"),
    node_data=["transmembrane_voltage.dat"],
    elem_data=["fibers.lon", "activation_times.dat"],
    trim_names=True
)
```

##### `igb_extract(igb_file: Path, output_file: Path, output_format: str = "ascii", first_frame: int = 0, last_frame: int = 0) -> None`

Extract data from IGB (openCARP binary) files.

**Command:** `igbextract`

```python
carp_wrapper.igb_extract(
    igb_file=Path("vm.igb"),
    output_file=Path("vm_frame0.dat"),
    output_format="ascii",
    first_frame=0,
    last_frame=0
)
```

---

#### `Meshtools3DWrapper`

Low-level wrapper for the `meshtools3d` binary.

**Constructor:**
```python
class Meshtools3DWrapper:
    def __init__(self, runner: CommandRunner, meshtools3d_path: Path)
```

**Initialization:**
```python
from pycemrg.system import CommandRunner

runner = CommandRunner()
meshtools3d = Meshtools3DWrapper(
    runner=runner,
    meshtools3d_path=Path("/opt/meshtools3d/meshtools3d")
)
```

##### `run(parameter_file: Path, expected_outputs: List[Path]) -> None`

Execute meshtools3d with parameter file.

```python
meshtools3d.run(
    parameter_file=Path("meshing.par"),
    expected_outputs=[
        Path("mesh.pts"),
        Path("mesh.elem"),
        Path("mesh.vtk")
    ]
)
```

---

### Path Contracts

Path contracts are frozen dataclasses that explicitly define all input/output paths for workflows. They eliminate ambiguity and make I/O transparent.

#### `MeshingPaths`

Defines paths for volumetric meshing workflow.

**Fields:**
```python
@dataclass(frozen=True)
class MeshingPaths:
    # Input
    input_segmentation_nifti: Path
    
    # Directories
    output_dir: Path
    tmp_dir: Path
    
    # Intermediate files (in tmp_dir)
    intermediate_inr: Path
    intermediate_parameter_file: Path
    
    # Final output (in output_dir)
    output_mesh_base: Path  # Base name without extension
```

---

#### `MeshPostprocessingPaths`

Defines paths for mesh refinement workflow.

**Fields:**
```python
@dataclass(frozen=True)
class MeshPostprocessingPaths:
    # Input
    input_mesh_base: Path  # Raw mesh from meshtools3d
    
    # Directories
    output_dir: Path
    tmp_dir: Path
    
    # Intermediate mesh (in tmp_dir)
    intermediate_myocardium_mesh: Path
    
    # Final output mesh (in output_dir)
    output_mesh_base: Path  # Clean, relabeled mesh
```

---

#### `VentricularSurfacePaths`

Defines all paths for ventricular (BiV) surface extraction.

**Fields:**
```python
@dataclass(frozen=True)
class VentricularSurfacePaths:
    # Input
    mesh: Path  # Four-chamber mesh base name
    
    # Directories
    output_dir: Path
    tmp_dir: Path
    
    # Intermediate surfaces (in tmp_dir)
    base_surface: Path
    epi_endo_combined: Path
    epi_endo_cc_base: Path          # Connected component base name
    septum_raw: Path
    septum_cc_base: Path            # Connected component base name
    lv_epi_intermediate: Path
    
    # Final surfaces (in output_dir)
    epi_surface: Path
    lv_endo_surface: Path
    rv_endo_surface: Path
    septum_surface: Path
    
    # VTX files for UVC (in output_dir)
    base_vtx: Path
    epi_vtx: Path
    lv_endo_vtx: Path
    rv_endo_vtx: Path
    septum_vtx: Path
    rv_septum_point_vtx: Path       # RV septum reference point
```

---

#### `AtrialSurfacePaths`

Defines paths for atrial (LA or RA) surface extraction.

**Fields:**
```python
@dataclass(frozen=True)
class AtrialSurfacePaths:
    # Input
    mesh: Path
    
    # Directories
    output_dir: Path
    tmp_dir: Path
    
    # Intermediate surfaces (in tmp_dir)
    base_surface: Path
    epi_endo_combined: Path
    
    # Final surfaces (in output_dir)
    epi_surface: Path
    endo_surface: Path
    
    # VTX files (in output_dir)
    base_vtx: Path
    epi_vtx: Path
    endo_vtx: Path
    
    # Blank compatibility files
    apex_vtx: Path
    rv_septum_point_vtx: Path
```

---

#### `BiVMeshPaths`

Defines paths for biventricular submesh extraction.

**Fields:**
```python
@dataclass(frozen=True)
class BiVMeshPaths:
    source_mesh: Path               # Four-chamber mesh
    output_mesh: Path               # BiV submesh
    output_dir: Path
    
    vtx_files_to_map: List[Path]   # VTX files to map from full mesh
    mapped_vtx_output_dir: Path     # Directory for mapped VTX files
```

---

#### `AtrialMeshPaths`

Defines paths for atrial submesh extraction.

**Fields:**
```python
@dataclass(frozen=True)
class AtrialMeshPaths:
    source_mesh: Path
    output_mesh: Path
    output_dir: Path
    
    vtx_files_to_map: List[Path]
    mapped_vtx_output_dir: Path
    
    # Template files for compatibility
    apex_template: Path
    rv_septum_template: Path
    apex_output: Path
    rv_septum_output: Path
```

---

#### `UVCSurfaceExtractionPaths`

Master path contract containing all sub-contracts for complete UVC workflow.

**Fields:**
```python
@dataclass(frozen=True)
class UVCSurfaceExtractionPaths:
    ventricular: VentricularSurfacePaths
    left_atrial: AtrialSurfacePaths
    right_atrial: AtrialSurfacePaths
    biv_mesh: BiVMeshPaths
    la_mesh: AtrialMeshPaths
    ra_mesh: AtrialMeshPaths
```

---

#### `VentricularUVCPaths`

Path contract for ventricular UVC calculation. This is the only **frozen** contract — all fields are immutable once built.

**Critical layout requirement:** `mguvc` expects the BiV mesh and all six VTX boundary files to be co-located in the same directory with standard names. The builder enforces this layout.

**Fields:**
```python
@dataclass(frozen=True)
class VentricularUVCPaths:
    # Input: BiV submesh (without extension)
    biv_mesh: Path          # e.g. /data/BiV/BiV

    # Input: VTX boundary files — must be in biv_mesh.parent
    base_vtx: Path          # BiV.base.vtx
    epi_vtx: Path           # BiV.epi.vtx
    lv_endo_vtx: Path       # BiV.lvendo.vtx
    rv_endo_vtx: Path       # BiV.rvendo.vtx
    septum_vtx: Path        # BiV.rvsept.vtx
    rvendo_nosept_vtx: Path # BiV.rvendo_nosept.vtx

    # Input: etags script — written to biv_mesh.parent (NOT inside output_dir)
    etags_file: Path        # BiV.etags.sh

    # Output directory — created by mguvc, not by the builder
    output_dir: Path        # biv_mesh.parent / "uvc"

    # Primary UVC coordinate outputs (in output_dir)
    uvc_z: Path             # Apico-basal
    uvc_rho: Path           # Transmural
    uvc_phi: Path           # Rotational
    uvc_ven: Path           # Ventricular identifier

    # Intermediate Laplace solutions (in output_dir)
    sol_apba: Path
    sol_endoepi: Path
    sol_lvendo: Path
    sol_rvendo: Path

    # Mapping files (in output_dir)
    aff_dat: Path
    m2s_dat: Path
```

**Note on etags placement:** The etags file is written to `biv_mesh.parent` (alongside the mesh), not inside `output_dir`. This is intentional — `mguvc` prompts interactively if `output_dir` already exists, so the directory must not be created before the tool runs. The etags file must be written somewhere accessible before `mguvc` starts; placing it next to the mesh avoids the problem.

---

### Path Builders

Builders simplify path contract creation using standard naming conventions. They encapsulate directory structure decisions.

#### `MeshingPathBuilder`

Constructs path contracts for meshing and refinement workflows.

**Constructor:**
```python
class MeshingPathBuilder:
    def __init__(self, output_dir: Union[Path, str])
```

**Directory structure created:**
```
output_dir/
├── 01_raw/          # Raw mesh outputs from meshtools3d
├── 02_refined/      # Refined mesh outputs
└── tmp/             # Intermediate files
```

**Methods:**

##### `build_meshing_paths(input_image: Path, raw_mesh_basename: str = "heart_mesh") -> MeshingPaths`

Build paths for initial meshing workflow.

```python
builder = MeshingPathBuilder(output_dir="/data/patient_001/meshing")

meshing_paths = builder.build_meshing_paths(
    input_image=Path("/data/patient_001/segmentation.nii.gz"),
    raw_mesh_basename="heart_mesh"
)
```

**Generated paths:**
- Input: `/data/patient_001/segmentation.nii.gz`
- Output: `/data/patient_001/meshing/01_raw/heart_mesh.{pts,elem,vtk}`
- Temp: `/data/patient_001/meshing/tmp/`

##### `build_postprocessing_paths(input_mesh_base: Path, refined_mesh_basename: str = "myocardium_clean") -> MeshPostprocessingPaths`

Build paths for refinement workflow.

```python
refinement_paths = builder.build_postprocessing_paths(
    input_mesh_base=meshing_paths.output_mesh_base,
    refined_mesh_basename="myocardium_clean"
)
```

**Generated paths:**
- Input: `/data/patient_001/meshing/01_raw/heart_mesh`
- Output: `/data/patient_001/meshing/02_refined/myocardium_clean.{pts,elem,vtk}`

---

#### `ModelCreationPathBuilder`

Constructs path contracts for UVC surface extraction workflows.

**Constructor:**
```python
class ModelCreationPathBuilder:
    def __init__(self, output_dir: Union[Path, str])
```

**Directory structure created:**
```
output_dir/
├── BiV/
│   ├── tmp/         # Intermediate ventricular files
│   └── biv/         # Mapped BiV VTX files
├── LA/
│   ├── tmp/         # Intermediate LA files
│   └── la/          # Mapped LA VTX files
└── RA/
    ├── tmp/         # Intermediate RA files
    └── ra/          # Mapped RA VTX files
```

**Methods:**

##### `build_ventricular_paths(mesh_base_path: Path) -> VentricularSurfacePaths`

Build paths for ventricular surface extraction.

```python
builder = ModelCreationPathBuilder(output_dir="/data/patient_001/surfaces")

ventricular_paths = builder.build_ventricular_paths(
    mesh_base_path=Path("/data/patient_001/mesh/four_chamber")
)
```

**Generated output paths:**
- `BiV/biv_epi.{surf,vtk}`
- `BiV/biv_lvendo.{surf,vtk}`
- `BiV/biv_rvendo.{surf,vtk}`
- `BiV/biv_septum.{surf,vtk}`
- `BiV/biv.base.vtx`
- `BiV/biv.epi.vtx`
- etc.

##### `build_atrial_paths(mesh_base_path: Path, chamber_prefix: str) -> AtrialSurfacePaths`

Build paths for atrial surface extraction.

**Parameters:**
- `chamber_prefix`: `"la"` or `"ra"`

```python
la_paths = builder.build_atrial_paths(
    mesh_base_path=Path("/data/patient_001/mesh/four_chamber"),
    chamber_prefix="la"
)
```

##### `build_biv_mesh_paths(mesh_base_path: Path, ventricular_paths: VentricularSurfacePaths) -> BiVMeshPaths`

Build paths for BiV submesh extraction.

```python
biv_mesh_paths = builder.build_biv_mesh_paths(
    mesh_base_path=Path("/data/patient_001/mesh/four_chamber"),
    ventricular_paths=ventricular_paths
)
```

##### `build_atrial_mesh_paths(mesh_base_path: Path, atrial_paths: AtrialSurfacePaths, blank_files_dir: Path, chamber_prefix: str) -> AtrialMeshPaths`

Build paths for atrial submesh extraction.

**Parameters:**
- `blank_files_dir`: Directory containing blank template VTX files

##### `build_all(mesh_base_path: Path, blank_files_dir: Path) -> UVCSurfaceExtractionPaths`

Build complete path contract for entire UVC workflow.

```python
all_paths = builder.build_all(
    mesh_base_path=Path("/data/patient_001/mesh/four_chamber"),
    blank_files_dir=Path("/templates/blank_vtx")
)

# Access specific path contracts
ventricular_paths = all_paths.ventricular
la_paths = all_paths.left_atrial
biv_mesh_paths = all_paths.biv_mesh
```

---

##### `build_ventricular_uvc_paths(biv_mesh: Path, output_subdir: str = "uvc", overwrite_existing: bool = True, backup_existing: bool = True) -> VentricularUVCPaths`

Build path contract for ventricular UVC calculation.

**Parameters:**
- `biv_mesh`: Base path to the BiV submesh (without extension). **The output location is derived from this path** — outputs always land in `biv_mesh.parent / output_subdir`. There is no separate `output_dir` argument.
- `output_subdir`: Name of the output subdirectory (default: `"uvc"`)
- `backup_existing`: If `True` (default), an existing `output_subdir` is renamed with a timestamp before proceeding
- `overwrite_existing`: If `True` and `backup_existing=False`, deletes the existing directory instead

**Side effects at build time:**
- If `output_subdir` already exists and `backup_existing=True`, it is moved to `{output_subdir}_backup_{timestamp}` immediately when this method is called — before any logic runs
- The output directory is **not** created by the builder; `mguvc` creates it during execution

**Path layout produced:**
```
biv_mesh.parent/            # e.g. /data/surfaces/BiV/
├── BiV.pts                 # BiV mesh (pre-existing)
├── BiV.elem
├── BiV.base.vtx            # VTX boundary files (pre-existing)
├── BiV.epi.vtx
├── BiV.lvendo.vtx
├── BiV.rvendo.vtx
├── BiV.rvsept.vtx
├── BiV.rvendo_nosept.vtx
├── BiV.etags.sh            # Written by UvcLogic._generate_etags
└── uvc/                    # Created by mguvc during execution
    ├── BiV.uvc_z.dat
    ├── BiV.uvc_rho.dat
    ├── BiV.uvc_phi.dat
    ├── BiV.uvc_ven.dat
    ├── BiV.sol_*_lap.dat
    ├── BiV.aff.dat
    └── BiV.m2s.dat
```

**Example:**
```python
# Assumes biv_mesh_paths.output_mesh was produced by run_biv_mesh_extraction
# and VTX files have been mapped into the same directory
biv_mesh = Path("/data/surfaces/BiV/BiV")

uvc_paths = builder.build_ventricular_uvc_paths(biv_mesh=biv_mesh)

uvc_logic.run_ventricular_uvc_calculation(
    paths=uvc_paths,
    lv_tag=1,
    rv_tag=2
)
```

---

### Utilities

Pure, stateless helper functions for low-level operations.

#### Mesh I/O (`utilities/mesh.py`)

##### `read_carp_mesh(mesh_base_path: Path, elem_type: ElemType = ElemType.Tt, read_tags: bool = True) -> Tuple[np.ndarray, np.ndarray]`

Read CARP mesh files (.pts and .elem).

**Parameters:**
- `elem_type`: `ElemType.Tt` (tetrahedra), `ElemType.Tr` (triangles), `ElemType.Ln` (lines)
- `read_tags`: Include element tag column

**Returns:** `(points_array, elements_array)`

```python
from pycemrg_model_creation.utilities.mesh import read_carp_mesh, ElemType

# Read tetrahedral mesh with tags
points, tets = read_carp_mesh(
    Path("mesh"),
    elem_type=ElemType.Tt,
    read_tags=True
)
# points: (N, 3) coordinates
# tets: (M, 5) [v0, v1, v2, v3, tag]

# Read surface triangles without tags
points, tris = read_carp_mesh(
    Path("surface"),
    elem_type=ElemType.Tr,
    read_tags=False
)
# tris: (M, 3) [v0, v1, v2]
```

##### `read_pts(pts_path: Path) -> np.ndarray`

Read CARP points file.

**Returns:** `(N, 3)` array of coordinates

##### `read_elem(elem_path: Path, elem_type: ElemType = ElemType.Tt, read_tags: bool = False) -> np.ndarray`

Read CARP element file.

##### `read_surf(surface_path: Path) -> np.ndarray`

Read CARP .surf file (triangle connectivity).

**Returns:** `(N, 3)` array of triangle vertex indices

##### `write_surf(surface_cells: np.ndarray, output_path: Path) -> None`

Write CARP .surf file.

##### `write_vtx(vertex_indices: np.ndarray, output_path: Path) -> None`

Write CARP .vtx file (vertex list).

##### `write_pts(points: np.ndarray, output_path: Path) -> None`

Write CARP .pts file.

##### `surf2vtx(surf: np.ndarray) -> np.ndarray`

Extract unique vertex indices from surface connectivity.

```python
triangles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
vertices = surf2vtx(triangles)  # [0, 1, 2, 3, 4]
```

##### `surf2vtk(mesh_base_path: Path, surface_path: Path, output_vtk_path: Path) -> None`

Convert CARP surface to VTK PolyData with proper vertex re-indexing.

```python
from pycemrg_model_creation.utilities.mesh import surf2vtk

surf2vtk(
    mesh_base_path=Path("full_mesh"),  # Source of .pts coordinates
    surface_path=Path("surface"),       # .surf connectivity
    output_vtk_path=Path("surface.vtk")
)
```

##### `generate_vtx_from_surf(input_surf_path: Path, output_vtx_path: Path) -> None`

Generate .vtx file from .surf file (extracts unique vertices).

##### `relabel_carp_elem_file(input_elem_path: Path, output_elem_path: Path, tag_mapping: Dict[int, int]) -> None`

Relabel element tags in CARP .elem file.

```python
relabel_carp_elem_file(
    input_elem_path=Path("mesh_old.elem"),
    output_elem_path=Path("mesh_new.elem"),
    tag_mapping={1: 10, 2: 20, 3: 30}  # old -> new
)
```

##### `find_numbered_parts(directory: Path, base_prefix: str) -> List[str]`

Find numbered connected component files (e.g., `mesh.part0`, `mesh.part1`).

##### `keep_largest_n_components(component_names: List[str], directory: Path, keep_n: int, delete_smaller: bool = True) -> List[str]`

Keep N largest connected components, optionally delete others.

**Returns:** List of kept component names, sorted by size (largest first)

##### `remove_septum_from_endo(endo_surface_path: Path, septum_surface_path: Path, output_path: Path) -> None`

Remove septal triangles from endocardial surface (creates free wall).

##### `connected_component_to_surface(eidx_path: Path, input_surface_path: Path, output_surface_path: Path) -> None`

Convert connected component indices (.eidx, .nod) to .surf/.vtx files.

---

#### Geometry Utilities (`utilities/geometry.py`)

##### `compute_surface_center_of_gravity(pts: np.ndarray) -> np.ndarray`

Compute center of gravity for surface points.

**Returns:** `(3,)` array

##### `compute_mesh_region_cog(mesh_pts: np.ndarray, mesh_elem: np.ndarray, tag_value: int) -> np.ndarray`

Compute center of gravity for all elements with specific tag.

```python
lv_center = compute_mesh_region_cog(points, elements, tag_value=1)
```

##### `identify_surface_orientation(pts: np.ndarray, surf: np.ndarray, reference_point: np.ndarray) -> float`

Determine fraction of surface normals pointing outward from reference point.

**Returns:** Fraction [0.0, 1.0] of outward-pointing normals

---

#### Image Utilities (`utilities/image.py`)

##### `convert_image_to_inr(nifti_path: Path, inr_path: Path) -> None`

Convert NIfTI file to INRIMAGE-4 format (legacy meshtools3d requirement).

**Supported dtypes:** bool, uint8, uint16, int16, float32, float64

---

#### Configuration Utilities (`utilities/config.py`)

##### `Meshtools3DParameters`

Helper class to generate meshtools3d parameter files.

```python
from pycemrg_model_creation.utilities.config import Meshtools3DParameters

params = Meshtools3DParameters()

# Update parameters
params.update('meshing', 'facet_size', '0.7')
params.update('meshing', 'cell_size', '0.8')
params.update('output', 'out_vtk', '1')

# Save to file
params.save(Path("meshing.par"))
```

**Default parameters:**
```python
DEFAULT_VALUES = {
    'segmentation': {
        'seg_dir': './',
        'seg_name': 'seg.inr',
        'mesh_from_segmentation': 1,
        'boundary_relabeling': 0,
    },
    'meshing': {
        'facet_angle': 30,
        'facet_size': 0.8,
        'facet_distance': 4,
        'cell_rad_edge_ratio': 2.0,
        'cell_size': 0.8,
        'rescaleFactor': 1000
    },
    'output': {
        'outdir': './out',
        'name': 'mesh',
        'out_carp': 1,
        'out_vtk': 0,
    }
}
```

---

## Complete Workflows

### End-to-End Model Creation Pipeline

```python
from pathlib import Path
import logging
from pycemrg.core import setup_logging
from pycemrg.data import LabelManager, LabelMapper
from pycemrg.system import CommandRunner
from pycemrg_model_creation import (
    MeshingLogic,
    RefinementLogic,
    SurfaceLogic,
    MeshingPathBuilder,
    ModelCreationPathBuilder,
    MeshtoolWrapper,
    Meshtools3DWrapper,
    TagsConfig
)
from pycemrg_model_creation.types import Chamber

# --- Configuration ---
setup_logging(log_level=logging.INFO, log_file="pipeline.log")

patient_dir = Path("/data/patient_001")
segmentation = patient_dir / "segmentation.nii.gz"
source_labels_config = patient_dir / "source_labels.yaml"
target_labels_config = patient_dir / "target_labels.yaml"
output_dir = patient_dir / "model_outputs"
blank_files_dir = Path("/templates/blank_vtx")

# --- Initialize Dependencies ---
runner = CommandRunner()

meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool")
)
meshtools3d = Meshtools3DWrapper(
    runner=runner,
    meshtools3d_path=Path("/opt/meshtools3d/meshtools3d")
)

source_label_manager = LabelManager(config_path=source_labels_config)
target_label_manager = LabelManager(config_path=target_labels_config)

# --- Stage 1: Generate Volumetric Mesh ---
logging.info("STAGE 1: Volumetric Meshing")

meshing_builder = MeshingPathBuilder(output_dir=output_dir / "meshing")
meshing_paths = meshing_builder.build_meshing_paths(
    input_image=segmentation,
    raw_mesh_basename="heart_mesh"
)

meshing_logic = MeshingLogic(meshtools3d_wrapper=meshtools3d)
meshing_logic.run_meshing(
    paths=meshing_paths,
    meshing_params={
        'meshing': {
            'facet_size': '0.7',
            'cell_size': '0.7'
        }
    },
    cleanup=True
)

# --- Stage 2: Refine and Relabel Mesh ---
logging.info("STAGE 2: Mesh Refinement")

refinement_paths = meshing_builder.build_postprocessing_paths(
    input_mesh_base=meshing_paths.output_mesh_base,
    refined_mesh_basename="myocardium_clean"
)

# Get myocardium tags from source labels
myocardium_tags = source_label_manager.get_values_from_names(
    ["LV", "RV", "LA", "RA"]
)

# Create mapping from source to target labeling scheme
label_mapper = LabelMapper(
    source=source_label_manager,
    target=target_label_manager
)
tag_mapping = label_mapper.get_source_to_target_mapping()

refinement_logic = RefinementLogic(meshtool_wrapper=meshtool)
refinement_logic.run_myocardium_postprocessing(
    paths=refinement_paths,
    myocardium_tags=myocardium_tags,
    tag_mapping=tag_mapping,
    simplify=True
)

# --- Stage 3: Extract UVC Surfaces ---
logging.info("STAGE 3: UVC Surface Extraction")

surface_builder = ModelCreationPathBuilder(output_dir=output_dir / "surfaces")
surface_paths = surface_builder.build_all(
    mesh_base_path=refinement_paths.output_mesh_base,
    blank_files_dir=blank_files_dir
)

surface_logic = SurfaceLogic(meshtool, target_label_manager)

# Create tags config for surface extraction
tags = TagsConfig(
    LV=target_label_manager.get_value("LV"),
    RV=target_label_manager.get_value("RV"),
    LA=target_label_manager.get_value("LA"),
    RA=target_label_manager.get_value("RA"),
    MV=target_label_manager.get_value("MV"),
    TV=target_label_manager.get_value("TV"),
    AV=target_label_manager.get_value("AV"),
    PV=target_label_manager.get_value("PV"),
    PArt=target_label_manager.get_value("PArt")
)

# Extract all surfaces and submeshes
surface_logic.run_all(
    paths=surface_paths,
    tags=tags
)

logging.info("=" * 60)
logging.info("MODEL CREATION COMPLETE")
logging.info(f"Outputs: {output_dir}")
logging.info("=" * 60)
```

---

### Ventricular-Only Workflow

```python
from pathlib import Path
from pycemrg.data import LabelManager
from pycemrg_model_creation import (
    SurfaceLogic,
    MeshtoolWrapper,
    ModelCreationPathBuilder,
    TagsConfig
)

# Initialize
label_manager = LabelManager("labels.yaml")
meshtool = MeshtoolWrapper.from_system_path()

# Build paths
output_dir = Path("/data/patient_001/surfaces")
mesh_base = Path("/data/patient_001/mesh/four_chamber")

builder = ModelCreationPathBuilder(output_dir=output_dir)
all_paths = builder.build_all(
    mesh_base_path=mesh_base,
    blank_files_dir=Path("/templates/blank_vtx")
)

# Extract ventricular surfaces only
surface_logic = SurfaceLogic(meshtool, label_manager)
surface_logic.run_ventricular_extraction(paths=all_paths.ventricular)

# Extract BiV submesh
tags = TagsConfig(
    LV=label_manager.get_value("LV"),
    RV=label_manager.get_value("RV"),
    LA=label_manager.get_value("LA"),
    RA=label_manager.get_value("RA"),
    MV=label_manager.get_value("MV"),
    TV=label_manager.get_value("TV"),
    AV=label_manager.get_value("AV"),
    PV=label_manager.get_value("PV"),
    PArt=label_manager.get_value("PArt")
)

surface_logic.run_biv_mesh_extraction(
    paths=all_paths.biv_mesh,
    tags=tags
)

print(f"✓ Ventricular surfaces ready for UVC calculation")
```

---

### UVC Coordinate Calculation

Once the BiV submesh and VTX files are in place (output of the ventricular-only workflow above), run UVC calculation:

```python
from pathlib import Path
from pycemrg.system import CarpRunner, CommandRunner
from pycemrg_model_creation.tools.wrappers import CarpWrapper
from pycemrg_model_creation.logic.uvc import UvcLogic
from pycemrg_model_creation.logic.builders import ModelCreationPathBuilder

# Initialize CARPentry
carp_runner = CarpRunner(
    runner=CommandRunner(),
    carp_config_path=Path("/opt/carpentry/config.sh")
)
carp_wrapper = CarpWrapper(carp_runner)
uvc_logic = UvcLogic(carp_wrapper)

# The BiV submesh produced by run_biv_mesh_extraction.
# The builder derives output_dir from this path: biv_mesh.parent / "uvc"
biv_mesh = Path("/data/patient_001/surfaces/BiV/BiV")

builder = ModelCreationPathBuilder(output_dir="/data/patient_001/surfaces")
uvc_paths = builder.build_ventricular_uvc_paths(biv_mesh=biv_mesh)

# lv_tag and rv_tag must match the element tags in the BiV mesh
uvc_logic.run_ventricular_uvc_calculation(
    paths=uvc_paths,
    lv_tag=1,
    rv_tag=2,
    np=4
)

# Outputs are in biv_mesh.parent / "uvc":
print(uvc_paths.output_dir)      # /data/patient_001/surfaces/BiV/uvc
print(uvc_paths.uvc_z)           # .../uvc/BiV.uvc_z.dat
print(uvc_paths.uvc_rho)         # .../uvc/BiV.uvc_rho.dat
```

**Re-running on existing data:**

If `uvc/` already exists from a previous run, `build_ventricular_uvc_paths` backs it up automatically (default `backup_existing=True`). To delete instead of backup:

```python
uvc_paths = builder.build_ventricular_uvc_paths(
    biv_mesh=biv_mesh,
    backup_existing=False,
    overwrite_existing=True
)
```

---

## Error Handling

### Exception Hierarchy

```python
SurfaceExtractionError (RuntimeError)
└── SurfaceIdentificationError
```

### `SurfaceExtractionError`

Base exception for all surface extraction errors.

**Common causes:**
- Missing input files
- Tool execution failures
- Invalid mesh topology

### `SurfaceIdentificationError`

Raised when surfaces cannot be automatically identified.

**Common causes:**
- Connected component count mismatch
- Ambiguous geometry (normals not clearly outward/inward)
- Multiple components with similar characteristics

**Example:**
```python
from pycemrg_model_creation.logic.surfaces import (
    SurfaceExtractionError,
    SurfaceIdentificationError
)

try:
    surface_logic.run_ventricular_extraction(paths=paths)
except SurfaceIdentificationError as e:
    logging.error(f"Could not automatically identify surfaces: {e}")
    # Manual intervention required - check intermediate files in tmp_dir
    # Look at connected components: {tmp_dir}/epi_endo_CC.part*.vtk
except SurfaceExtractionError as e:
    logging.error(f"Surface extraction failed: {e}")
    # Check logs for tool execution errors
```

---

## Best Practices

### 1. Always Use Path Builders

Path builders encapsulate naming conventions and directory structures. They reduce errors and improve maintainability.

```python
# ✓ GOOD: Use builder
builder = ModelCreationPathBuilder(output_dir)
paths = builder.build_all(mesh_base, blank_files_dir)

# ✗ AVOID: Manual construction
paths = VentricularSurfacePaths(
    mesh=mesh,
    output_dir=output_dir,
    tmp_dir=tmp_dir,
    # ... 20+ more fields to fill manually, error-prone
)
```

### 2. Validate Inputs Before Long Workflows

Check file existence early to fail fast.

```python
# Pre-flight checks
assert segmentation.exists(), f"Segmentation not found: {segmentation}"
assert mesh_base.with_suffix(".pts").exists(), f"Mesh not found: {mesh_base}"
assert labels_config.exists(), f"Labels config not found: {labels_config}"

# Only then start workflow
meshing_logic.run_meshing(paths=meshing_paths)
```

### 3. Enable Detailed Logging

```python
from pycemrg.core import setup_logging
import logging

# File logging for debugging
setup_logging(
    log_level=logging.DEBUG,
    log_file="pipeline.log"
)

# Console logging for progress
setup_logging(log_level=logging.INFO)
```

### 4. Organize Patient Data Consistently

Recommended structure:

```
/data/
└── patient_001/
    ├── input/
    │   ├── segmentation.nii.gz
    │   ├── source_labels.yaml
    │   └── target_labels.yaml
    ├── meshing/
    │   ├── 01_raw/
    │   │   ├── heart_mesh.pts
    │   │   ├── heart_mesh.elem
    │   │   └── heart_mesh.vtk
    │   ├── 02_refined/
    │   │   ├── myocardium_clean.pts
    │   │   ├── myocardium_clean.elem
    │   │   └── myocardium_clean.vtk
    │   └── tmp/
    └── surfaces/
        ├── BiV/
        │   ├── biv_epi.surf
        │   ├── biv_lvendo.surf
        │   ├── biv.base.vtx
        │   └── ...
        ├── LA/
        └── RA/
```

### 5. Keep Intermediate Files During Development

Set `cleanup=False` during development to inspect intermediate files:

```python
meshing_logic.run_meshing(
    paths=meshing_paths,
    cleanup=False  # Keep .inr and .par for debugging
)
```

Check `tmp_dir` for intermediate surfaces during surface extraction.

### 6. Use Label Managers for Flexibility

Never hardcode anatomical label values. Use `LabelManager` to support different labeling schemes.

```python
# ✓ GOOD: Flexible
label_manager = LabelManager("labels.yaml")
lv_tag = label_manager.get_value("LV")

# ✗ AVOID: Hardcoded
lv_tag = 1  # What if labels.yaml uses different values?
```

### 7. Leverage Label Mapping for Standardization

When working with multiple datasets with different labeling schemes:

```python
from pycemrg.data import LabelMapper

source_labels = LabelManager("dataset_labels.yaml")
target_labels = LabelManager("standard_labels.yaml")

mapper = LabelMapper(source=source_labels, target=target_labels)
tag_mapping = mapper.get_source_to_target_mapping()

# Use mapping in refinement
refinement_logic.run_myocardium_postprocessing(
    paths=paths,
    myocardium_tags=source_labels.get_values_from_names(["LV", "RV"]),
    tag_mapping=tag_mapping
)
```

### 8. Check Tool Availability

For optional tools like `simplify_tag_topology`:

```python
meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool")
)

if meshtool.is_simplify_topology_available:
    refinement_logic.run_myocardium_postprocessing(
        paths=paths,
        myocardium_tags=tags,
        tag_mapping=mapping,
        simplify=True
    )
else:
    logging.warning("Topology simplification not available, skipping")
    refinement_logic.run_myocardium_postprocessing(
        paths=paths,
        myocardium_tags=tags,
        tag_mapping=mapping,
        simplify=False
    )
```

---

## Utility Types & Enums

### `Chamber`

Enum for cardiac chamber identifiers.

```python
from pycemrg_model_creation.types import Chamber

Chamber.LV  # "LV"
Chamber.RV  # "RV"
Chamber.LA  # "LA"
Chamber.RA  # "RA"
```

### `SurfaceType`

Enum for surface type identifiers.

```python
from pycemrg_model_creation.types import SurfaceType

SurfaceType.EPI     # "epi"
SurfaceType.ENDO    # "endo"
SurfaceType.BASE    # "base"
SurfaceType.SEPTUM  # "septum"
```

### `ElemType`

Enum for CARP element types (in `utilities/mesh.py`).

```python
from pycemrg_model_creation.utilities.mesh import ElemType

ElemType.Tt  # Tetrahedra (connectivity columns 1,2,3,4)
ElemType.Tr  # Triangles (connectivity columns 1,2,3)
ElemType.Ln  # Lines (connectivity columns 1,2)
```

---

## Version History

### v1.0.0 (Current)
- Initial public API
- Core workflows:
  - Volumetric meshing (meshtools3d)
  - Mesh refinement and relabeling
  - Ventricular surface extraction for UVC
  - Atrial surface extraction
  - BiV and atrial submesh generation
- Path contract system with builders
- Support for BiV and atrial geometries
- Integration with pycemrg core (LabelManager, LabelMapper, CommandRunner, CarpRunner)

---

## Dependencies

### Required
- `pycemrg` (core library)
- `numpy`
- `pyvista` (for VTK operations)
- `SimpleITK` (for image I/O)

### Optional
- `meshtools3d` binary (for volumetric meshing)
- `meshtool` binary (for surface extraction and mesh operations)
- CARPentry/openCARP installation (for fiber generation and simulations)

---

## Further Reading

- **pycemrg Core Documentation**: For `LabelManager`, `LabelMapper`, `CarpRunner`, and system utilities
- **CARPentry Documentation**: For underlying tool specifications
- **openCARP**: For cardiac simulation capabilities
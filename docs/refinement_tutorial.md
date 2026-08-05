# Tutorial — refining a heart mesh from an external orchestrator

This guide covers **one stage**: turning a raw volumetric heart mesh into a clean, relabelled
myocardium mesh. It assumes you have never read this library's source.

The package also contains surface extraction and UVC calculation. Those are under active
development and are deliberately not documented here yet.

**The division of labour:** this library is stateless logic. It does not look for files, read
environment variables, or decide where anything lives — your orchestrator does all of that and
hands the library explicit paths. Nothing here reads config from ambient state.

---

## 1. What you need before you start

### Python

Python ≥ 3.10. Not yet on PyPI, so install from a checkout:

```bash
pip install /path/to/pycemrg-model-creation
```

That pulls `pycemrg`, `numpy`, `pyvista` and `pyyaml`. The import name uses underscores:
`import pycemrg_model_creation`.

### The `meshtool` binary

`meshtool` must be on your `PATH`. Nothing else is required for a basic run.

If you also want **topology simplification**, pass your meshtool installation directory so the
library can find `standalones/simplify_tag_topology` inside it. If you omit the directory, or the
standalone is not there, the library logs a warning and continues *without* simplifying — the run
still succeeds and still produces output. See the gotcha in §4.

### An input mesh

CARP text format, addressed by its **stem** — the path with no extension:

```
heart_mesh.pts     node coordinates    (required, micrometres by default)
heart_mesh.elem    connectivity + tags (required)
heart_mesh.lon     fibre orientations  (optional)
```

A missing `.lon` is normal and is **not an error**: meshtool fabricates default `[1 0 0]` fibres and
logs a message to stderr saying so. Real fibres are generated later in the pipeline.

This library does not build the mesh from a segmentation. That belongs to the sibling package
`pycemrg-meshing`.

### Two label YAML files

Refinement renumbers element tags, so it needs to know what the numbers mean now (`source`) and what
you want them to be (`target`). Both files use the same format:

```yaml
labels:
  LV_myo: 103
  RV_myo: 104
  LA_myo: 105

groups:
  MYOCARDIUM: [LV_myo, RV_myo, LA_myo]
```

`labels` maps a name to an integer tag. `groups` names a set of labels, and groups may reference
other groups. Both files are yours to write; no examples ship with the library.

Two things worth knowing, because neither is obvious and neither raises an error:

- The mapping is built **by matching label names** between the two files. A name present in source
  but absent from target is silently ignored — that tag is left alone.
- Only labels whose value actually **differs** between the files appear in the mapping.

---

## 2. The orchestration

Five steps. This is the whole integration.

```python
import logging
from pathlib import Path

from pycemrg.core import setup_logging
from pycemrg.data import LabelManager, LabelMapper
from pycemrg_model_creation.logic import RefinementLogic, RefinementPathBuilder
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools import MeshtoolWrapper

setup_logging(log_level=logging.INFO, log_file=Path("refinement.log"))

# 1. Name the paths. This touches no disk -- you can build a contract purely to
#    inspect where things would go.
builder = RefinementPathBuilder(output_dir=Path("/work/case01"))
paths = builder.build_postprocessing_paths(
    input_mesh_base=CarpMesh("/data/case01/heart_mesh"),  # a stem, not a file
)

# 2. Wire up meshtool. Omit meshtool_install_dir if you do not need simplification.
meshtool = MeshtoolWrapper.from_system_path(
    meshtool_install_dir=Path("/opt/meshtool"),
)

# 3. Resolve the tags from your two label files.
source = LabelManager("/data/case01/source_labels.yaml")
target = LabelManager("/data/case01/target_labels.yaml")
myocardium_tags = source.get_values_from_names(["MYOCARDIUM"])   # a group name
tag_mapping = LabelMapper(source=source, target=target).get_source_to_target_mapping()

# 4. Run. This is where directories get created and meshtool is invoked.
RefinementLogic(meshtool_wrapper=meshtool).run_myocardium_postprocessing(
    paths=paths,
    myocardium_tags=myocardium_tags,
    tag_mapping=tag_mapping,
    simplify=True,
)

# 5. Consume the result. Ask the handle for the family member you want.
print(paths.output_mesh_base.pts)   # /work/case01/refined/myocardium_clean.pts
print(paths.output_mesh_base.elem)
print(paths.output_mesh_base.vtk)
```

### What lands on disk

```
/work/case01/
├── refined/
│   ├── myocardium_clean.elem     relabelled tags
│   ├── myocardium_clean.pts
│   ├── myocardium_clean.lon
│   └── myocardium_clean.vtk      for visualisation
└── tmp/
    └── myocardium_intermediate.*  extraction output, kept for inspection
```

Rename either directory when you build the contract:

```python
RefinementPathBuilder(output_dir=..., refined_subdir="02_refined")
```

and the final mesh's basename with `build_postprocessing_paths(..., refined_mesh_basename="myo")`.
`tmp/` is currently fixed.

---

## 3. `CarpMesh`: pass stems, not files

A CARP mesh is not one file, it is a stem plus a family that shares it. `CarpMesh` is the handle for
that family, and it is what the contract holds.

```python
mesh = CarpMesh("/data/case01/heart_mesh")
mesh.pts            # PosixPath('/data/case01/heart_mesh.pts')
mesh.elem, mesh.lon, mesh.vtk
mesh.directory      # PosixPath('/data/case01')
mesh.name           # 'heart_mesh'
str(mesh)           # '/data/case01/heart_mesh'  -- also os.PathLike
```

Two rules:

- **Construct it from a stem.** `CarpMesh("heart_mesh.pts")` raises `ValueError` telling you so.
  `build_postprocessing_paths` accepts a `str`, a `Path` or a `CarpMesh` and normalises for you.
- **Never call `pathlib.with_suffix` on a mesh path.** It *replaces* the last dot-segment rather
  than appending, so `Path("biv.base").with_suffix(".vtx")` silently gives you `biv.vtx`. Ask the
  handle instead. This is why the contract holds handles at all.

The contract is frozen, and `output_dir` / `tmp_dir` are **properties** derived from the handles,
not stored fields — they cannot drift out of step with the meshes they belong to.

---

## 4. Things that will bite you

- **Simplification fails quietly by design.** `simplify=True` with no
  `meshtool_install_dir`, or a missing standalone, logs a warning and carries on unsimplified. The
  run reports success. If simplification matters to you, assert
  `meshtool.is_simplify_topology_available` before calling.
- **The two branches produce the same files by different routes.** With simplification, meshtool
  writes the output mesh and the tags are relabelled **in place**. Without it, the tags are
  relabelled out of the intermediate and the `.pts`/`.lon` are copied across. Same result, different
  provenance — worth knowing when a log looks unfamiliar.
- **An existing output `.lon` is moved aside, not overwritten.** You get a timestamped
  `myocardium_clean.lon.20260805-143012.bak` and a warning showing its first few lines. Five rows of
  `1 0 0` means it was meshtool's fabricated default and the backup is disposable; anything else
  means real fibre data was in the way. Only happens on the no-simplification path.
- **Directories are created by the run, not by the builder.** Build a contract as often as you like
  to inspect paths; nothing appears on disk until `run_myocardium_postprocessing` is called.
- **Output is not cleared between runs.** Re-running over a populated directory overwrites file by
  file. If you need a clean slate, remove it yourself first.
- **Logging is where the detail lives.** The library raises on genuine failures but reports
  everything else — skipped simplification, fabricated fibres, backed-up files — through the
  standard `logging` module. Configure a handler or you will not see any of it.

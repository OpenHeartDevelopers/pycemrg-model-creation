# src/pycemrg_model_creation/logic/contracts.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..meshpaths import CarpMesh


@dataclass(frozen=True)
class MeshPostprocessingPaths:
    """
    Path contract for the mesh post-processing (refinement) workflow.

    Every field that is really a *mesh* is a ``CarpMesh`` handle rather than a
    bare stem, so the extensions are asked of the handle (``mesh.pts``,
    ``mesh.elem``) instead of being reattached by each caller with
    ``with_suffix`` — which truncates dotted stems and is banned here.

    Note the wrapper boundary: ``MeshtoolWrapper`` takes plain ``Path`` stems
    and builds its own expected-output names, so the logic layer passes
    ``mesh.stem`` when it calls a wrapper and uses the handle everywhere else.
    """

    # Input
    # Raw volumetric mesh (e.g. .../heart_mesh), produced upstream of this
    # library. Its .lon is frequently absent; meshtool fabricates default
    # [1 0 0] fibres when it is, which is expected, not an error.
    input_mesh_base: CarpMesh

    # Intermediate mesh (in tmp_dir): the myocardium after tag extraction and
    # before optional simplification. Load-bearing rather than scratch — it is
    # the simplify input when simplification runs, and both the relabel input
    # and the copy source when it is skipped.
    intermediate_myocardium_mesh: CarpMesh

    # Final output mesh (in output_dir): the cleaned, relabelled mesh.
    output_mesh_base: CarpMesh

    @property
    def output_dir(self) -> Path:
        """Directory holding the refined mesh. The output mesh's own directory."""
        return self.output_mesh_base.directory

    @property
    def tmp_dir(self) -> Path:
        """Temporary directory for intermediate files. The intermediate mesh's own directory."""
        return self.intermediate_myocardium_mesh.directory


@dataclass(frozen=True)
class VentricularSurfacePaths:
    """
    Path contract for ventricular (BiV) surface extraction.

    **Derived names are asked for, not stored.** Every name below that is a
    fixed convention relative to ``output_dir`` is a property, so the contract
    and the builder cannot drift apart: there is one spelling of ``biv_epi``,
    and it lives here. Only genuine inputs, and names not yet reducible, are
    fields.

    Frozen, like the other refactored contracts: a derived name that could be
    reassigned would defeat the point. Constructing this touches no disk —
    directory *creation* belongs to ``SurfaceLogic``, not to naming.

    Note the wrapper boundary: ``MeshtoolWrapper`` and the ``utilities.mesh``
    readers take plain ``Path`` stems and reattach extensions themselves, so
    the logic layer passes ``mesh.stem`` when it calls one and uses the handle
    everywhere else. ``surf2vtk`` and ``read_carp_mesh`` would raise on a
    handle; ``extract_surface`` would happen to work, and is given the stem
    anyway so the boundary is one rule rather than three special cases.

    The boundary ``*_vtx`` names were fields until we established what each
    file is for. They are now properties over ``submesh``, so mguvc's stem and
    its role vocabulary have one definition between them.
    """

    # -- Inputs ---------------------------------------------------------

    # The four-chamber mesh this chamber is cut from. A handle, so the
    # extensions are asked of it rather than reattached by each caller.
    # The wrappers still take plain stems, so the logic hands them
    # `paths.mesh.stem` — see the class docstring on that boundary.
    mesh: CarpMesh

    # Output root for this chamber, e.g. <root>/BiV. Not derivable from
    # `mesh`: the input mesh lives in a different tree from the outputs.
    output_dir: Path

    # The BiV submesh these boundaries are named against. `mguvc` builds the
    # filenames from this stem, so the contract asks the handle rather than
    # storing five spellings that can drift from it.
    submesh: CarpMesh

    # -- Boundary VTX files (beside the submesh) -------------------------
    #
    # The role strings are mguvc's own vocabulary, and there is no translation
    # layer — see `.claude/evidence_biv_prefix.md`. `rvsept` is spelled that
    # way here because that is what mguvc opens; do not reintroduce `septum`.
    #
    # mguvc reads four of these: `base`, `lvendo`, `rvendo` and `rvsept`. It
    # derives `epi` itself and never opens `epi_vtx`. That file exists for the
    # later fibre step, which is not ported yet, so it is not unused.
    #
    # `apex_vtx` is absent on purpose: for BiV, mguvc *writes* `BiV.lvapex.vtx`.
    # `rv_septum_point_vtx` is absent too. mguvc asks for `rvsept_pt` only under
    # `--output-model lv`. The atria borrow that mode, so the field lives on
    # `AtrialSurfacePaths`. It belongs to the mode, not to a chamber.

    @property
    def base_vtx(self) -> Path:
        """Valve-plane boundary. An mguvc input."""
        return self.submesh.role("base", ".vtx")

    @property
    def epi_vtx(self) -> Path:
        """Epicardium. **Not** an mguvc input; read by the fibre step."""
        return self.submesh.role("epi", ".vtx")

    @property
    def lv_endo_vtx(self) -> Path:
        """LV endocardium. An mguvc input."""
        return self.submesh.role("lvendo", ".vtx")

    @property
    def rv_endo_vtx(self) -> Path:
        """RV endocardium. An mguvc input."""
        return self.submesh.role("rvendo", ".vtx")

    @property
    def rvsept_vtx(self) -> Path:
        """Interventricular septum. An mguvc input."""
        return self.submesh.role("rvsept", ".vtx")

    # -- Derived directories ---------------------------------------------

    @property
    def tmp_dir(self) -> Path:
        """Scratch directory for intermediate surfaces."""
        return self.output_dir / "tmp"

    # -- Intermediate surfaces (in tmp_dir) ------------------------------

    @property
    def base_surface(self) -> Path:
        """Ventricle/valve-plane interface, before VTX conversion."""
        return self.tmp_dir / "base"

    @property
    def epi_endo_combined(self) -> Path:
        """Combined epi/endo surface, before separation into components."""
        return self.tmp_dir / "epi_endo"

    @property
    def septum_raw(self) -> Path:
        """LV surface facing the RV, before component separation."""
        return self.tmp_dir / "septum"

    @property
    def lv_epi_intermediate(self) -> Path:
        """The non-septum component of the septum extraction."""
        return self.tmp_dir / "lv_epi_intermediate"

    # -- Connected-component prefixes (in tmp_dir) -----------------------
    #
    # These are *predicted* names, not discovered ones: each is the stem
    # handed to `meshtool extract unreachable` via `-submsh=`, and then to
    # `find_numbered_parts` as the prefix to search for. The tool appends
    # `.partN` at an arity only known at run time, and that arity comes back
    # as a return value — it is never named here.

    @property
    def epi_endo_cc_base(self) -> Path:
        """Prefix for the epi/endo connected components."""
        return self.tmp_dir / "epi_endo_cc"

    @property
    def septum_cc_base(self) -> Path:
        """Prefix for the septum connected components."""
        return self.tmp_dir / "septum_cc"

    # -- Final surfaces (in output_dir) ----------------------------------

    @property
    def epi_surface(self) -> Path:
        """Epicardium."""
        return self.output_dir / "biv_epi"

    @property
    def lv_endo_surface(self) -> Path:
        """LV endocardium."""
        return self.output_dir / "biv_lvendo"

    @property
    def rv_endo_surface(self) -> Path:
        """RV endocardium."""
        return self.output_dir / "biv_rvendo"

    @property
    def septum_surface(self) -> Path:
        """Interventricular septum."""
        return self.output_dir / "biv_septum"


@dataclass
class AtrialSurfacePaths:
    """
    Path contract for atrial surface extraction (LA or RA).
    """

    # Input
    mesh: Path  # Full four-chamber mesh base name

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

    # Blank files for compatibility
    apex_vtx: Path
    rv_septum_point_vtx: Path


@dataclass
class BiVMeshPaths:
    """
    Path contract for BiV mesh extraction and mapping.
    """

    source_mesh: Path  # Full four-chamber mesh

    # The BiV submesh. A handle, so the boundary VTX names can be asked of it
    # rather than restated: `mguvc` builds them from this stem, as
    # `<stem>.base.vtx` and so on. The stem is `BiV`, which is mguvc's own
    # vocabulary — see `.claude/evidence_biv_prefix.md`.
    #
    # The wrappers still reattach extensions themselves, so callers hand them
    # `paths.output_mesh.stem`. `MeshtoolWrapper.extract_mesh` would raise on a
    # handle, because it calls `with_suffix`.
    output_mesh: CarpMesh

    output_dir: Path

    # VTX files to map from four-chamber to BiV
    vtx_files_to_map: List[Path]
    mapped_vtx_output_dir: Path


@dataclass
class AtrialMeshPaths:
    """
    Path contract for LA/RA mesh extraction and mapping.
    """

    source_mesh: Path
    output_mesh: Path
    output_dir: Path

    # VTX files to map
    vtx_files_to_map: List[Path]
    mapped_vtx_output_dir: Path

    # Template files for apex/septum (blank files)
    apex_template: Path
    rv_septum_template: Path
    apex_output: Path
    rv_septum_output: Path


@dataclass
class UVCSurfaceExtractionPaths:
    """
    Master path contract for complete UVC surface extraction workflow.
    """

    ventricular: VentricularSurfacePaths
    left_atrial: AtrialSurfacePaths
    right_atrial: AtrialSurfacePaths
    biv_mesh: BiVMeshPaths
    la_mesh: AtrialMeshPaths
    ra_mesh: AtrialMeshPaths


@dataclass(frozen=True)
class VentricularUVCPaths:
    """
    Path contract for ventricular UVC (Universal Ventricular Coordinate)
    calculation.

    Three fields, because three is how many choices mguvc actually leaves the
    caller: which mesh, where the outputs go, and where the etags script is
    written. Everything else is a function of those and is asked for rather
    than stored:

    - boundary inputs — ``tools.mguvc.boundary_inputs(biv_mesh)``
    - outputs — ``tools.mguvc.outputs_for(biv_mesh, output_dir)``

    Storing derived names here is what let this contract and the validator
    disagree about them in the past. Ask; do not restate.

    mguvc reads four boundary VTX files from the mesh's own directory,
    stem-prefixed (``BiV.base.vtx``). It writes everything under
    ``output_dir`` and nothing beside the mesh.
    """

    # Input: BiV submesh extracted from the four-chamber mesh. Its directory
    # must also hold the four boundary VTX files.
    biv_mesh: CarpMesh

    # Output: directory for the UVC coordinate files. Must not exist when
    # mguvc runs — mguvc prompts interactively if it does.
    output_dir: Path

    # Input: element-tags script (bash, T_LV/T_RV definitions) written by this
    # library and passed to mguvc as --tags-file. Explicit because the
    # location is the caller's to choose, not mguvc's to mandate; it is put
    # beside the mesh so that output_dir is not created early.
    etags_file: Path


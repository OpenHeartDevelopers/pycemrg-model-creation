# src/pycemrg_model_creation/tools/mguvc.py

"""
What mguvc reads and what it writes.

mguvc's filenames are a pure function of the mesh stem and the output
directory, so they can be computed before the tool runs. That knowledge lives
here, beside the wrapper that invokes mguvc, rather than in a path builder or
a workflow contract: when mguvc changes its output names, one file changes.

Nothing here touches the file system. `outputs_for` predicts names; whether
those files exist is the caller's question.

Everything in this module was verified against a native mguvc run recorded in
`scripts/integration/run_mguvc_native.sh` (CARPentry_ICL_latest, exit 0).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Union

from ..meshpaths import CarpMesh


class UvcBoundary(str, Enum):
    """
    The boundary-condition VTX files mguvc reads.

    These four, and no others. The native run reports ``checking data
    '<role>'`` for exactly these and copies them into ``<output_dir>/src/``.

    ``epi``, ``lvepi``, ``rvendo_nosept``, ``rvjunc`` and ``tissue`` are
    **not** inputs — mguvc derives them itself (``extracting surface
    '<name>' ...``). Files bearing those names may sit beside the mesh, left
    over from surface extraction; mguvc never opens them. Do not add them
    here, and do not require them when validating.
    """

    BASE = "base"
    LV_ENDO = "lvendo"
    RV_ENDO = "rvendo"
    RV_SEPT = "rvsept"


def boundary_inputs(mesh: CarpMesh) -> Dict[UvcBoundary, Path]:
    """
    The VTX files mguvc expects beside `mesh`, keyed by role.

    Names are stem-prefixed — ``BiV.base.vtx``, not ``base.vtx`` — and must
    sit in the mesh's own directory.

    Args:
        mesh: The BiV submesh mguvc will run on.

    Returns:
        One entry per `UvcBoundary`. Use it to validate inputs; the roles come
        from one place so a validator cannot drift from a builder.

    Example:
        >>> paths = boundary_inputs(CarpMesh("/data/BiV/BiV"))
        >>> paths[UvcBoundary.RV_SEPT].name
        'BiV.rvsept.vtx'
    """
    return {boundary: mesh.role(boundary.value, ".vtx") for boundary in UvcBoundary}


@dataclass(frozen=True)
class MguvcOutputs:
    """
    The mguvc outputs this library consumes.

    A real run emits roughly 75 files. This models the ten we act on — the
    four UVC coordinate fields, the four Laplace solutions and the two mapping
    files — because those are the ones the library reads back or asserts on.

    Deliberately not modelled, all seen in the native run:

    - the surfaces mguvc derives for its own use (``epi``, ``lvepi``,
      ``rvendo_nosept``, ``rvjunc``, ``tissue``), each as ``.surf`` and
      ``.surf.vtx``, plus fourteen ``*_face.surf`` pairs;
    - ``<stem>.axis``, ``<stem>.phi.tags``, ``<stem>.uvc``,
      ``<stem>.lvapex.vtx``, ``<stem>.rvapex.vtx``;
    - a ``<stem>.lon`` — mguvc generates default fibres when the input mesh
      has none, which is normal here; real fibres arrive later from
      `GlRuleFibres`;
    - copies of the mesh and of the four boundary VTX files, and the whole
      ``src/`` subdirectory.

    Add a member when the library starts using one, not before.

    All outputs land in `output_dir`; mguvc writes nothing beside the mesh.
    Their stem is the mesh's basename, *not* the mesh's directory.
    """

    mesh: CarpMesh
    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    # -- The four UVC coordinate fields ---------------------------------

    @property
    def uvc_z(self) -> Path:
        """Apico-basal coordinate: apex 0, base 1."""
        return self._in_output.role("uvc_z", ".dat")

    @property
    def uvc_rho(self) -> Path:
        """Transmural coordinate: endo 0, epi 1."""
        return self._in_output.role("uvc_rho", ".dat")

    @property
    def uvc_phi(self) -> Path:
        """Rotational coordinate."""
        return self._in_output.role("uvc_phi", ".dat")

    @property
    def uvc_ven(self) -> Path:
        """Ventricular identifier, LV against RV."""
        return self._in_output.role("uvc_ven", ".dat")

    # -- Intermediate Laplace solutions ---------------------------------
    # Written only when mguvc is invoked with --laplace-solution.

    @property
    def sol_apba(self) -> Path:
        """Apico-basal Laplace solution."""
        return self._in_output.role("sol_apba_lap", ".dat")

    @property
    def sol_endoepi(self) -> Path:
        """Endo-epi Laplace solution."""
        return self._in_output.role("sol_endoepi_lap", ".dat")

    @property
    def sol_lvendo(self) -> Path:
        """LV endo Laplace solution."""
        return self._in_output.role("sol_lvendo_lap", ".dat")

    @property
    def sol_rvendo(self) -> Path:
        """RV endo Laplace solution."""
        return self._in_output.role("sol_rvendo_lap", ".dat")

    # -- Mapping files --------------------------------------------------

    @property
    def aff_dat(self) -> Path:
        """Affine transformation data."""
        return self._in_output.role("aff", ".dat")

    @property
    def m2s_dat(self) -> Path:
        """Mesh-to-surface mapping."""
        return self._in_output.role("m2s", ".dat")

    # -- Groupings ------------------------------------------------------

    @property
    def coordinates(self) -> Tuple[Path, ...]:
        """The four UVC fields. These are what a run must produce to count."""
        return (self.uvc_z, self.uvc_rho, self.uvc_phi, self.uvc_ven)

    @property
    def laplace_solutions(self) -> Tuple[Path, ...]:
        """Present only when mguvc ran with --laplace-solution."""
        return (self.sol_apba, self.sol_endoepi, self.sol_lvendo, self.sol_rvendo)

    @property
    def mappings(self) -> Tuple[Path, ...]:
        """Affine and mesh-to-surface mapping data."""
        return (self.aff_dat, self.m2s_dat)

    # -- Internals ------------------------------------------------------

    @property
    def _in_output(self) -> CarpMesh:
        """
        The output stem: the mesh's basename, relocated into `output_dir`.

        mguvc names its outputs after the mesh but writes them elsewhere, so
        this is a different stem from `self.mesh` and the two must not be
        confused.
        """
        return CarpMesh(self.output_dir / self.mesh.name)


def outputs_for(
    mesh: CarpMesh, output_dir: Union[Path, str]
) -> MguvcOutputs:
    """
    Predict what mguvc will write, without running it.

    Pure: performs no I/O and does not require the run to have happened, so it
    can be used to validate before invoking and to assert afterwards.

    Args:
        mesh: The BiV submesh passed to mguvc as ``--model-name``.
        output_dir: The directory passed as ``--output-dir``.

    Returns:
        An `MguvcOutputs` describing the files this library consumes.

    Example:
        >>> outputs = outputs_for(CarpMesh("/data/BiV/BiV"), "/data/BiV/uvc")
        >>> outputs.uvc_z
        PosixPath('/data/BiV/uvc/BiV.uvc_z.dat')
    """
    return MguvcOutputs(mesh=mesh, output_dir=Path(output_dir))

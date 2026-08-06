# src/pycemrg_model_creation/meshpaths.py

"""
Path handles for the file families CARP and meshtool work with.

A CARP "mesh" is not one file: it is a stem plus a family of siblings that
share it (``BiV.pts``, ``BiV.elem``, ``BiV.lon``). Every layer of this
library passes those stems around, and every layer has to reattach the
extensions itself. The handles here own that reattachment so it is written
once.

Extensions are appended, never substituted. ``pathlib.Path.with_suffix``
*replaces* the final dot-segment, so on the dotted names CARP and meshtool
routinely emit it silently truncates::

    Path("biv.base").with_suffix(".vtx")   -> biv.vtx      # wrong
    CarpMesh("biv.base").vtx               -> biv.base.vtx # intended

These handles hold no state beyond the stem and never touch the file
system. Existence checks belong to the caller.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class _StemHandle:
    """
    Shared mechanics for every handle here: a stem, and siblings built by
    *appending* to it.

    Private on purpose. It exists to state the stem arithmetic once, not to
    make the handles interchangeable — nothing dispatches on this type, and
    an ``isinstance`` check against it would be asserting something the
    design does not promise.

    The rule for what may live here: **anything derivable from the stem
    alone.** ``name``, ``directory`` and ``_sibling`` qualify — every handle
    can answer them honestly. A *family member* (``.pts``, ``.surf``,
    ``.nod``) never does: naming one asserts that a tool writes it, and that
    claim differs per handle. It is why ``SurfaceMesh`` is not a
    ``CarpMesh``, and this base must not undo it.

    Subclasses declare which concrete-file extensions they refuse via
    ``_refuses``. It is a ``ClassVar``, so the dataclass machinery leaves it
    as a class attribute rather than turning it into a second constructor
    argument. The default ``()`` refuses nothing, since ``x in ()`` is always
    false — that is a real setting, not a placeholder.

    Every subclass must itself be ``@dataclass(frozen=True)``: mixing frozen
    and non-frozen in one hierarchy raises ``TypeError`` at class creation.
    """

    stem: Path

    # Extensions this handle refuses as a stem. A stem ending in one of them is
    # almost always a caller that passed a concrete file where a stem was wanted.
    _refuses: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # Accept str for convenience, but store a Path. Assigning through
        # object.__setattr__ is how a frozen dataclass normalises its fields.
        object.__setattr__(self, "stem", Path(self.stem))

        if self.stem.suffix in self._refuses:
            name = type(self).__name__
            raise ValueError(
                f"{name} takes a stem without an extension, got '{self.stem}'. "
                f"Use {name}('{self.stem.with_suffix('')}') instead."
            )

    # -- Identity -------------------------------------------------------

    @property
    def name(self) -> str:
        """The stem's basename, e.g. 'BiV'. Used by tools that want a bare name."""
        return self.stem.name

    @property
    def directory(self) -> Path:
        """The directory holding the family."""
        return self.stem.parent

    # -- Internals ------------------------------------------------------

    def _sibling(self, extension: str) -> Path:
        """
        Append an extension to the stem.

        Deliberately not `with_suffix`: stems here routinely carry dots
        (`epi_endo.surfmesh`, `septum_cc.part0`) and those dot-segments are
        meaningful, not extensions to be replaced.
        """
        return self.stem.parent / f"{self.stem.name}{extension}"

    # -- Interop --------------------------------------------------------

    def __str__(self) -> str:
        return str(self.stem)

    def __fspath__(self) -> str:
        # Makes the handle os.PathLike, so it can be handed to a wrapper
        # wherever a stem is expected without an explicit conversion.
        return str(self.stem)


@dataclass(frozen=True)
class SubmeshIndex(_StemHandle):
    """
    The ``.nod``/``.eidx`` pair written when a submesh is carved from a parent.

    ``meshtool extract unreachable`` and the submesh extractions emit, beside
    the mesh family itself, two binary index files: ``.nod`` maps the submesh's
    node ids back to the parent's, and ``.eidx`` does the same for its
    elements. They are what lets a surface extracted from one connected
    component be expressed again in the coordinates of the mesh it came from.

    Deliberately its own handle rather than two more properties on
    ``CarpMesh``. A mesh that was never carved out of anything — the trunk
    ``BiV`` — has no index pair, so advertising one on every ``CarpMesh``
    would be the same contract-that-lies that keeps ``SurfaceMesh`` separate.
    Reach it through :attr:`CarpMesh.index` when the mesh is a submesh.

    A ``.surfmesh`` carries a ``.nod`` but no ``.eidx``, so ``SurfaceMesh``
    delegates its ``.nod`` here and does not expose ``.eidx`` at all.

    Example:
        >>> SubmeshIndex("/tmp/epi_endo_cc.part0").eidx
        PosixPath('/tmp/epi_endo_cc.part0.eidx')
    """

    _refuses: ClassVar[tuple[str, ...]] = (".nod", ".eidx")

    @property
    def nod(self) -> Path:
        """Node index map back to the parent mesh, binary."""
        return self._sibling(".nod")

    @property
    def eidx(self) -> Path:
        """Element index map back to the parent mesh, binary."""
        return self._sibling(".eidx")


@dataclass(frozen=True)
class CarpMesh(_StemHandle):
    """
    A CARP volumetric mesh, addressed by its stem.

    The stem is the path *without* an extension, which is also the form
    meshtool and the CARPentry binaries take on the command line
    (``meshtool extract mesh -msh=/data/BiV ...``). Interpolating or
    ``os.fspath``-ing this handle yields that stem, so it can be passed
    straight to a wrapper.

    ``.vtk`` is included alongside the CARP triple because converting a mesh
    to VTK for visualisation conventionally keeps the same stem.

    Example:
        >>> mesh = CarpMesh("/data/surfaces_uvc/BiV/BiV")
        >>> mesh.pts
        PosixPath('/data/surfaces_uvc/BiV/BiV.pts')
        >>> str(mesh)
        '/data/surfaces_uvc/BiV/BiV'
    """

    # A stem ending in one of these is almost always a caller that passed a
    # concrete file where a stem was wanted.
    _refuses: ClassVar[tuple[str, ...]] = (
        ".pts",
        ".elem",
        ".lon",
        ".vtk",
        ".bpts",
        ".belem",
    )

    # -- The family -----------------------------------------------------

    @property
    def pts(self) -> Path:
        """Node coordinates."""
        return self._sibling(".pts")

    @property
    def elem(self) -> Path:
        """Element connectivity and region tags."""
        return self._sibling(".elem")

    @property
    def lon(self) -> Path:
        """Fibre orientations."""
        return self._sibling(".lon")

    @property
    def vtk(self) -> Path:
        """VTK conversion of the mesh, for visualisation."""
        return self._sibling(".vtk")

    # meshtool's carp_bin counterparts to .pts/.elem. Needed by the later
    # simulation stages; there is no binary counterpart to .lon.
    @property
    def bpts(self) -> Path:
        """Node coordinates, binary."""
        return self._sibling(".bpts")

    @property
    def belem(self) -> Path:
        """Element connectivity, binary."""
        return self._sibling(".belem")

    # -- Submesh index --------------------------------------------------

    @property
    def index(self) -> SubmeshIndex:
        """
        The ``.nod``/``.eidx`` pair mapping this mesh back to its parent.

        Meaningful only when this mesh was carved out of a larger one — a
        connected component, or an extracted submesh. The trunk mesh has no
        such pair, which is why these two extensions are reached through a
        nested handle rather than sitting alongside ``.pts`` and ``.elem``.
        As everywhere else here, existence checks belong to the caller.
        """
        return SubmeshIndex(self.stem)

    # -- Role-named siblings --------------------------------------------

    def role(self, name: str, extension: str) -> Path:
        """
        Build a role-named sibling: ``<stem>.<name><extension>``.

        CARP and mguvc address boundary sets and derived surfaces by a role
        slotted between the stem and the extension — ``BiV.base.vtx``,
        ``BiV.rvsept.vtx``, ``BiV.lvfree_face.surf``. This is the mechanism
        for building such a name. *Which* roles are legal belongs to the tool
        that mandates them, not to this handle.

        Args:
            name: The role, without surrounding dots, e.g. "base", "rvsept",
                  "lvfree_face".
            extension: The extension, leading dot included, e.g. ".vtx".
                       May itself be multi-part — ".surf.vtx" is a real one.

        Example:
            >>> mesh = CarpMesh("/data/BiV/BiV")
            >>> mesh.role("base", ".vtx")
            PosixPath('/data/BiV/BiV.base.vtx')
            >>> mesh.role("epi", ".surf.vtx")
            PosixPath('/data/BiV/BiV.epi.surf.vtx')
        """
        if not name or name.startswith("."):
            raise ValueError(
                f"Role name must be non-empty and carry no leading dot, got '{name}'. "
                f"The separating dot is added for you."
            )
        if not extension.startswith("."):
            raise ValueError(
                f"Extension must start with a dot, got '{extension}'. "
                f"Did you mean '.{extension}'?"
            )
        return self._sibling(f".{name}{extension}")


@dataclass(frozen=True)
class SurfaceMesh(_StemHandle):
    """
    meshtool's ``.surfmesh`` companion to an extracted surface.

    Deliberately *not* a ``CarpMesh``. The name suggests one, but the recorded
    output of real runs shows this family carries only ``.vtk``, ``.nod`` and
    ``.fcon`` — never ``.pts`` or ``.elem``. Reusing ``CarpMesh`` here would
    advertise five extensions that are never written, which is the kind of
    contract-that-lies this handle exists to prevent.

    ``.fcon`` is not always present: it appeared for two of the three surfmesh
    stems in the recorded runs. Existence checks belong to the caller.

    Example:
        >>> SurfaceMesh("/data/tmp/septum.surfmesh").vtk
        PosixPath('/data/tmp/septum.surfmesh.vtk')
    """

    # Refuses nothing: a `.surfmesh` stem legitimately ends in a dot-segment,
    # and no guard was ever asserted here. `()` is the setting, not a gap.
    _refuses: ClassVar[tuple[str, ...]] = ()

    @property
    def vtk(self) -> Path:
        """VTK rendering of the surface."""
        return self._sibling(".vtk")

    @property
    def nod(self) -> Path:
        """
        Node index map back to the parent mesh, binary.

        The same concept as a submesh's ``.nod``, so it is delegated rather
        than restated — one definition of where ``.nod`` sits. There is no
        matching ``.eidx``: no recorded run has produced one for a
        ``.surfmesh``, so the pair is not exposed here as a pair.
        """
        return SubmeshIndex(self.stem).nod

    @property
    def fcon(self) -> Path:
        """Face connectivity. Not always written."""
        return self._sibling(".fcon")


@dataclass(frozen=True)
class CarpSurface(_StemHandle):
    """
    A surface extracted by ``meshtool extract surface``, addressed by its stem.

    Like ``CarpMesh``, a surface is a family rather than a file. meshtool emits
    the triangle list as ``<stem>.surf``, the node set that spans it as
    ``<stem>.surf.vtx``, and — depending on the operation — a ``<stem>.neubc``
    and a ``<stem>.surfmesh`` family.

    Note that the VTX companion is ``<stem>.surf.vtx``, *not* ``<stem>.vtx``.
    This is exactly the case ``pathlib.with_suffix`` gets wrong, and it is why
    the surface stage needs a handle rather than string surgery.

    Example:
        >>> surface = CarpSurface("/data/tmp/septum")
        >>> surface.surf
        PosixPath('/data/tmp/septum.surf')
        >>> surface.surf_vtx
        PosixPath('/data/tmp/septum.surf.vtx')
        >>> surface.surfmesh.nod
        PosixPath('/data/tmp/septum.surfmesh.nod')
    """

    stem: Path

    # `.vtx` is here because a surface's companion is `<stem>.surf.vtx`, so a
    # stem ending in `.vtx` is a caller one level too deep.
    _refuses: ClassVar[tuple[str, ...]] = (".surf", ".surfmesh", ".neubc", ".vtx")

    # -- The family -----------------------------------------------------

    @property
    def surf(self) -> Path:
        """Triangle list."""
        return self._sibling(".surf")

    @property
    def surf_vtx(self) -> Path:
        """Node indices spanned by the surface. Note the doubled extension."""
        return self._sibling(".surf.vtx")

    @property
    def neubc(self) -> Path:
        """Neumann boundary conditions. meshtool writes it; nothing here reads it."""
        return self._sibling(".neubc")

    @property
    def surfmesh(self) -> SurfaceMesh:
        """The ``.surfmesh`` companion family, as its own handle."""
        return SurfaceMesh(self._sibling(".surfmesh"))

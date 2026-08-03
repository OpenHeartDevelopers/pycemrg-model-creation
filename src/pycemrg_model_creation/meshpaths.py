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

# Extensions CarpMesh owns. A stem ending in one of these is almost always a
# caller that passed a concrete file where a stem was wanted.
_CARP_MESH_EXTENSIONS = (".pts", ".elem", ".lon", ".vtk", ".bpts", ".belem")


@dataclass(frozen=True)
class CarpMesh:
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

    stem: Path

    def __post_init__(self) -> None:
        # Accept str for convenience, but store a Path. Assigning through
        # object.__setattr__ is how a frozen dataclass normalises its fields.
        object.__setattr__(self, "stem", Path(self.stem))

        if self.stem.suffix in _CARP_MESH_EXTENSIONS:
            raise ValueError(
                f"CarpMesh takes a stem without an extension, got '{self.stem}'. "
                f"Use CarpMesh('{self.stem.with_suffix('')}') instead."
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

    # -- Internals ------------------------------------------------------

    def _sibling(self, extension: str) -> Path:
        """
        Append an extension to the stem.

        Deliberately not `with_suffix`: the stem may itself contain dots and
        those dot-segments are meaningful, not extensions to be replaced.
        """
        return self.stem.parent / f"{self.stem.name}{extension}"

    # -- Interop --------------------------------------------------------

    def __str__(self) -> str:
        return str(self.stem)

    def __fspath__(self) -> str:
        # Makes the handle os.PathLike, so it can be handed to a wrapper
        # wherever a stem is expected without an explicit conversion.
        return str(self.stem)

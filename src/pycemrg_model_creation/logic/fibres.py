# src/pycemrg_model_creation/logic/fibres.py
"""
Rule-based ventricular fibre generation logic.

This module orchestrates GlRuleFibres to generate biventricular fibre
orientations from a completed UVC (Universal Ventricular Coordinate) run.
"""

import logging
import shutil
from typing import Dict, Optional

from pathlib import Path

from pycemrg_model_creation.logic.contracts import VentricularFibrePaths
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.tools.mguvc import outputs_for
from pycemrg_model_creation.tools.wrappers import DEFAULT_FIBRE_ANGLES, CarpWrapper
from pycemrg_model_creation.utilities import mesh as mshu

# The element-centres file GlElemCenters writes beside the mesh. The suffix is
# an underscore, not a dotted role, because that is the spelling
# cemrg-heartbuilder uses and `correct_fibres` there reads it back by that
# name. It is derived here rather than stored on the contract, and it is not a
# `CarpMesh` member: only a mesh that has been through GlElemCenters has one.
ELEMENT_CENTRES_SUFFIX = "_elem_centres.pts"

# The rule-based fibre file GlRuleFibers writes, before it is copied onto the
# mesh's own `.lon`. Underscore-spelled and angle-stamped, as
# cemrg-heartbuilder spells it (`FourChamberModeler.get_meshname_fibres`), so
# two runs at different angles leave two distinguishable files rather than one
# that silently changed. Derived here for the same reason as
# ELEMENT_CENTRES_SUFFIX: only a mesh that has been through GlRuleFibers has
# one, and there is a single call site.
RULE_FIBRES_PREFIX = "_fibres_bayer"


class FibreLogic:
    """
    Stateless logic for rule-based ventricular fibre generation.

    Orchestrates CARPentry's GlRuleFibres tool to assign fibre orientations
    to a biventricular mesh from the four Laplace solutions a prior UVC run
    already produced. `VentricularFibrePaths` carries the two choices the
    caller actually makes — which mesh, and where the UVC run wrote its
    Laplace solutions. The Laplace inputs and the `.lon` output are asked of
    `tools.mguvc` and the mesh handle rather than restated here.
    """

    def __init__(self, carp_wrapper: CarpWrapper):
        """
        Initialize FibreLogic with dependencies.

        Args:
            carp_wrapper: Initialized CarpWrapper for running GlRuleFibres
        """
        self.carp = carp_wrapper
        self.logger = logging.getLogger(__name__)

    def run_ventricular_fibre_generation(
        self,
        paths: VentricularFibrePaths,
        angles: Optional[Dict[str, float]] = None,
        fibre_type: str = "biv",
    ) -> None:
        """
        Complete ventricular fibre generation workflow.

        Executes the full pipeline:
        1. Validate the mesh and the four Laplace solutions exist
        2. Run GlRuleFibres to assign fibre orientations

        Args:
            paths: Path contract for the fibre generation workflow
            angles: Fibre rotation angles (alpha_endo, alpha_epi, beta_endo,
                    beta_epi). Defaults to `CarpWrapper`'s own module default.
            fibre_type: Fibre type passed to GlRuleFibres (default: "biv")

        Raises:
            FileNotFoundError: If the mesh or a Laplace solution is missing

        Example:
            >>> fibre_logic.run_ventricular_fibre_generation(paths=fibre_paths)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ventricular fibre generation")
        self.logger.info("=" * 60)

        self._validate_inputs(paths)

        # Resolved here rather than left to the wrapper, because the output
        # filename is stamped with the angles actually used.
        angles_used = self._validated_angles(angles)
        generated = self._rule_fibres_path(paths.biv_mesh, angles_used)

        outputs = outputs_for(paths.biv_mesh, paths.uvc_output_dir)
        self.carp.gl_rule_fibres(
            mesh_name=paths.biv_mesh.stem,
            uvc_apba=outputs.sol_apba,
            uvc_epi=outputs.sol_endoepi,
            uvc_lv=outputs.sol_lvendo,
            uvc_rv=outputs.sol_rvendo,
            output_name=generated,
            angles=angles_used,
            fibre_type=fibre_type,
        )

        # GlRuleFibers reads the mesh's own `.lon` as an input, so it must not
        # be the thing GlRuleFibers writes: a run that died mid-write would
        # leave the mesh unusable for a retry. Write beside it, then copy on.
        shutil.copyfile(generated, paths.biv_mesh.lon)

        self.logger.info("=" * 60)
        self.logger.info("Fibre generation completed successfully")
        self.logger.info(f"Fibres generated at: {generated}")
        self.logger.info(f"Copied onto mesh fibres: {paths.biv_mesh.lon}")
        self.logger.info("=" * 60)

    def correct_ventricular_fibres(
        self,
        paths: VentricularFibrePaths,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Repair degenerate fibre vectors left by GlRuleFibers, in place.

        Runs `GlElemCenters` to get one centre per element, then replaces
        every fibre whose z-component is below `tolerance` with the whole
        fibre row of the nearest well-defined element. The mesh's `.lon` is
        overwritten with the result.

        Deliberately **not** called by
        :meth:`run_ventricular_fibre_generation`. It is a separate step an
        orchestrator invokes after generation, which is how
        `cemrg-heartbuilder` sequences it (fibres, then centres, then
        correction) and what `cemrg-bivbuilder` asked to be able to do.

        Args:
            paths: Path contract for the fibre generation workflow
            tolerance: Magnitude below which a z-component counts as
                degenerate. The reference pipeline uses 1e-6.

        Raises:
            FileNotFoundError: If the mesh or its `.lon` is missing
            ValueError: If the centres and fibres disagree on element count

        Example:
            >>> fibre_logic.correct_ventricular_fibres(paths=fibre_paths)
        """
        self.logger.info("Correcting degenerate ventricular fibres")

        mesh = paths.biv_mesh
        for description, mesh_file in (
            ("points", mesh.pts),
            ("elements", mesh.elem),
            ("fibres", mesh.lon),
        ):
            if not mesh_file.exists():
                raise FileNotFoundError(
                    f"Mesh {description} file not found: {mesh_file}. "
                    f"Fibre correction runs after generation, not before."
                )

        centres_path = self._element_centres_path(mesh)
        self.carp.gl_elem_centers(meshname=mesh.stem, output=centres_path)

        fibres = mshu.read_lon(mesh.lon)
        element_centres = mshu.read_pts(centres_path)

        corrected = mshu.correct_degenerate_fibres(
            fibres, element_centres, tolerance=tolerance
        )

        mshu.write_lon(corrected, mesh.lon)
        self.logger.info(f"Corrected fibres written to: {mesh.lon}")

    @staticmethod
    def _validated_angles(angles: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Resolve the angle set, refusing a partial one.

        `angles` replaces the default wholesale; it is not merged with it. A
        dict missing a key would therefore send an incomplete angle set to
        GlRuleFibers, and -- since the output filename is stamped with
        `alpha_endo` and `alpha_epi` -- would fail as a bare `KeyError` if the
        missing key happened to be one of those two. Name the problem instead.

        Args:
            angles: The caller's angles, or None for the library default.

        Returns:
            A complete angle set.

        Raises:
            ValueError: If `angles` is given but omits any required key.
        """
        if angles is None:
            return DEFAULT_FIBRE_ANGLES

        missing = sorted(set(DEFAULT_FIBRE_ANGLES) - set(angles))
        if missing:
            raise ValueError(
                f"Incomplete fibre angles: missing {', '.join(missing)}. "
                f"`angles` replaces the default set wholesale rather than "
                f"merging with it, so pass all of "
                f"{', '.join(sorted(DEFAULT_FIBRE_ANGLES))} or omit the "
                f"argument to use the default."
            )
        return angles

    @staticmethod
    def _rule_fibres_path(mesh: CarpMesh, angles: Dict[str, float]) -> Path:
        """
        Where GlRuleFibers writes this run's fibres, before they are copied
        onto the mesh's `.lon`.

        Stamped with the two alpha angles, as cemrg-heartbuilder stamps it,
        so a second run at different angles produces a second file instead of
        overwriting the first with no record of which angles are in it.
        """
        stamp = f"{angles['alpha_endo']}_{angles['alpha_epi']}"
        return mesh.directory / f"{mesh.name}{RULE_FIBRES_PREFIX}_{stamp}.lon"

    @staticmethod
    def _element_centres_path(mesh: CarpMesh) -> Path:
        """
        Where GlElemCenters writes this mesh's element centres.

        Asked for rather than stored, and built from the handle's own
        directory and name so the two cannot drift apart.
        """
        return mesh.directory / f"{mesh.name}{ELEMENT_CENTRES_SUFFIX}"

    def _validate_inputs(self, paths: VentricularFibrePaths) -> None:
        """
        Validate the files GlRuleFibres will read.

        That is the whole mesh family (.pts, .elem, .lon) and the four
        Laplace solutions a prior UVC run wrote under
        `paths.uvc_output_dir`.

        The `.lon` is not optional here. GlRuleFibers loads the mesh behind
        `-m` in full, so it opens the `.lon` before it writes one, and a
        missing file surfaces as an opaque non-zero exit rather than a
        named error. A real chain always has one: `meshtool extract mesh`
        carries the parent mesh's fibres across into the submesh. This
        differs from `mguvc`, which fabricates `[1 0 0]` when the `.lon` is
        absent -- do not carry that tolerance over to this step.

        Args:
            paths: VentricularFibrePaths contract

        Raises:
            FileNotFoundError: If any required file is missing
        """
        self.logger.debug("Validating input files")

        mesh = paths.biv_mesh
        for description, mesh_file in (
            ("points", mesh.pts),
            ("elements", mesh.elem),
            ("fibres", mesh.lon),
        ):
            if not mesh_file.exists():
                raise FileNotFoundError(
                    f"Mesh {description} file not found: {mesh_file}"
                )

        self.logger.debug(f"✓ Mesh files validated: {mesh}")

        outputs = outputs_for(paths.biv_mesh, paths.uvc_output_dir)
        missing_files = []
        for name, solution_path in (
            ("apico-basal", outputs.sol_apba),
            ("endo-epi", outputs.sol_endoepi),
            ("LV endo", outputs.sol_lvendo),
            ("RV endo", outputs.sol_rvendo),
        ):
            if solution_path.exists():
                self.logger.debug(f"✓ {name} Laplace solution: {solution_path.name}")
            else:
                missing_files.append(f"  - {name}: {solution_path}")

        if missing_files:
            raise FileNotFoundError(
                "Required Laplace solution files not found:\n"
                + "\n".join(missing_files)
            )

        self.logger.debug("All input files validated successfully")

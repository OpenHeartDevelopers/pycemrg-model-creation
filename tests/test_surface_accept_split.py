"""
How `extract_ventricular_surfaces` treats an accepted split on disk.

The stage extracts the epi/endo surface again on every run, and meshtool
writes the same triangles in a new row order each time. An accepted split
(`write_accepted_split`) therefore applies while the new surface has the same
triangles, and the stage writes the parts again from the new surface instead
of trusting `.eidx` row positions from the old one. A stale or absent split
must never stop the flow, and must never be counted alongside meshtool's parts.

The stage needs more than this test supplies after the component step (a
tetrahedral mesh, labels), so each run ends in a `SurfaceExtractionError`.
The assertions are on what happened before that point.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pathlib import Path

import pycemrg_model_creation.utilities.mesh as mshu

from pycemrg_model_creation.logic.surfaces import (
    SurfaceExtractionError,
    SurfaceLogic,
)
from pycemrg_model_creation.meshpaths import CarpMesh
from pycemrg_model_creation.utilities.mesh import write_pts, write_surf


# Three triangles pinched at vertex 0: meshtool sees one component, edge
# adjacency sees the three the stage needs.
PINCHED = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6]])
POINTS = np.vstack([np.zeros(3), np.eye(3), -np.eye(3)])


class _Labels:
    def get_tags_string(self, names):
        return ",".join(str(i) for i, _ in enumerate(names, start=1))


class _StubMeshtool:
    """Writes `surf` (and a matching `.neubc`) as the extracted surface."""

    def __init__(self, surf):
        self.surf = surf
        self.unreachable_calls = 0

    def extract_surface(self, input_mesh_path, output_surface_path, **_):
        write_surf(self.surf, Path(f"{output_surface_path}.surf"))
        _write_neubc(self.surf, Path(f"{output_surface_path}.neubc"))

    def extract_unreachable(self, **_):
        self.unreachable_calls += 1


def _write_neubc(surf, path):
    rows = [f"{a} {b} {c} 1 0 0" for a, b, c in surf]
    path.write_text(f"{len(rows)}\n" + "\n".join(rows) + "\n")


@pytest.fixture
def paths(tmp_path):
    write_pts(POINTS, tmp_path / "mesh.pts")
    return SimpleNamespace(
        mesh=CarpMesh(tmp_path / "mesh"),
        tmp_dir=tmp_path,
        epi_endo_combined=tmp_path / "epi_endo",
        epi_endo_cc_base=tmp_path / "epi_endo_cc",
    )


def _accept_split_of(surf, paths):
    surface = Path(f"{paths.epi_endo_combined}.surf")
    write_surf(surf, surface)
    mshu.write_accepted_split(surface, paths.mesh.pts, paths.epi_endo_cc_base)


def _run(meshtool, paths):
    logic = SurfaceLogic(meshtool, _Labels())
    with pytest.raises(SurfaceExtractionError) as caught:
        logic.extract_ventricular_surfaces(paths)
    return str(caught.value)


def _triangles_of_part(k, paths, surf):
    eidx = np.fromfile(f"{paths.epi_endo_cc_base}.part{k}.eidx", dtype=np.int64)
    return {tuple(row) for row in surf[eidx]}


def test_a_matching_split_is_split_again_and_meshtool_is_skipped(paths):
    _accept_split_of(PINCHED, paths)
    meshtool = _StubMeshtool(PINCHED)

    _run(meshtool, paths)

    assert meshtool.unreachable_calls == 0
    assert mshu.split_sidecar_path(paths.epi_endo_cc_base).exists()


def test_a_new_row_order_still_matches_and_the_parts_follow_it(paths):
    """The production case: same triangles, new order, fresh `.eidx`."""
    _accept_split_of(PINCHED, paths)
    before = [_triangles_of_part(k, paths, PINCHED) for k in range(3)]

    reordered = PINCHED[::-1].copy()
    meshtool = _StubMeshtool(reordered)
    _run(meshtool, paths)

    assert meshtool.unreachable_calls == 0
    after = [_triangles_of_part(k, paths, reordered) for k in range(3)]
    assert sorted(map(sorted, after)) == sorted(map(sorted, before))


def test_a_stale_split_is_removed_and_meshtool_runs(paths, caplog):
    _accept_split_of(PINCHED, paths)
    changed = PINCHED.copy()
    changed[2, 2] = 2  # now shares an edge with the first triangle
    meshtool = _StubMeshtool(changed)

    message = _run(meshtool, paths)

    assert meshtool.unreachable_calls == 2
    assert "stale" in caplog.text
    assert not mshu.split_sidecar_path(paths.epi_endo_cc_base).exists()
    assert "found 0" in message  # the stub writes no parts


def test_without_a_sidecar_old_parts_are_removed_before_meshtool(paths):
    old = Path(f"{paths.epi_endo_cc_base}.part0.elem")
    old.write_text("left by an earlier meshtool run")
    meshtool = _StubMeshtool(PINCHED)

    message = _run(meshtool, paths)

    assert meshtool.unreachable_calls == 2
    assert not old.exists()
    assert "found 0" in message


# --- Hand-made surface roles --------------------------------------------- #

def _with_outputs(paths, tmp_path):
    paths.epi_surface = tmp_path / "out_epi"
    paths.lv_endo_surface = tmp_path / "out_lvendo"
    paths.rv_endo_surface = tmp_path / "out_rvendo"
    return paths


def test_roles_survive_the_re_split_and_choose_the_outputs(paths, tmp_path):
    from pycemrg_model_creation.utilities.topology import SurfaceRoles

    _with_outputs(paths, tmp_path)
    surface = Path(f"{paths.epi_endo_combined}.surf")
    write_surf(PINCHED, surface)
    roles = SurfaceRoles(epi=2, lv_endo=0, rv_endo=1)
    mshu.write_accepted_split(
        surface, paths.mesh.pts, paths.epi_endo_cc_base, roles=roles
    )

    # A new row order: the stage splits again and must keep the roles.
    meshtool = _StubMeshtool(PINCHED[::-1].copy())
    SurfaceLogic(meshtool, _Labels()).extract_ventricular_surfaces(paths)

    assert mshu.read_split_roles(paths.epi_endo_cc_base) == roles
    base = paths.epi_endo_cc_base
    for out, number in (("out_epi", 2), ("out_lvendo", 0), ("out_rvendo", 1)):
        assert (tmp_path / f"{out}.elem").read_text() == (
            Path(f"{base}.part{number}.elem").read_text()
        )


def test_a_role_naming_a_dropped_part_raises(paths, tmp_path):
    import json

    from pycemrg_model_creation.logic.surfaces import SurfaceIdentificationError

    # Three three-triangle sheets and a two-triangle sliver, all pinched at 0.
    # The sliver is part3, and the stage keeps only the largest three. (Not a
    # one-triangle sliver: `read_elem` returns a one-row file as 1-D, and
    # `keep_largest_n_components` would then size it by its column count.)
    four = np.array([[0, 1, 2], [2, 1, 9], [9, 1, 13],
                     [0, 3, 4], [4, 3, 10], [10, 3, 14],
                     [0, 5, 6], [6, 5, 11], [11, 5, 15],
                     [0, 7, 8], [8, 7, 12]])
    write_pts(np.vstack([POINTS, np.ones((9, 3))]), paths.mesh.pts)
    _with_outputs(paths, tmp_path)
    _accept_split_of(four, paths)
    sidecar = mshu.split_sidecar_path(paths.epi_endo_cc_base)
    data = json.loads(sidecar.read_text())
    data["roles"] = {"epi": 3, "lv_endo": 0, "rv_endo": 1}  # edited by hand
    sidecar.write_text(json.dumps(data))

    with pytest.raises(SurfaceIdentificationError, match="part3"):
        SurfaceLogic(_StubMeshtool(four), _Labels()).extract_ventricular_surfaces(
            paths
        )


# --- The septum takes the same accept path --------------------------------- #

def test_an_accepted_septum_split_is_used_and_meshtool_is_skipped(paths, tmp_path):
    # A three-triangle sheet and a two-triangle sheet pinched at 0: meshtool
    # sees one component, the septum step needs two (largest first).
    septum = np.array([[0, 1, 2], [2, 1, 7], [7, 1, 8], [0, 3, 4], [4, 3, 9]])
    write_pts(np.vstack([POINTS, np.ones((3, 3))]), paths.mesh.pts)
    paths.septum_raw = tmp_path / "septum"
    paths.septum_cc_base = tmp_path / "septum_cc"
    paths.lv_epi_intermediate = tmp_path / "out_lv_epi"
    paths.septum_surface = tmp_path / "out_septum"

    surface = Path(f"{paths.septum_raw}.surf")
    write_surf(septum, surface)
    mshu.write_accepted_split(surface, paths.mesh.pts, paths.septum_cc_base)

    meshtool = _StubMeshtool(septum[::-1].copy())
    SurfaceLogic(meshtool, _Labels()).extract_septum(paths)

    assert meshtool.unreachable_calls == 0
    assert (tmp_path / "out_septum.elem").read_text() == (
        Path(f"{paths.septum_cc_base}.part1.elem").read_text()
    )

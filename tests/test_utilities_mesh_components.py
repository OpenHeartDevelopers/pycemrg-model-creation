"""Unit tests for connected-component discovery and pruning in `utilities.mesh`.

These two functions are the *discovered*-outputs mechanism: `meshtool extract
unreachable` splits a surface into however many components the data happens to
have, so the arity can only be learned by looking. Names computable in advance
belong to `tools.mguvc.outputs_for` instead, and the two are not unified.

Unlike `test_utilities_mesh_io.py`, nothing here is a round trip. These
functions scan a directory and delete files, so every test works against real
files under `tmp_path`.
"""

from pathlib import Path

import pytest

from pycemrg_model_creation.meshpaths import CarpMesh, SubmeshParts
from pycemrg_model_creation.utilities.mesh import (
    find_numbered_parts,
    keep_largest_n_components,
)

# The family meshtool writes beside each component. `.elem` is the one both
# functions read; the rest exist so the deletion can be seen to take the whole
# family and not just the file it measured.
FAMILY = (".elem", ".pts", ".nod", ".eidx")


def make_component(directory: Path, name: str, triangles: int) -> CarpMesh:
    """Write one component's family. `triangles` fixes its measured size."""
    body = "\n".join("Tr 0 1 2" for _ in range(triangles))
    (directory / f"{name}.elem").write_text(f"{triangles}\n{body}\n")
    for extension in FAMILY[1:]:
        (directory / f"{name}{extension}").write_text("placeholder\n")
    return CarpMesh(directory / name)


class TestFindNumberedParts:
    def test_finds_the_sequential_run(self, tmp_path):
        for index in range(3):
            make_component(tmp_path, f"epi_endo_cc.part{index}", triangles=4)

        found = find_numbered_parts(tmp_path, "epi_endo_cc")

        assert [part.name for part in found] == [
            "epi_endo_cc.part0",
            "epi_endo_cc.part1",
            "epi_endo_cc.part2",
        ]

    def test_returns_handles_rooted_in_the_search_directory(self, tmp_path):
        # The defect this return type removes: callers used to get bare names
        # and re-join `tmp_dir` themselves at every site.
        make_component(tmp_path, "epi_endo_cc.part0", triangles=4)

        found = find_numbered_parts(tmp_path, "epi_endo_cc")

        assert isinstance(found, SubmeshParts)
        assert found[0].directory == tmp_path
        assert found[0].elem == tmp_path / "epi_endo_cc.part0.elem"

    def test_stops_at_the_first_gap(self, tmp_path):
        # Numbering is sequential from 0. A missing part1 ends the run, even
        # though part2 exists -- the loop's own contract.
        make_component(tmp_path, "cc.part0", triangles=4)
        make_component(tmp_path, "cc.part2", triangles=4)

        assert [part.name for part in find_numbered_parts(tmp_path, "cc")] == [
            "cc.part0"
        ]

    def test_finding_nothing_is_an_empty_answer_not_an_error(self, tmp_path):
        found = find_numbered_parts(tmp_path, "absent")

        assert isinstance(found, SubmeshParts)
        assert len(found) == 0

    def test_a_different_prefix_is_not_matched(self, tmp_path):
        make_component(tmp_path, "septum_cc.part0", triangles=4)

        assert len(find_numbered_parts(tmp_path, "epi_endo_cc")) == 0

    def test_the_part_segment_is_not_treated_as_an_extension(self, tmp_path):
        # `with_suffix` would turn `cc.part0` into `cc.elem`. The handle
        # appends instead, which is the whole reason these are handles.
        make_component(tmp_path, "cc.part0", triangles=4)

        assert find_numbered_parts(tmp_path, "cc")[0].elem.name == "cc.part0.elem"


class TestKeepLargestNComponents:
    def test_keeps_the_largest_and_orders_them_by_size(self, tmp_path):
        make_component(tmp_path, "cc.part0", triangles=5)
        make_component(tmp_path, "cc.part1", triangles=9)
        make_component(tmp_path, "cc.part2", triangles=2)

        kept = keep_largest_n_components(find_numbered_parts(tmp_path, "cc"), keep_n=2)

        assert [part.name for part in kept] == ["cc.part1", "cc.part0"]

    def test_discarded_components_lose_their_whole_family(self, tmp_path):
        # Not just the `.elem` the size was measured from.
        make_component(tmp_path, "cc.part0", triangles=9)
        make_component(tmp_path, "cc.part1", triangles=2)

        keep_largest_n_components(find_numbered_parts(tmp_path, "cc"), keep_n=1)

        for extension in FAMILY:
            assert not (tmp_path / f"cc.part1{extension}").exists()

    def test_kept_components_are_left_alone(self, tmp_path):
        make_component(tmp_path, "cc.part0", triangles=9)
        make_component(tmp_path, "cc.part1", triangles=2)

        keep_largest_n_components(find_numbered_parts(tmp_path, "cc"), keep_n=1)

        for extension in FAMILY:
            assert (tmp_path / f"cc.part0{extension}").exists()

    def test_deletion_can_be_declined(self, tmp_path):
        make_component(tmp_path, "cc.part0", triangles=9)
        make_component(tmp_path, "cc.part1", triangles=2)

        kept = keep_largest_n_components(
            find_numbered_parts(tmp_path, "cc"), keep_n=1, delete_smaller=False
        )

        assert [part.name for part in kept] == ["cc.part0"]
        assert (tmp_path / "cc.part1.elem").exists()

    def test_asking_for_more_than_exist_is_refused(self, tmp_path):
        make_component(tmp_path, "cc.part0", triangles=4)

        with pytest.raises(ValueError, match="only 1 found"):
            keep_largest_n_components(find_numbered_parts(tmp_path, "cc"), keep_n=3)

    def test_refusal_happens_before_anything_is_deleted(self, tmp_path):
        make_component(tmp_path, "cc.part0", triangles=4)

        with pytest.raises(ValueError):
            keep_largest_n_components(find_numbered_parts(tmp_path, "cc"), keep_n=3)

        assert (tmp_path / "cc.part0.elem").exists()


class TestKeepLargestAcceptsAnyIterable:
    # Strict in what it returns, liberal in what it accepts. A caller holding
    # a plain list of handles must not have to wrap it first -- that friction
    # is what makes someone copy the function instead of importing it.

    def test_a_plain_list_is_accepted(self, tmp_path):
        components = [
            make_component(tmp_path, "cc.part0", triangles=5),
            make_component(tmp_path, "cc.part1", triangles=9),
        ]

        kept = keep_largest_n_components(components, keep_n=1, delete_smaller=False)

        assert [part.name for part in kept] == ["cc.part1"]

    def test_a_generator_is_accepted(self, tmp_path):
        # It is counted before it is walked, so a one-shot iterable has to be
        # materialised internally rather than consumed twice.
        make_component(tmp_path, "cc.part0", triangles=5)
        make_component(tmp_path, "cc.part1", triangles=9)
        components = (part for part in find_numbered_parts(tmp_path, "cc"))

        kept = keep_largest_n_components(components, keep_n=1, delete_smaller=False)

        assert [part.name for part in kept] == ["cc.part1"]

    @pytest.mark.parametrize("wrap", [list, tuple, SubmeshParts], ids=["list", "tuple", "SubmeshParts"])
    def test_the_return_type_does_not_follow_the_input_type(self, tmp_path, wrap):
        components = [
            make_component(tmp_path, "cc.part0", triangles=5),
            make_component(tmp_path, "cc.part1", triangles=9),
        ]

        kept = keep_largest_n_components(
            wrap(components), keep_n=1, delete_smaller=False
        )

        assert isinstance(kept, SubmeshParts)

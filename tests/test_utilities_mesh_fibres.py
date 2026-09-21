"""Unit tests for `correct_degenerate_fibres`.

A pure array transform: no files, no tags, no mesh connectivity. The
behaviour pinned here is deliberately the reference pipeline's, quirks
included, because changing any of it changes the fibres a mesh gets. See
`.claude/evidence_tagsconfig_150926.md` for the `cemrg-heartbuilder` source
this was ported from.
"""

import numpy as np
import pytest

from pycemrg_model_creation.utilities.mesh import correct_degenerate_fibres

# A well-defined row: fibre plus sheet, six floats, non-zero z.
GOOD_A = [1.0, 0.0, 0.9, 0.0, 1.0, 0.1]
GOOD_B = [0.0, 1.0, 0.8, 1.0, 0.0, 0.2]
# Degenerate: z below tolerance. Note it is otherwise a perfectly good unit
# vector -- the test is on the z-component alone, not on the norm.
FLAT = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]


class TestWhichElementsAreCorrected:
    def test_a_mesh_with_no_degenerate_fibres_is_unchanged(self):
        fibres = np.array([GOOD_A, GOOD_B])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result, fibres)

    def test_the_input_is_never_modified(self):
        fibres = np.array([GOOD_A, FLAT])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        before = fibres.copy()

        correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(fibres, before)

    def test_the_test_is_on_the_z_component_alone_not_the_norm(self):
        # A unit vector lying flat in xy counts as degenerate; a short vector
        # with a real z does not. This is the reference behaviour and it is
        # surprising enough to be worth pinning.
        flat_unit = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        short_but_tilted = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
        fibres = np.array([GOOD_A, flat_unit, short_but_tilted])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert not np.array_equal(result[1], flat_unit)  # corrected
        assert np.array_equal(result[2], short_but_tilted)  # left alone

    def test_the_tolerance_is_honoured(self):
        nearly_flat = [1.0, 0.0, 1e-4, 0.0, 1.0, 0.0]
        fibres = np.array([GOOD_A, nearly_flat])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        assert np.array_equal(
            correct_degenerate_fibres(fibres, centres)[1], nearly_flat
        )
        assert not np.array_equal(
            correct_degenerate_fibres(fibres, centres, tolerance=1e-3)[1],
            nearly_flat,
        )


class TestWhichNeighbourIsChosen:
    def test_nearest_is_measured_between_element_centres_not_indices(self):
        # The discriminating case. Element 2 is degenerate. By *index* its
        # neighbours are 1 and 3; by *centre* the nearest is element 0. A
        # port that walked the array instead of measuring would pass every
        # other test here and fail this one.
        fibres = np.array([GOOD_A, GOOD_B, FLAT, GOOD_B])
        centres = np.array(
            [[0.0, 0, 0], [100.0, 0, 0], [1.0, 0, 0], [200.0, 0, 0]]
        )

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result[2], GOOD_A)

    def test_the_whole_row_travels_including_the_sheet_direction(self):
        # Six columns: fibre then sheet. Copying only the first three would
        # leave a mismatched sheet behind.
        fibres = np.array([GOOD_A, FLAT])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result[1], GOOD_A)

    def test_a_degenerate_element_never_inherits_from_another_degenerate_one(self):
        # Element 1 is closest to element 2, but both are degenerate. The
        # donor must be element 0, the only well-defined one.
        fibres = np.array([GOOD_A, FLAT, FLAT])
        centres = np.array([[10.0, 0, 0], [0.0, 0, 0], [0.1, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result[1], GOOD_A)
        assert np.array_equal(result[2], GOOD_A)

    def test_equidistant_neighbours_do_not_raise(self):
        # The upstream defect this port does not carry: resolving ties with
        # `where(d == d.min())` yields two indices and the row assignment
        # raises. Taking the first is the deviation, and it is deliberate.
        fibres = np.array([GOOD_A, FLAT, GOOD_A])
        centres = np.array([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result[1], GOOD_A)

    def test_fibre_only_arrays_are_supported(self):
        # Three columns rather than six: fibre direction with no sheet.
        fibres = np.array([[1.0, 0.0, 0.9], [0.0, 1.0, 0.0]])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        result = correct_degenerate_fibres(fibres, centres)

        assert np.array_equal(result[1], [1.0, 0.0, 0.9])


class TestRefusals:
    def test_every_element_degenerate_is_refused(self):
        fibres = np.array([FLAT, FLAT])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        with pytest.raises(ValueError, match="no .*well-defined neighbour"):
            correct_degenerate_fibres(fibres, centres)

    def test_disagreeing_element_counts_are_refused(self):
        # A stale centres file is the realistic cause, and silently
        # mis-pairing rows would corrupt the whole mesh.
        fibres = np.array([GOOD_A, FLAT, GOOD_B])
        centres = np.array([[0.0, 0, 0], [1.0, 0, 0]])

        with pytest.raises(ValueError, match="disagree on the element count"):
            correct_degenerate_fibres(fibres, centres)

    @pytest.mark.parametrize(
        "bad_fibres",
        [np.zeros((3,)), np.zeros((3, 4))],
        ids=["one-dimensional", "not a multiple of three"],
    )
    def test_malformed_fibre_arrays_are_refused(self, bad_fibres):
        with pytest.raises(ValueError, match="shape"):
            correct_degenerate_fibres(bad_fibres, np.zeros((3, 3)))

    def test_malformed_centres_are_refused(self):
        with pytest.raises(ValueError, match="Element centres must have shape"):
            correct_degenerate_fibres(np.array([GOOD_A]), np.zeros((1, 2)))

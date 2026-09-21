# tests/test_config.py
"""
`label_manager_from_mapping`: building a LabelManager from data in memory.

These pin the behaviour the retired `TagsConfig` used to provide, plus the
one it did not -- strictness about names it does not recognise.
"""

import pytest

from pycemrg.data.labels import LabelManager

from pycemrg_model_creation.config import label_manager_from_mapping


class TestItBuildsARealLabelManager:
    def test_the_result_is_a_plain_label_manager(self):
        """
        Deliberately not a subclass.

        One type reaches the rest of the library however it was built, so
        nothing downstream has two cases to handle. If this ever becomes a
        subclass, that decision is being reversed.
        """
        labels = label_manager_from_mapping({"LV": 1})
        assert type(labels) is LabelManager

    def test_it_leaves_no_file_behind(self, tmp_path, monkeypatch):
        """
        The manifest goes to disk only long enough to be read back.

        LabelManager does not retain its config_path, so the temporary
        directory can go as soon as the object exists.
        """
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        labels = label_manager_from_mapping({"LV": 1, "RV": 2})

        assert labels.get_tags_string(["LV", "RV"]) == "1,2"
        assert list(tmp_path.iterdir()) == []


class TestTagValues:
    def test_a_single_integer_becomes_a_label(self):
        labels = label_manager_from_mapping({"LV": 1, "RV": 2, "PArt": 6})
        assert labels.get_value("LV") == 1
        assert labels.get_tags_string(["LV", "RV", "PArt"]) == "1,2,6"

    def test_a_list_of_integers_resolves_to_all_of_them(self):
        """
        `TagsConfig` allowed `Union[int, List[int]]`, and this keeps that.

        LabelManager's `labels:` map holds one integer per name -- adding a
        list to its value set raises -- so a list becomes a `groups:` entry
        whose members are the values spelled as digits.
        """
        labels = label_manager_from_mapping({"LV": [1, 3], "RV": 2})

        assert labels.get_values_from_names(["LV"]) == [1, 3]
        assert labels.get_tags_string(["LV", "RV"]) == "1,2,3"

    def test_named_groups_pass_through(self):
        labels = label_manager_from_mapping(
            {"LV_myo": 1, "LV_bp": 3, "RV": 2},
            groups={"LV": ["LV_myo", "LV_bp"]},
        )
        assert labels.get_tags_string(["LV"]) == "1,3"


class TestItRefusesWhatTagsConfigAccepted:
    def test_an_unknown_name_raises_rather_than_being_skipped(self):
        """
        The whole point of the fold.

        `TagsConfig.get_tags_string` did `getattr(self, key, None)` and
        skipped what it did not recognise, so a typo produced a short tag
        string and a quietly wrong mesh -- the same silent class of defect as
        the dropped PArt.
        """
        labels = label_manager_from_mapping({"LV": 1, "PArt": 6})

        with pytest.raises(KeyError, match="PArtt"):
            labels.get_tags_string(["LV", "PArtt"])

    def test_a_boolean_is_not_a_tag_value(self):
        """bool is an int subclass; `True` is a mistake, not a tag of 1."""
        with pytest.raises(ValueError, match="boolean"):
            label_manager_from_mapping({"LV": True})

    @pytest.mark.parametrize(
        "bad_mapping",
        [{"LV": "1"}, {"LV": 1.5}, {"LV": None}],
        ids=["string", "float", "none"],
    )
    def test_a_non_integer_value_raises(self, bad_mapping):
        with pytest.raises(ValueError, match="must be an integer"):
            label_manager_from_mapping(bad_mapping)

    def test_a_non_integer_list_member_raises(self):
        with pytest.raises(ValueError, match="non-integer member"):
            label_manager_from_mapping({"LV": [1, "x"]})

    def test_a_name_cannot_be_both_a_value_list_and_a_group(self):
        with pytest.raises(ValueError, match="both"):
            label_manager_from_mapping({"LV": [1, 3]}, groups={"LV": ["RV"]})

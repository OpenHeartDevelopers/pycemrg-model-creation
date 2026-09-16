# src/pycemrg_model_creation/config.py
"""
Building a `LabelManager` from data already in memory.

`LabelManager` (from the sibling `pycemrg` package) is this library's single
vocabulary for anatomical tags: it reads a YAML manifest and translates names
and groups into the integer tag values `meshtool` wants. Its only constructor
takes a path to that file.

This module exists for callers who hold their tags as a mapping rather than a
file -- a test fixture, or an orchestrator assembling tags from somewhere
other than disk. It replaces the former `TagsConfig`, which was a second,
overlapping tag vocabulary with nine hardcoded fields. `TagsConfig` skipped
names it did not recognise, so a typo produced a short tag string and a
quietly wrong mesh; `LabelManager` raises and names the valid keys.

`from_mapping` is a candidate for `LabelManager` itself in a later `pycemrg`
release. When it lands upstream, this module delegates to it and then goes
away -- nothing here reaches into `LabelManager`'s internals, so that
migration is a deletion rather than a rewrite.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import yaml

from pycemrg.data.labels import LabelManager

TagValue = Union[int, List[int]]


def label_manager_from_mapping(
    tags: Mapping[str, TagValue],
    groups: Optional[Mapping[str, List[str]]] = None,
) -> LabelManager:
    """
    Build a `LabelManager` from an in-memory mapping of tag names to values.

    A value may be a single integer or a list of them, which is what the old
    `TagsConfig` allowed. The two are not the same thing to `LabelManager`:
    its `labels:` map holds one integer per name, while several values under
    one name is what `groups:` is for. A list is therefore written as a group
    whose members are the values spelled as digits, which
    `get_values_from_names` resolves natively.

    Args:
        tags: Tag names to their integer value, or to a list of values.
        groups: Optional named collections of other tag names, passed through
            to the manifest's `groups:` section unchanged.

    Returns:
        A plain `LabelManager`. Deliberately not a subclass: one type reaches
        the rest of the library whichever way it was built.

    Raises:
        ValueError: If a tag value is neither an integer nor a list of them,
            or if a name is defined as both a tag list and a group.

    Example:
        >>> labels = label_manager_from_mapping({"LV": 1, "RV": 2})
        >>> labels.get_tags_string(["LV", "RV"])
        '1,2'
        >>> label_manager_from_mapping({"LV": [1, 3]}).get_tags_string(["LV"])
        '1,3'
    """
    manifest_labels: Dict[str, int] = {}
    manifest_groups: Dict[str, List[str]] = {
        name: list(members) for name, members in (groups or {}).items()
    }

    for name, value in tags.items():
        if isinstance(value, bool):
            # bool is an int subclass; a tag value of True is a mistake, not a 1.
            raise ValueError(f"Tag '{name}' has a boolean value: {value!r}")
        if isinstance(value, int):
            manifest_labels[name] = value
            continue
        if isinstance(value, (list, tuple)):
            if name in manifest_groups:
                raise ValueError(
                    f"Tag '{name}' is defined both as a list of values and as "
                    f"a group. It can be one or the other, not both."
                )
            members = []
            for member in value:
                if isinstance(member, bool) or not isinstance(member, int):
                    raise ValueError(
                        f"Tag '{name}' has a non-integer member: {member!r}"
                    )
                members.append(str(member))
            manifest_groups[name] = members
            continue
        raise ValueError(
            f"Tag '{name}' must be an integer or a list of integers, "
            f"got {type(value).__name__}: {value!r}"
        )

    manifest = {"labels": manifest_labels, "groups": manifest_groups}

    # LabelManager's only constructor takes a path, so the mapping goes to disk
    # just long enough to be read back. It does not retain the path -- __init__
    # loads the file into _labels/_groups and closes it -- so the directory can
    # go as soon as the object exists. The alternative, setting those private
    # attributes directly, would couple this module to pycemrg's internals and
    # break silently if a later release added a fourth.
    with tempfile.TemporaryDirectory(prefix="pycemrg-labels-") as tmp_dir:
        manifest_path = Path(tmp_dir) / "labels.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        return LabelManager(config_path=manifest_path)

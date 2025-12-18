# src/pycemrg_model_creation/config.py

from dataclasses import dataclass
from typing import Dict, List, Union, Optional


@dataclass
class TagsConfig:
    """
    Configuration for mesh element tags.

    Maps anatomical regions to their corresponding mesh element tag values.
    Tags can be single integers or lists of integers.
    """

    LV: Union[int, List[int]]
    RV: Union[int, List[int]]
    LA: Union[int, List[int]]
    RA: Union[int, List[int]]
    MV: Union[int, List[int]]  # Mitral valve
    TV: Union[int, List[int]]  # Tricuspid valve
    AV: Union[int, List[int]]  # Aortic valve
    PV: Union[int, List[int]]  # Pulmonary valve
    PArt: Union[int, List[int]]  # Pulmonary artery


    @classmethod
    def from_dict(cls, tags_dict: Dict[str, Union[int, List[int]]]) -> "TagsConfig":
        """Create TagsConfig from dictionary"""
        return cls(**tags_dict)

    def to_dict(self) -> Dict[str, Union[int, List[int]]]:
        """Convert TagsConfig to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def get_tags_string(self, keys: List[str]) -> str:
        """
        Get comma-separated string of tags for specified keys.

        Args:
            keys: List of tag names (e.g., ["LV", "RV"])

        Returns:
            Comma-separated string (e.g., "1,2" or "1,2,3,4")
        """
        tags = []
        for key in keys:
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, list):
                tags.extend(map(str, value))
            else:
                tags.append(str(value))
        return ",".join(tags)

    def get_tags_list(self, keys: List[str]) -> List[int]:
        """
        Get list of tag integers for specified keys.

        Args:
            keys: List of tag names (e.g., ["LV", "RV"])

        Returns:
            Flat list of integers
        """
        tags = []
        for key in keys:
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, list):
                tags.extend(value)
            else:
                tags.append(value)
        return tags

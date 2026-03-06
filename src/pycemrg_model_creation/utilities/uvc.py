# src/pycemrg_model_creation/utilities/uvc.py
"""
Utility functions for UVC (Universal Ventricular Coordinate) workflows.

This module provides helper functions for UVC calculation, including
etags script generation for element region mapping.
"""

from pathlib import Path
from typing import Dict, Optional


# Unused tag sentinel value for structures not present in the mesh
UNUSED_TAG = 200

# Default anatomical tag definitions for mguvc
DEFAULT_ETAGS = {
    'T_LV': 1,                      # Left ventricular myocardium
    'T_RV': 2,                      # Right ventricular myocardium
    'T_UNUSED': UNUSED_TAG,         # Unused tag
    'T_LA': UNUSED_TAG,             # Left atrial wall
    'T_LABP': UNUSED_TAG,           # Left atrial blood pool
    'T_LINFPULMVEINCUT': UNUSED_TAG,  # Left inferior pulmonary vein (cut)
    'T_LSUPPULMVEINCUT': UNUSED_TAG,  # Left superior pulmonary vein (cut)
    'T_RINFPULMVEINCUT': UNUSED_TAG,  # Right inferior pulmonary vein (cut)
    'T_RSUPPULMVEINCUT': UNUSED_TAG,  # Right superior pulmonary vein (cut)
    'T_RA': UNUSED_TAG,             # Right atrial wall
    'T_RABP': UNUSED_TAG,           # Right atrial blood pool
    'T_LVBP': UNUSED_TAG,           # Left ventricular blood pool
    'T_AORTA': UNUSED_TAG,          # Aorta
    'T_AORTABP': UNUSED_TAG,        # Aortic blood pool
    'T_MITRALVV': UNUSED_TAG,       # Mitral valve
    'T_AORTICVV': UNUSED_TAG,       # Aortic valve
    'T_RVBP': UNUSED_TAG,           # Right ventricular blood pool
    'T_VCINF': UNUSED_TAG,          # Vena cava inferior
    'T_VCSUP': UNUSED_TAG,          # Vena cava superior
    'T_PULMARTERY': UNUSED_TAG,     # Pulmonary artery
    'T_PULMARTERYBP': UNUSED_TAG,   # Pulmonary artery blood pool
    'T_TRICUSPVV': UNUSED_TAG,      # Tricuspid valve
    'T_PULMVV': UNUSED_TAG          # Pulmonic valve
}


class ETagsParameters:
    """
    Manages etags parameter generation for UVC calculation.
    
    The etags file is a bash script that defines anatomical region tags
    for the mguvc tool. It maps mesh element tags to standardized
    anatomical identifiers that mguvc uses to solve Laplace equations.
    
    Attributes:
        type: UVC mode - 'base' (BiV), 'la' (left atrium), 'ra' (right atrium)
        tags: Dictionary mapping anatomical names to tag values
    """
    
    def __init__(self, mode: str = 'base') -> None:
        """
        Initialize etags parameters.
        
        Args:
            mode: UVC calculation mode - 'base' (BiV), 'la', or 'ra'
        
        Raises:
            ValueError: If mode is not one of the valid options
        """
        valid_modes = ['base', 'la', 'ra']
        if mode not in valid_modes:
            raise ValueError(
                f'Invalid mode for ETagsParameters. Must be one of {valid_modes}'
            )
        
        self.mode = mode
        self.tags = DEFAULT_ETAGS.copy()
        self._update_tags_for_mode()
    
    def update_mode(self, mode: str) -> None:
        """
        Change the UVC mode and update tags accordingly.
        
        Args:
            mode: New mode - 'base', 'la', or 'ra'
        """
        self.mode = mode
        self._update_tags_for_mode()
    
    def set_tag(self, tag_name: str, tag_value: int) -> None:
        """
        Manually set a specific tag value.
        
        Args:
            tag_name: Anatomical tag name (e.g., 'T_LV', 'T_RV')
            tag_value: Integer tag value from the mesh
        
        Example:
            >>> etags = ETagsParameters('base')
            >>> etags.set_tag('T_LV', 10)
            >>> etags.set_tag('T_RV', 20)
        """
        if tag_name not in self.tags:
            raise KeyError(f"Unknown tag name: {tag_name}")
        self.tags[tag_name] = tag_value
    
    def _update_tags_for_mode(self) -> None:
        """
        Update tag values based on current mode.
        
        Mode-specific behavior:
        - base: T_LV=1, T_RV=2 (biventricular)
        - la: T_LV=3 (left atrium uses LV slot)
        - ra: T_LV=4 (right atrium uses LV slot)
        
        All other tags set to UNUSED_TAG.
        """
        # Reset all tags to unused
        for tag_name in self.tags:
            if tag_name != 'T_UNUSED':
                self.tags[tag_name] = UNUSED_TAG
        
        # Set mode-specific tags
        if self.mode == 'base':
            self.tags['T_LV'] = 1
            self.tags['T_RV'] = 2
        elif self.mode == 'la':
            self.tags['T_LV'] = 3
        elif self.mode == 'ra':
            self.tags['T_LV'] = 4
    
    def generate_script_content(self) -> str:
        """
        Generate bash script content for etags file.
        
        Returns:
            Bash script as string with shebang and variable definitions
        
        Example:
            >>> etags = ETagsParameters('base')
            >>> etags.set_tag('T_LV', 10)
            >>> etags.set_tag('T_RV', 20)
            >>> script = etags.generate_script_content()
        """
        lines = ['#!/bin/bash', '']
        
        # Add header comment
        if self.mode != 'base':
            lines.append(
                f'## CHANGE ONLY THIS LABEL SO THAT T_LV = '
                f'THE LABEL OF YOUR {self.mode.upper()}'
            )
        else:
            lines.append('## ONLY CHANGE THESE LABELS TO MATCH YOUR MESH LABELS')
        
        lines.append('')
        
        # Separate used and unused tags
        tags_used = {k: v for k, v in self.tags.items() if v != UNUSED_TAG}
        tags_unused = {k: v for k, v in self.tags.items() if v == UNUSED_TAG}
        
        # Write used tags first
        for tag_name, tag_value in tags_used.items():
            lines.append(f'{tag_name}={tag_value}')
        
        lines.append('')
        
        # Write unused tags
        for tag_name, tag_value in tags_unused.items():
            lines.append(f'{tag_name}={tag_value}')
        
        return '\n'.join(lines) + '\n'
    
    def save_to_file(self, output_path: Path) -> None:
        """
        Save etags parameters as bash script.
        
        Automatically adds .sh extension if not present.
        Makes file executable (chmod +x).
        
        Args:
            output_path: Path to save the etags script
        
        Example:
            >>> etags = ETagsParameters('base')
            >>> etags.set_tag('T_LV', 10)
            >>> etags.set_tag('T_RV', 20)
            >>> etags.save_to_file(Path('mesh.etags'))
            # Creates mesh.etags.sh
        """
        # Ensure .sh extension
        if not str(output_path).endswith('.sh'):
            output_path = Path(str(output_path) + '.sh')
        
        # Write script
        content = self.generate_script_content()
        output_path.write_text(content)
        
        # Make executable
        import stat
        output_path.chmod(output_path.stat().st_mode | stat.S_IEXEC)


def write_etags_file(
    output_path: Path,
    lv_tag: int,
    rv_tag: Optional[int] = None,
    mode: str = 'base',
    custom_tags: Optional[Dict[str, int]] = None
) -> None:
    """
    Convenience function to write etags file for mguvc.
    
    Creates a bash script that maps mesh element tags to anatomical
    regions for UVC coordinate calculation.
    
    Args:
        output_path: Path to write etags script (will add .sh if needed)
        lv_tag: Element tag for LV myocardium in the mesh
        rv_tag: Element tag for RV myocardium (required for 'base' mode)
        mode: UVC mode - 'base' (BiV), 'la', or 'ra' (default: 'base')
        custom_tags: Optional dict of additional tag overrides
    
    Raises:
        ValueError: If rv_tag not provided for 'base' mode
        IOError: If file cannot be written
    
    Example:
        >>> # Biventricular UVC
        >>> write_etags_file(
        ...     Path("mesh.etags"),
        ...     lv_tag=10,
        ...     rv_tag=20,
        ...     mode='base'
        ... )
        
        >>> # Left atrial UVC
        >>> write_etags_file(
        ...     Path("la_mesh.etags"),
        ...     lv_tag=3,
        ...     mode='la'
        ... )
    """
    if mode == 'base' and rv_tag is None:
        raise ValueError("rv_tag is required for 'base' mode")
    
    # Create ETagsParameters
    etags = ETagsParameters(mode=mode)
    
    # Set primary tags
    etags.set_tag('T_LV', lv_tag)
    if rv_tag is not None:
        etags.set_tag('T_RV', rv_tag)
    
    # Apply custom tag overrides
    if custom_tags:
        for tag_name, tag_value in custom_tags.items():
            etags.set_tag(tag_name, tag_value)
    
    # Save to file
    etags.save_to_file(output_path)
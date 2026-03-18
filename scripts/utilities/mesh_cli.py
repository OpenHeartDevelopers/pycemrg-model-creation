#!/usr/bin/env python3
"""
mesh_cli.py - Command-line utilities for CARP cardiac mesh analysis.

Subcommands:
    volumes  Compute total volume per element label.

Examples:
    python scripts/utilities/mesh_cli.py volumes --input path/to/myocardium
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --labels 2 3
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --format vtk
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --input-mm --output-mm3
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------

def compute_tet_volumes(points: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """
    Vectorised tetrahedral volume: V = |det([b-a, c-a, d-a])| / 6.

    Args:
        points:       (N, 3) node coordinates.
        connectivity: (M, 4) zero-based node indices per element.

    Returns:
        (M,) array of element volumes in the same cubic units as the input coordinates.
    """
    a = points[connectivity[:, 0]]
    b = points[connectivity[:, 1]]
    c = points[connectivity[:, 2]]
    d = points[connectivity[:, 3]]
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def _load_carp(input_path: Path):
    """Returns (points, connectivity, tags) from a CARP .pts/.elem pair."""
    from pycemrg_model_creation.utilities.mesh import read_pts, read_elem, ElemType

    pts_path = input_path.with_suffix(".pts")
    elem_path = input_path.with_suffix(".elem")

    if not pts_path.exists():
        raise FileNotFoundError(f"Points file not found: {pts_path}")
    if not elem_path.exists():
        raise FileNotFoundError(f"Elements file not found: {elem_path}")

    points = read_pts(pts_path)
    elements = read_elem(elem_path, elem_type=ElemType.Tt, read_tags=True)

    return points, elements[:, :4], elements[:, 4]


def _load_vtk(input_path: Path, tag_field: str):
    """Returns (points, connectivity, tags) from a VTK unstructured grid."""
    import pyvista as pv

    vtk_path = input_path.with_suffix(".vtk")
    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK file not found: {vtk_path}")

    mesh = pv.read(vtk_path)

    if pv.CellType.TETRA not in mesh.cells_dict:
        raise ValueError(f"No tetrahedral cells found in {vtk_path.name}")

    if len(mesh.cells_dict) > 1:
        print(
            f"Warning: {vtk_path.name} contains {len(mesh.cells_dict)} cell types. "
            "Only tetrahedra are analysed. Tag indexing assumes a pure-tet mesh.",
            file=sys.stderr,
        )

    if tag_field not in mesh.cell_data:
        available = list(mesh.cell_data.keys()) or ["(none)"]
        raise ValueError(
            f"Tag field '{tag_field}' not found in cell data. "
            f"Available: {available}. Use --tag-field to specify the correct name."
        )

    connectivity = mesh.cells_dict[pv.CellType.TETRA]
    points = np.array(mesh.points)
    tags = np.array(mesh.cell_data[tag_field])

    return points, connectivity, tags


# ---------------------------------------------------------------------------
# Subcommand: volumes
# ---------------------------------------------------------------------------

def cmd_volumes(args: argparse.Namespace) -> None:
    input_path = Path(args.input)

    if args.format == "carp":
        points, connectivity, tags = _load_carp(input_path)
    else:
        points, connectivity, tags = _load_vtk(input_path, args.tag_field)

    raw_volumes = compute_tet_volumes(points, connectivity)

    # Unit conversion: raw volumes are in (input unit)³
    # μm³ → cm³: 1 cm = 1e4 μm  → 1 cm³ = 1e12 μm³
    # μm³ → mm³: 1 mm = 1e3 μm  → 1 mm³ = 1e9  μm³
    # mm³ → cm³: 1 cm = 10  mm  → 1 cm³ = 1e3  mm³
    if args.input_mm:
        scale = 1.0 if args.output_mm3 else 1e-3
        input_unit = "mm"
    else:
        scale = 1e-9 if args.output_mm3 else 1e-12
        input_unit = "μm"

    output_unit = "mm³" if args.output_mm3 else "cm³"

    all_labels = np.unique(tags)

    if args.labels:
        requested = set(args.labels)
        unknown = requested - set(all_labels.tolist())
        if unknown:
            print(f"Warning: labels not found in mesh: {sorted(unknown)}", file=sys.stderr)
        labels_to_report = sorted(requested & set(all_labels.tolist()))
    else:
        labels_to_report = sorted(all_labels.tolist())

    # Output
    label_col = max((len(str(l)) for l in labels_to_report), default=5)
    label_col = max(label_col, len("Label"))
    value_col = 16

    print(f"\nInput:  {input_path} ({input_unit})")
    print(f"Output: {output_unit}\n")
    print(f"  {'Label':>{label_col}}    {'Volume':>{value_col}} ({output_unit})")
    print(f"  {'-' * label_col}    {'-' * (value_col + 6)}")

    grand_total = 0.0
    for label in labels_to_report:
        vol = raw_volumes[tags == label].sum() * scale
        grand_total += vol
        print(f"  {label:>{label_col}}    {vol:>{value_col}.6f}")

    print(f"  {'-' * label_col}    {'-' * (value_col + 6)}")
    print(f"  {'Total':>{label_col}}    {grand_total:>{value_col}.6f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_cli.py",
        description="Command-line utilities for CARP cardiac mesh analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vol = subparsers.add_parser(
        "volumes",
        help="Compute total volume per element label.",
        description=(
            "Computes total tetrahedral volume for each element label. "
            "Coordinates default to micrometers; output defaults to cm³."
        ),
    )
    vol.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Mesh base path without extension (e.g. path/to/myocardium).",
    )
    vol.add_argument(
        "--labels",
        type=int,
        nargs="+",
        metavar="N",
        help="Labels to include. Omit to report all labels in the mesh.",
    )
    vol.add_argument(
        "--format",
        choices=["carp", "vtk"],
        default="carp",
        help="Mesh format: 'carp' reads .pts/.elem; 'vtk' reads .vtk. (default: carp)",
    )
    vol.add_argument(
        "--tag-field",
        default="elemTag",
        metavar="FIELD",
        help="Cell data field containing element tags in VTK files. (default: elemTag)",
    )
    vol.add_argument(
        "--input-mm",
        action="store_true",
        help="Input coordinates are in millimeters (default: micrometers).",
    )
    vol.add_argument(
        "--output-mm3",
        action="store_true",
        help="Output volumes in mm³ (default: cm³).",
    )
    vol.set_defaults(func=cmd_volumes)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

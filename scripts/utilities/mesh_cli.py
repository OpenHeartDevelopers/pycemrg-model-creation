#!/usr/bin/env python3
"""
mesh_cli.py - Command-line utilities for CARP cardiac mesh analysis.

Subcommands:
    volumes   Compute total volume per element label.
    topology  Report the edge-adjacency topology of a surface, and the defects
              that make `meshtool extract unreachable` disagree with it.
    split     Write that edge-adjacency split as the .partN families the
              pipeline reads back, in place of meshtool's.

Examples:
    python scripts/utilities/mesh_cli.py volumes --input path/to/myocardium
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --labels 2 3
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --format vtk
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --input-mm --output-mm3
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --total-only
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --plain --show-header
    python scripts/utilities/mesh_cli.py volumes --input path/to/mesh --plain --total-only

    python scripts/utilities/mesh_cli.py topology --input tmp/epi_endo.surfmesh.vtk
    python scripts/utilities/mesh_cli.py topology --input tmp/epi_endo.surf --pts refined/mesh.pts
    python scripts/utilities/mesh_cli.py topology --input tmp/epi_endo.surf --pts refined/mesh.pts \
        --neubc tmp/epi_endo.neubc

    python scripts/utilities/mesh_cli.py split --input tmp/epi_endo.surf \
        --pts refined/mesh.pts --prefix epi_endo_cc --neubc tmp/epi_endo.neubc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

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

    label_volumes = {label: raw_volumes[tags == label].sum() * scale for label in labels_to_report}
    grand_total = sum(label_volumes.values())

    if args.plain:
        _print_plain(input_path, label_volumes, grand_total, input_unit, output_unit, args)
    else:
        _print_table(input_path, label_volumes, grand_total, input_unit, output_unit, args)


def _print_table(input_path, label_volumes, grand_total, input_unit, output_unit, args):
    print(f"\nInput:  {input_path} ({input_unit})")
    print(f"Output: {output_unit}\n")

    if args.total_only:
        print(f"  TOTAL    {grand_total:.6f} ({output_unit})\n")
        return

    label_col = max((len(str(l)) for l in label_volumes), default=5)
    label_col = max(label_col, len("Label"))
    value_col = 16

    print(f"  {'Label':>{label_col}}    {'Volume':>{value_col}} ({output_unit})")
    print(f"  {'-' * label_col}    {'-' * (value_col + 6)}")

    for label, vol in label_volumes.items():
        print(f"  {label:>{label_col}}    {vol:>{value_col}.6f}")

    print(f"  {'-' * label_col}    {'-' * (value_col + 6)}")
    print(f"  {'Total':>{label_col}}    {grand_total:>{value_col}.6f}\n")


def _print_plain(input_path, label_volumes, grand_total, input_unit, output_unit, args):
    if args.show_header:
        if args.total_only:
            print("path,total_volume,input_unit,output_unit")
        else:
            label_headers = ",".join(f"vol_{label}" for label in label_volumes)
            print(f"path,{label_headers},total_volume,input_unit,output_unit")

    if args.total_only:
        print(f"{input_path},{grand_total:.6f},{input_unit},{output_unit}")
    else:
        label_values = ",".join(f"{vol:.6f}" for vol in label_volumes.values())
        print(f"{input_path},{label_values},{grand_total:.6f},{input_unit},{output_unit}")



# ---------------------------------------------------------------------------
# Subcommand: topology
# ---------------------------------------------------------------------------

def _load_surface(input_path: Path, pts_path: Path = None):
    """
    Returns `(points, triangles)` from either a VTK surface or a CARP `.surf`.

    The two differ in what they carry. A `.surf` is triangles alone, indexed
    into its parent mesh, so it needs that mesh's `.pts`. A `.vtk` surface is
    self-contained but its indices are its own, which is why only the `.surf`
    route can produce index files the pipeline will accept.

    Points may be `None` when a `.surf` is given without `--pts`: the topology
    is still computable, only the diagnostic mesh is not.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Surface not found: {input_path}")

    if input_path.suffix.lower() == ".vtk":
        mesh = pv.read(input_path)
        if pv.CellType.TRIANGLE not in mesh.cells_dict:
            raise ValueError(f"No triangles found in {input_path.name}")
        return np.array(mesh.points), mesh.cells_dict[pv.CellType.TRIANGLE]

    from pycemrg_model_creation.utilities.mesh import read_surf, read_pts

    triangles = read_surf(input_path)
    if pts_path is None:
        return None, triangles

    if not pts_path.exists():
        raise FileNotFoundError(f"Points file not found: {pts_path}")
    return read_pts(pts_path), triangles


def cmd_topology(args: argparse.Namespace) -> None:
    from pycemrg_model_creation.utilities.mesh import read_neubc, write_topology_vtk
    from pycemrg_model_creation.utilities.topology import analyse_surface_topology

    input_path = Path(args.input)
    pts_path = Path(args.pts) if args.pts else None
    points, triangles = _load_surface(input_path, pts_path)

    report = analyse_surface_topology(triangles)

    print(f"{input_path.name}: {triangles.shape[0]} triangles, "
          f"{len(np.unique(triangles))} vertices")
    print(report.summary())

    tags = None
    if args.neubc:
        neubc_path = Path(args.neubc)
        if not neubc_path.exists():
            raise FileNotFoundError(f"neubc file not found: {neubc_path}")
        rows = read_neubc(neubc_path)
        if rows.shape[0] != triangles.shape[0]:
            raise ValueError(
                f"{neubc_path.name} has {rows.shape[0]} rows but the surface has "
                f"{triangles.shape[0]} triangles; they do not describe each other."
            )
        tags = rows[:, 3]

    print()
    header = f"{'comp':>5} {'triangles':>10}"
    if tags is not None:
        header += "   tags"
    print(header)
    for component in range(report.n_components):
        selected = report.triangle_component == component
        line = f"{component:>5} {int(selected.sum()):>10}"
        if tags is not None:
            counts = {int(t): int((tags[selected] == t).sum())
                      for t in np.unique(tags[selected])}
            line += f"   {counts}"
        print(line)

    if report.pinch_vertices.size:
        print(f"\n{len(report.pinch_vertices)} pinch vertices, touching "
              f"{int(report.pinched_triangles(triangles).sum())} triangles.")
        fans, counts = np.unique(report.pinch_fan_counts, return_counts=True)
        for fan, count in zip(fans, counts):
            print(f"  {count} vertices carrying {fan} fans")
    if report.nonmanifold_edges.size:
        print(f"\n{len(report.nonmanifold_edges)} non-manifold edges "
              f"(shared by more than two triangles):")
        for edge in report.nonmanifold_edges[:10]:
            print(f"  {edge[0]} - {edge[1]}")

    if args.no_vtk:
        return
    if points is None:
        print("\nNo --pts given, so no diagnostic mesh was written.", file=sys.stderr)
        return

    output_path = Path(args.output) if args.output else Path(
        f"{input_path.parent / input_path.name.rsplit('.', 1)[0]}_topology.vtk"
    )
    write_topology_vtk(points, triangles, report, output_path)
    if tags is not None:
        mesh = pv.read(output_path)
        mesh.cell_data["elemTag"] = tags.astype(np.int8)
        mesh.save(output_path, binary=False)
    print(f"\nWrote {output_path}")
    print("  colour by 'component' -> the proposed split")
    print("  colour by 'pinch'     -> the defects")



# ---------------------------------------------------------------------------
# Subcommand: split
# ---------------------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> None:
    from pycemrg_model_creation.utilities.mesh import write_accepted_split
    from pycemrg_model_creation.utilities.topology import SurfaceRoles

    role_args = (args.epi, args.lv_endo, args.rv_endo)
    if any(r is not None for r in role_args) and None in role_args:
        raise SystemExit("Give all of --epi, --lv-endo and --rv-endo, or none.")

    input_path = Path(args.input)
    if not args.neubc:
        print(
            "No --neubc given, so every triangle is tagged 1 and the elemTag "
            "meshtool writes is lost.\nThat distinction is what shows the septum "
            "inside the RV endocardium. Pass the\nmatching .neubc to keep it.",
            file=sys.stderr,
        )

    directory = Path(args.output_dir) if args.output_dir else input_path.parent
    try:
        # Writes the parts and the sidecar that lets the surface stage accept
        # them; see `write_accepted_split` for why the sidecar exists.
        split = write_accepted_split(
            surface_path=input_path,
            pts_path=Path(args.pts),
            base=directory / args.prefix,
            neubc_path=Path(args.neubc) if args.neubc else None,
            expect=args.expect,
            write_vtk=not args.no_vtk,
            roles=SurfaceRoles(*role_args) if args.epi is not None else None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    components, tags = split.components, split.tags
    print(split.topology.summary())

    print()
    for component in components:
        summary = (f"  {component.name}: {component.n_triangles} triangles, "
                   f"{len(component.vertices)} vertices")
        if tags is not None:
            counts = {int(t): int((tags[component.triangles] == t).sum())
                      for t in np.unique(tags[component.triangles])}
            summary += f", tags {counts}"
        print(summary)

    overlaps = [(c, other, shared)
                for c in components
                for other, shared in c.shared_with.items()
                if c.name < components[other].name]
    if overlaps:
        print("\nShared vertices, where sheets pinch:")
        for component, other, shared in overlaps:
            print(f"  {component.name} n {components[other].name}: "
                  f"{len(shared)} nodes")
        print(
            "\nThese nodes appear in both components' .vtx sets. That is only a "
            "problem\nif both become mguvc boundary inputs: it reads base, lvendo, "
            "rvendo and\nrvsept, and derives epi. Check which components these are "
            "before relying\non the result.",
            file=sys.stderr,
        )

    print(f"\nWrote {len(components)} components and {args.prefix}.split.json "
          f"to {directory}")
    print("Re-run the surface stage. It uses these parts for as long as the "
          "surface it extracts\nmatches the one split here.")


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
    vol.add_argument(
        "--total-only",
        action="store_true",
        help="Report only the grand total volume, not per-label breakdown.",
    )
    vol.add_argument(
        "--plain",
        action="store_true",
        help="CSV output suitable for piping: path,vol_1,...,total_volume,input_unit,output_unit.",
    )
    vol.add_argument(
        "--show-header",
        action="store_true",
        help="Print a CSV header row (only used with --plain).",
    )
    vol.set_defaults(func=cmd_volumes)

    topo = subparsers.add_parser(
        "topology",
        help="Report a surface's edge-adjacency components and topological defects.",
        description=(
            "Reads a surface under edge adjacency -- two triangles connect only "
            "if they share an edge -- and names the defects that make "
            "`meshtool extract unreachable`, which follows nodes, disagree. Use "
            "it on any surface that came back with the wrong number of "
            "components, or on one that merely looks suspicious."
        ),
    )
    topo.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="A .vtk surface (self-contained) or a CARP .surf file, extension included.",
    )
    topo.add_argument(
        "--pts",
        metavar="PATH",
        help="Parent mesh .pts, required with a .surf input to write a diagnostic mesh.",
    )
    topo.add_argument(
        "--neubc",
        metavar="PATH",
        help="Matching .neubc, to break each component down by element tag.",
    )
    topo.add_argument(
        "--output",
        metavar="PATH",
        help="Diagnostic .vtk to write. (default: <input>_topology.vtk)",
    )
    topo.add_argument(
        "--no-vtk",
        action="store_true",
        help="Report only; write nothing.",
    )
    topo.set_defaults(func=cmd_topology)

    spl = subparsers.add_parser(
        "split",
        help="Write the edge-adjacency split as .partN families.",
        description=(
            "Writes each edge-adjacency component as the CARP submesh family "
            "`meshtool extract unreachable` would have written, so the pipeline "
            "reads them back with no change. Nothing is dropped: a vertex where "
            "two sheets pinch belongs to both, and the overlap is reported "
            "rather than resolved."
        ),
    )
    spl.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="The CARP .surf that was split, extension included.",
    )
    spl.add_argument(
        "--pts",
        required=True,
        metavar="PATH",
        help="Parent mesh .pts, whose indexing the .surf uses.",
    )
    spl.add_argument(
        "--prefix",
        required=True,
        metavar="NAME",
        help="Base name for the parts, e.g. epi_endo_cc -> epi_endo_cc.part0.",
    )
    spl.add_argument(
        "--neubc",
        metavar="PATH",
        help="Matching .neubc, whose tags are carried into each part's .elem and .vtk.",
    )
    spl.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Where to write. (default: beside the input)",
    )
    spl.add_argument(
        "--expect",
        type=int,
        metavar="N",
        help="Refuse unless edge adjacency finds at least N components.",
    )
    spl.add_argument(
        "--no-vtk",
        action="store_true",
        help="Skip the per-component .vtk.",
    )
    for flag, role in (("--epi", "epicardium"), ("--lv-endo", "LV endocardium"),
                       ("--rv-endo", "RV endocardium")):
        spl.add_argument(
            flag,
            type=int,
            metavar="N",
            help=f"Part number of the {role}. Recorded in the sidecar so the "
                 "surface stage uses it instead of its geometric test. Give "
                 "all three or none.",
        )
    spl.set_defaults(func=cmd_split)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

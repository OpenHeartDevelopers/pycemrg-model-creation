# src/pycemrg_carp/tools.py

import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence, Union, Optional

# This library's core dependency is on pycemrg's system runners
from pycemrg.system import CarpRunner
from pycemrg.system import CommandRunner

# Define constants at the module level for clarity and reusability
DEFAULT_FIBRE_ANGLES: Dict[str, float] = {
    "alpha_endo": 60,
    "alpha_epi": -60,
    "beta_endo": -65,
    "beta_epi": 25,
}


class MeshDataOperation(Enum):
    """Defines operations for command: [meshtool insert mesh]"""

    ONLY_TAGS = 0
    ONLY_FIBRES = 1
    BOTH = 2


class CarpWrapper:
    """
    A high-level SDK for common CARPentry command-line tools.

    This class provides Pythonic wrappers around tools like GlRuleFibres,
    GlVTKConvert, and igbextract, abstracting away the command-line details.
    """

    def __init__(self, carp_runner: CarpRunner):
        """
        Initializes the CarpWrapper SDK.

        Args:
            carp_runner (CarpRunner): An initialized CarpRunner instance that
                knows how to set up the CARP environment and execute commands.
        """
        self.runner = carp_runner
        self.logger = logging.getLogger(__name__)

    def gl_rule_fibres(
        self,
        mesh_name: Path,
        uvc_apba: Path,
        uvc_epi: Path,
        uvc_lv: Path,
        uvc_rv: Path,
        output_name: Path,
        angles: Dict[str, float] = None,
        fibre_type: str = "biv",
    ) -> None:
        """
        Wrapper for the GlRuleFibres command.

        Args:
            mesh_name: Path to the input mesh.
            uvc_apba: Path to the apicobasal UVC coordinates.
            uvc_epi: Path to the epicardial UVC coordinates.
            uvc_lv: Path to the left ventricle UVC coordinates.
            uvc_rv: Path to the right ventricle UVC coordinates.
            output_name: Path for the output fibre data.
            angles: Dictionary of fibre angles. Defaults to module constant.
            fibre_type: Fibre type, e.g., 'biv'.
        """
        # Use default angles if none are provided
        angles_to_use = angles if angles is not None else DEFAULT_FIBRE_ANGLES

        self.logger.info(f"Generating rule-based fibres for {mesh_name.name}")
        cmd = [
            "GlRuleFibres",
            "-m",
            mesh_name,
            "-t",
            fibre_type,
            "-a",
            uvc_apba,
            "-e",
            uvc_epi,
            "-l",
            uvc_lv,
            "-r",
            uvc_rv,
            "-o",
            output_name,
        ]
        for key, value in angles_to_use.items():
            cmd.extend([f"--{key}", str(value)])

        self.runner.run(cmd, expected_outputs=[output_name])
        self.logger.info(f"Successfully created fibres at {output_name}")

    def gl_vtk_convert(
        self,
        mesh_name: Path,
        output_name: Path,
        node_data: Sequence[str] = (),
        elem_data: Sequence[str] = (),
        trim_names: bool = True,
    ) -> None:
        """Wrapper for the GlVTKConvert command."""
        self.logger.info(f"Converting {mesh_name.name} to VTK format")
        cmd: List[Union[str, Path]] = [
            "GlVTKConvert",
            "-m",
            mesh_name,
            "-o",
            output_name,
        ]
        for data in node_data:
            cmd.extend(["-n", data])
        for data in elem_data:
            cmd.extend(["-e", data])
        if trim_names:
            cmd.append("--trim-names")

        self.runner.run(cmd, expected_outputs=[output_name])
        self.logger.info(f"Successfully converted mesh to {output_name}")

    def igb_extract(
        self,
        igb_file: Path,
        output_file: Path,
        output_format: str = "ascii",
        first_frame: int = 0,
        last_frame: int = 0,
    ) -> None:
        """
        Wrapper for the igbextract command.

        Args:
            igb_file: Path to the input .igb file.
            output_file: Path for the extracted output.
            output_format: Output format (e.g., 'ascii', 'ascii_1pLn').
            first_frame: First frame to extract (inclusive).
            last_frame: Last frame to extract (inclusive).
        """
        self.logger.info(f"Extracting frames from {igb_file.name}")
        cmd = [
            "igbextract",
            igb_file,
            "-o",
            output_format,
            f"--f0={first_frame}",
            f"--f1={last_frame}",
            "-O",
            output_file,
        ]
        self.runner.run(cmd, expected_outputs=[output_file])
        self.logger.info(f"Successfully extracted data to {output_file}")

    def gl_elem_centers(
        self, meshname: Path, output: Path, fmt: str = "0", pstrat: str = "0"
    ) -> None:
        """
        Wrapper for the GlElemCenters command.

        Args:
            meshname: Path to the input mesh.
            output: Path for the output element centers.
            fmt: Output format (default is '0').
            pstrat: Partitioning strategy (default is '0').
        """
        self.logger.info(f"Calculating element centers for {meshname.name}")
        cmd = [
            "GlElemCenters",
            "-m",
            meshname,
            f"-f {fmt}",
            f"-p {pstrat}",
            "-o",
            output,
        ]
        self.runner.run(cmd, expected_outputs=[output])
        self.logger.info(f"Successfully calculated element centers at {output}")

    def carp_pt(
        self, parfile: Path, simID: Path, meshname: Path, stim0: Path, stim1: Path
    ) -> None:
        """
        Wrapper for the carp.pt command.
        Args:
            parfile: Path to the parameter file.
            simID: Simulation ID.
            meshname: Path to the mesh name.
            stim0: Path to the first stimulus vertex file.
            stim1: Path to the second stimulus vertex file.
        """
        cmd = [
            "carp.pt",
            f"+F {parfile}",
            f"-simID {simID}",
            f"-meshname {meshname}",
            f"-stimulus[0].vtx_file {stim0}",
            f"-stimulus[1].vtx_file {stim1}",
        ]
        self.logger.info(f"Running carp.pt with simulation ID {simID}")
        self.runner.run(cmd)

    def ek_batch(
        self, meshpath: Path, initfiles_str: str, tags_str: Path, tagfile=""
    ) -> None:
        """
        Wrapper for the ekbatch command.
        Args:
            meshpath: Path to the mesh.
            initfiles_str: Space-separated string of initialization files.
            tags_str: Space-separated string of tags.
            tagfile: Optional path to a tag file.
        """
        cmd = ["ekbatch", f"{meshpath}", f"{initfiles_str}", f"{tags_str}"]

        if tagfile:
            cmd += f" --tagfile {tagfile}"

        outputs_list = [f"{ofile}.dat" for ofile in initfiles_str.split(" ")]
        self.runner.run(cmd, outputs_list)


# ... (imports and CarpWrapper class remain the same) ...


class MeshtoolWrapper:
    """
    A high-level SDK for the meshtool command-line utility.

    This class provides Pythonic wrappers for common meshtool operations like
    extracting, converting, and manipulating meshes. It can be initialized to
    use a system-wide 'meshtool' or one from a CARPentry environment.
    """

    def __init__(
        self,
        runner: Union[CommandRunner, CarpRunner],
        meshtool_install_dir: Optional[Path] = None,
    ):
        """
        Initializes the MeshtoolWrapper with a specific command runner.

        Args:
            runner: An initialized runner that will be used for execution.
            meshtool_install_dir: Path to the root meshtool installation directory.
                                  Required to locate standalone tools like
                                  simplify_tag_topology.
        """
        self.runner = runner
        self.logger = logging.getLogger(__name__)
        
        # Configure standalone tools
        self._simplify_topology_cmd: Optional[Path] = None
        self._simplify_topology_available = False

        if meshtool_install_dir:
            self._simplify_topology_cmd = meshtool_install_dir / "standalones/simplify_tag_topology"
            self._simplify_topology_available = self._simplify_topology_cmd.is_file()

            if not self._simplify_topology_available:
                self.logger.warning(
                    f"simplify_tag_topology not found at {self._simplify_topology_cmd}. "
                    "Topology simplification will not be available."
                )

    @property
    def is_simplify_topology_available(self) -> bool:
        """Returns True if the simplify_tag_topology command is available."""
        return self._simplify_topology_available

    @classmethod
    def from_system_path(
        cls, logger: Optional[logging.Logger] = None
    ) -> "MeshtoolWrapper":
        """
        Creates a MeshtoolWrapper instance that uses a system-wide 'meshtool'.

        This assumes 'meshtool' is available in the shell's PATH.

        Args:
            logger: An optional logger instance.
        """
        # Uses the generic CommandRunner from pycemrg
        runner = CommandRunner(logger=logger)
        return cls(runner)

    @classmethod
    def from_carp_runner(cls, carp_runner: CarpRunner) -> "MeshtoolWrapper":
        """
        Creates a MeshtoolWrapper instance that uses the 'meshtool' from a
        CARPentry environment.

        Args:
            carp_runner: An initialized CarpRunner that provides the environment.
        """
        # The provided CarpRunner already has its environment configured.
        return cls(carp_runner)

    # --- Refactored Methods (with improvements) ---

    def extract_mesh(
        self,
        input_mesh_path: Path,
        output_submesh_path: Path,
        tags: Sequence[Union[str, int]],
        ifmt: str = "carp_txt",
        normalise: bool = False,
    ) -> None:
        """Extracts a submesh based on element tags. Wrapper for `meshtool extract mesh`."""
        tags_str = ",".join(map(str, tags))
        self.logger.info(f"Extracting tags [{tags_str}] from {input_mesh_path.name}")

        cmd: List[Union[str, Path]] = [
            "meshtool",
            "extract",
            "mesh",
            f"-msh={input_mesh_path}",
            f"-tags={tags_str}",
            f"-submsh={output_submesh_path}",
            f"-ifmt={ifmt}",
        ]
        if normalise:
            cmd.append("-norm")

        exts = ["elem", "pts"] if ifmt == "carp_txt" else ["vtk"]
        expected = [output_submesh_path.with_suffix(f".{ext}") for ext in exts]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info(f"Successfully created submesh at {output_submesh_path}")

    def extract_surface(
        self,
        input_mesh_path: Path,
        output_surface_path: Path,
        ofmt: str = "vtk",
        op_tag_base: Optional[str] = None,
    ) -> None:
        """Extracts a surface from a volume mesh. Wrapper for `meshtool extract surface`."""
        self.logger.info(f"Extracting surface from {input_mesh_path.name}")
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "extract",
            "surface",
            f"-msh={input_mesh_path}",
            f"-surf={output_surface_path}",
            f"-ofmt={ofmt}",
        ]
        if op_tag_base:
            cmd.append(f"-op={op_tag_base}")

        exts = ["pts", "elem"] if ofmt == "carp_txt" else ["vtk"]
        expected = [output_surface_path.with_suffix(f".surfmesh.{ext}") for ext in exts]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info(
            f"Successfully extracted surface to {output_surface_path}.surfmesh.*"
        )

    def extract_unreachable(
        self, 
        input_mesh_path: Path, 
        submsh_path: Path, 
        ofmt: str = 'vtk', 
        ifmt: str = '',
    ) -> None: 
        self.logger.info(f"Extracting unreachable components from {input_mesh_path.name}")
        cmd: List[Union[str, Path]] = [
            "meshtool", "extract", "unreachable",
            f"-msh={input_mesh_path}"
        ]
        if ifmt:
            cmd.append(f"-ifmt={ifmt}")
        
        cmd.append(f"-ofmt={ofmt}")
        cmd.append(f"-submsh={submsh_path}")

        self.runner.run(cmd)  # TODO: find expected outputs and include here
        self.logger.info(f"Successfully extracted unreachable components to {submsh_path}")

    def convert(
        self,
        input_mesh_path: Path,
        output_mesh_path: Path,
        ofmt: str = "vtk",
        ifmt: Optional[str] = None,
    ) -> None:
        """Converts a mesh from one format to another. Wrapper for `meshtool convert`."""
        self.logger.info(
            f"Converting {input_mesh_path} to {ofmt} at {output_mesh_path}"
        )
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "convert",
            f"-imsh={input_mesh_path}",
            f"-omsh={output_mesh_path}",
            f"-ofmt={ofmt}",
        ]
        if ifmt:
            cmd.append(f"-ifmt={ifmt}")

        if ofmt in ("carp_txt", "carp_bin"):
            exts = ["bpts", "belem"] if ofmt == "carp_bin" else ["pts", "elem"]
            expected = [output_mesh_path.with_suffix(f".{ext}") for ext in exts]
        else:
            expected = [output_mesh_path.with_suffix(f".{ofmt}")]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info("Successfully converted mesh.")

    def smooth(
        self,
        input_mesh_path: Path,
        output_mesh_path: Path,
        smoothing_params: str,
        tags: Optional[Sequence[Union[str, int]]] = None,
        ifmt: str = "carp_txt",
        ofmt: str = "carp_txt",
    ) -> None:
        """Smooths a mesh. Wrapper for `meshtool smooth mesh`."""
        self.logger.info(
            f"Smoothing {input_mesh_path.name} with params: {smoothing_params}"
        )
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "smooth",
            "mesh",
            f"-msh={input_mesh_path}",
            f"-smth={smoothing_params}",
            f"-outmsh={output_mesh_path}",
            f"-ifmt={ifmt}",
            f"-ofmt={ofmt}",
        ]
        if tags:
            cmd.append(f"-tags={','.join(map(str, tags))}")

        exts = ["elem", "pts"] if ofmt == "carp_txt" else ["vtk"]
        expected = [output_mesh_path.with_suffix(f".{ext}") for ext in exts]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info(f"Successfully created smoothed mesh at {output_mesh_path}")

    def interpolate(
        self, mesh_path: Path, input_data_path: Path, output_data_path: Path, mode: str
    ) -> None:
        """Interpolates data between nodes and elements. Wrapper for `meshtool interpolate`."""
        if mode not in ["node2elem", "elem2node"]:
            raise ValueError("Interpolation mode must be 'node2elem' or 'elem2node'")

        self.logger.info(
            f"Interpolating {input_data_path.name} to {output_data_path.name} ({mode})"
        )
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "interpolate",
            mode,
            f"-omsh={mesh_path}",
            f"-idat={input_data_path}",
            f"-odat={output_data_path}",
        ]
        self.runner.run(cmd, expected_outputs=[output_data_path])
        self.logger.info("Interpolation successful.")

    def insert_data(
        self,
        target_mesh_path: Path,
        source_submesh_path: Path,
        source_data_path: Path,
        output_data_path: Path,
        mode: str,
    ) -> None:
        """Inserts data from a submesh into a larger mesh. Wrapper for `meshtool insert data`."""
        self.logger.info(
            f"Inserting data from {source_data_path.name} into {target_mesh_path.name}"
        )
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "insert",
            "data",
            f"-msh={target_mesh_path}",
            f"-submsh={source_submesh_path}",
            f"-submsh_data={source_data_path}",
            f"-odat={output_data_path}",
            f"-mode={mode}",
        ]
        self.runner.run(cmd, expected_outputs=[output_data_path])
        self.logger.info(f"Successfully created data file at {output_data_path}")

    def node2elem(
        self,
        target_mesh_path: Path,
        source_submesh_path: Path,
        source_data_path: Path,
        output_data_path: Path,
    ) -> None:
        """
        insert_data on mode 'node2elem'
        """
        self.insert_data(
            target_mesh_path=target_mesh_path,
            source_submesh_path=source_submesh_path,
            source_data_path=source_data_path,
            output_data_path=output_data_path,
            mode="node2elem",
        )

    def elem2node(
        self,
        target_mesh_path: Path,
        source_submesh_path: Path,
        source_data_path: Path,
        output_data_path: Path,
    ) -> None:
        """
        insert_data on mode 'elem2node'
        """
        self.insert_data(
            target_mesh_path=target_mesh_path,
            source_submesh_path=source_submesh_path,
            source_data_path=source_data_path,
            output_data_path=output_data_path,
            mode="elem2node",
        )

    def insert_submesh(
        self,
        target_mesh_path: Path,
        source_submesh_path: Path,
        output_mesh_path: Path,
        ofmt: str,
    ) -> None:
        """Inserts a submesh into a larger mesh. Wrapper for `meshtool insert submesh`."""
        self.logger.info(
            f"Inserting {source_submesh_path.name} into {target_mesh_path.name}"
        )
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "insert",
            "submesh",
            f"-msh={target_mesh_path}",
            f"-submsh={source_submesh_path}",
            f"-outmsh={output_mesh_path}",
            f"-ofmt={ofmt}",
        ]
        exts = ["pts", "elem"] if ofmt == "carp_txt" else ["vtk"]
        expected = [output_mesh_path.with_suffix(f".{ext}") for ext in exts]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info(f"Successfully created combined mesh at {output_mesh_path}")

    def insert_meshdata(
        self,
        mesh_from: Path,
        mesh_into: Path,
        operation: MeshDataOperation,
        output_mesh: Path,
    ) -> None:
        """
        insert meshdata: the fiber and tag data of a mesh is inserted into another mesh
                 based on vertex locations
        """
        msg = (
            f"Inserting fibre and tag data from {mesh_from.name} into {mesh_into.name}"
        )
        self.logger.info(msg)
        op_dict = {"only_tags": 0, "only_fibres": 1, "both": 2}
        if operation not in op_dict.keys():
            msg = f"Operation {operation} not supported, choose from [{op_dict.keys()}]"
            self.logger.error(msg)
            raise ValueError(msg)

        cmd: List[Union[str, Path]] = [
            "meshtool",
            "insert",
            "meshdata",
            f"-msh={mesh_from}",
            f"-imsh={mesh_into}",
            f"-op={op_dict[operation]}",
            f"-outmsh={output_mesh}",
        ]
        self.runner.run(cmd)  # TODO: find expected outputs and include here

    def map(
        self,
        submesh_path: Path,
        files_list: List[str],
        output_folder: Path,
        mode: str = "m2s",
    ) -> None:
        msg = f"Mapping {len(files_list)} onto {submesh_path.name}"
        self.logger.info(msg)
        self.logger.debug(f"Mapping files {[f.name for f in files_list]}")

        if mode not in ["m2s", "s2m"]:
            msg = f'Mode {mode} not supported, use "m2s" or "s2m"'
            self.logger.error(msg)
            raise ValueError(msg)

        files_str = ",".join(files_list)
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "map",
            f"-submsh={submesh_path}",
            f"-files={files_str}",
            f"-outdir={output_folder}",
            f"-mode={mode}",
        ]
        self.runner.run(cmd)  # TODO: find expected outputs and include
        self.logger.info(f"Successfully mapped files to {output_folder}")

    def generate_fibres(
        self,
        mesh_path: Path,
        output_mesh_path: Path,
        num_fibre_directions: int = 2,
    ) -> None:
        """
        generate fibres: generate default fibers for a given mesh file
        """
        self.logger.info(f"Generating default fibre files in {output_mesh_path}")
        cmd: List[Union[str, Path]] = [
            "meshtool",
            "generate",
            "fibres",
            f"-msh={mesh_path}",
            f"-outmsh={output_mesh_path}",
            f"-op={num_fibre_directions}",
        ]
        expected_output = output_mesh_path.with_suffix(".vtk")
        self.runner.run(cmd, [expected_output])

        self.logger.info(f"Successfully generated fibres at {expected_output}")

    def simplify_topology(
        self,
        input_mesh_path: Path,
        output_mesh_path: Path,
        neighbors: int = 50,
        ifmt: str = "carp_txt",
        ofmt: str = "carp_txt",
    ) -> None:
        """
        Simplifies the topology of a mesh using the simplify_tag_topology tool.

        Args:
            input_mesh_path: Base path to the input mesh.
            output_mesh_path: Base path for the output mesh.
            neighbors: Number of neighbors to consider for simplification.
            ifmt: Input format of the mesh.
            ofmt: Output format for the mesh.

        Raises:
            RuntimeError: If the simplify_tag_topology command is not available.
        """
        if not self.is_simplify_topology_available:
            raise RuntimeError(
                "Cannot simplify topology: 'simplify_tag_topology' command not found. "
                "Ensure 'meshtool_install_dir' is provided during initialization."
            )

        self.logger.info(f"Simplifying topology for {input_mesh_path.name}")
        cmd: List[Union[str, Path]] = [
            self._simplify_topology_cmd,
            f"-msh={input_mesh_path}",
            f"-outmsh={output_mesh_path}",
            f"-neigh={neighbors}",
            f"-ifmt={ifmt}",
            f"-ofmt={ofmt}",
        ]

        exts = ["elem", "pts"] if ofmt == "carp_txt" else [ofmt]
        expected = [output_mesh_path.with_suffix(f".{ext}") for ext in exts]
        self.runner.run(cmd, expected_outputs=expected)
        self.logger.info(
            f"Successfully created simplified mesh at {output_mesh_path}"
        )



class Meshtools3DWrapper:
    """
    A low-level wrapper for the meshtools3d binary.

    This wrapper's sole responsibility is to execute the meshtools3d command
    with a given parameter file. It does not create the parameter file.
    """

    def __init__(self, runner: CommandRunner, meshtools3d_path: Path):
        """
        Initializes the wrapper.

        Args:
            runner: A CommandRunner to execute the process.
            meshtools3d_path: The absolute path to the meshtools3d binary.
        """
        self.runner = runner
        self.meshtools3d_path = meshtools3d_path
        self.logger = logging.getLogger(__name__)

        if not self.meshtools3d_path.is_file():
            raise FileNotFoundError(
                f"meshtools3d binary not found at the specified path: "
                f"{self.meshtools3d_path}"
            )

    def run(self, parameter_file: Path, expected_outputs: List[Path]) -> None:
        """
        Executes the meshtools3d binary with a given parameter file.

        Args:
            parameter_file: Path to the .par file for configuration.
            expected_outputs: A list of files the command is expected to create.
        """
        if not parameter_file.is_file():
            raise FileNotFoundError(f"Parameter file not found: {parameter_file}")

        self.logger.info(
            f"Executing meshtools3d with parameter file: {parameter_file.name}"
        )
        cmd = [str(self.meshtools3d_path), "-f", str(parameter_file)]

        self.runner.run(cmd=cmd, expected_outputs=expected_outputs)
        self.logger.info("meshtools3d execution completed successfully.")

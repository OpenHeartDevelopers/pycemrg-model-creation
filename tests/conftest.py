import pytest
import os
from pathlib import Path

# This file is automatically discovered by pytest.
# Fixtures defined here are available to all tests.

@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """
    A pytest fixture that provides the root path to the integration test data.

    It reads the path from the 'PYCEMRG_TEST_DATA_ROOT' environment variable.
    If the variable is not set, the tests that depend on this fixture will be
    skipped with a clear message.
    """
    data_path_str = os.environ.get("PYCEMRG_TEST_DATA_ROOT")

    if not data_path_str:
        pytest.skip(
            "Skipping integration tests: PYCEMRG_TEST_DATA_ROOT environment variable not set."
        )

    data_path = Path(data_path_str)

    if not data_path.is_dir():
        pytest.fail(
            f"PYCEMRG_TEST_DATA_ROOT path does not exist or is not a directory: {data_path}"
        )

    return data_path

@pytest.fixture(scope="session")
def test_meshtool_root() -> Path:
    """
    User can set this to point to a custom Meshtool installation for testing.

    PYCEMRG_MESHTOOL_DIR environment variable can be used to specify the path.
    """

    meshtool_path_str = os.environ.get("PYCEMRG_MESHTOOL_DIR")

    if not meshtool_path_str:
        pytest.skip(
            "Skipping integration tests: PYCEMRG_MESHTOOL_DIR environment variable not set."
        )

    meshtool_path = Path(meshtool_path_str)

    if not meshtool_path.is_dir():
        pytest.fail(
            f"PYCEMRG_MESHTOOL_DIR path does not exist or is not a directory: {meshtool_path}"
        )

    return meshtool_path

@pytest.fixture(scope="session")
def test_m3d_root() -> Path:
    """
    User can set this to point to a custom Meshtools3D installation for testing.

    PYCEMRG_MESHTOOLS3D_DIR environment variable can be used to specify the path.
    """

    m3d_path_str = os.environ.get("PYCEMRG_MESHTOOLS3D_DIR")

    if not m3d_path_str:
        pytest.skip(
            "Skipping integration tests: PYCEMRG_MESHTOOLS3D_DIR environment variable not set."
        )

    m3d_path = Path(m3d_path_str)

    if not m3d_path.is_dir():
        pytest.fail(
            f"PYCEMRG_MESHTOOLS3D_DIR path does not exist or is not a directory: {m3d_path}"
        )

    return m3d_path

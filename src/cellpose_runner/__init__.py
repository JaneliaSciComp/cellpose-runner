"""Run CellPose with formalized configuration logging and visualization."""

from importlib.metadata import PackageNotFoundError, version

from ._config import CellposeConfig
from ._run import prepare_run, run, segment
from ._script_support import run_with_logging

try:
    __version__ = version("cellpose-runner")
except PackageNotFoundError:  # package is not installed
    __version__ = "uninstalled"

__all__ = [
    "CellposeConfig",
    "__version__",
    "prepare_run",
    "run",
    "run_with_logging",
    "segment",
]

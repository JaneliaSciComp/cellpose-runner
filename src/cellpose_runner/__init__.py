"""Run CellPose with formalized configuration logging and visualization."""

from importlib.metadata import PackageNotFoundError, version

from ._config import CellposeConfig, ModelConfig, NormalizeConfig
from ._neuroglancer import neuroglancer_url, serve_view, view
from ._report import fileglancer_url, report
from ._run import prepare_run, run, segment
from ._script_support import LoadVolume, cli_main, resolve_run_dir, run_with_logging

try:
    __version__ = version("cellpose-runner")
except PackageNotFoundError:  # package is not installed
    __version__ = "uninstalled"

__all__ = [
    "CellposeConfig",
    "LoadVolume",
    "ModelConfig",
    "NormalizeConfig",
    "__version__",
    "cli_main",
    "fileglancer_url",
    "neuroglancer_url",
    "prepare_run",
    "report",
    "resolve_run_dir",
    "run",
    "run_with_logging",
    "segment",
    "serve_view",
    "view",
]

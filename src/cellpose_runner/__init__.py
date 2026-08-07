"""Run CellPose with formalized configuration logging and visualization."""

from importlib.metadata import PackageNotFoundError, version

from ._config import CellposeConfig

try:
    __version__ = version("cellpose-runner")
except PackageNotFoundError:  # package is not installed
    __version__ = "uninstalled"

__all__ = ["CellposeConfig", "__version__"]

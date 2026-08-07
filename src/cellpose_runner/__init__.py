"""Run CellPose with formalized configuration logging and visualization."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellpose-runner")
except PackageNotFoundError:  # package is not installed
    __version__ = "uninstalled"

__all__ = ["__version__"]

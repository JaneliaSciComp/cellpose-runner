import shutil
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Any

import tomli_w

from cellpose_runner._config import CellposeConfig

CONFIG_FILENAME = "config.toml"
LOCK_FILENAME = "uv.lock"

# The top-level package directory, resolved from the package rather than from
# this module's location, so moving modules around does not shift the search
# below.
_PACKAGE_ROOT = Path(files(__package__.split(".")[0])).resolve()  # type: ignore[arg-type]


def _find_lock_file() -> Path:
    """Find the `uv.lock` describing the environment this run executes in.

    This is whichever uv project the package is installed under -- this
    repository when working from a clone, or the consuming project when
    installed as a dependency. Either way it pins the versions that actually
    did the segmenting.

    Raises:
        RuntimeError: If there is no lock file, so the environment could not be
            recorded.
    """
    for directory in _PACKAGE_ROOT.parents:
        candidate = directory / LOCK_FILENAME
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"No {LOCK_FILENAME} found above {_PACKAGE_ROOT}. Every run records the environment "
        "that produced it, so cellpose_runner must be run from a uv-managed project "
        "(`uv run` / `uv sync`)."
    )


def _package_version() -> str | None:
    """The installed version.

    Working from a clone, `uv.lock` records this package as `editable = "."`
    with no version, so this is the only record of the code that ran --
    setuptools-scm embeds the commit in it (`0.1.dev4+g6037df459`). Installed as
    a dependency, the lock pins it properly and this agrees with the lock.
    """
    try:
        return version("cellpose-runner")
    except PackageNotFoundError:
        return None


def write_run_config(
    run_dir: Path,
    config: CellposeConfig,
    run_name: str,
    input_shape: tuple[int, ...],
    input_dtype: str,
) -> Path:
    """Write `config.toml` and copy `uv.lock` into `run_dir`.

    The resolved config is written in full, including fields left at their
    default, so the file records what ran rather than what was typed.

    Args:
        run_dir: The run directory to write into.
        config: The resolved segmentation parameters.
        run_name: The run's name, as returned by `create_run_dir`.
        input_shape: Shape of the volume being segmented.
        input_dtype: Dtype of the volume being segmented.

    Returns:
        The path to the written config file.

    Raises:
        RuntimeError: If no `uv.lock` can be found, so the environment that
            produced the run could not be recorded.
    """
    run: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
    }

    # TOML cannot represent null, so anything unavailable is omitted rather
    # than recorded as empty. Reading the file back yields the same defaults.
    if (package_version := _package_version()) is not None:
        run["cellpose_runner_version"] = package_version

    shutil.copy(_find_lock_file(), run_dir / LOCK_FILENAME)

    cellpose = {k: v for k, v in config.model_dump().items() if v is not None}

    config_path = run_dir / CONFIG_FILENAME
    with config_path.open("wb") as f:
        tomli_w.dump({"cellpose": cellpose, "run": run}, f)
    return config_path

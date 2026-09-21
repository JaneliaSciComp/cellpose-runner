"""Shared LSF submission helpers for submit_cluster.py / submit_cluster_views.py.

Not a script itself -- imported by the two scripts above, both of which call
`prepare` in-process (cheap: memmaps the volume just for its shape, no pixel
data read) before any GPU job exists, so LSF's own -o/-e logs can point
straight at the run directory, and `bsub` shells out to submit the GPU/CPU
jobs themselves.
"""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

from cellpose_runner import prepare_run
from cellpose_runner._config import CellposeConfig

PKG_DIR = Path(__file__).resolve().parent.parent
SCRIPT = PKG_DIR / "scripts" / "run_timepoint.py"

# run_timepoint.py is a standalone script, not an importable package, so its
# load_volume is loaded directly from its file path here (same approach as
# serve_view.py).
_spec = importlib.util.spec_from_file_location("run_timepoint", SCRIPT)
assert _spec is not None and _spec.loader is not None
_run_timepoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_timepoint)
load_volume = _run_timepoint.load_volume


def uv_run_cmd(mode: str) -> list[str]:
    """The `uv run` prefix for invoking run_timepoint.py's `segment`/`run-view`/`consolidate`.

    three_d_dino needs cellpose3d (torch/opencv/etc.), an optional `dino`
    extra -- every other mode runs fine without it, so it's only added when
    asked for.
    """
    cmd = ["uv", "run", "--no-dev", "--project", str(PKG_DIR)]
    if mode == "three_d_dino":
        cmd += ["--extra", "dino"]
    return cmd


def read_config(config_path: Path) -> tuple[CellposeConfig, dict, Path, str]:
    with config_path.open("rb") as f:
        toml = tomllib.load(f)
    config = CellposeConfig(**toml["cellpose"])
    output_root = Path(toml["output_root"]).expanduser()
    # lsf_project lives in the config (top-level, alongside output_root)
    # rather than being hardcoded in a submission script, since the billing
    # project is a property of the dataset/run, not of the script.
    return config, toml["data-loader"], output_root, toml["lsf_project"]


def prepare(config_path: Path) -> Path:
    """Load the config's volume and create its run directory, in-process."""
    config, data_loader, output_root, _lsf_project = read_config(config_path)
    volume = load_volume(data_loader)
    # [data-loader] is recorded in config.toml, alongside [cellpose]/[run] --
    # so a run directory says which raw path/timepoint/channel produced it,
    # not just which cellpose config ran.
    return prepare_run(volume, config, output_root, extra_metadata={"data-loader": data_loader})


def bsub(args: list[str]) -> int:
    """Run bsub, returning the job ID LSF assigns.

    bsub prints "Job <12345> is submitted to queue <...>." on stdout.
    """
    result = subprocess.run(["bsub", *args], capture_output=True, text=True, check=True)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    start = result.stdout.index("<") + 1
    end = result.stdout.index(">", start)
    return int(result.stdout[start:end])

"""Load one timepoint of the raw p4 nuclear channel and run cellpose_runner on it.

Everything except `load_volume` below is generic -- see cellpose_runner.cli_main.
One config file (with its own timepoint) fully describes one run.

Usage:
    uv run scripts/run_p4_timepoint.py prepare <config.toml>
    uv run scripts/run_p4_timepoint.py segment <run_dir> <config.toml>

    # or, for one call that does both, as on a laptop:
    uv run scripts/run_p4_timepoint.py run <config.toml>
"""

import numpy as np
import tifffile
from janelia_pathlib import JaneliaPath

from cellpose_runner import cli_main


def load_volume(data_loader: dict) -> np.ndarray:
    # raw_path is recorded as whichever OS wrote the config (e.g. the
    # cluster's /groups/... form); translate it to this machine's own form
    # (e.g. /Volumes/... on a Mac) rather than assuming it's already correct.
    raw_path = JaneliaPath(data_loader["raw_path"])
    if not raw_path.exists():
        raw_path.mount()
    timepoint = data_loader["timepoint"]
    nuclear_channel = data_loader["nuclear_channel"]

    # memmap rather than imread: this ImageJ hyperstack is one contiguous page,
    # so imread would decode the whole 5GB file before we slice one timepoint.
    # memmap is lazy, so this is cheap even when only shape/dtype are needed
    # (as in `prepare`) -- no pixel data is read until the array is used.
    mapped = tifffile.memmap(raw_path)
    volume = mapped[timepoint, :, nuclear_channel].astype(mapped.dtype.newbyteorder("="))
    return volume[..., None]  # ZYX -> ZYXC, single channel


if __name__ == "__main__":
    cli_main(load_volume)

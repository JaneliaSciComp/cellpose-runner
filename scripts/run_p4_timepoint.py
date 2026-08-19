"""Load one timepoint of the raw p4 nuclear channel and run cellpose_runner on it.

Everything except `load_volume` below is generic -- see cellpose_runner.cli_main.
One config file (with its own timepoint) fully describes one run.

Usage:
    uv run scripts/run_p4_timepoint.py prepare <config.toml>
    uv run scripts/run_p4_timepoint.py segment <run_dir> <config.toml>

    # or, for one call that does both, as on a laptop:
    uv run scripts/run_p4_timepoint.py run <config.toml>
"""

from pathlib import Path

import numpy as np
import tifffile

from cellpose_runner import cli_main
from cellpose_runner._paths import resolve_janelia_path


def _center_third(volume: np.ndarray) -> np.ndarray:
    """The middle third of `volume` along each of Z, Y and X.

    A do_3D run over the full 73x1280x1280 volume takes hours, which is too
    slow to sweep parameters against. The embryo occupies a small part of the
    field of view, so a center crop keeps plenty of nuclei to judge a
    parameter set by while cutting the work by ~27x. Edges of the embryo do get
    cut off -- fine for comparing parameters, not for a final segmentation.
    """
    slices = tuple(slice(length // 3, 2 * (length // 3)) for length in volume.shape)
    return volume[slices]


def load_volume(data_loader: dict) -> np.ndarray:
    if "timepoint" in data_loader:
        return load_volume_raw(data_loader)
    else:
        return load_volume_processed(data_loader)


def load_volume_processed(data_loader: dict) -> np.ndarray:
    raw_path = resolve_janelia_path(Path(data_loader["raw_path"]))
    mapped = tifffile.memmap(raw_path)

    # Crop the memmap before materializing it: the upsampled file is ~4.5GB
    # over SMB, of which a center_third run reads ~1/27th. Slicing first means
    # only the bytes actually wanted cross the network.
    # Defaults to the full volume, so an existing config without this key
    # keeps segmenting what it always did.
    if data_loader.get("center_third", False):
        mapped = _center_third(mapped)
    volume = np.asarray(mapped).astype(mapped.dtype.newbyteorder("="))
    return volume[..., None]  # ZYX -> ZYXC, single channel


def load_volume_raw(data_loader: dict) -> np.ndarray:
    # raw_path is recorded as whichever OS wrote the config (e.g. the
    # cluster's /groups/... form); translate it to this machine's own form
    # (e.g. /Volumes/... on a Mac) rather than assuming it's already correct.
    raw_path = resolve_janelia_path(Path(data_loader["raw_path"]))
    timepoint = data_loader["timepoint"]
    nuclear_channel = data_loader["nuclear_channel"]

    # memmap rather than imread: this ImageJ hyperstack is one contiguous page,
    # so imread would decode the whole 5GB file before we slice one timepoint.
    # memmap is lazy, so this is cheap even when only shape/dtype are needed
    # (as in `prepare`) -- no pixel data is read until the array is used.
    mapped = tifffile.memmap(raw_path)
    volume = mapped[timepoint, :, nuclear_channel].astype(mapped.dtype.newbyteorder("="))

    # Defaults to the full volume, so an existing config without this key
    # keeps segmenting what it always did.
    if data_loader.get("center_third", False):
        volume = _center_third(volume)
    return volume[..., None]  # ZYX -> ZYXC, single channel


if __name__ == "__main__":
    cli_main(load_volume)

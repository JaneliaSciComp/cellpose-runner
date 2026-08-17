"""Load one timepoint of the raw p4 nuclear channel and run cellpose_runner on it.

Usage: uv run scripts/run_p4_timepoint.py <timepoint> <config.toml>
"""

import sys
import tomllib
from pathlib import Path

import tifffile

from cellpose_runner import CellposeConfig, run_with_logging

RAW_PATH = Path("~/data/HighSNR_embryo_SLS161/p4/raw/p4.tif").expanduser()
NUCLEAR_CHANNEL = 1  # 488, confirmed by eye against processed/p4-488_t1.tif
OUTPUT_ROOT = Path("~/experiments/HighSNR_embryo_SLS161/p4").expanduser()


def main() -> None:
    timepoint = int(sys.argv[1])
    config_path = Path(sys.argv[2])

    with config_path.open("rb") as f:
        config = CellposeConfig(**tomllib.load(f)["cellpose"])

    # memmap rather than imread: this ImageJ hyperstack is one contiguous page,
    # so imread would decode the whole 5GB file before we slice one timepoint.
    mapped = tifffile.memmap(RAW_PATH)
    volume = mapped[timepoint, :, NUCLEAR_CHANNEL].astype(mapped.dtype.newbyteorder("="))
    volume = volume[..., None]  # ZYX -> ZYXC, single channel

    run_with_logging(volume, config, OUTPUT_ROOT)


if __name__ == "__main__":
    main()

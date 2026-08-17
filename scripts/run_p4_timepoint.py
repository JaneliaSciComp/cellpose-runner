"""Load one timepoint of the raw p4 nuclear channel and run cellpose_runner on it.

The data location (raw file path, nuclear channel index, output root) lives in
the config's [data] table alongside [cellpose], so one TOML fully describes a
run regardless of where it's invoked from -- laptop or cluster.

Split into `prepare` and `segment` so a caller (e.g. a cluster submission
script) can create the run directory -- and know its path, to point LSF's own
logs at it -- before the GPU job that does the actual segmentation starts.

Usage:
    uv run scripts/run_p4_timepoint.py prepare <timepoint> <config.toml>
    uv run scripts/run_p4_timepoint.py segment <run_dir> <timepoint> <config.toml>

    # or, for one call that does both, as on a laptop:
    uv run scripts/run_p4_timepoint.py run <timepoint> <config.toml>
"""

import argparse
import logging
import tomllib
from pathlib import Path

import numpy as np
import tifffile

from cellpose_runner import CellposeConfig, prepare_run, run_with_logging, segment

logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> tuple[CellposeConfig, dict]:
    with config_path.open("rb") as f:
        toml = tomllib.load(f)
    return CellposeConfig(**toml["cellpose"]), toml["data"]


def _load_volume(data: dict, timepoint: int) -> np.ndarray:
    raw_path = Path(data["raw_path"]).expanduser()
    nuclear_channel = data["nuclear_channel"]

    # memmap rather than imread: this ImageJ hyperstack is one contiguous page,
    # so imread would decode the whole 5GB file before we slice one timepoint.
    # memmap is lazy, so this is cheap even when only shape/dtype are needed
    # (as in `prepare`) -- no pixel data is read until the array is used.
    mapped = tifffile.memmap(raw_path)
    volume = mapped[timepoint, :, nuclear_channel].astype(mapped.dtype.newbyteorder("="))
    return volume[..., None]  # ZYX -> ZYXC, single channel


def cmd_prepare(args: argparse.Namespace) -> None:
    config, data = _load_config(args.config_path)
    volume = _load_volume(data, args.timepoint)
    output_root = Path(data["output_root"]).expanduser()

    run_dir = prepare_run(volume, config, output_root)
    print(run_dir)  # noqa: T201 -- the one line a caller needs to capture


def cmd_segment(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.run_dir / "script.log"),
        ],
    )
    config, data = _load_config(args.config_path)
    volume = _load_volume(data, args.timepoint)

    logger.info("running cellpose on volume %s %s", volume.shape, volume.dtype)
    masks = segment(args.run_dir, volume, config)
    logger.info("segmented %s, %d labels", masks.shape, masks.max())


def cmd_run(args: argparse.Namespace) -> None:
    config, data = _load_config(args.config_path)
    volume = _load_volume(data, args.timepoint)
    output_root = Path(data["output_root"]).expanduser()

    run_with_logging(volume, config, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("timepoint", type=int)
    prepare_parser.add_argument("config_path", type=Path)
    prepare_parser.set_defaults(func=cmd_prepare)

    segment_parser = subparsers.add_parser("segment")
    segment_parser.add_argument("run_dir", type=Path)
    segment_parser.add_argument("timepoint", type=int)
    segment_parser.add_argument("config_path", type=Path)
    segment_parser.set_defaults(func=cmd_segment)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("timepoint", type=int)
    run_parser.add_argument("config_path", type=Path)
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

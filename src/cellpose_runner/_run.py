import logging
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import zarr

from cellpose_runner._config import CellposeConfig
from cellpose_runner._config_file import (
    check_library_is_committed,
    read_run_config,
    write_run_config,
)
from cellpose_runner._rundir import create_run_dir

MASKS_FILENAME = "masks.zarr"
FLOWS_FILENAME = "flows.zarr"
STYLES_FILENAME = "styles.npy"


class Segmenter(Protocol):
    """The part of `cellpose.models.CellposeModel` that `run` uses."""

    def eval(
        self, x: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]: ...


def smallest_label_dtype(max_label: int) -> np.dtype:
    """The smallest unsigned integer dtype that holds labels 1..max_label.

    A volume with a few hundred objects fits in `uint16`, halving the stored
    size against a blanket `uint32`.
    """
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_label <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.uint64)


# `chunks` is a fixed target length per axis; it need not divide the array's
# shape. `shards` only needs to be a multiple of `chunks` per axis -- zarr
# clips both to the array's actual extent for free, so rounding a shard up to
# the next multiple of the chunk length is enough to cover the whole array in
# one shard, however awkward its shape.
_TARGET_CHUNK_LENGTH = 128


def _one_shard(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Chunk and shard shapes covering all of `shape` in a single shard.

    Each axis's chunk is `_TARGET_CHUNK_LENGTH`, capped at that axis's own
    length in `shape` -- so a small axis (e.g. the 3-element component axis of
    a flow array) gets one chunk covering it, rather than one padded out to
    the target.
    """
    chunks = tuple(min(length, _TARGET_CHUNK_LENGTH) for length in shape)
    shards = tuple(-(-length // chunk) * chunk for length, chunk in zip(shape, chunks, strict=True))
    return chunks, shards


def _build_model(config: CellposeConfig) -> Segmenter:
    from cellpose.models import CellposeModel

    # cellpose is untyped, so this is an assertion rather than a check. The
    # tests in test_config.py pin our kwargs against its real signatures.
    return cast("Segmenter", CellposeModel(**config.model_kwargs()))


def _reset_cellpose_logging() -> None:
    """Undo `cellpose.io.logger_setup`, so ordinary logging config applies.

    That function attaches handlers writing to a shared `~/.cellpose/run.log`
    and to stdout, and sets `propagate = False`, which stops cellpose's records
    from ever reaching the root logger. Anything the caller configures would
    then silently capture nothing.
    """
    logger = logging.getLogger("cellpose")
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def _write_masks(run_dir: Path, masks: np.ndarray) -> np.dtype:
    """Write `masks.zarr` at the narrowest dtype that fits, and return it."""
    # Checked before the cast: a wraparound afterwards leaves no evidence. Valid
    # as an object count only because cellpose labels consecutively from 1.
    dtype = smallest_label_dtype(int(masks.max()))
    chunks, shards = _one_shard(masks.shape)
    zarr.create_array(
        store=run_dir / MASKS_FILENAME,
        data=masks.astype(dtype),
        # One shard for the whole array, so a run never scatters many small
        # chunk files across a filesystem.
        chunks=chunks,
        shards=shards,
        overwrite=False,
    )
    return dtype


# CellposeSAM's eval() always returns exactly these three, in this order:
# `masks, [plot.dx_to_circ(dP), dP, cellprob], styles`. Pinned to cellpose>=4.1,<5,
# so this ordering can be hardcoded rather than kept generic; supporting older
# cellposes (e.g. for legacy fine-tuned models) would need this to vary again.
_FLOW_NAMES = ("rgb", "dP", "cellprob")


def _write_flows(run_dir: Path, flows: list[np.ndarray]) -> None:
    """Write `flows.zarr` as a group with one array per element, named by meaning."""
    group = zarr.create_group(store=run_dir / FLOWS_FILENAME, overwrite=False)
    for name, flow in zip(_FLOW_NAMES, flows, strict=True):
        flow = np.asarray(flow)
        chunks, shards = _one_shard(flow.shape)
        group.create_array(name=name, data=flow, chunks=chunks, shards=shards)


def prepare_run(
    volume: np.ndarray,
    config: CellposeConfig,
    output_root: Path,
    name: str | None = None,
    extra_metadata: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Validate `volume`, create a new run directory, and record what will run.

    Writes `config.toml` and a copy of `uv.lock` before anything touches the
    GPU, so a crashed `segment()` call is still identifiable. Split from
    `segment()` so a caller can learn the run directory -- to log into it, for
    instance -- before segmentation starts.

    Args:
        volume: A 3D (YXC) or 4D (ZYXC) array to segment, channels last.
            Single-channel data still needs the axis, e.g. `volume[..., None]`.
        config: The segmentation parameters. Mutated in place: `channel_axis`
            and `z_axis` are overwritten from `volume`'s own shape before
            being recorded, so `config.toml` never disagrees with the volume
            it actually ran against, no matter what the caller passed in.
        output_root: Directory to create the run directory in.
        name: Name for the run directory, in place of a generated slug.
        extra_metadata: Additional tables to write into `config.toml`, keyed
            by table name -- e.g. how the volume was loaded, or cluster job
            metadata. See `write_run_config`.

    Returns:
        The new run directory, ready for `segment()`.

    Raises:
        ValueError: If `volume` is not 3D or 4D, or `extra_metadata` uses a
            reserved table name.
        DirtyLibraryError: If this package has uncommitted changes.
    """
    if volume.ndim not in (3, 4):
        raise ValueError(
            "volume must be 3D (YXC) or 4D (ZYXC), channels last, got "
            f"{volume.ndim}D with shape {volume.shape}."
        )
    # Both checks come before the run directory exists, so a rejected call
    # leaves nothing behind.
    check_library_is_committed()

    # channel_axis is always the fixed, last axis of our contract; z_axis is
    # the axis before it exactly when the array carries a Z dimension. Set
    # here, before config.toml is written, so `segment()` -- which reads the
    # config back from config.toml rather than trusting a caller's copy --
    # always sees the axes that actually match volume's shape.
    config.preprocess.channel_axis = -1
    config.preprocess.z_axis = 0 if volume.ndim == 4 else None

    run_dir, run_name = create_run_dir(output_root, name=name)
    write_run_config(
        run_dir,
        config,
        run_name=run_name,
        input_shape=volume.shape,
        input_dtype=str(volume.dtype),
        extra=extra_metadata,
    )
    return run_dir


def segment(run_dir: Path, volume: np.ndarray) -> np.ndarray:
    """Segment `volume` and write its outputs into `run_dir`.

    Writes `masks.zarr`, plus `flows.zarr` and `styles.npy` if the config asks
    for them. Call `prepare_run()` first to get `run_dir`.

    Reads the config back from `run_dir`'s own `config.toml`, rather than
    taking one as an argument -- that file is the one source of truth for
    what a run segments with (including `channel_axis`/`z_axis`, which
    `prepare_run()` sets from `volume`'s shape before writing it), so this
    can't drift from it even across a `prepare`/`segment` split across
    separate processes.

    Args:
        run_dir: A run directory, as returned by `prepare_run()`.
        volume: The same array passed to `prepare_run()`.

    Returns:
        The label array, at the dtype it was stored as.
    """
    config = read_run_config(run_dir)

    _reset_cellpose_logging()
    model = _build_model(config)
    masks, flows, styles = model.eval(volume, **config.eval_kwargs())

    dtype = _write_masks(run_dir, masks)
    if config.save_flows:
        _write_flows(run_dir, flows)
    if config.save_styles:
        np.save(run_dir / STYLES_FILENAME, styles)

    return masks.astype(dtype)


def run(
    volume: np.ndarray,
    config: CellposeConfig,
    output_root: Path,
    name: str | None = None,
    extra_metadata: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Segment one volume into a new run directory.

    Equivalent to `prepare_run()` followed by `segment()`, for the common case
    where the run directory itself isn't needed until after segmentation. The
    run directory is new every time, so runs never overwrite each other.

    Args:
        volume: A 3D (YXC) or 4D (ZYXC) array to segment, channels last.
            Single-channel data still needs the axis, e.g. `volume[..., None]`.
        config: The segmentation parameters.
        output_root: Directory to create the run directory in.
        name: Name for the run directory, in place of a generated slug.
        extra_metadata: Additional tables to write into `config.toml`. See
            `prepare_run`.

    Returns:
        The label array, at the dtype it was stored as.

    Raises:
        ValueError: If `volume` is not 3D or 4D.
        DirtyLibraryError: If this package has uncommitted changes.
    """
    run_dir = prepare_run(volume, config, output_root, name=name, extra_metadata=extra_metadata)
    return segment(run_dir, volume)

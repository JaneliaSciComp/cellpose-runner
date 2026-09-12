import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from cellpose_runner._config import (
    CellposeConfig,
    ThreeDFlowsInferenceConfig,
    ThreeDFlowsPostprocessConfig,
)
from cellpose_runner._run import STYLES_FILENAME, _one_shard, _write_flows, _write_masks
from cellpose_runner._view_fusion import (
    TRANSPOSE_TO_VIEW,
    VIEWS,
    consolidate_masks,
    normalize_and_resize,
)

logger = logging.getLogger(__name__)

VIEW_FILENAME_TEMPLATE = "view_{view}.zarr"
_Y_NAME = "y"
_STYLE_NAME = "style"


def run_view(
    volume: np.ndarray, config: CellposeConfig, view: str
) -> tuple[np.ndarray, np.ndarray]:
    """Run one of `core.run_3D`'s orthogonal-view GPU forward passes in isolation.

    Builds its own model (runs in its own process/LSF job). Normalizes and
    resizes the volume exactly as `CellposeModel.eval()` does before calling
    `core.run_net` for this one view's transposed axis order.

    Returns `(y, style)` exactly as `core.run_net` returns them, in this
    view's OWN transposed axis order -- not yet transposed back to
    `(Lz, Ly, Lx, ...)`; that inversion happens in `consolidate()`.
    """
    from cellpose import core
    from cellpose.models import CellposeModel

    inference = config.inference
    if not isinstance(inference, ThreeDFlowsInferenceConfig):
        raise TypeError(
            f"view splitting is only meaningful for mode='three_d_flows', got {config.mode!r}"
        )
    volume, _rescale = normalize_and_resize(
        volume,
        config.preprocess.normalize.to_eval_arg(),
        config.preprocess.diameter,
        inference.anisotropy,
    )

    # Needs direct access to the underlying network (.net), which the
    # Segmenter protocol _run.py uses doesn't expose -- built directly here
    # rather than via _build_model.
    model: Any = CellposeModel(**config.model_kwargs())
    bsize = inference.bsize
    if bsize is None:
        bsize = 256 if model.backbone == "sam_vitl" else 384

    xsl = volume.transpose(TRANSPOSE_TO_VIEW[view])
    logger.info(
        "running %s view: %d planes of size (%d, %d)",
        view,
        xsl.shape[0],
        xsl.shape[1],
        xsl.shape[2],
    )
    y, style = core.run_net(
        model.net,
        xsl,
        batch_size=inference.batch_size,
        augment=inference.augment,
        bsize=bsize,
        tile_overlap=inference.tile_overlap,
        rsz=None,
    )
    logger.info("finished %s view", view)
    return y, style


def write_view(run_dir: Path, view: str, y: np.ndarray, style: np.ndarray) -> None:
    """Persist one view's raw `run_net()` output as `view_<VIEW>.zarr`."""
    group = zarr.create_group(
        store=run_dir / VIEW_FILENAME_TEMPLATE.format(view=view), overwrite=False
    )
    y = np.asarray(y)
    chunks, shards = _one_shard(y.shape)
    group.create_array(name=_Y_NAME, data=y, chunks=chunks, shards=shards)
    style = np.asarray(style)
    group.create_array(name=_STYLE_NAME, data=style, chunks=style.shape, shards=style.shape)


def read_view(run_dir: Path, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Read back one view's `(y, style)`, as written by `write_view()`."""
    store = run_dir / VIEW_FILENAME_TEMPLATE.format(view=view)
    y = zarr.open_array(store=store, path=_Y_NAME)[:]
    style = zarr.open_array(store=store, path=_STYLE_NAME)[:]
    return np.asarray(y), np.asarray(style)


def consolidate(run_dir: Path, config: CellposeConfig, cleanup: bool = True) -> np.ndarray:
    """Combine the 3 per-view zarr outputs into final masks.

    Reads `view_YX.zarr`/`view_ZY.zarr`/`view_ZX.zarr` (all 3 must exist),
    then delegates the fuse/smooth/resize/compute-masks pipeline -- shared
    with `_dino_views.consolidate` -- to `_view_fusion.consolidate_masks`.
    Deliberately CPU-only: no GPU work happens here.

    Writes `masks.zarr` via `_write_masks`, and `flows.zarr`/`styles.npy` via
    `_write_flows`/`np.save` if `config.save_flows`/`save_styles`.

    Args:
        run_dir: A run directory with all 3 `view_*.zarr` written.
        config: The segmentation parameters.
        cleanup: Delete the 3 `view_*.zarr` directories once the final
            outputs are written, since they're large (raw per-view flow
            fields) and only useful as this function's own intermediate
            input. Defaults to True; set False to keep them for debugging.

    Returns:
        The label array at its stored dtype, same contract as `segment()`.
    """
    from cellpose import plot

    postprocess = config.postprocess
    if not isinstance(postprocess, ThreeDFlowsPostprocessConfig):
        raise TypeError(
            f"view splitting is only meaningful for mode='three_d_flows', got {config.mode!r}"
        )
    inference = config.inference
    if not isinstance(inference, ThreeDFlowsInferenceConfig):
        raise TypeError(
            f"view splitting is only meaningful for mode='three_d_flows', got {config.mode!r}"
        )

    ys = {}
    style = None
    for view in VIEWS:
        y, view_style = read_view(run_dir, view)
        ys[view] = y
        style = view_style  # only the last view's style survives, matching run_3D

    masks, dP, cellprob = consolidate_masks(
        run_dir, ys, postprocess, config.preprocess.diameter, resample=inference.resample
    )

    dtype = _write_masks(run_dir, masks)
    if config.save_flows:
        _write_flows(run_dir, [plot.dx_to_circ(dP), dP, cellprob])
    if config.save_styles:
        assert style is not None  # always set by the loop over the 3 fixed views above
        np.save(run_dir / STYLES_FILENAME, style)

    if cleanup:
        for view in VIEWS:
            shutil.rmtree(run_dir / VIEW_FILENAME_TEMPLATE.format(view=view))
        logger.info("removed intermediate view_*.zarr")

    return np.asarray(masks.astype(dtype))

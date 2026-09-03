import logging
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from cellpose_runner._config import (
    CellposeConfig,
    ThreeDFlowsInferenceConfig,
    ThreeDFlowsPostprocessConfig,
)
from cellpose_runner._config_file import CONFIG_FILENAME
from cellpose_runner._run import STYLES_FILENAME, _one_shard, _write_flows, _write_masks

logger = logging.getLogger(__name__)

# Matches cellpose.core.run_3D's own tables exactly -- these
# are local variables inside run_3D, not exported constants, so copied here
# verbatim rather than imported.
_VIEWS = ("YX", "ZY", "ZX")
_TRANSPOSE_TO_VIEW = {"YX": (0, 1, 2, 3), "ZY": (1, 0, 2, 3), "ZX": (2, 0, 1, 3)}
_TRANSPOSE_FROM_VIEW = {"YX": (0, 1, 2), "ZY": (1, 0, 2), "ZX": (1, 2, 0)}
# Which 2 of the combined flow's 3 channels each view contributes to, and
# which of that view's own 2 flow channels (y[..., 0] and y[..., 1])
_DEST_CHANNELS = {"YX": (1, 2), "ZY": (0, 2), "ZX": (0, 1)}
_SOURCE_CHANNELS = (0, 1)

VIEW_FILENAME_TEMPLATE = "view_{view}.zarr"
_Y_NAME = "y"
_STYLE_NAME = "style"


def _normalize_and_resize(volume: np.ndarray, config: CellposeConfig) -> tuple[np.ndarray, float]:
    """Normalize and resize `volume` exactly as `CellposeModel.eval()` does for do_3D.

    Redone independently by every view job rather than shared, since it's
    cheap CPU work relative to the GPU forward pass. Returns the volume ready
    for `core.run_net`, plus `rescale` (needed again, unchanged, by
    `consolidate()` to resize the combined output back).
    """
    from cellpose import transforms

    normalize_arg = config.preprocess.normalize.to_eval_arg()
    if isinstance(normalize_arg, dict):
        volume = transforms.normalize_img(volume, **normalize_arg)

    rescale = 1.0
    diameter = config.preprocess.diameter
    if diameter is not None and diameter > 0:
        rescale = 30.0 / diameter

    inference = config.inference
    if not isinstance(inference, ThreeDFlowsInferenceConfig):
        raise TypeError(
            f"view splitting is only meaningful for mode='three_d_flows', got {config.mode!r}"
        )
    anisotropy = inference.anisotropy
    if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
        Lz, Ly, Lx = volume.shape[:3]
        effective_anisotropy = 1.0 if anisotropy is None else anisotropy
        new_shape = (int(Lz * effective_anisotropy * rescale), int(Ly * rescale), int(Lx * rescale))
        volume = transforms.resize_image_3d(volume, new_shape, no_channels=False)

    return volume, rescale


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

    volume, _rescale = _normalize_and_resize(volume, config)

    # Needs direct access to the underlying network (.net), which the
    # Segmenter protocol _run.py uses doesn't expose -- built directly here
    # rather than via _build_model.
    model: Any = CellposeModel(**config.model_kwargs())
    inference = config.inference
    if not isinstance(inference, ThreeDFlowsInferenceConfig):
        raise TypeError(
            f"view splitting is only meaningful for mode='three_d_flows', got {config.mode!r}"
        )
    bsize = inference.bsize
    if bsize is None:
        bsize = 256 if model.backbone == "sam_vitl" else 384

    xsl = volume.transpose(_TRANSPOSE_TO_VIEW[view])
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


def _read_input_shape(run_dir: Path) -> tuple[int, ...]:
    with (run_dir / CONFIG_FILENAME).open("rb") as f:
        return tuple(tomllib.load(f)["run"]["input_shape"])


def consolidate(run_dir: Path, config: CellposeConfig, cleanup: bool = True) -> np.ndarray:
    """Combine the 3 per-view zarr outputs into final masks.

    Reads `view_YX.zarr`/`view_ZY.zarr`/`view_ZX.zarr` (all 3 must exist) and
    the original (pre-resize) `input_shape` from `config.toml`'s `[run]`
    table. Reconstructs the combined flow/cellprob field exactly as
    `core.run_3D`'s own accumulation does, resizes back to the original
    shape if `config.inference.resample` calls for it, applies
    `flow3D_smooth` to the combined field (never per-view -- `run_3D`'s
    accumulation happens first in real cellpose too, so per-view smoothing
    would double-count), derives `niter`, then computes masks.
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
    from cellpose import dynamics, plot, transforms

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

    logger.info("consolidating views for %s", run_dir)

    # input_shape includes the channel axis (volume.shape as recorded by
    # prepare_run); only the spatial (Z, Y, X) part matters for resizing back.
    original_shape = _read_input_shape(run_dir)[:3]

    ys = {}
    style = None
    for view in _VIEWS:
        y, view_style = read_view(run_dir, view)
        ys[view] = y
        style = view_style  # only the last view's style survives, matching run_3D

    shape = ys["YX"].shape[:-1]  # (Lz, Ly, Lx) in the resized (post-run_view) space
    yf = np.zeros((*shape, 4), dtype="float32")
    for view in _VIEWS:
        y = ys[view]
        inverse = _TRANSPOSE_FROM_VIEW[view]
        yf[..., -1] += y[..., -1].transpose(inverse)
        for source, dest in zip(_SOURCE_CHANNELS, _DEST_CHANNELS[view], strict=True):
            yf[..., dest] += y[..., source].transpose(inverse)

    rescale = 1.0
    diameter = config.preprocess.diameter
    if diameter is not None and diameter > 0:
        rescale = 30.0 / diameter

    resample = inference.resample
    if resample and (rescale != 1.0 or original_shape[0] != yf.shape[0]):
        logger.info("resizing 3D flows and cellprob to original image size")
        yf = transforms.resize_image_3d(yf, original_shape, no_channels=False)

    cellprob = yf[..., -1]
    dP = yf[..., :-1].transpose((3, 0, 1, 2))

    flow3D_smooth = postprocess.flow3D_smooth
    if flow3D_smooth:
        from scipy.ndimage import gaussian_filter

        sigma = (
            [flow3D_smooth] * 3 if isinstance(flow3D_smooth, (int, float)) else list(flow3D_smooth)
        )
        if len(sigma) == 3 and any(v > 0 for v in sigma):
            logger.info("smoothing flows with ZYX sigma=%s", sigma)
            dP = gaussian_filter(dP, [0, *sigma])

    niter_scale = rescale if resample else 1
    niter = postprocess.niter
    niter = int(200 / niter_scale) if niter is None or niter == 0 else niter

    resize = original_shape if tuple(dP.shape[-3:]) != tuple(original_shape) else None
    masks = dynamics.resize_and_compute_masks(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=postprocess.cellprob_threshold,
        flow_threshold=postprocess.flow_threshold,
        do_3D=True,
        min_size=postprocess.min_size,
        max_size_fraction=postprocess.max_size_fraction,
        resize=resize,
        device=torch.device("cpu"),
    )
    masks = masks.squeeze()
    dP = dP.squeeze()
    cellprob = cellprob.squeeze()

    dtype = _write_masks(run_dir, masks)
    if config.save_flows:
        _write_flows(run_dir, [plot.dx_to_circ(dP), dP, cellprob])
    if config.save_styles:
        assert style is not None  # always set by the loop over the 3 fixed views above
        np.save(run_dir / STYLES_FILENAME, style)

    if cleanup:
        for view in _VIEWS:
            shutil.rmtree(run_dir / VIEW_FILENAME_TEMPLATE.format(view=view))
        logger.info("removed intermediate view_*.zarr")

    logger.info("consolidated %s, %d labels", masks.shape, masks.max())
    return np.asarray(masks.astype(dtype))

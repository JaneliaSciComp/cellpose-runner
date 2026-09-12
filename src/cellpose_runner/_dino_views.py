import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

from cellpose_runner._config import (
    CellposeConfig,
    CPDinoModelConfig,
    ThreeDDinoInferenceConfig,
    ThreeDDinoPostprocessConfig,
)
from cellpose_runner._run import _one_shard, _write_flows, _write_masks
from cellpose_runner._view_fusion import (
    TRANSPOSE_TO_VIEW,
    VIEWS,
    consolidate_masks,
    normalize_and_resize,
)

if TYPE_CHECKING:
    from cellpose3d import CPDINO_3D

logger = logging.getLogger(__name__)

VIEW_FILENAME_TEMPLATE = "dino_view_{view}.zarr"
_Y_NAME = "y"


def _require_dino_configs(
    config: CellposeConfig,
) -> tuple[CPDinoModelConfig, ThreeDDinoInferenceConfig, ThreeDDinoPostprocessConfig]:
    """Narrow `config`'s stage configs to their `three_d_dino` subclasses, or raise."""
    model, inference, postprocess = config.model, config.inference, config.postprocess
    if (
        not isinstance(model, CPDinoModelConfig)
        or not isinstance(inference, ThreeDDinoInferenceConfig)
        or not isinstance(postprocess, ThreeDDinoPostprocessConfig)
    ):
        raise TypeError(
            f"dino view splitting is only meaningful for mode='three_d_dino', got {config.mode!r}"
        )
    return model, inference, postprocess


def _build_net(model_config: CPDinoModelConfig) -> "CPDINO_3D":
    """Build a `CPDINO_3D` and load its trained 3D conv stem.

    Two loads, matching `cellpose3d`'s own usage: the constructor call loads
    `base_model_path`'s 2D backbone while the patch-embedding stem is still
    2D; the explicit `load_model()` call below loads `checkpoint_path`'s
    retrained 3D conv stem (and backbone) afterward, once that stem has
    become the 3D module its checkpoint's keys actually match.
    """
    from cellpose3d import CPDINO_3D

    kwargs = model_config.to_init_kwargs()
    device = kwargs.pop("device")
    net = CPDINO_3D(device=device, **kwargs)
    net.load_model(model_config.checkpoint_path, device, strict=False)
    net.eval()
    return net


def run_view(volume: np.ndarray, config: CellposeConfig, view: str) -> np.ndarray:
    """Run one orthogonal view's Z-windowed `CPDINO_3D` forward pass in isolation.

    Builds its own net (runs in its own process/LSF job). Normalizes and
    resizes the volume for `diameter`/`anisotropy` first (see
    `_view_fusion.normalize_and_resize`) -- the `three_d_dino` example
    notebook always normalizes before calling `eval_3d`, which never
    normalizes internally -- then slides a `wsize`-thick Z window along this
    view's own transposed Z axis, matching `cellpose3d.eval_3d`'s per-view
    loop.

    Returns `y`, this view's raw `(Lz, Ly, Lx, 3)` flow/cellprob field, in the
    view's OWN transposed axis order -- not yet transposed back to
    `(Lz, Ly, Lx, ...)`; that inversion happens in `consolidate()`.
    """
    from cellpose3d.utils3d import run_net_3d
    from tqdm import trange

    model_config, inference, _postprocess = _require_dino_configs(config)
    volume, _rescale = normalize_and_resize(
        volume,
        config.preprocess.normalize.to_eval_arg(),
        config.preprocess.diameter,
        inference.anisotropy,
    )
    if volume.shape[-1] == 1:
        volume = np.concatenate([volume, np.zeros_like(volume)], axis=-1)

    net = _build_net(model_config)
    wsize = model_config.wsize

    xsl = volume.transpose(TRANSPOSE_TO_VIEW[view])
    Ly, Lx = xsl.shape[-3:-1]
    Lz = xsl.shape[0]
    logger.info("running %s view: %d planes of size (%d, %d)", view, Lz, Ly, Lx)
    xsl = np.pad(xsl, ((wsize // 2, wsize // 2), (0, 0), (0, 0), (0, 0)))

    y = np.zeros((Lz, Ly, Lx, 3), dtype="float32")
    for z in trange(Lz):
        y[z] = run_net_3d(
            net,
            xsl[z : z + wsize],
            bsize=inference.bsize,
            tile_overlap=inference.tile_overlap,
            batch_size=inference.batch_size,
        )
    logger.info("finished %s view", view)
    return y


def write_view(run_dir: Path, view: str, y: np.ndarray) -> None:
    """Persist one view's raw `run_view()` output as `dino_view_<VIEW>.zarr`."""
    group = zarr.create_group(
        store=run_dir / VIEW_FILENAME_TEMPLATE.format(view=view), overwrite=False
    )
    y = np.asarray(y)
    chunks, shards = _one_shard(y.shape)
    group.create_array(name=_Y_NAME, data=y, chunks=chunks, shards=shards)


def read_view(run_dir: Path, view: str) -> np.ndarray:
    """Read back one view's `y`, as written by `write_view()`."""
    store = run_dir / VIEW_FILENAME_TEMPLATE.format(view=view)
    return np.asarray(zarr.open_array(store=store, path=_Y_NAME)[:])


def consolidate(run_dir: Path, config: CellposeConfig, cleanup: bool = True) -> np.ndarray:
    """Combine the 3 per-view `dino_view_*.zarr` outputs into final masks.

    Reads `dino_view_YX.zarr`/`dino_view_ZY.zarr`/`dino_view_ZX.zarr` (all 3
    must exist), then delegates the fuse/smooth/resize/compute-masks pipeline
    -- shared with `_views.consolidate` -- to `_view_fusion.consolidate_masks`,
    with `dino=True` for `eval_3d`'s one real algorithmic difference (halving
    `dP` before mask computation). Deliberately CPU-only: no GPU work happens
    here.

    Writes `masks.zarr` via `_write_masks`, and `flows.zarr` via `_write_flows`
    if `config.save_flows` -- `three_d_dino` has no per-forward-pass style
    vector, so `styles.npy` is never written regardless of `save_styles`.

    Args:
        run_dir: A run directory with all 3 `dino_view_*.zarr` written.
        config: The segmentation parameters.
        cleanup: Delete the 3 `dino_view_*.zarr` directories once the final
            outputs are written. Defaults to True; set False for debugging.

    Returns:
        The label array at its stored dtype, same contract as `segment()`.
    """
    _model_config, _inference, postprocess = _require_dino_configs(config)

    ys = {view: read_view(run_dir, view) for view in VIEWS}

    # three_d_dino has no `resample` field of its own (see
    # ThreeDDinoInferenceConfig) -- it always resizes back to the original
    # input shape, unlike three_d_flows which only does so when
    # config.inference.resample is set.
    masks, dP, cellprob = consolidate_masks(
        run_dir, ys, postprocess, config.preprocess.diameter, resample=True, dino=True
    )

    dtype = _write_masks(run_dir, masks)
    if config.save_flows:
        _write_flows(run_dir, [dP, dP, cellprob])

    if cleanup:
        for view in VIEWS:
            shutil.rmtree(run_dir / VIEW_FILENAME_TEMPLATE.format(view=view))
        logger.info("removed intermediate dino_view_*.zarr")

    return np.asarray(masks.astype(dtype))

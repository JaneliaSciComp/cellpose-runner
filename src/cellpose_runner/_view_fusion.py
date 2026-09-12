import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cellpose_runner._config import ThreeDDinoPostprocessConfig, ThreeDFlowsPostprocessConfig
from cellpose_runner._config_file import CONFIG_FILENAME

logger = logging.getLogger(__name__)

# Matches cellpose.core.run_3D's own local variables exactly (`three_d_flows`),
# and cellpose3d.utils3d.eval_3d's own local variables exactly (`three_d_dino`)
# -- both split a volume into the same 3 axis-aligned transposes, so the
# tables coincide even though what runs on each view differs. Copied here
# verbatim rather than imported from either upstream, since neither exports
# them as constants.
VIEWS = ("YX", "ZY", "ZX")
TRANSPOSE_TO_VIEW = {"YX": (0, 1, 2, 3), "ZY": (1, 0, 2, 3), "ZX": (2, 0, 1, 3)}
TRANSPOSE_FROM_VIEW = {"YX": (0, 1, 2), "ZY": (1, 0, 2), "ZX": (1, 2, 0)}
# Which 2 of the combined flow's 3 channels each view contributes to, and
# which of that view's own 2 flow channels (y[..., 0] and y[..., 1]).
DEST_CHANNELS = {"YX": (1, 2), "ZY": (0, 2), "ZX": (0, 1)}
SOURCE_CHANNELS = (0, 1)


def fuse_views(ys: dict[str, np.ndarray]) -> np.ndarray:
    """Combine 3 orthogonal views' `(Lz, Ly, Lx, 3)` flow/cellprob fields into one.

    `ys` maps each of `VIEWS` to that view's own raw per-view output, still in
    its own transposed axis order (as `run_view()` returns it in both
    `_views.py` and `_dino_views.py`). Returns a `(Lz, Ly, Lx, 4)` array in
    the untransposed axis order: channels 0-2 are the summed dP components,
    channel 3 is the summed cellprob -- the same accumulation both
    `cellpose.core.run_3D` and `cellpose3d.eval_3d` do, transposing each
    view's contribution back before summing.
    """
    shape = ys["YX"].shape[:-1]
    yf = np.zeros((*shape, 4), dtype="float32")
    for view in VIEWS:
        y = ys[view]
        inverse = TRANSPOSE_FROM_VIEW[view]
        yf[..., -1] += y[..., -1].transpose(inverse)
        for source, dest in zip(SOURCE_CHANNELS, DEST_CHANNELS[view], strict=True):
            yf[..., dest] += y[..., source].transpose(inverse)
    return yf


def read_input_shape(run_dir: Path) -> tuple[int, ...]:
    """The `input_shape` (including channel axis) recorded in `run_dir`'s `config.toml`."""
    with (run_dir / CONFIG_FILENAME).open("rb") as f:
        return tuple(tomllib.load(f)["run"]["input_shape"])


def normalize_and_resize(
    volume: np.ndarray,
    normalize_arg: bool | dict[str, Any],
    diameter: float | None,
    anisotropy: float | None,
) -> tuple[np.ndarray, float]:
    """Normalize, then resize `volume` to a 30px cell diameter, correcting Z for `anisotropy`.

    Shared by `_views.run_view` (`three_d_flows`) and `_dino_views.run_view`
    (`three_d_dino`): both normalize and rescale their volume identically
    before their respective network forward passes, per view. Normalizing
    before resizing matches `CellposeModel.eval()`'s own order (`do_3D`'s
    `_run_3D`), and matches the `three_d_dino` example notebook, which calls
    `transforms.normalize_img` before `eval_3d` -- unlike `eval_3d` itself,
    which never normalizes internally and expects an already-normalized
    volume.

    Unlike upstream `cellpose3d.eval_3d`, which rescales Z by the same
    diameter-derived `scale` as XY with no anisotropy correction, this
    applies `anisotropy` to Z on top of `scale`, so a volume with a
    coarser-sampled Z axis is treated at the same physical scale as XY before
    the network ever sees it.

    Returns the volume ready for the forward pass, plus `rescale` (needed
    again, unchanged, by `consolidate()` to resize the combined output back).
    """
    from cellpose import transforms

    if isinstance(normalize_arg, dict):
        volume = transforms.normalize_img(volume, **normalize_arg)

    rescale = 1.0
    if diameter is not None and diameter > 0:
        rescale = 30.0 / diameter

    if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
        Lz, Ly, Lx = volume.shape[:3]
        effective_anisotropy = 1.0 if anisotropy is None else anisotropy
        new_shape = (int(Lz * effective_anisotropy * rescale), int(Ly * rescale), int(Lx * rescale))
        volume = transforms.resize_image_3d(volume, new_shape, no_channels=False)

    return volume, rescale


def consolidate_masks(
    run_dir: Path,
    ys: dict[str, np.ndarray],
    postprocess: ThreeDFlowsPostprocessConfig | ThreeDDinoPostprocessConfig,
    diameter: float | None,
    resample: bool,
    dino: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse `ys`'s 3 orthogonal-view outputs into masks, matching `core.run_3D`/`eval_3d`.

    Shared by `_views.consolidate` (`three_d_flows`) and `_dino_views.consolidate`
    (`three_d_dino`): both fuse views, optionally smooth the combined flow
    field, resize back to the original input shape, and call
    `dynamics.resize_and_compute_masks` -- identically, once `dino=True`
    accounts for `three_d_dino`'s one real algorithmic difference from
    `three_d_flows` (see below). Writing masks/flows/styles to disk, and
    everything specific to *reading* each mode's own per-view zarr layout,
    stays in each module's own `consolidate()`.

    Args:
        run_dir: A run directory with `config.toml`'s `[run]` table recording
            the original (pre-resize) `input_shape`.
        ys: Maps each of `VIEWS` to that view's own raw per-view output, as
            returned by `run_view()` -- see `fuse_views()`.
        postprocess: The mode's postprocess config. `min_size`, `max_size_fraction`,
            `flow_threshold`, `niter`, `cellprob_threshold`, `flow3D_smooth` are
            read from here; `flow_threshold` is a no-op either way, since
            `dynamics.compute_masks` only applies it when `do_3D=False`.
        diameter: `config.preprocess.diameter`, to recompute the same `rescale`
            factor `run_view()`'s `normalize_and_resize()` used.
        resample: Whether to resize the combined field back to the original
            input shape before computing masks (matching cellpose's own
            `resample` semantics) -- `three_d_flows` only resizes if
            `config.inference.resample` is set; `three_d_dino` always resizes
            back, having no `resample` field of its own to gate on.
        dino: `eval_3d` halves `dP` before calling `compute_masks`
            (`dP / 2`) -- a real difference in how `cellpose3d`'s DINO-based
            network's flow field is scaled relative to `CellposeModel`'s own,
            not an oversight to fix, so this is applied only when `dino=True`.

    Returns:
        `(masks, dP, cellprob)`, each already squeezed, as needed by each
        caller's own mask-writing/flow-writing/logging.
    """
    from cellpose import dynamics, transforms

    logger.info("consolidating views for %s", run_dir)

    # input_shape includes the channel axis (volume.shape as recorded by
    # prepare_run); only the spatial (Z, Y, X) part matters for resizing back.
    original_shape = read_input_shape(run_dir)[:3]

    yf = fuse_views(ys)

    rescale = 1.0
    if diameter is not None and diameter > 0:
        rescale = 30.0 / diameter

    if resample and (rescale != 1.0 or original_shape[0] != yf.shape[0]):
        logger.info("resizing 3D flows and cellprob to original image size")
        yf = transforms.resize_image_3d(yf, original_shape, no_channels=False)

    cellprob = yf[..., -1]
    dP = yf[..., :-1].transpose((3, 0, 1, 2))
    if dino:
        dP = dP / 2

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

    logger.info("consolidated %s, %d labels", masks.shape, masks.max())
    return masks, dP, cellprob

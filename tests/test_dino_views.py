"""Runs real CPDINO_3D end to end. Excluded by default; run with `pytest -m slow`.

Requires a real CPDINO_3D checkpoint, a GPU, and cellpose3d's own
test_img_liu.tif fixture (present in an editable local checkout, per this
repo's `dino` extra). Set CPDINO3D_MODEL_PATH to the checkpoint's path; the
module skips at collection time if either prerequisite is missing.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from cellpose_runner import (
    CellposeConfig,
    CPDinoModelConfig,
    ThreeDDinoInferenceConfig,
    ThreeDDinoPostprocessConfig,
    prepare_run,
)
from cellpose_runner._dino_views import consolidate, read_view, run_view, write_view
from cellpose_runner._run import FLOWS_FILENAME


def _find_test_image() -> Path | None:
    """`cellpose3d`'s own `test_img_liu.tif`, alongside its editable checkout.

    `cellpose3d.__file__` resolves to `<repo>/src/cellpose3d/__init__.py`; the
    fixture lives at `<repo>/test_img_liu.tif`, two directories up. Returns
    `None` (rather than raising) when `cellpose3d` isn't installed at all, or
    is installed as a non-editable build without the repo's other files --
    the `dino` extra guarantees the package, not this specific test asset.
    """
    try:
        import cellpose3d
    except ImportError:
        return None
    candidate = Path(cellpose3d.__file__).resolve().parents[2] / "test_img_liu.tif"
    return candidate if candidate.is_file() else None


# Real cellpose3d/torch code, unlike ours, isn't warnings-clean; the
# project-wide `filterwarnings = ["error"]` would otherwise fail this test on
# warnings we don't control and aren't testing for.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("default"),
    pytest.mark.skipif(
        "CPDINO3D_MODEL_PATH" not in os.environ,
        reason="requires CPDINO3D_MODEL_PATH pointing at a real CPDINO_3D checkpoint",
    ),
    pytest.mark.skipif(
        _find_test_image() is None,
        reason="requires cellpose3d's own test_img_liu.tif, found via an editable local checkout",
    ),
]


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    """Neutralise the dirty-library guard, which fails on any working tree.

    The guard has its own tests; here it would only assert that this repository
    happens to be committed right now.
    """
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


def _real_volume() -> np.ndarray:
    """A crop of `cellpose3d`'s own `test_img_liu.tif`.

    Unlike `test_views.py`'s synthetic Gaussian-blob fixture, CPDINO_3D is a
    trained model that detects nothing in a small/faint synthetic volume (its
    own `wsize=25` alone exceeds a tiny synthetic Z extent) -- a crop of real
    tissue, at a scale close to the model's own example notebook, is what
    actually exercises its flow predictions. Full-resolution (300, 300, 300)
    is too slow to run routinely, so this takes a modest interior crop.
    """
    import tifffile

    image_path = _find_test_image()
    assert image_path is not None  # already gated by this module's skipif
    volume = tifffile.imread(image_path)[100:164, 100:196, 100:196]
    return volume.astype(np.float32)[..., None]


def _test_device() -> str:
    """Whichever device `cellpose` itself would pick on this machine (CUDA/MPS/CPU).

    Unlike `ModelConfig`, `CPDinoModelConfig.to_init_kwargs()` always resolves
    a concrete `torch.device` and `CPDINO_3D.__init__` eagerly calls
    `.to(device)`, so an unset device errors immediately on a CUDA-less
    machine rather than falling back gracefully the way `CellposeModel` does
    -- this reuses cellpose's own `assign_device` so the test picks whatever
    this machine actually has (e.g. MPS on Apple Silicon) instead of forcing
    CPU everywhere.
    """
    from cellpose.core import assign_device

    device, _gpu = assign_device(gpu=True)
    return str(device)


def _config() -> CellposeConfig:
    return CellposeConfig(
        mode="three_d_dino",
        model=CPDinoModelConfig(
            checkpoint_path=os.environ["CPDINO3D_MODEL_PATH"], device=_test_device()
        ),
        inference=ThreeDDinoInferenceConfig(),
        # eval_3d's own dynamics.compute_masks call never passes min_size, so
        # it gets compute_masks's own default of -1 (off) rather than
        # PostprocessConfig's min_size=15 -- matched here (-1, not 0; 0 still
        # runs fill_holes_and_remove_small_masks, just removing nothing) so
        # this test's exact-equivalence assertion isn't comparing filtered
        # masks against unfiltered ones. max_size_fraction's default (0.4) is
        # unchanged: compute_masks applies it unconditionally either way. See
        # test_split_views_and_consolidate_match_eval_3d for the corresponding
        # fill_holes_and_remove_small_masks() call this test makes on
        # eval_3d's own output, since compute_masks (unlike our shared
        # resize_and_compute_masks path) skips hole-filling entirely when
        # min_size<=0.
        postprocess=ThreeDDinoPostprocessConfig(min_size=-1),
        save_flows=True,
    )


def test_split_views_and_consolidate_match_eval_3d(tmp_path):
    """`run_view`/`consolidate` must reproduce `cellpose3d.eval_3d`'s own
    output on the same (normalized) volume -- not just internally consistent
    bookkeeping, but numerically equivalent to what the upstream function
    already does, aside from this port's anisotropy fix (see
    `_dino_views.py`), which `eval_3d` itself doesn't apply and which is a
    no-op here since `test_img_liu.tif`'s crop has isotropic voxel spacing.

    `eval_3d` never normalizes internally -- like the example notebook, this
    normalizes the volume itself before passing it to `eval_3d`, matching
    what `run_view` now does internally via `normalize_and_resize`.

    `eval_3d`'s own `dynamics.compute_masks` call skips hole-filling
    entirely (`min_size<=0`, `compute_masks`'s own default), while our
    shared `_view_fusion.consolidate_masks` always calls
    `resize_and_compute_masks`, which always fills holes regardless of
    `min_size` -- so `expected_masks` gets the same
    `fill_holes_and_remove_small_masks(min_size=-1)` call applied
    explicitly below, to keep this an exact-equivalence test rather than
    one that happens to pass only because this crop's masks have no
    interior holes.
    """
    from cellpose import transforms
    from cellpose.utils import fill_holes_and_remove_small_masks
    from cellpose3d.utils3d import eval_3d

    from cellpose_runner._dino_views import _build_net

    volume = _real_volume()
    config = _config()

    run_dir = prepare_run(volume, config, tmp_path, name="split")
    for view in ("YX", "ZY", "ZX"):
        y = run_view(volume, config, view)
        write_view(run_dir, view, y)
    for view in ("YX", "ZY", "ZX"):
        # Round-tripping through zarr, not just reusing the in-memory y, is
        # what an independent LSF job actually does.
        read_view(run_dir, view)
    split_masks = consolidate(run_dir, config)

    import torch

    net = _build_net(config.model)
    normalized_volume = transforms.normalize_img(volume.astype(np.float32), axis=-1)
    expected_masks, _yf = eval_3d(
        net,
        normalized_volume,
        diameter=30.0,
        wsize=config.model.wsize,
        flow3D_smooth=config.postprocess.flow3D_smooth,
        cellprob_threshold=config.postprocess.cellprob_threshold,
        device=torch.device(_test_device()),
    )
    expected_masks = fill_holes_and_remove_small_masks(expected_masks, min_size=-1)

    # Guards against a vacuous pass: if the blob volume stopped producing any
    # detected objects, comparing two empty mask arrays would trivially
    # "match" without exercising the per-view transpose/accumulation logic.
    assert expected_masks.max() > 0
    assert np.array_equal(split_masks, expected_masks)

    flows = zarr.open_group(store=run_dir / FLOWS_FILENAME)
    assert flows["dP"].shape[0] == 3

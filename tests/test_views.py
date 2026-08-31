"""Runs real cellpose end to end. Excluded by default; run with `pytest -m slow`."""

import numpy as np
import pytest
import zarr
from scipy.ndimage import gaussian_filter

from cellpose_runner import (
    CellposeConfig,
    ThreeDFlowsInferenceConfig,
    ThreeDFlowsPostprocessConfig,
    prepare_run,
    segment,
)
from cellpose_runner._run import FLOWS_FILENAME
from cellpose_runner._views import consolidate, read_view, run_view, write_view

# Real cellpose/torch code, unlike ours, isn't warnings-clean; the project-wide
# `filterwarnings = ["error"]` would otherwise fail this test on warnings we
# don't control and aren't testing for.
pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("default")]


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    """Neutralise the dirty-library guard, which fails on any working tree.

    The guard has its own tests; here it would only assert that this repository
    happens to be committed right now.
    """
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


def _blob_volume() -> np.ndarray:
    """A tiny synthetic volume with 5 separated 3D Gaussian blobs.

    Random binary noise (as used elsewhere in this repo for cheap synthetic
    tests) has no coherent structure, so cellpose detects zero objects in it
    -- an equivalence check against an empty mask array would pass trivially
    without exercising any of the per-view transpose/accumulation logic this
    test exists to verify. Actual blobs give cellpose real flow fields to
    combine.
    """
    shape = (8, 64, 64)
    volume = np.zeros(shape, dtype=np.float32)
    centers = [(2, 16, 16), (2, 48, 48), (5, 32, 16), (5, 16, 48), (6, 48, 20)]
    for z, y, x in centers:
        volume[z, y, x] = 1000
    volume = gaussian_filter(volume, sigma=(1.2, 4, 4))
    volume = (volume / volume.max() * 255).astype(np.uint8)
    return volume[..., None]


def test_split_views_and_consolidate_match_monolithic_do_3d(tmp_path):
    """The whole point of `_views.py`: splitting do_3D into 3 jobs + consolidation
    must produce the exact same masks (and dP/cellprob) as one monolithic
    `do_3D=True` `eval()` call -- not just internally consistent bookkeeping,
    but numerically equivalent to what real cellpose already does.
    """
    volume = _blob_volume()

    config = CellposeConfig(
        mode="three_d_flows",
        inference=ThreeDFlowsInferenceConfig(),
        postprocess=ThreeDFlowsPostprocessConfig(),
        save_flows=True,
    )

    monolithic_run_dir = prepare_run(volume, config, tmp_path, name="monolithic")
    monolithic_masks = segment(monolithic_run_dir, volume)

    split_run_dir = prepare_run(volume, config, tmp_path, name="split")
    for view in ("YX", "ZY", "ZX"):
        y, style = run_view(volume, config, view)
        write_view(split_run_dir, view, y, style)
    for view in ("YX", "ZY", "ZX"):
        # Round-tripping through zarr, not just reusing the in-memory y/style,
        # is what an independent LSF job actually does.
        read_view(split_run_dir, view)
    split_masks = consolidate(split_run_dir, config)

    # Guards against a vacuous pass: if the blob volume stopped producing
    # any detected objects (e.g. a future change to the blob parameters),
    # comparing two empty mask arrays would trivially "match" without
    # exercising the per-view transpose/accumulation logic at all.
    assert monolithic_masks.max() > 0
    assert np.array_equal(split_masks, monolithic_masks)

    monolithic_flows = zarr.open_group(store=monolithic_run_dir / FLOWS_FILENAME)
    split_flows = zarr.open_group(store=split_run_dir / FLOWS_FILENAME)
    for name in ("rgb", "dP", "cellprob"):
        assert np.array_equal(monolithic_flows[name][:], split_flows[name][:])

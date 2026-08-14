"""Runs real cellpose end to end. Excluded by default; run with `pytest -m slow`."""

import numpy as np
import pytest
import zarr

from cellpose_runner import CellposeConfig, run
from cellpose_runner._config_file import CONFIG_FILENAME, LOCK_FILENAME
from cellpose_runner._run import MASKS_FILENAME

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


@pytest.mark.parametrize(
    "shape, config",
    [
        ((64, 64, 1), CellposeConfig()),
        ((8, 64, 64, 1), CellposeConfig(stitch_threshold=0.1)),
        ((8, 64, 64, 1), CellposeConfig(do_3D=True)),
    ],
    ids=["2d", "3d_stitched", "3d_do_3D"],
)
def test_run_segments_a_synthetic_volume(tmp_path, shape, config):
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 2, size=shape, dtype=np.uint8) * 255

    masks = run(volume, config, tmp_path)

    run_dir = next(tmp_path.iterdir())
    assert (run_dir / CONFIG_FILENAME).is_file()
    assert (run_dir / LOCK_FILENAME).is_file()

    stored = zarr.open_array(store=run_dir / MASKS_FILENAME)[:]
    assert np.array_equal(stored, masks)
    assert stored.shape == volume.shape[:-1]


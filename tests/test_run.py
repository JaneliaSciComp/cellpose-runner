import logging

import numpy as np
import pytest
import zarr

from cellpose_runner import CellposeConfig
from cellpose_runner._config_file import CONFIG_FILENAME, LOCK_FILENAME
from cellpose_runner._run import (
    FLOWS_FILENAME,
    MASKS_FILENAME,
    STYLES_FILENAME,
    run,
    smallest_label_dtype,
)


class FakeModel:
    """Stands in for CellposeModel, recording how it was called."""

    def __init__(self, n_labels=3, shape=(4, 8, 8)):
        self.n_labels = n_labels
        self.shape = shape
        self.eval_kwargs = None

    def eval(self, x, **kwargs):
        self.eval_kwargs = kwargs
        masks = np.zeros(self.shape, dtype=np.int32)
        # Label consecutively from 1, as cellpose does.
        for label in range(1, self.n_labels + 1):
            masks.reshape(-1)[label] = label
        flows = [
            np.zeros((3, *self.shape), dtype=np.uint8),
            np.zeros((3, *self.shape), dtype=np.float32),
            np.zeros(self.shape, dtype=np.float32),
        ]
        styles = np.zeros(256, dtype=np.float32)
        return masks, flows, styles


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    """Neutralise the dirty-library guard, which fails on any working tree.

    The guard has its own tests; here it would only assert that this repository
    happens to be committed right now.
    """
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


@pytest.fixture
def fake_model(monkeypatch):
    """Segment with a stub, so these tests never load cellpose or torch."""
    model = FakeModel()
    monkeypatch.setattr("cellpose_runner._run._build_model", lambda _: model)
    return model


@pytest.fixture
def volume():
    return np.zeros((4, 8, 8), dtype=np.uint16)


@pytest.fixture
def segmented(tmp_path, volume, fake_model):
    """A completed run, returning its directory and the model that served it."""
    masks = run(volume, CellposeConfig(), tmp_path)
    run_dir = next(tmp_path.iterdir())
    return run_dir, fake_model, masks


@pytest.mark.parametrize(
    "max_label, expected",
    [(0, np.uint8), (255, np.uint8), (256, np.uint16), (65535, np.uint16), (65536, np.uint32)],
)
def test_smallest_label_dtype(max_label, expected):
    assert smallest_label_dtype(max_label) == expected


def test_writes_the_expected_files(segmented):
    run_dir, _, _ = segmented
    assert {p.name for p in run_dir.iterdir()} == {
        CONFIG_FILENAME,
        LOCK_FILENAME,
        MASKS_FILENAME,
    }


def test_masks_round_trip_at_the_narrowest_dtype(segmented):
    run_dir, _, masks = segmented
    stored = zarr.open_array(store=run_dir / MASKS_FILENAME)[:]
    assert stored.dtype == np.uint8
    assert np.array_equal(stored, masks)
    assert stored.max() == 3


def test_eval_receives_the_config(segmented):
    _, model, _ = segmented
    assert model.eval_kwargs == CellposeConfig().eval_kwargs()


def test_flows_and_styles_are_opt_in(tmp_path, volume, fake_model):
    run(volume, CellposeConfig(save_flows=True, save_styles=True), tmp_path)
    run_dir = next(tmp_path.iterdir())

    group = zarr.open_group(store=run_dir / FLOWS_FILENAME)
    assert [group[name].shape for name in ("rgb", "dP", "cellprob")] == [
        (3, 4, 8, 8),
        (3, 4, 8, 8),
        (4, 8, 8),
    ]
    assert np.load(run_dir / STYLES_FILENAME).shape == (256,)


def test_config_is_written_before_segmenting(tmp_path, volume, monkeypatch):
    # A crashed run must still be identifiable rather than an anonymous
    # half-populated directory.
    def explode(_):
        raise RuntimeError("no gpu")

    monkeypatch.setattr("cellpose_runner._run._build_model", explode)

    with pytest.raises(RuntimeError, match="no gpu"):
        run(volume, CellposeConfig(), tmp_path)

    run_dir = next(tmp_path.iterdir())
    assert (run_dir / CONFIG_FILENAME).is_file()
    assert not (run_dir / MASKS_FILENAME).exists()


@pytest.mark.parametrize("shape", [(8,), (2, 4, 8, 8)])
def test_rejects_volumes_that_are_not_2d_or_3d(tmp_path, shape):
    with pytest.raises(ValueError, match="2D \\(YX\\) or 3D \\(ZYX\\)"):
        run(np.zeros(shape, dtype=np.uint16), CellposeConfig(), tmp_path)

    # Rejected before anything was created.
    assert not list(tmp_path.iterdir())


def test_cellpose_logging_is_left_to_the_caller(tmp_path, volume, fake_model, monkeypatch):
    # cellpose's logger_setup detaches its logger from the root, so anything the
    # caller configures would capture nothing.
    cellpose_logger = logging.getLogger("cellpose")
    monkeypatch.setattr(cellpose_logger, "handlers", [logging.NullHandler()])
    monkeypatch.setattr(cellpose_logger, "propagate", False)

    run(volume, CellposeConfig(), tmp_path)

    assert cellpose_logger.handlers == []
    assert cellpose_logger.propagate is True


def test_each_run_gets_its_own_directory(tmp_path, volume, fake_model):
    for _ in range(2):
        run(volume, CellposeConfig(), tmp_path)
    assert len(list(tmp_path.iterdir())) == 2

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
    _one_shard,
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
    return np.zeros((4, 8, 8, 1), dtype=np.uint16)


@pytest.fixture
def segmented(tmp_path, volume, fake_model):
    """A completed run, returning its directory and the model that served it."""
    masks = run(volume, CellposeConfig(stitch_threshold=0.1), tmp_path)
    run_dir = next(tmp_path.iterdir())
    return run_dir, fake_model, masks


@pytest.mark.parametrize(
    "max_label, expected",
    [(0, np.uint8), (255, np.uint8), (256, np.uint16), (65535, np.uint16), (65536, np.uint32)],
)
def test_smallest_label_dtype(max_label, expected):
    assert smallest_label_dtype(max_label) == expected


@pytest.mark.parametrize(
    "shape, chunks, shards",
    [
        # Smaller than the target chunk length: one chunk covers the axis.
        ((3, 8, 8), (3, 8, 8), (3, 8, 8)),
        # Larger and evenly divisible: chunk is the target, shard matches shape.
        ((256, 256), (128, 128), (256, 256)),
        # Larger and not evenly divisible: shard rounds up past the array's
        # own shape rather than needing the chunk length to divide it.
        ((200, 300), (128, 128), (256, 384)),
        # Mixed: a small leading axis (e.g. the 3-element component axis of a
        # flow array) alongside large, awkward spatial axes.
        ((3, 200, 300), (3, 128, 128), (3, 256, 384)),
    ],
)
def test_one_shard(shape, chunks, shards):
    assert _one_shard(shape) == (chunks, shards)


def test_one_shard_covers_the_whole_array():
    # The point of a single shard: whatever the shape, one shard must be able
    # to hold the whole array (zarr clips shard/chunk overhang for free).
    for shape in [(3, 8, 8), (256, 256), (200, 300), (3, 200, 300), (1, 1), (129, 129)]:
        _, shards = _one_shard(shape)
        assert all(s >= length for s, length in zip(shards, shape, strict=True))


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
    # z_axis and channel_axis are not part of the config: they're fixed by our
    # ZYXC contract (channel_axis always last, z_axis present whenever the
    # array is 4D), not something a caller resolves per run.
    config = CellposeConfig(stitch_threshold=0.1)
    _, model, _ = segmented
    assert model.eval_kwargs == {**config.eval_kwargs(), "z_axis": 0, "channel_axis": -1}


def test_eval_receives_no_z_axis_for_a_2d_image(tmp_path, fake_model):
    fake_model.shape = (8, 8)
    run(np.zeros((8, 8, 1), dtype=np.uint16), CellposeConfig(), tmp_path)
    assert fake_model.eval_kwargs["z_axis"] is None
    assert fake_model.eval_kwargs["channel_axis"] == -1


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


@pytest.mark.parametrize("shape", [(8,), (8, 8), (2, 4, 8, 8, 1)])
def test_rejects_volumes_that_are_not_3d_or_4d(tmp_path, shape):
    with pytest.raises(ValueError, match="3D \\(YXC\\) or 4D \\(ZYXC\\)"):
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

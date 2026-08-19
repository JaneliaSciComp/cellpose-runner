import numpy as np
import pytest

from cellpose_runner import CellposeConfig, prepare_run
from cellpose_runner._report import (
    MASKS_FILENAME,
    _discover_runs,
    _row,
    _varies_across_rows,
    fileglancer_url,
)


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


def test_fileglancer_url_appends_the_relative_run_path(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000000_agile-seahorse"
    assert (
        fileglancer_url("https://fileglancer.example.org/runs", run_dir, runs_root)
        == "https://fileglancer.example.org/runs/20260101T000000_agile-seahorse"
    )


def test_fileglancer_url_strips_a_trailing_slash_on_the_base(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101T000000_agile-seahorse"
    assert fileglancer_url("https://fileglancer.example.org/runs/", run_dir, runs_root) == (
        "https://fileglancer.example.org/runs/20260101T000000_agile-seahorse"
    )


def test_discover_runs_finds_only_directories_with_a_config(tmp_path):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path)
    (tmp_path / "not_a_run").mkdir()

    assert _discover_runs(tmp_path) == [run_dir]


def test_row_reads_cellpose_and_run_fields(tmp_path):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    config = CellposeConfig(do_3D=True)
    config.preprocess.diameter = 30.0
    run_dir = prepare_run(volume, config, tmp_path)

    row = _row(run_dir)

    # Every [cellpose] field, not a hand-picked subset -- including ones the
    # test didn't set, since config.toml records resolved defaults too.
    # Nested stage tables (model, preprocess) come through as nested dicts,
    # same as config.toml itself, rather than being flattened.
    assert row["do_3D"] is True
    assert row["preprocess"]["diameter"] == 30.0
    assert row["batch_size"] == CellposeConfig.model_fields["batch_size"].default
    # Every [run] field alongside them.
    assert row["input_shape"] == list(volume.shape)
    assert row["input_dtype"] == str(volume.dtype)
    assert row["cellpose_runner_version"]

    # Identified by its run_name slug, not a path or link.
    assert "path" not in row
    # No masks.zarr yet: prepare_run() only sets the run directory up.
    assert row["status"] == f"no {MASKS_FILENAME}"


def test_varies_across_rows_true_when_values_differ():
    rows = [{"diameter": 30.0}, {"diameter": 40.0}]
    assert _varies_across_rows("diameter", rows) is True


def test_varies_across_rows_false_when_values_match():
    rows = [{"diameter": 30.0}, {"diameter": 30.0}]
    assert _varies_across_rows("diameter", rows) is False


def test_varies_across_rows_compares_unhashable_values():
    # input_shape is a list; a set-based comparison would need hashability.
    rows = [{"input_shape": [4, 8, 8]}, {"input_shape": [4, 8, 8]}]
    assert _varies_across_rows("input_shape", rows) is False


def test_varies_across_rows_true_when_missing_from_some_rows():
    rows = [{"diameter": 30.0}, {}]
    assert _varies_across_rows("diameter", rows) is True


def test_report_returns_a_servable_app(tmp_path):
    # The Panel/Tabulator widget tree itself isn't worth asserting against
    # directly -- it's implementation detail that shifts with Panel versions.
    # _varies_across_rows has its own tests for the logic that actually
    # matters (which columns default to hidden).
    pytest.importorskip("panel")
    from cellpose_runner import report

    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    config = CellposeConfig()
    config.preprocess.diameter = 30.0
    prepare_run(volume, config, tmp_path)

    app = report(tmp_path)

    assert list(app.main)


def test_report_raises_when_no_runs_are_found(tmp_path):
    pytest.importorskip("panel")
    from cellpose_runner import report

    with pytest.raises(ValueError, match="No runs found"):
        report(tmp_path)

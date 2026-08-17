import numpy as np
import pytest

from cellpose_runner import CellposeConfig, prepare_run
from cellpose_runner._report import MASKS_FILENAME, _discover_runs, _row, fileglancer_url


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
    run_dir = prepare_run(volume, CellposeConfig(diameter=30.0, do_3D=True), tmp_path)

    row = _row(run_dir, tmp_path, fileglancer_base_url=None)

    # Every [cellpose] field, not a hand-picked subset -- including ones the
    # test didn't set, since config.toml records resolved defaults too.
    assert row["do_3D"] is True
    assert row["diameter"] == 30.0
    assert row["batch_size"] == CellposeConfig.model_fields["batch_size"].default
    # Every [run] field alongside them.
    assert row["input_shape"] == list(volume.shape)
    assert row["input_dtype"] == str(volume.dtype)
    assert row["cellpose_runner_version"]

    assert row["path"] == str(run_dir)
    # No masks.zarr yet: prepare_run() only sets the run directory up.
    assert row["status"] == f"no {MASKS_FILENAME}"


def test_row_uses_a_fileglancer_link_when_given_a_base_url(tmp_path):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path)

    row = _row(run_dir, tmp_path, fileglancer_base_url="https://fileglancer.example.org/runs")

    assert row["path"] == fileglancer_url("https://fileglancer.example.org/runs", run_dir, tmp_path)


def test_report_writes_an_html_file(tmp_path):
    pytest.importorskip("panel")
    from cellpose_runner import report

    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    prepare_run(volume, CellposeConfig(diameter=30.0), tmp_path)

    output_path = tmp_path / "report.html"
    result = report(tmp_path, output_path)

    assert result == output_path
    html = output_path.read_text()
    assert "<html" in html.lower()

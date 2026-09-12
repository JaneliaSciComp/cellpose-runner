import logging

import numpy as np
import pytest
import tomli_w

from cellpose_runner import CellposeConfig, prepare_run
from cellpose_runner._script_support import cli_main, resolve_run_dir


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


def _close_root_logging_handlers():
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


@pytest.fixture(autouse=True)
def _reset_root_logging_handlers():
    """Close root logger handlers `cli_main` adds via `logging.basicConfig(force=True)`.

    Each CLI test's `_configure_logging()` call opens a `FileHandler` into a
    `tmp_path`-scoped log file; closing it here (rather than leaving that to
    the next `force=True` call) means the handle doesn't outlive `tmp_path`'s
    own cleanup, which pytest's unraisable-exception hook would otherwise
    flag on a later garbage collection.
    """
    yield
    _close_root_logging_handlers()


def _write_config(config_path, output_root):
    with config_path.open("wb") as f:
        tomli_w.dump({"output_root": str(output_root)}, f)


def test_resolve_run_dir_finds_the_matching_slug(tmp_path):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path, name="agile-seahorse")

    config_path = tmp_path / "config.toml"
    _write_config(config_path, tmp_path)

    assert resolve_run_dir(config_path, "agile-seahorse") == run_dir


def test_resolve_run_dir_raises_when_no_run_matches(tmp_path):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, tmp_path)

    with pytest.raises(ValueError, match="expected exactly one run"):
        resolve_run_dir(config_path, "no-such-slug")


def test_resolve_run_dir_raises_when_multiple_runs_match(tmp_path):
    (tmp_path / "20260101T000000_agile-seahorse").mkdir()
    (tmp_path / "20260102T000000_agile-seahorse").mkdir()

    config_path = tmp_path / "config.toml"
    _write_config(config_path, tmp_path)

    with pytest.raises(ValueError, match="expected exactly one run"):
        resolve_run_dir(config_path, "agile-seahorse")


def _write_full_config(config_path, output_root):
    with config_path.open("wb") as f:
        tomli_w.dump(
            {
                "output_root": str(output_root),
                "cellpose": {},
                "data-loader": {},
            },
            f,
        )


def _write_full_dino_config(config_path, output_root):
    with config_path.open("wb") as f:
        tomli_w.dump(
            {
                "output_root": str(output_root),
                "cellpose": {
                    "mode": "three_d_dino",
                    "model": {"checkpoint_path": "/fake/cpdino3d.pt"},
                    "inference": {},
                    "postprocess": {},
                },
                "data-loader": {},
            },
            f,
        )


def test_cli_run_view_calls_run_view_and_write_view(tmp_path, monkeypatch):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path, name="agile-seahorse")
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path, tmp_path)

    calls = {}

    def fake_run_view(passed_volume, config, view):
        calls["run_view"] = (passed_volume, view)
        return np.zeros((1,)), np.zeros((1,))

    def fake_write_view(passed_run_dir, view, y, style):
        calls["write_view"] = (passed_run_dir, view)

    monkeypatch.setattr("cellpose_runner._views.run_view", fake_run_view)
    monkeypatch.setattr("cellpose_runner._views.write_view", fake_write_view)
    monkeypatch.setattr("sys.argv", ["prog", "run-view", str(run_dir), "YX", str(config_path)])

    cli_main(lambda _data_loader: volume)

    assert calls["run_view"][1] == "YX"
    assert calls["write_view"] == (run_dir, "YX")


def test_cli_consolidate_does_not_call_load_volume(tmp_path, monkeypatch):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path, name="agile-seahorse")
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path, tmp_path)

    load_volume_calls = []

    def fake_load_volume(data_loader):
        load_volume_calls.append(data_loader)
        return volume

    def fake_consolidate(passed_run_dir, config):
        assert passed_run_dir == run_dir
        return np.zeros((4, 8, 8), dtype=np.uint8)

    monkeypatch.setattr("cellpose_runner._views.consolidate", fake_consolidate)
    monkeypatch.setattr("sys.argv", ["prog", "consolidate", str(run_dir), str(config_path)])

    cli_main(fake_load_volume)

    assert load_volume_calls == []


def test_cli_run_view_dispatches_to_dino_views_for_three_d_dino_mode(tmp_path, monkeypatch):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path, name="agile-seahorse")
    config_path = tmp_path / "config.toml"
    _write_full_dino_config(config_path, tmp_path)

    calls = {}

    def fake_run_view(passed_volume, config, view):
        calls["run_view"] = (passed_volume, view)
        return np.zeros((1,))

    def fake_write_view(passed_run_dir, view, y):
        calls["write_view"] = (passed_run_dir, view)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("three_d_dino mode must not dispatch to _views")

    monkeypatch.setattr("cellpose_runner._dino_views.run_view", fake_run_view)
    monkeypatch.setattr("cellpose_runner._dino_views.write_view", fake_write_view)
    monkeypatch.setattr("cellpose_runner._views.run_view", fail_if_called)
    monkeypatch.setattr("cellpose_runner._views.write_view", fail_if_called)
    monkeypatch.setattr("sys.argv", ["prog", "run-view", str(run_dir), "YX", str(config_path)])

    cli_main(lambda _data_loader: volume)

    assert calls["run_view"][1] == "YX"
    assert calls["write_view"] == (run_dir, "YX")


def test_cli_consolidate_dispatches_to_dino_views_for_three_d_dino_mode(tmp_path, monkeypatch):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(volume, CellposeConfig(), tmp_path, name="agile-seahorse")
    config_path = tmp_path / "config.toml"
    _write_full_dino_config(config_path, tmp_path)

    def fake_consolidate(passed_run_dir, config):
        assert passed_run_dir == run_dir
        return np.zeros((4, 8, 8), dtype=np.uint8)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("three_d_dino mode must not dispatch to _views")

    monkeypatch.setattr("cellpose_runner._dino_views.consolidate", fake_consolidate)
    monkeypatch.setattr("cellpose_runner._views.consolidate", fail_if_called)
    monkeypatch.setattr("sys.argv", ["prog", "consolidate", str(run_dir), str(config_path)])

    cli_main(lambda _data_loader: volume)

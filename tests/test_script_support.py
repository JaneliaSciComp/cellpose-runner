import numpy as np
import pytest
import tomli_w

from cellpose_runner import CellposeConfig, prepare_run
from cellpose_runner._script_support import resolve_run_dir


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


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

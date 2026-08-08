import tomllib
from datetime import datetime

import pytest

from cellpose_runner import CellposeConfig
from cellpose_runner._config_file import LOCK_FILENAME, _find_lock_file, write_run_config


@pytest.fixture
def written(tmp_path):
    """A run directory with config.toml written into it, and its parsed contents."""
    config_path = write_run_config(
        tmp_path,
        CellposeConfig(diameter=30.0, do_3D=True),
        run_name="agile-seahorse",
        input_shape=(8, 64, 64),
        input_dtype="uint16",
    )
    with config_path.open("rb") as f:
        return tmp_path, tomllib.load(f)


def test_config_round_trips(written):
    _, parsed = written
    config = CellposeConfig(**parsed["cellpose"])
    assert config == CellposeConfig(diameter=30.0, do_3D=True)


def test_every_set_config_field_is_recorded(written):
    # Defaults are written explicitly, so the file records what ran rather than
    # what was typed. Only fields that are None are absent, since TOML has no
    # way to represent them.
    _, parsed = written
    config = CellposeConfig(diameter=30.0, do_3D=True)
    expected = {name for name in CellposeConfig.model_fields if getattr(config, name) is not None}
    assert set(parsed["cellpose"]) == expected


def test_run_table_describes_the_run(written):
    _, parsed = written
    run = parsed["run"]
    assert run["run_name"] == "agile-seahorse"
    assert run["input_shape"] == [8, 64, 64]
    assert run["input_dtype"] == "uint16"
    assert datetime.fromisoformat(run["timestamp"])
    # The only record of the code that segmented, since uv.lock cannot pin an
    # editable install.
    assert run["cellpose_runner_version"]


def test_lock_file_is_copied(written):
    run_dir, _ = written
    assert (run_dir / LOCK_FILENAME).read_bytes() == _find_lock_file().read_bytes()


def test_refuses_to_run_without_a_lock_file(tmp_path, monkeypatch):
    # Outside a uv-managed project there is no lock to describe the environment,
    # so the run would be quietly unreproducible.
    monkeypatch.setattr("cellpose_runner._config_file._PACKAGE_ROOT", tmp_path / "site-packages")

    with pytest.raises(RuntimeError, match="uv-managed project"):
        _find_lock_file()

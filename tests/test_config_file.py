import subprocess
import tomllib
from datetime import datetime

import pytest

from cellpose_runner import CellposeConfig
from cellpose_runner._config_file import (
    LOCK_FILENAME,
    DirtyLibraryError,
    _find_lock_file,
    check_library_is_committed,
    write_run_config,
)


def _sample_config() -> CellposeConfig:
    config = CellposeConfig(do_3D=True)
    config.preprocess.diameter = 30.0
    return config


@pytest.fixture
def written(tmp_path):
    """A run directory with config.toml written into it, and its parsed contents."""
    config_path = write_run_config(
        tmp_path,
        _sample_config(),
        run_name="agile-seahorse",
        input_shape=(8, 64, 64),
        input_dtype="uint16",
    )
    with config_path.open("rb") as f:
        return tmp_path, tomllib.load(f)


def test_config_round_trips(written):
    _, parsed = written
    config = CellposeConfig(**parsed["cellpose"])
    assert config == _sample_config()


def test_every_set_config_field_is_recorded(written):
    # Defaults are written explicitly, so the file records what ran rather than
    # what was typed. Only fields that are None are absent, since TOML has no
    # way to represent them; nested stage configs (model, preprocess) are
    # never None themselves, so they're always present as nested tables.
    _, parsed = written
    config = _sample_config()
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


def test_extra_tables_are_written(tmp_path):
    config_path = write_run_config(
        tmp_path,
        CellposeConfig(),
        run_name="agile-seahorse",
        input_shape=(8, 64, 64),
        input_dtype="uint16",
        extra={"data-loader": {"raw_path": "/data/p4.tif", "timepoint": 3}},
    )
    with config_path.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["data-loader"] == {"raw_path": "/data/p4.tif", "timepoint": 3}


def test_multiple_extra_tables_coexist(tmp_path):
    config_path = write_run_config(
        tmp_path,
        CellposeConfig(),
        run_name="agile-seahorse",
        input_shape=(8, 64, 64),
        input_dtype="uint16",
        extra={"data-loader": {"raw_path": "/data/p4.tif"}, "cluster": {"queue": "gpu_l4"}},
    )
    with config_path.open("rb") as f:
        parsed = tomllib.load(f)
    assert parsed["data-loader"] == {"raw_path": "/data/p4.tif"}
    assert parsed["cluster"] == {"queue": "gpu_l4"}


def test_extra_none_leaves_config_toml_unchanged(written):
    # extra=None (the default, used by the written fixture) must not add any
    # tables beyond cellpose/run -- existing callers see no change.
    _, parsed = written
    assert set(parsed) == {"cellpose", "run"}


@pytest.mark.parametrize("reserved", ["cellpose", "run"])
def test_extra_rejects_reserved_table_names(tmp_path, reserved):
    with pytest.raises(ValueError, match=reserved):
        write_run_config(
            tmp_path,
            CellposeConfig(),
            run_name="agile-seahorse",
            input_shape=(8, 64, 64),
            input_dtype="uint16",
            extra={reserved: {"oops": True}},
        )


def test_refuses_to_run_without_a_lock_file(tmp_path, monkeypatch):
    # Outside a uv-managed project there is no lock to describe the environment,
    # so the run would be quietly unreproducible.
    monkeypatch.setattr("cellpose_runner._config_file._PACKAGE_ROOT", tmp_path / "site-packages")

    with pytest.raises(RuntimeError, match="uv-managed project"):
        _find_lock_file()


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@e", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A committed repo laid out like ours, with the package root patched to it."""
    package_root = tmp_path / "src" / "cellpose_runner"
    package_root.mkdir(parents=True)
    (package_root / "_run.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("readme\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")

    monkeypatch.setattr("cellpose_runner._config_file._PACKAGE_ROOT", package_root)
    return tmp_path, package_root


def test_clean_checkout_passes(checkout):
    check_library_is_committed()


def test_modified_library_file_raises(checkout):
    _, package_root = checkout
    (package_root / "_run.py").write_text("x = 2\n")

    with pytest.raises(DirtyLibraryError, match=r"_run\.py"):
        check_library_is_committed()


def test_untracked_library_file_raises(checkout):
    # A new module the run imports is invisible to the recorded version, so it
    # matters more than a modified one, not less.
    _, package_root = checkout
    (package_root / "_experiment.py").write_text("x = 3\n")

    with pytest.raises(DirtyLibraryError, match=r"_experiment\.py"):
        check_library_is_committed()


def test_dirty_file_outside_the_package_is_ignored(checkout):
    # Configs and scratch files are dirty almost always; blocking on them would
    # make the check something to disable.
    repo, _ = checkout
    (repo / "README.md").write_text("edited\n")
    (repo / "my_config.toml").write_text("diameter = 30\n")

    check_library_is_committed()


def test_outside_a_checkout_does_not_raise(tmp_path, monkeypatch):
    # An installed package cannot be edited in place, and uv.lock pins it.
    installed = tmp_path / "site-packages" / "cellpose_runner"
    installed.mkdir(parents=True)
    monkeypatch.setattr("cellpose_runner._config_file._PACKAGE_ROOT", installed)

    check_library_is_committed()

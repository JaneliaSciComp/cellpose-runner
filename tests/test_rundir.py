from datetime import datetime

import pytest

from cellpose_runner._rundir import create_run_dir


class _FrozenDatetime:
    """Pins the timestamp so directory names are deterministic."""

    @staticmethod
    def now() -> datetime:
        return datetime(2026, 8, 7, 15, 22, 31)


@pytest.fixture
def fixed_name(monkeypatch):
    """Make both the timestamp and the generated slug deterministic."""
    monkeypatch.setattr("cellpose_runner._rundir.datetime", _FrozenDatetime)
    monkeypatch.setattr("cellpose_runner._rundir.generate_slug", lambda _: "stub-slug")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("run", "20260807T152231_run"),
        (None, "20260807T152231_stub-slug"),
    ],
)
def test_run_dir_is_timestamp_then_name(tmp_path, fixed_name, name, expected):
    run_dir, run_name = create_run_dir(tmp_path, name=name)
    assert run_dir.name == expected
    assert run_dir.parent == tmp_path
    assert run_dir.name.endswith(f"_{run_name}")


def test_creates_missing_output_root(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    run_dir, _ = create_run_dir(root)
    assert run_dir.is_dir()


def test_collision_raises_rather_than_reusing(tmp_path, fixed_name):
    # Two runs must never share a directory: the second one would write its
    # outputs over the first.
    create_run_dir(tmp_path, name="fixed")

    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, name="fixed")

import numpy as np
import pytest
import zarr

from cellpose_runner import CellposeConfig, prepare_run
from cellpose_runner._neuroglancer import (
    _as_source,
    _persist_viewer_table,
    _read_viewer_table,
    _to_xyz,
    view,
)
from cellpose_runner._run import MASKS_FILENAME


@pytest.fixture(autouse=True)
def _committed(monkeypatch):
    monkeypatch.setattr("cellpose_runner._run.check_library_is_committed", lambda: None)


@pytest.fixture
def run_dir(tmp_path):
    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    return prepare_run(volume, CellposeConfig(), tmp_path)


def test_as_source_appends_the_format_suffix():
    assert _as_source("https://fileglancer.example.org/runs/t0", "zarr2") == (
        "https://fileglancer.example.org/runs/t0/|zarr2:"
    )


def test_as_source_strips_a_trailing_slash():
    assert _as_source("https://fileglancer.example.org/runs/t0/", "zarr2") == (
        "https://fileglancer.example.org/runs/t0/|zarr2:"
    )


def test_as_source_percent_encodes_the_path():
    assert "Channel%20390" in _as_source("https://fileglancer.example.org/Channel 390", "zarr2")


def test_viewer_table_round_trips(run_dir):
    config_path = run_dir / "config.toml"
    assert _read_viewer_table(config_path) == {}

    _persist_viewer_table(config_path, {"raw_fileglancer_url": "https://example.org/raw"})

    assert _read_viewer_table(config_path) == {"raw_fileglancer_url": "https://example.org/raw"}


def test_persisting_the_viewer_table_does_not_disturb_cellpose_or_run(run_dir):
    import tomllib

    config_path = run_dir / "config.toml"
    with config_path.open("rb") as f:
        before = tomllib.load(f)

    _persist_viewer_table(config_path, {"raw_fileglancer_url": "https://example.org/raw"})

    with config_path.open("rb") as f:
        after = tomllib.load(f)
    assert after["cellpose"] == before["cellpose"]
    assert after["run"] == before["run"]


def test_view_persists_urls_given_as_arguments(run_dir, monkeypatch):
    monkeypatch.setattr("cellpose_runner._neuroglancer.webbrowser.open", lambda url: None)
    monkeypatch.setattr(
        "cellpose_runner._neuroglancer.neuroglancer_url", lambda image, masks: "http://ng/state"
    )

    view(
        run_dir,
        raw_fileglancer_url="https://example.org/raw",
        runs_root_fileglancer_url="https://example.org/runs",
    )

    assert _read_viewer_table(run_dir / "config.toml") == {
        "raw_fileglancer_url": "https://example.org/raw",
        "runs_root_fileglancer_url": "https://example.org/runs",
    }


def test_view_reuses_persisted_urls_without_prompting(run_dir, monkeypatch):
    _persist_viewer_table(
        run_dir / "config.toml",
        {
            "raw_fileglancer_url": "https://example.org/raw",
            "runs_root_fileglancer_url": "https://example.org/runs",
        },
    )
    monkeypatch.setattr("cellpose_runner._neuroglancer.webbrowser.open", lambda url: None)
    monkeypatch.setattr(
        "cellpose_runner._neuroglancer.neuroglancer_url", lambda image, masks: "http://ng/state"
    )

    def explode(prompt):
        raise AssertionError("should not prompt when already persisted")

    monkeypatch.setattr("builtins.input", explode)

    view(run_dir)


def test_view_opens_the_built_url(run_dir, monkeypatch):
    opened = []
    monkeypatch.setattr("cellpose_runner._neuroglancer.webbrowser.open", opened.append)
    monkeypatch.setattr(
        "cellpose_runner._neuroglancer.neuroglancer_url", lambda image, masks: "http://ng/state"
    )

    result = view(
        run_dir,
        raw_fileglancer_url="https://example.org/raw",
        runs_root_fileglancer_url="https://example.org/runs",
    )

    assert result == "http://ng/state"
    assert opened == ["http://ng/state"]


def test_neuroglancer_url_builds_a_link():
    pytest.importorskip("neuroglancer")
    from cellpose_runner import neuroglancer_url

    url = neuroglancer_url(
        "https://fileglancer.example.org/raw", "https://fileglancer.example.org/masks"
    )
    assert url.startswith("https://neuroglancer-demo.appspot.com/")


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        # ZYXC -> drop C, reverse ZYX -> XYZ
        ((4, 8, 16, 1), (16, 8, 4)),
        # YXC -> drop C, reverse YX -> XY
        ((8, 16, 1), (16, 8)),
    ],
)
def test_to_xyz_drops_channel_and_reverses_spatial_axes(shape, expected_shape):
    assert _to_xyz(np.zeros(shape, dtype=np.uint16), has_channel_axis=True).shape == expected_shape


def test_to_xyz_passes_through_masks_with_no_channel_axis():
    # masks.zarr has no channel axis at all (ZYX/YX already); ndim alone can't
    # distinguish this rank-3 array from a channelled YXC one, hence the flag.
    array = np.zeros((4, 8, 16), dtype=np.uint16)
    assert _to_xyz(array, has_channel_axis=False).shape == (16, 8, 4)


def test_serve_view_requires_a_data_loader_table(run_dir, monkeypatch):
    from cellpose_runner._neuroglancer import serve_view

    with pytest.raises(ValueError, match=r"\[data-loader\]"):
        serve_view(run_dir, load_volume=lambda data_loader: np.zeros((4, 8, 8, 1)))


def test_serve_view_loads_the_volume_and_masks(tmp_path, monkeypatch):
    pytest.importorskip("neuroglancer")
    from cellpose_runner._neuroglancer import serve_view

    volume = np.zeros((4, 8, 8, 1), dtype=np.uint16)
    run_dir = prepare_run(
        volume,
        CellposeConfig(),
        tmp_path,
        extra_metadata={"data-loader": {"raw_path": "/data/p4.tif"}},
    )
    masks = np.zeros((4, 8, 8), dtype=np.uint16)
    zarr.create_array(store=run_dir / MASKS_FILENAME, data=masks, overwrite=False)

    loaded_data_loader = {}

    def load_volume(data_loader):
        loaded_data_loader.update(data_loader)
        return volume

    opened = []
    monkeypatch.setattr("cellpose_runner._neuroglancer.webbrowser.open", opened.append)
    # serve_view blocks on input() so its background HTTP server stays up
    # until the caller is done viewing; that's not testable here.
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    url = serve_view(run_dir, load_volume)

    assert loaded_data_loader == {"raw_path": "/data/p4.tif"}
    assert isinstance(url, str) and url
    assert opened == [url]

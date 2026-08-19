import tomllib
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import quote

import numpy as np
import tomli_w

from cellpose_runner._config_file import CONFIG_FILENAME
from cellpose_runner._paths import resolve_janelia_path
from cellpose_runner._report import fileglancer_url
from cellpose_runner._run import FLOWS_FILENAME, MASKS_FILENAME

if TYPE_CHECKING:
    from cellpose_runner._script_support import LoadVolume

VIEWER_TABLE = "viewer"


def _resolve(path: Path) -> Path:
    """`path` translated to this machine's own form, auto-mounted if needed."""
    return resolve_janelia_path(path)


def _as_source(url: str, fmt: str) -> str:
    """A bare fileglancer URL as a neuroglancer datasource string.

    Percent-encodes the path (fileglancer URLs can contain spaces/commas from
    filenames), then appends the format suffix neuroglancer's `fmt:` driver
    syntax expects (e.g. `zarr2:` for zarr).
    """
    return f"{quote(url.rstrip('/'), safe='/:%')}/|{fmt}:"


def neuroglancer_url(
    image_fileglancer_url: str,
    masks_fileglancer_url: str,
    prefix: str = "https://neuroglancer-demo.appspot.com/",
) -> str:
    """A static neuroglancer link viewing a raw image with masks as labels.

    Built entirely from a resolved `neuroglancer.ViewerState` -- no server
    involved, unlike `neuroglancer.LocalVolume`-based viewers: both layers
    reference their fileglancer HTTPS URLs directly, exactly as fileglancer
    itself would serve them to any HTTP client. `masks_fileglancer_url` is
    loaded as a segmentation layer rather than fileglancer's own auto-inferred
    generic image, so labels render with per-object colors/selection.

    Args:
        image_fileglancer_url: Fileglancer URL for the raw image (zarr).
        masks_fileglancer_url: Fileglancer URL for `masks.zarr`.
        prefix: The neuroglancer instance to encode the link for.

    Returns:
        A complete, shareable neuroglancer URL.
    """
    import neuroglancer

    state = neuroglancer.ViewerState()
    image_layer = neuroglancer.ImageLayer(source=_as_source(image_fileglancer_url, "zarr2"))
    state.layers.append(name="image", layer=image_layer)
    masks_layer = neuroglancer.SegmentationLayer(source=_as_source(masks_fileglancer_url, "zarr2"))
    state.layers.append(name="masks", layer=masks_layer)
    # neuroglancer ships no type information, so this is untyped even under our override.
    return cast("str", neuroglancer.to_url(state, prefix=prefix))


def _read_viewer_table(config_path: Path) -> dict[str, str]:
    with config_path.open("rb") as f:
        toml = tomllib.load(f)
    return cast("dict[str, str]", toml.get(VIEWER_TABLE, {}))


def _persist_viewer_table(config_path: Path, viewer: dict[str, str]) -> None:
    with config_path.open("rb") as f:
        toml = tomllib.load(f)
    toml[VIEWER_TABLE] = viewer
    with config_path.open("wb") as f:
        tomli_w.dump(toml, f)


def view(
    run_dir: Path,
    raw_fileglancer_url: str | None = None,
    runs_root_fileglancer_url: str | None = None,
) -> str:
    """Open `run_dir`'s masks in neuroglancer alongside its raw image.

    The two URLs a link needs -- where the raw image lives, and the
    fileglancer base URL for the runs root `run_dir` sits under -- are
    prompted for on the terminal if not given and not already recorded, then
    persisted into `config.toml`'s `[viewer]` table so later calls for the
    same run (or, for `runs_root_fileglancer_url`, any run under the same
    root) need neither argument.

    Args:
        run_dir: A run directory, as returned by `run()`/`prepare_run()`.
        raw_fileglancer_url: Fileglancer URL for the raw image this run
            segmented. Prompted for if not given and not already recorded.
        runs_root_fileglancer_url: Fileglancer URL for `run_dir`'s parent
            (the runs root). Prompted for if not given and not already
            recorded.

    Returns:
        The neuroglancer URL, also opened in a browser.
    """
    run_dir = _resolve(run_dir)
    config_path = run_dir / CONFIG_FILENAME
    viewer = _read_viewer_table(config_path)

    raw_fileglancer_url = raw_fileglancer_url or viewer.get("raw_fileglancer_url")
    if not raw_fileglancer_url:
        raw_fileglancer_url = input(f"Fileglancer URL for the raw image behind {run_dir}: ")

    runs_root_fileglancer_url = runs_root_fileglancer_url or viewer.get("runs_root_fileglancer_url")
    if not runs_root_fileglancer_url:
        runs_root_fileglancer_url = input(f"Fileglancer URL for {run_dir.parent}: ")

    viewer = {
        "raw_fileglancer_url": raw_fileglancer_url,
        "runs_root_fileglancer_url": runs_root_fileglancer_url,
    }
    _persist_viewer_table(config_path, viewer)

    masks_url = fileglancer_url(runs_root_fileglancer_url, run_dir / "masks.zarr", run_dir.parent)
    url = neuroglancer_url(raw_fileglancer_url, masks_url)
    webbrowser.open(url)
    return url


def _to_xyz(array: np.ndarray, *, has_channel_axis: bool) -> np.ndarray:
    """`array`'s spatial axes reversed to neuroglancer's XYZ convention.

    Our contract is channels-last `ZYXC`/`YXC` for a loaded volume -- drop
    that channel axis (a single raw-image channel is what `load_volume`
    returns per its own contract) -- then reverse the remaining axes, `ZYX`
    -> `XYZ` / `YX` -> `XY`. `masks.zarr` has no channel axis at all (already
    `ZYX`/`YX`), and `ndim` alone can't tell a channelled `YXC` array from an
    unchannelled `ZYX` one -- both are rank 3 -- so the caller says which.
    """
    spatial = array[..., 0] if has_channel_axis else array
    return spatial.transpose(*reversed(range(spatial.ndim)))


def serve_view(run_dir: Path, load_volume: "LoadVolume") -> str:
    """Serve `run_dir`'s masks alongside its raw image, loaded directly.

    For data that isn't already zarr/n5/precomputed-backed (so has no
    fileglancer-servable URL `view()` could point neuroglancer at) --
    `load_volume` is called here, in-process, and both it and `masks.zarr`
    are served locally via `neuroglancer.LocalVolume`. This keeps a Python
    process running for as long as the link is being viewed, unlike `view()`.

    Args:
        run_dir: A run directory, as returned by `run()`/`prepare_run()`.
        load_volume: The same loader a script passed to `cli_main()`, reading
            `run_dir`'s own recorded `[data-loader]` table.

    Returns:
        The neuroglancer URL, also opened in a browser. The server keeps
        running (this call blocks) until interrupted.
    """
    run_dir = _resolve(run_dir)
    with (run_dir / CONFIG_FILENAME).open("rb") as f:
        toml = tomllib.load(f)
    if "data-loader" not in toml:
        raise ValueError(
            f"{run_dir / CONFIG_FILENAME} has no [data-loader] table, so there's no record of "
            "how to load its volume. serve_view() only works for runs made through cli_main()."
        )

    import neuroglancer
    import zarr

    volume = load_volume(toml["data-loader"])
    masks = np.asarray(zarr.open_array(store=run_dir / MASKS_FILENAME)[:])

    if save_flows := toml["cellpose"]["save_flows"]:
        cellprob = np.asarray(zarr.open_array(store=run_dir / FLOWS_FILENAME / "cellprob")[:])

    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.layers["image"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(data=_to_xyz(volume, has_channel_axis=True))
        )
        if save_flows:
            state.layers["cellprob"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(data=_to_xyz(cellprob, has_channel_axis=False))
            )

        state.layers["masks"] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(
                data=_to_xyz(masks, has_channel_axis=False), volume_type="segmentation"
            )
        )

    url = str(viewer)
    webbrowser.open(url)
    # neuroglancer.Viewer() serves on a background thread and doesn't block on
    # its own -- without this, the process (and its server) would exit the
    # instant this function returns, and the just-opened tab would go dead.
    input(f"Serving {url}\nPress enter to stop serving and exit: ")
    return url

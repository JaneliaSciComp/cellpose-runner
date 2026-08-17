import tomllib
from pathlib import Path
from typing import Any

from cellpose_runner._config_file import CONFIG_FILENAME
from cellpose_runner._run import MASKS_FILENAME

_LEADING_COLUMNS = ("run_name",)
_TRAILING_COLUMNS = ("status", "path")


def fileglancer_url(base_url: str, run_dir: Path, runs_root: Path) -> str:
    """A fileglancer link for `run_dir`, given a base URL for `runs_root`.

    There is no fileglancer API yet, so `base_url` is obtained by hand from
    the fileglancer UI for `runs_root` itself; this appends the run's path
    relative to that root, rather than constructing a URL from scratch.

    Args:
        base_url: The fileglancer URL for `runs_root`, as shown in its UI.
        run_dir: A run directory under `runs_root`.
        runs_root: The root passed to `report()`.

    Returns:
        `base_url` with `run_dir`'s relative path appended.
    """
    return f"{base_url.rstrip('/')}/{run_dir.relative_to(runs_root)}"


def _discover_runs(runs_root: Path) -> list[Path]:
    return sorted(p.parent for p in runs_root.glob(f"*/{CONFIG_FILENAME}"))


def _row(run_dir: Path, runs_root: Path, fileglancer_base_url: str | None) -> dict[str, Any]:
    with (run_dir / CONFIG_FILENAME).open("rb") as f:
        toml = tomllib.load(f)
    cellpose = toml.get("cellpose", {})
    run = toml.get("run", {})
    path = (
        fileglancer_url(fileglancer_base_url, run_dir, runs_root)
        if fileglancer_base_url
        else str(run_dir)
    )
    return {
        "run_name": run.get("run_name", run_dir.name),
        # Every [run] and [cellpose] field config.toml recorded, defaults
        # included -- not a hand-picked subset, so a newly added field (in
        # either table, or one this report doesn't know about) still shows
        # up for comparison. run_name is pulled out above so it stays first.
        **{k: v for k, v in run.items() if k != "run_name"},
        **cellpose,
        "status": "done" if (run_dir / MASKS_FILENAME).exists() else f"no {MASKS_FILENAME}",
        "path": path,
    }


def report(
    runs_root: Path, output_path: Path, fileglancer_base_url: str | None = None
) -> Path:
    """Write a self-contained, sortable HTML table of every run under `runs_root`.

    One row per run directory (`<runs_root>/*/config.toml`). Rendered with
    Panel's `Tabulator` widget -- real interactive sorting/filtering with far
    less code than a hand-rolled table, at the cost of needing internet
    access to view (its JS/CSS load from a CDN, not vendored). No live
    server is needed to *view* the saved HTML, only to generate it once.
    Requires the `report` extra (`pandas`, `panel`).

    Args:
        runs_root: Directory containing run directories directly under it.
        output_path: Where to write the report HTML file.
        fileglancer_base_url: The fileglancer URL for `runs_root`, obtained by
            hand from its UI (no fileglancer API exists yet). If given, each
            row's path is a fileglancer link via `fileglancer_url()`. If not,
            the plain filesystem path is shown instead.

    Returns:
        `output_path`.
    """
    import pandas as pd
    import panel as pn

    pn.extension("tabulator")

    rows = [
        _row(run_dir, runs_root, fileglancer_base_url) for run_dir in _discover_runs(runs_root)
    ]
    # Column order: run_name first, whatever [run]/[cellpose] fields any row
    # has in between (a run's own recorded metadata and resolved config,
    # which may vary as either gains fields over time), status/path last.
    # dict preserves insertion order, so this also dedupes while keeping
    # order stable across runs that share fields.
    middle_columns = dict.fromkeys(
        key for row in rows for key in row if key not in (*_LEADING_COLUMNS, *_TRAILING_COLUMNS)
    )
    df = pd.DataFrame(rows, columns=[*_LEADING_COLUMNS, *middle_columns, *_TRAILING_COLUMNS])

    table = pn.widgets.Tabulator(df, show_index=False, disabled=True, pagination=None)
    pn.Column(f"# cellpose_runner runs: {runs_root}", table).save(output_path, embed=True)
    return output_path

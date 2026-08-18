import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpose_runner._config_file import CONFIG_FILENAME
from cellpose_runner._run import MASKS_FILENAME

if TYPE_CHECKING:
    import panel

_LEADING_COLUMNS = ("run_name",)
_TRAILING_COLUMNS = ("status",)


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


def _row(run_dir: Path) -> dict[str, Any]:
    with (run_dir / CONFIG_FILENAME).open("rb") as f:
        toml = tomllib.load(f)
    cellpose = toml.get("cellpose", {})
    run = toml.get("run", {})
    return {
        "run_name": run.get("run_name", run_dir.name),
        # Every [run] and [cellpose] field config.toml recorded, defaults
        # included -- not a hand-picked subset, so a newly added field (in
        # either table, or one this report doesn't know about) still shows
        # up for comparison. run_name is pulled out above so it stays first.
        **{k: v for k, v in run.items() if k != "run_name"},
        **cellpose,
        "status": "done" if (run_dir / MASKS_FILENAME).exists() else f"no {MASKS_FILENAME}",
    }


_MISSING = object()


def _varies_across_rows(column: str, rows: list[dict[str, Any]]) -> bool:
    """Whether `column` differs across `rows`, or is missing from some.

    Some recorded values (e.g. `input_shape`) are lists, so this compares by
    equality rather than deduplicating through a `set`, which would need
    every value to be hashable.
    """
    values = [row.get(column, _MISSING) for row in rows]
    return any(v != values[0] for v in values[1:])


_ACCENT_COLOR = "#8B3FE0"


def report(runs_root: Path) -> "panel.template.FastListTemplate":
    """A live, sortable table of every run under `runs_root`, with column checkboxes.

    One row per run directory (`<runs_root>/*/config.toml`), identified by its
    `run_name` slug -- not a path or link, since that's what a caller passes
    to e.g. `resolve_run_dir()` / `serve_view()` to act on a specific run.
    Rendered with Panel's `Tabulator` widget for real interactive
    sorting/filtering, plus a checkbox per column toggling which show --
    both need a live Python process to react to clicks (Tabulator's own
    column-visibility toggling isn't reachable through Panel's static export,
    and Panel's reactive callbacks don't survive one either), so this is
    meant to be run with `panel serve`, not saved to a static file.

    Requires the `report` extra (`pandas`, `panel`).

    Args:
        runs_root: Directory containing run directories directly under it.

    Returns:
        A servable Panel template (`panel serve` it, or call `.show()`).

    Raises:
        ValueError: If `runs_root` has no run directories under it -- serving
            an empty table would just look like a blank/broken page.
    """
    import pandas as pd
    import panel

    panel.extension("tabulator")

    run_dirs = _discover_runs(runs_root)
    if not run_dirs:
        raise ValueError(f"No runs found under {runs_root} (no */{CONFIG_FILENAME} matches).")

    rows = [_row(run_dir) for run_dir in run_dirs]
    # Column order: run_name first, whatever [run]/[cellpose] fields any row
    # has in between (a run's own recorded metadata and resolved config,
    # which may vary as either gains fields over time), status/path last.
    # dict preserves insertion order, so this also dedupes while keeping
    # order stable across runs that share fields.
    middle_columns = dict.fromkeys(
        key for row in rows for key in row if key not in (*_LEADING_COLUMNS, *_TRAILING_COLUMNS)
    )
    columns = [*_LEADING_COLUMNS, *middle_columns, *_TRAILING_COLUMNS]
    df = pd.DataFrame(rows, columns=columns)

    table = panel.widgets.Tabulator(
        df,
        show_index=False,
        disabled=True,
        pagination=None,
        theme="materialize",
        sizing_mode="stretch_width",
    )
    # A column the same for every run isn't useful for comparison, so it
    # starts unchecked -- except run_name/status, which stay visible even
    # when constant (e.g. a report with only one run).
    default_visible = [
        c
        for c in columns
        if c in (*_LEADING_COLUMNS, *_TRAILING_COLUMNS) or _varies_across_rows(c, rows)
    ]

    # One Checkbox per column in a wrapping FlexBox, rather than a single
    # CheckBoxGroup(inline=True) -- that lays out as a non-wrapping flex row,
    # so with many columns it just grows a horizontal scrollbar.
    def _update_hidden_column(column: str, checkbox: "panel.widgets.Checkbox") -> None:
        def _on_change(event: object) -> None:
            hidden = set(table.hidden_columns)
            hidden.discard(column) if checkbox.value else hidden.add(column)
            table.hidden_columns = list(hidden)

        checkbox.param.watch(_on_change, "value")

    column_checkboxes = []
    for c in columns:
        checkbox = panel.widgets.Checkbox(label=c, value=c in default_visible)
        _update_hidden_column(c, checkbox)
        column_checkboxes.append(checkbox)
    checkboxes = panel.Card(
        panel.FlexBox(*column_checkboxes, flex_wrap="wrap"),
        title="Columns",
        collapsible=False,
    )
    table.hidden_columns = [c for c in columns if c not in default_visible]

    template = panel.template.FastListTemplate(
        title="cellpose_runner runs",
        accent_base_color=_ACCENT_COLOR,
        header_background=_ACCENT_COLOR,
        main=[
            panel.pane.Markdown(f"### `{runs_root}`"),
            checkboxes,
            table,
        ],
    )
    template.servable()
    return template

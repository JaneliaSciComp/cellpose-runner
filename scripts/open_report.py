"""Serve a live, sortable report of every run under a runs root.

Dataset-agnostic: works for any run() output, not just p4. Interactive
column checkboxes need a live Python process to react to clicks, so this is
served, not saved to a static file -- run it with `panel serve`, not `python`.

Usage:
    uv run --extra report panel serve scripts/open_report.py --show \
        --args <config.toml | output_root>
"""

import sys
import tomllib
from pathlib import Path

from cellpose_runner import report
from cellpose_runner._paths import resolve_janelia_path


def _runs_root(arg: Path) -> Path:
    if arg.is_dir():
        runs_root = arg
    else:
        with arg.open("rb") as f:
            runs_root = Path(tomllib.load(f)["output_root"]).expanduser()

    # output_root may be recorded in whichever OS wrote the config (e.g. the
    # cluster's /nrs/... form); translate/mount it for this machine.
    return resolve_janelia_path(runs_root)


report(_runs_root(Path(sys.argv[1])))

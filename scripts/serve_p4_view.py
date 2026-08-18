"""Serve a p4 run's raw volume and masks together in neuroglancer.

raw/p4.tif isn't zarr/n5/precomputed-backed, so it has no fileglancer URL
neuroglancer can fetch directly -- this loads it in-process (same load_volume
as run_p4_timepoint.py) and serves both layers via neuroglancer.LocalVolume.
Keeps this process alive (and its server) until you press enter.

Usage:
    uv run --extra view scripts/serve_p4_view.py <config.toml> <run-slug>
    uv run --extra view scripts/serve_p4_view.py <run_dir>
"""

import importlib.util
import sys
from pathlib import Path

from cellpose_runner import resolve_run_dir
from cellpose_runner._neuroglancer import serve_view

# run_p4_timepoint.py is a standalone script, not an importable package, so
# its load_volume is loaded directly from its file path here.
_spec = importlib.util.spec_from_file_location(
    "run_p4_timepoint", Path(__file__).parent / "run_p4_timepoint.py"
)
assert _spec is not None and _spec.loader is not None
_run_p4_timepoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_p4_timepoint)
load_volume = _run_p4_timepoint.load_volume


def main() -> None:
    if len(sys.argv) == 3:
        run_dir = resolve_run_dir(Path(sys.argv[1]), sys.argv[2])
    else:
        run_dir = Path(sys.argv[1])
    serve_view(run_dir, load_volume)


if __name__ == "__main__":
    main()

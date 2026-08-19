import time
from pathlib import Path
from typing import cast

from janelia_pathlib import JaneliaPath


def resolve_janelia_path(path: Path, timeout: float = 30) -> Path:
    """`path` translated to this machine's own form, auto-mounted if needed.

    A recorded path (a run directory, a raw data file) may be in whichever OS
    wrote it -- e.g. the cluster's `/nrs/...` or `/groups/...` form -- so this
    is run over any path before touching the filesystem, not just the ones a
    caller already knows are Janelia paths.

    Waits for `path` itself, not just the share root. `JaneliaPath.mount()`
    returns as soon as the share's mount point appears, but SMB makes deeper
    directories listable a moment later, so globbing or reading immediately
    after a cold mount would see an empty or missing directory.

    Args:
        path: The path to translate and mount.
        timeout: Seconds to wait for `path` to appear after mounting.

    Returns:
        The translated path, whether or not it ended up existing -- a missing
        path is left for the caller to report in its own terms.
    """
    resolved = JaneliaPath(path)
    if not resolved.exists():
        resolved.mount(timeout=timeout)
        # mount() waited for the share root; wait for this path in particular.
        deadline = time.monotonic() + timeout
        while not resolved.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
    return cast("Path", resolved)

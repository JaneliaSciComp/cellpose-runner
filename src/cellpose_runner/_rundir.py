"""Creation of timestamped run directories."""

from datetime import datetime
from pathlib import Path

from coolname import generate_slug

# ISO 8601 basic format: no separators, so it is safe in a filename.
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


def create_run_dir(output_root: Path, name: str | None = None) -> tuple[Path, str]:
    """Create a new run directory under `output_root`.

    The directory is named `<YYYYMMDD>T<HHMMSS>_<name>`, so runs sort
    chronologically and a day's work is one glob.

    The run directory is derived from `output_root` rather than rebuilt, so a
    `Path` subclass such as `JaneliaPath` is preserved.

    Args:
        output_root: Directory to create the run directory in. Created if it
            does not exist.
        name: Name to use instead of a randomly generated slug.

    Returns:
        The newly created run directory, and the name it was given.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = name or generate_slug(2)
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    run_dir = output_root / f"{timestamp}_{run_name}"
    run_dir.mkdir()
    return run_dir, run_name

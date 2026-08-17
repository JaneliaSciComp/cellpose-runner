import logging
from pathlib import Path

import numpy as np

from cellpose_runner._config import CellposeConfig
from cellpose_runner._run import prepare_run, segment

LOG_FILENAME = "script.log"


def run_with_logging(volume: np.ndarray, config: CellposeConfig, output_root: Path) -> np.ndarray:
    """Run `prepare_run()` then `segment()`, logging into the run directory.

    Convenience for one-off scripts: logging setup is the same regardless of
    dataset, so a script only needs to load its volume and build its config.

    Args:
        volume: A 3D (YXC) or 4D (ZYXC) array to segment, channels last.
        config: The segmentation parameters.
        output_root: Directory to create the run directory in.

    Returns:
        The label array, as returned by `segment()`.
    """
    run_dir = prepare_run(volume, config, output_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / LOG_FILENAME)],
    )
    logger = logging.getLogger(__name__)
    logger.info("config: %s", config)
    logger.info("run directory: %s", run_dir)

    logger.info("running cellpose on volume %s %s", volume.shape, volume.dtype)
    masks = segment(run_dir, volume, config)
    logger.info("segmented %s, %d labels", masks.shape, masks.max())
    return masks

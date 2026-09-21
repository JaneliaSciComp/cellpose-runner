"""Submit one cellpose_runner segmentation as an LSF job.

Usage:
    uv run scripts/submit_cluster.py <config.toml>

One bsub job, no array, no sweep -- validates the cluster path for a single
run before anything more ambitious. See scratch/CLUSTER_DESIGN.md.
"""

import sys
from pathlib import Path

from _cluster_submit import SCRIPT, bsub, prepare, read_config, uv_run_cmd

# no wall-time cap; cheapest uncapped GPU queue -- do_3D=True runs exceed gpu_short's 1hr cap
QUEUE = "gpu_l4"
SLOTS = 8  # gpu_l4 is 15GB/slot; observed local peak RSS is ~8GB, well under
WALLTIME = "4:00"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <config.toml>")
    config_path = Path(sys.argv[1])

    config, _data_loader, _output_root, project = read_config(config_path)
    uv_run = uv_run_cmd(config.mode)

    run_dir = prepare(config_path)
    print(f"run directory: {run_dir}")  # noqa: T201

    # umask 002: keep output group-writable on shared /nrs storage.
    # No -R rusage[mem=...]: Janelia GPU queues allocate memory per slot via
    # -n, so passing rusage alongside it is redundant, and multiplies under
    # RESOURCE_RESERVE_PER_SLOT=Y. See scratch/CLUSTER_DESIGN.md.
    bsub(
        [
            "-J",
            f"cellpose-runner-{run_dir.name}",
            "-n",
            str(SLOTS),
            "-gpu",
            "num=1",
            "-q",
            QUEUE,
            "-P",
            project,
            "-W",
            WALLTIME,
            "-o",
            str(run_dir / "lsf.out"),
            "-e",
            str(run_dir / "lsf.err"),
            f"umask 002; {' '.join(uv_run)} {SCRIPT} segment {run_dir} {config_path}",
        ]
    )


if __name__ == "__main__":
    main()

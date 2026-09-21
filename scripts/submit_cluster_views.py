"""Submit a mode="three_d_flows" or mode="three_d_dino" cellpose_runner
segmentation as 4 dependent LSF jobs: 3 parallel per-view GPU forward
passes, then 1 CPU-only consolidation.

Usage:
    uv run scripts/submit_cluster_views.py <config.toml>

Splits do_3D's 3 orthogonal-view forward passes (see cellpose_runner._views
for three_d_flows, cellpose_runner._dino_views for three_d_dino) across
independent GPU jobs that run concurrently instead of sequentially inside
one segment() call, then a consolidation job combines their outputs into
final masks. See scratch/parallel_3d_views_plan.md. Which implementation
runs is decided inside cellpose_runner's own CLI dispatch, by the config's
`mode` -- this script only needs to know whether that mode requires the
`dino` extra (cellpose3d: torch/opencv/etc., not part of the base install).
"""

import sys
from pathlib import Path

from _cluster_submit import SCRIPT, bsub, prepare, read_config, uv_run_cmd

# no wall-time cap; cheapest uncapped GPU queue -- view runs can exceed gpu_short's 1hr cap
GPU_QUEUE = "gpu_l4"
# gpu_l4 is 15GB/slot; observed local peak RSS is ~8GB for three_d_flows -- no cluster
# measurement yet for three_d_dino's own footprint, so this is shared as a starting
# point, to adjust per-mode once one exists
GPU_SLOTS = 8
GPU_WALLTIME = "4:00"

# CPU-only; no GPU flag. 14-day max, default batch queue for runtime > 1hr.
CONSOLIDATE_QUEUE = "local"
CONSOLIDATE_SLOTS = 4
CONSOLIDATE_WALLTIME = "1:00"

VIEWS = ("YX", "ZY", "ZX")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <config.toml>")
    config_path = Path(sys.argv[1])

    config, _data_loader, _output_root, project = read_config(config_path)
    uv_run = uv_run_cmd(config.mode)

    run_dir = prepare(config_path)
    print(f"run directory: {run_dir}")  # noqa: T201

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # umask 002: keep output group-writable on shared /nrs storage.
    # No -R rusage[mem=...]: Janelia GPU/CPU queues allocate memory per slot via
    # -n, so passing rusage alongside it is redundant, and multiplies under
    # RESOURCE_RESERVE_PER_SLOT=Y. See scratch/CLUSTER_DESIGN.md.
    view_job_ids = []
    for view in VIEWS:
        job_id = bsub(
            [
                "-J",
                f"cellpose-runner-view-{view}-{run_dir.name}",
                "-n",
                str(GPU_SLOTS),
                "-gpu",
                "num=1",
                "-q",
                GPU_QUEUE,
                "-P",
                project,
                "-W",
                GPU_WALLTIME,
                "-o",
                str(log_dir / f"lsf.view-{view}.out"),
                "-e",
                str(log_dir / f"lsf.view-{view}.err"),
                f"umask 002; {' '.join(uv_run)} {SCRIPT} run-view {run_dir} {view} {config_path}",
            ]
        )
        view_job_ids.append(job_id)

    # done() requires success, so a failed view job leaves consolidation pending
    # forever (visible via `bjobs`) rather than running against incomplete data.
    dependency = " && ".join(f"done({job_id})" for job_id in view_job_ids)

    bsub(
        [
            "-J",
            f"cellpose-runner-consolidate-{run_dir.name}",
            "-n",
            str(CONSOLIDATE_SLOTS),
            "-q",
            CONSOLIDATE_QUEUE,
            "-P",
            project,
            "-W",
            CONSOLIDATE_WALLTIME,
            "-w",
            dependency,
            "-o",
            str(log_dir / "lsf.consolidate.out"),
            "-e",
            str(log_dir / "lsf.consolidate.err"),
            f"umask 002; {' '.join(uv_run)} {SCRIPT} consolidate {run_dir} {config_path}",
        ]
    )


if __name__ == "__main__":
    main()

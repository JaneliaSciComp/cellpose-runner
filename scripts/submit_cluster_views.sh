#!/usr/bin/env bash
# Submit a mode="three_d_flows" or mode="three_d_dino" cellpose_runner
# segmentation as 4 dependent LSF jobs: 3 parallel per-view GPU forward
# passes, then 1 CPU-only consolidation.
#
# Usage: scripts/submit_cluster_views.sh <config.toml>
#
# Splits do_3D's 3 orthogonal-view forward passes (see cellpose_runner._views
# for three_d_flows, cellpose_runner._dino_views for three_d_dino) across
# independent GPU jobs that run concurrently instead of sequentially inside
# one segment() call, then a consolidation job combines their outputs into
# final masks. See scratch/parallel_3d_views_plan.md. Which implementation
# runs is decided inside cellpose_runner's own CLI dispatch, by the config's
# `mode` -- this script only needs to know whether that mode requires the
# `dino` extra (cellpose3d: torch/opencv/etc., not part of the base install).
#
# `prepare` runs here, on the submitting node: cheap (memmaps the volume just
# for its shape, no pixel data read) and creates the run directory before any
# GPU job exists, so LSF's own -o/-e logs can point straight at it.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.toml>" >&2
    exit 1
fi

CONFIG="$1"

GPU_QUEUE=gpu_l4  # no wall-time cap; cheapest uncapped GPU queue -- view runs can exceed gpu_short's 1hr cap
GPU_SLOTS=8       # gpu_l4 is 15GB/slot; observed local peak RSS is ~8GB for three_d_flows -- no
                  # cluster measurement yet for three_d_dino's own footprint, so this is shared
                  # as a starting point, to adjust per-mode once one exists
GPU_WALLTIME=4:00

CONSOLIDATE_QUEUE=local  # CPU-only; no GPU flag. 14-day max, default batch queue for runtime > 1hr.
CONSOLIDATE_SLOTS=4
CONSOLIDATE_WALLTIME=1:00

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PKG_DIR/scripts/run_timepoint.py"

UV_RUN=(uv run --no-dev --project "$PKG_DIR")
# three_d_dino needs cellpose3d (torch/opencv/etc.), an optional `dino` extra
# -- every other mode runs fine without it, so it's only added when asked for.
MODE="$(uv run --no-dev --project "$PKG_DIR" python3 -c "
import tomllib, sys
with open(sys.argv[1], 'rb') as f:
    print(tomllib.load(f)['cellpose'].get('mode', 'two_d'))
" "$CONFIG")"
if [ "$MODE" = "three_d_dino" ]; then
    UV_RUN+=(--extra dino)
fi

# lsf_project lives in the config (top-level, alongside output_root) rather
# than being hardcoded here, since the billing project is a property of the
# dataset/run, not of this script.
PROJECT="$("${UV_RUN[@]}" python3 -c "
import tomllib, sys
with open(sys.argv[1], 'rb') as f:
    print(tomllib.load(f)['lsf_project'])
" "$CONFIG")"

RUN_DIR="$("${UV_RUN[@]}" "$SCRIPT" prepare "$CONFIG")"
echo "run directory: $RUN_DIR"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# umask 002: keep output group-writable on shared /nrs storage.
# No -R rusage[mem=...]: Janelia GPU/CPU queues allocate memory per slot via
# -n, so passing rusage alongside it is redundant, and multiplies under
# RESOURCE_RESERVE_PER_SLOT=Y. See scratch/CLUSTER_DESIGN.md.
VIEW_JOB_IDS=()
for VIEW in YX ZY ZX; do
    JOB_OUTPUT="$(bsub \
        -J "cellpose-runner-view-$VIEW-$(basename "$RUN_DIR")" \
        -n "$GPU_SLOTS" \
        -gpu "num=1" \
        -q "$GPU_QUEUE" \
        -P "$PROJECT" \
        -W "$GPU_WALLTIME" \
        -o "$LOG_DIR/lsf.view-$VIEW.out" \
        -e "$LOG_DIR/lsf.view-$VIEW.err" \
        "umask 002; ${UV_RUN[*]} $SCRIPT run-view $RUN_DIR $VIEW $CONFIG")"
    echo "$JOB_OUTPUT"
    # bsub prints "Job <12345> is submitted to queue <...>." on stdout.
    JOB_ID="$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | head -1)"
    VIEW_JOB_IDS+=("$JOB_ID")
done

# done() requires success, so a failed view job leaves consolidation pending
# forever (visible via `bjobs`) rather than running against incomplete data.
DEPENDENCY="done(${VIEW_JOB_IDS[0]}) && done(${VIEW_JOB_IDS[1]}) && done(${VIEW_JOB_IDS[2]})"

bsub \
    -J "cellpose-runner-consolidate-$(basename "$RUN_DIR")" \
    -n "$CONSOLIDATE_SLOTS" \
    -q "$CONSOLIDATE_QUEUE" \
    -P "$PROJECT" \
    -W "$CONSOLIDATE_WALLTIME" \
    -w "$DEPENDENCY" \
    -o "$LOG_DIR/lsf.consolidate.out" \
    -e "$LOG_DIR/lsf.consolidate.err" \
    "umask 002; ${UV_RUN[*]} $SCRIPT consolidate $RUN_DIR $CONFIG"

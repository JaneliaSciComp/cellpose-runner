#!/usr/bin/env bash
# Submit one cellpose_runner segmentation for a p4 timepoint as an LSF job.
#
# Usage: scripts/submit_p4_cluster.sh <timepoint> <config.toml>
#
# One bsub job, no array, no sweep -- validates the cluster path for a single
# run before anything more ambitious. See scratch/CLUSTER_DESIGN.md.
#
# `prepare` runs here, on the submitting node: cheap (memmaps the volume just
# for its shape, no pixel data read) and creates the run directory before the
# GPU job exists, so LSF's own -o/-e logs can point straight at it instead of
# a separately tracked log path.
set -euo pipefail

TIMEPOINT="$1"
CONFIG="$2"

PROJECT=shroff        # LSF billing project
QUEUE=gpu_short        # 1-hour cap, any GPU type -- for this validation run
SLOTS=8                # gpu_short is 15GB/slot; observed local peak RSS is ~8GB, well under
WALLTIME=1:00

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PKG_DIR/scripts/run_p4_timepoint.py"

RUN_DIR="$(uv run --project "$PKG_DIR" "$SCRIPT" prepare "$TIMEPOINT" "$CONFIG")"
echo "run directory: $RUN_DIR"

# umask 002: keep output group-writable on shared /nrs storage.
# No -R rusage[mem=...]: Janelia GPU queues allocate memory per slot via -n,
# so passing rusage alongside it is redundant, and multiplies under
# RESOURCE_RESERVE_PER_SLOT=Y. See scratch/CLUSTER_DESIGN.md.
bsub \
    -J "cellpose-runner-p4-t${TIMEPOINT}" \
    -n "$SLOTS" \
    -gpu "num=1" \
    -q "$QUEUE" \
    -P "$PROJECT" \
    -W "$WALLTIME" \
    -o "$RUN_DIR/lsf.out" \
    -e "$RUN_DIR/lsf.err" \
    "umask 002; uv run --project $PKG_DIR $SCRIPT segment $RUN_DIR $TIMEPOINT $CONFIG"

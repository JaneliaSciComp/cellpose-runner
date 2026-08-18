#!/usr/bin/env bash
# Submit one cellpose_runner segmentation for a p4 timepoint as an LSF job.
#
# Usage: scripts/submit_p4_cluster.sh <config.toml>
#
# One bsub job, no array, no sweep -- validates the cluster path for a single
# run before anything more ambitious. See scratch/CLUSTER_DESIGN.md.
#
# `prepare` runs here, on the submitting node: cheap (memmaps the volume just
# for its shape, no pixel data read) and creates the run directory before the
# GPU job exists, so LSF's own -o/-e logs can point straight at it instead of
# a separately tracked log path.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config.toml>" >&2
    exit 1
fi

CONFIG="$1"

QUEUE=gpu_l4    # no wall-time cap; cheapest uncapped GPU queue -- do_3D=True runs exceed gpu_short's 1hr cap
SLOTS=8         # gpu_l4 is 15GB/slot; observed local peak RSS is ~8GB, well under
WALLTIME=4:00

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$PKG_DIR/scripts/run_p4_timepoint.py"

# lsf_project lives in the config (top-level, alongside output_root) rather
# than being hardcoded here, since the billing project is a property of the
# dataset/run, not of this script.
PROJECT="$(uv run --no-dev --project "$PKG_DIR" python3 -c "
import tomllib, sys
with open(sys.argv[1], 'rb') as f:
    print(tomllib.load(f)['lsf_project'])
" "$CONFIG")"

RUN_DIR="$(uv run --no-dev --project "$PKG_DIR" "$SCRIPT" prepare "$CONFIG")"
echo "run directory: $RUN_DIR"

# umask 002: keep output group-writable on shared /nrs storage.
# No -R rusage[mem=...]: Janelia GPU queues allocate memory per slot via -n,
# so passing rusage alongside it is redundant, and multiplies under
# RESOURCE_RESERVE_PER_SLOT=Y. See scratch/CLUSTER_DESIGN.md.
bsub \
    -J "cellpose-runner-$(basename "$RUN_DIR")" \
    -n "$SLOTS" \
    -gpu "num=1" \
    -q "$QUEUE" \
    -P "$PROJECT" \
    -W "$WALLTIME" \
    -o "$RUN_DIR/lsf.out" \
    -e "$RUN_DIR/lsf.err" \
    "umask 002; uv run --no-dev --project $PKG_DIR $SCRIPT segment $RUN_DIR $CONFIG"

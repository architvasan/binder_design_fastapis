#!/bin/bash
# Serve the RFD4-Proteina design API on a CUDA GPU node.
#
# RFD4-Proteina loads in-process, so it must be installed in the serving env
# with its CUDA extra. On a cluster, run this on a GPU node (e.g. an
# interactive PBS/Slurm allocation) and port-forward from your laptop:
#   ssh -L 8000:<node>:8000 <you>@<login-host>

set -euo pipefail

cd "$(dirname "$0")/.."

export RFD4_RUNS_ROOT="${RFD4_RUNS_ROOT:-$PWD/runs}"
export RFD4_HOST="${RFD4_HOST:-0.0.0.0}"
export RFD4_PORT="${RFD4_PORT:-8000}"
# export RFD4_HF_TOKEN=...        # if the checkpoint is gated
# export RFD4_PRELOAD_MODEL=rfd4-proteina

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install "rfd4-proteina[all-cuda13]"   # the model + CUDA runtime
  uv pip install -e ".[dev]"                    # this service
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install "rfd4-proteina[all-cuda13]"
  pip install -e ".[dev]"
fi

nvidia-smi || echo "WARNING: nvidia-smi not available on this node"
exec rfd4-proteina-serve

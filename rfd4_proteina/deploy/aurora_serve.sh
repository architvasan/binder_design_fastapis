#!/bin/bash
# Serve the RFD4 / Proteina-Complexa design API on an Aurora compute node.
#
# The service shells out to the `complexa` CLI, so it must run INSIDE the
# activated Proteina-Complexa environment (which puts `complexa` on PATH and
# exports PROTEINA_ACCELERATOR=xpu, CKPT_PATH, AF2_DIR, ESMFOLD_DIR, RF3_CKPT_PATH,
# and the project caches). Paths below come from the aurora-proteina-complexa
# recipe (profile.json); override via env vars if your install moved.
#
# Launch on an interactive PBS tile, e.g.:
#   qsub -I -l select=1 -l walltime=01:00:00 -A <Project> -q <queue>
#   bash deploy/aurora_serve.sh
# Then from your laptop, forward the port:
#   ssh -L 8000:<node>:8000 <you>@aurora.alcf.anl.gov

set -euo pipefail

ENV_ROOT="${PROTEINA_ENV_ROOT:-/lus/flare/projects/FRAME-IDP/mmeigooni/envs/proteina-complexa}"
export PROTEINA_REPO_DIR="${PROTEINA_REPO_DIR:-/lus/flare/projects/FRAME-IDP/mmeigooni/software/Proteina-Complexa}"
export PROTEINA_RUNS_ROOT="${PROTEINA_RUNS_ROOT:-$PROTEINA_REPO_DIR/runs}"
export PROTEINA_HOST="${PROTEINA_HOST:-0.0.0.0}"
export PROTEINA_PORT="${PROTEINA_PORT:-8000}"

# Activate the framework env (sets XPU accelerator, checkpoint + cache vars).
# shellcheck disable=SC1091
source "$ENV_ROOT/bin/activate"

# Keep this process pinned to ONE tile and serialize campaigns (gen_njobs=1 /
# eval_njobs=1 are already forced by the service). Respect the endpoint's mask.
echo "host=$(hostname)  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
command -v complexa >/dev/null 2>&1 || {
  echo "ERROR: 'complexa' not on PATH after activation" >&2
  exit 1
}

# Install the service into the active env if not present, then serve.
python -m pip install -e . >/dev/null 2>&1 || true
exec rfd4-proteina-serve

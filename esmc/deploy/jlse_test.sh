#!/bin/bash
# Install + test esmc-service on a JLSE GPU node (Nvidia A100 testbed).
#
# Scheduler: JLSE uses Cobalt. Confirm your project name, the queue, and whether
# your node has moved to PBS in the JLSE wiki:
#   https://apps.cels.anl.gov/confluence/spaces/JLSEdocs/overview
#
# Submit (Cobalt, from the esmc/ directory):
#   qsub -n 1 -t 60 -q gpu_a100 -A <YourProject> deploy/jlse_test.sh
# Interactive instead:
#   qsub -I -n 1 -t 60 -q gpu_a100 -A <YourProject>
# Monitor: qstat -u $USER      Cancel: qdel <jobid>
# Output:  <jobid>.output  <jobid>.error  <jobid>.cobaltlog  (in the submit dir)
#
# If your node is on PBS instead, submit with:
#   qsub -l select=1 -l walltime=01:00:00 -q gpu_a100 -A <YourProject> deploy/jlse_test.sh
# (the #COBALT lines below are ignored by PBS).

#COBALT -n 1
#COBALT -t 60
#COBALT -q gpu_a100
# #COBALT -A <YourProject>   # set here, or pass -A on the qsub command line

set -euo pipefail

# Start in the service directory under either scheduler.
cd "${PBS_O_WORKDIR:-${ESMC_DIR:-$PWD}}"
echo "host=$(hostname)  cwd=$PWD"
nvidia-smi || echo "WARNING: nvidia-smi not available on this node"

# Persist the Hugging Face weight cache across jobs on the shared filesystem.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

# --- environment (prefer uv; fall back to venv + pip) ---
module load cuda 2>/dev/null || true
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv-jlse
  # shellcheck disable=SC1091
  source .venv-jlse/bin/activate
  uv pip install -e ".[dev]"
else
  python3 -m venv .venv-jlse
  # shellcheck disable=SC1091
  source .venv-jlse/bin/activate
  pip install --upgrade pip
  pip install -e ".[dev]"
fi

# --- 1. static checks + unit tests (CPU; fake backend, no weights) ---
ruff check src tests
ruff format --check src tests
mypy src tests
pytest -q

# --- 2. real-weights GPU smoke: load esmc_300m on the A100, embed one sequence ---
python - <<'PY'
import torch
from esmc_service.server import EsmcEmbeddingBackend
from esmc_service.server import Settings

assert torch.cuda.is_available(), 'no CUDA visible on this node'
backend = EsmcEmbeddingBackend(Settings())        # downloads weights into HF_HOME
vectors = backend.embed('esmc_300m', ['MKTAYIAKQRQISFVKSH'], transformer_layer=12)
assert backend.loaded is not None
print('device:', backend.loaded.device, 'dim:', len(vectors[0]),
      'head:', [round(x, 3) for x in vectors[0][:4]])
backend.unload()
PY

# --- 3. live endpoint smoke ---
ESMC_PRELOAD_MODEL=esmc_300m python -m uvicorn esmc_service.server:app \
  --port 8000 --log-level warning &
SRV=$!
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf http://127.0.0.1:8000/health && echo
curl -s http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"esmc_300m","input":["MKTAYIAKQRQISFVKSH"]}' | head -c 200; echo
kill "$SRV" 2>/dev/null || true

echo 'JLSE test run complete.'

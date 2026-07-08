#!/bin/bash
# Shared environment for Neurolink on the cluster (SLURM).
# Login:  source scripts/cluster/env.sh
# SLURM:  sourced by job scripts (offline, no pip).

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NEUROLINK_ROOT:=$(cd "$CLUSTER_DIR/../.." && pwd)}"
cd "$NEUROLINK_ROOT"
export PYTHONPATH="${NEUROLINK_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

module purge
module load arch/v100
module load pytorch-gpu

export PYTHONUNBUFFERED=1
export PYTHONUSERBASE="${PYTHONUSERBASE:-${WORK:-$HOME}/.local_neurolink}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export HF_HOME="${HF_HOME:-${WORK:-$HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
else
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
fi

mkdir -p logs data eval "$HF_HOME"

neurolink() {
  python -m neurolink "$@"
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: export HF_TOKEN before sbatch (Mistral-7B is gated)." >&2
    exit 1
  fi
  if ! python "$CLUSTER_DIR/check_deps.py"; then
    echo "ERROR: run bash scripts/cluster/setup_login.sh on the login node first." >&2
    exit 1
  fi
  if ! python "$CLUSTER_DIR/verify_models.py"; then
    echo "ERROR: HuggingFace models not cached — run setup_login.sh on login node." >&2
    exit 1
  fi
fi

if [[ "${NEUROLINK_ENV_QUIET:-}" != "1" ]]; then
  echo "Neurolink env: $NEUROLINK_ROOT"
  echo "Python: $(which python)"
  echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
fi

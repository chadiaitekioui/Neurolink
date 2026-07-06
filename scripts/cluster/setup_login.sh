#!/bin/bash
# Install Neurolink on the cluster login node (HTTP allowed). Run once.
# Usage: export HF_TOKEN=hf_... ; bash scripts/cluster/setup_login.sh

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NEUROLINK_ENV_QUIET=1
# shellcheck disable=SC1091
source "$CLUSTER_DIR/env.sh"

export TRANSFORMERS_OFFLINE=0
unset HF_HUB_OFFLINE

echo "=== Neurolink — setup login ==="

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "WARNING: export HF_TOKEN before GPU jobs (Mistral-7B gated)." >&2
fi

if command -v idr_module_search >/dev/null 2>&1; then
  idr_module_search -f "$CLUSTER_DIR/module_search.txt" --arch v100 || true
fi

if ! python "$CLUSTER_DIR/check_deps.py" 2>/dev/null; then
  echo "=== pip install addons (see $CLUSTER_DIR/addons.txt) ==="
  pip install --user --no-cache-dir -r "$CLUSTER_DIR/addons.txt"
fi
pip install --user --no-cache-dir --no-deps -e "$NEUROLINK_ROOT"
python "$CLUSTER_DIR/check_deps.py"

MODELS=(
  mistralai/Mistral-7B-v0.1
  ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section
  BrainGPT/BrainGPT-7B-v0.2
)

if command -v huggingface-cli >/dev/null 2>&1; then
  for model in "${MODELS[@]}"; do
    name="${model##*/}"
    if [[ -d "$HF_HOME/models--${model//\//--}" ]] || [[ -d "$HF_HOME/hub/models--${model//\//--}" ]]; then
      echo "=== Model cached: $model ==="
    else
      echo "=== Download: $model → $HF_HOME ==="
      huggingface-cli download "$model"
    fi
  done
else
  echo "WARNING: huggingface-cli missing — use \$DSDIR or install huggingface_hub." >&2
fi

mkdir -p logs data eval
echo "=== Done. Next: bash scripts/cluster/login_index.sh ==="

#!/bin/bash
#SBATCH --job-name=nl-prompt-smoke
#SBATCH --output=logs/nl-prompt-smoke-%j.out
#SBATCH --error=logs/nl-prompt-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Parallel smoke: shared Year N → Year N+1 prompt (30 articles) on mistral_base + braingpt.
# Predict 2023 from Year 2022; then MiniLM P@k/R@k + contamination.
# Does NOT run Job 2 / neurolink compare / LoRA.
# Output: eval/prompt_smoke_*.json

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

mkdir -p logs eval
echo "=== Prompt-v2 smoke (independent of Job 2) ==="
python scripts/prompt_smoke_predict.py --config config/forecast/prompt_smoke.yaml
echo "=== Prompt-v2 smoke done ==="
ls -lt eval/prompt_smoke_*.json | head -3

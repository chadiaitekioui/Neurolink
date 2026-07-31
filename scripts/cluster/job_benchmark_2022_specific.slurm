#!/bin/bash
#SBATCH --job-name=nl-bench22-spec
#SBATCH --output=logs/nl-bench22-spec-%j.out
#SBATCH --error=logs/nl-bench22-spec-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=60:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Specific-direction experiment A — Benchmark 2022 on neurolink_specific.db.
# Needs literature_base_specific year_max=2022.
# LoRA-base vs mistral_base vs braingpt (years 2023–2025) + full eval.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

test -f data/neurolink_specific.db || {
  echo "ERROR: missing data/neurolink_specific.db"; exit 1; }

test -d data/models/literature_base_specific/year_max_2022/lora || {
  echo "ERROR: missing literature_base_specific adapter — run job_train_lora_base_specific_2022.slurm"; exit 1; }

echo "=== Compare: LoRA-base (specific) + mistral_base + braingpt ==="
neurolink compare \
  --lora-year-max 2022 \
  --config config/forecast/predict_compare_specific.yaml

echo "=== Eval: specific LoRA-base round ==="
neurolink eval --config config/eval/scenarios/eval_compare_2022_specific.yaml
cp -f eval/summary.md eval/summary_job2_specific.md

echo "=== Done ==="
echo "  eval/summary_job2_specific.md"
ls -la data/models/literature_base_specific/year_max_2022/lora/ | head

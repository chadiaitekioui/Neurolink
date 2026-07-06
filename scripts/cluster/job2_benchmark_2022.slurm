#!/bin/bash
#SBATCH --job-name=nl-bench22
#SBATCH --output=logs/nl-bench22-%j.out
#SBATCH --error=logs/nl-bench22-%j.err
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=REPLACE_WITH_YOUR_GPU_ACCOUNT
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Job 2 — Train LoRA year_max=2022, benchmark 2023-2025 (3 LLMs, temp=0), eval.
# Prompt and temperature are fixed in config/forecast/predict_compare.yaml.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

echo "=== Job 2: train LoRA 2022 ==="
neurolink train-literature \
  --year-max 2022 \
  --config config/forecast/predict_literature.yaml

echo "=== Job 2: benchmark anchor 2022 (2023-2025) ==="
neurolink compare \
  --lora-year-max 2022 \
  --config config/forecast/predict_compare.yaml

echo "=== Job 2: eval benchmark 2022 ==="
neurolink eval --config config/eval/scenarios/eval_compare_2022.yaml
cp -f eval/summary.md eval/summary_job2.md

echo "=== Job 2 done — adapter: data/models/literature/year_max_2022/lora/ ==="

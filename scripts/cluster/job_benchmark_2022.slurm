#!/bin/bash
#SBATCH --job-name=nl-bench22
#SBATCH --output=logs/nl-bench22-%j.out
#SBATCH --error=logs/nl-bench22-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=40:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Benchmark only (no train) — adapters already at year_max=2022.
# 1) LoRA-base vs mistral_base vs braingpt (2023-2025) + eval
# 2) LoRA-instruct vs same baselines + eval
# Note: mistral_base/braingpt are re-run in (2); literature_lora slot changes adapter.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

test -d data/models/literature_base/year_max_2022/lora || {
  echo "ERROR: missing literature_base adapter"; exit 1; }
test -d data/models/literature_instruct/year_max_2022/lora || {
  echo "ERROR: missing literature_instruct adapter"; exit 1; }

echo "=== Benchmark A: LoRA BASE + mistral_base + braingpt ==="
neurolink compare \
  --lora-year-max 2022 \
  --config config/forecast/predict_compare.yaml

echo "=== Eval A (base LoRA) ==="
neurolink eval --config config/eval/scenarios/eval_compare_2022.yaml
cp -f eval/summary.md eval/summary_job2_base.md

echo "=== Benchmark B: LoRA INSTRUCT + mistral_base + braingpt ==="
neurolink compare \
  --lora-year-max 2022 \
  --config config/forecast/predict_compare_instruct.yaml

echo "=== Eval B (instruct LoRA) ==="
neurolink eval --config config/eval/scenarios/eval_compare_2022_instruct.yaml
cp -f eval/summary.md eval/summary_job2_instruct.md

echo "=== Done ==="
echo "  eval/summary_job2_base.md"
echo "  eval/summary_job2_instruct.md"
ls -la data/models/literature_base/year_max_2022/lora/ data/models/literature_instruct/year_max_2022/lora/ | head

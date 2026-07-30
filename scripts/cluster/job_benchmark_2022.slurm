#!/bin/bash
#SBATCH --job-name=nl-bench22
#SBATCH --output=logs/nl-bench22-%j.out
#SBATCH --error=logs/nl-bench22-%j.err
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

# Benchmark only (no train) — needs literature_base year_max=2022.
# Round A: LoRA-base vs mistral_base vs braingpt (2023-2025) + eval
# Round B: auto if data/models/literature_instruct/year_max_2022/lora/ exists

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

test -d data/models/literature_base/year_max_2022/lora || {
  echo "ERROR: missing literature_base adapter"; exit 1; }

HAS_INSTRUCT=0
if test -d data/models/literature_instruct/year_max_2022/lora; then
  HAS_INSTRUCT=1
  echo "INFO: found literature_instruct adapter — will run Round B"
else
  echo "INFO: no literature_instruct adapter — Round B skipped"
fi

echo "=== Benchmark A: LoRA BASE + mistral_base + braingpt ==="
neurolink compare \
  --lora-year-max 2022 \
  --config config/forecast/predict_compare.yaml

echo "=== Eval A (base LoRA) ==="
neurolink eval --config config/eval/scenarios/eval_compare_2022.yaml
cp -f eval/summary.md eval/summary_job2_base.md

if [[ "$HAS_INSTRUCT" -eq 1 ]]; then
  echo "=== Benchmark B: LoRA INSTRUCT + mistral_base + braingpt ==="
  neurolink compare \
    --lora-year-max 2022 \
    --config config/forecast/predict_compare_instruct.yaml

  echo "=== Eval B (instruct LoRA) ==="
  neurolink eval --config config/eval/scenarios/eval_compare_2022_instruct.yaml
  cp -f eval/summary.md eval/summary_job2_instruct.md
else
  echo "=== Round B skipped ==="
fi

echo "=== Done ==="
echo "  eval/summary_job2_base.md"
[[ "$HAS_INSTRUCT" -eq 1 ]] && echo "  eval/summary_job2_instruct.md"
ls -la data/models/literature_base/year_max_2022/lora/ | head

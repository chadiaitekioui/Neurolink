#!/bin/bash
#SBATCH --job-name=nl-lora25
#SBATCH --output=logs/nl-lora25-%j.out
#SBATCH --error=logs/nl-lora25-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=40:00:00
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Job 3 — Train LoRA year_max=2025, benchmark 2026 (3 LLMs), predict 2027 (LoRA 2025).
# predict_2027.yaml: benchmark_lora_year_max=2025, greedy temp=0, no eval for 2027.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

echo "=== Job 3: train LoRA 2025 ==="
neurolink train-literature \
  --year-max 2025 \
  --config config/forecast/predict_literature.yaml

echo "=== Job 3: benchmark anchor 2025 (year 2026) ==="
neurolink compare \
  --lora-year-max 2025 \
  --config config/forecast/predict_compare.yaml

echo "=== Job 3: eval benchmark 2025 → 2026 ==="
neurolink eval --config config/eval/scenarios/eval_compare_2025.yaml

echo "=== Job 3: inference 2027 (LoRA 2025, no eval) ==="
neurolink predict --config config/forecast/scenarios/predict_2027.yaml

echo "=== Job 3: archive outputs for local download ==="
ARCHIVE="neurolink_outputs_${SLURM_JOB_ID}.tar.gz"
tar czf "$ARCHIVE" \
  data/neurolink.db \
  data/models/literature \
  eval/summary.md \
  eval/summary_job2.md 2>/dev/null || \
tar czf "$ARCHIVE" data/neurolink.db data/models/literature eval/
echo "Archive: $NEUROLINK_ROOT/$ARCHIVE"
ls -lh "$ARCHIVE"

echo "=== Job 3 done ==="

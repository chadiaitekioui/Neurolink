#!/bin/bash
#SBATCH --job-name=nl-stress-eval
#SBATCH --output=logs/nl-stress-eval-%j.out
#SBATCH --error=logs/nl-stress-eval-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Offline stress metrics on existing predictions (no LM regeneration).
# Needs: data/neurolink.db with Job-2 predict run_ids + MiniLM in HF_HOME.
#
# Optional env overrides:
#   PREDICT_RUN_A / PREDICT_RUN_B  — predict run_ids (defaults = bench22 442691)
#   EVAL_RUN_ID                   — Job-2 evaluations snapshot (Round A)
#   CORPUS_SAMPLE                 — default 4000
#   OUT_DIR                       — default eval
#
# Submit from Neurolink root:
#   sbatch scripts/cluster/job_eval_stress_metrics.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

DB_PATH="${DB_PATH:-data/neurolink.db}"
OUT_DIR="${OUT_DIR:-eval}"
CORPUS_SAMPLE="${CORPUS_SAMPLE:-4000}"
# Round A / B from nl-bench22-442691 (override if re-run)
PREDICT_RUN_A="${PREDICT_RUN_A:-benchmark_20260730T143510Z}"
PREDICT_RUN_B="${PREDICT_RUN_B:-benchmark_20260730T195112Z}"
EVAL_RUN_ID="${EVAL_RUN_ID:-eval_20260730T191521Z}"

test -f "$DB_PATH" || { echo "ERROR: missing $DB_PATH"; exit 1; }

echo "=== Stress metrics (MiniLM offline) ==="
echo "DB=$DB_PATH"
echo "OUT_DIR=$OUT_DIR"
echo "RUN_A=$PREDICT_RUN_A  RUN_B=$PREDICT_RUN_B"
echo "EVAL_RUN_ID=$EVAL_RUN_ID  CORPUS_SAMPLE=$CORPUS_SAMPLE"
echo "CUDA=$(python -c 'import torch; print(torch.cuda.is_available())')"

python scripts/eval_stress_metrics.py \
  --db-path "$DB_PATH" \
  --out-dir "$OUT_DIR" \
  --predict-config config/forecast/predict_compare.yaml \
  --predict-run-id "${PREDICT_RUN_A}:round_a" \
  --predict-run-id "${PREDICT_RUN_B}:round_b" \
  --eval-run-id "$EVAL_RUN_ID" \
  --corpus-sample "$CORPUS_SAMPLE"

echo "=== Done ==="
ls -lt "$OUT_DIR"/stress_metrics_*.json | head -5

#!/bin/bash
#SBATCH --job-name=nl-eval-thr55
#SBATCH --output=logs/nl-eval-thr55-%j.out
#SBATCH --error=logs/nl-eval-thr55-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Side experiment: re-eval existing predictions with semantic_threshold=0.55.
# No compare / retrain / reindex — needs prior benchmark preds in data/neurolink.db.
# Reads:  config/eval/scenarios/eval_compare_2022_thr55.yaml
# Writes: eval/summary_thr55.md (+ evaluations rows for this eval run)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

test -f data/neurolink.db || {
  echo "ERROR: missing data/neurolink.db"; exit 1; }

echo "=== Eval thr55: MiniLM match τ=0.55 (stress + corpus_minilm floor) ==="
echo "Config: config/eval/scenarios/eval_compare_2022_thr55.yaml"
neurolink eval --config config/eval/scenarios/eval_compare_2022_thr55.yaml
cp -f eval/summary.md eval/summary_thr55.md

echo "=== Done ==="
echo "  eval/summary_thr55.md"
echo "Look for baseline_corpus_minilm_precision @ k=50 (years 2023–2025)."
grep -n -i "minilm\|precision\|baseline\|threshold\|0\.55" eval/summary_thr55.md | head -80 || true

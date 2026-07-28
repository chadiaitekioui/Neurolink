#!/bin/bash
#SBATCH --job-name=nl-llm50
#SBATCH --output=logs/nl-llm50-%j.out
#SBATCH --error=logs/nl-llm50-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Smoke test: LLM subject extraction on 50 articles (~2-5 min GPU).
# Prefer a DB copy so impact does not wipe the full questions table on prod.
#
# Usage (copy DB first, recommended):
#   cp data/neurolink.db data/neurolink_sample50.db
#   sqlite3 data/neurolink_sample50.db "DELETE FROM questions; DELETE FROM article_segments;"
#
# Option A — first 50 articles without segments:
#   neurolink segment --config config/index/segment_sample50.yaml --device cuda --limit 50
#
# Option B — fixed PMID list (e.g. bench22_v7_results/gt_sample50.json):
#   neurolink segment --config config/index/segment_sample50.yaml --device cuda \
#     --pmids-file bench22_v7_results/gt_sample50.json --force

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

CONFIG="${NEUROLINK_SEGMENT_CONFIG:-config/index/segment_sample50.yaml}"
PMIDS_FILE="${NEUROLINK_PMIDS_FILE:-}"
LIMIT="${NEUROLINK_SEGMENT_LIMIT:-50}"

echo "=== LLM subject sample test ==="
neurolink --db data/neurolink_sample50.db status

SEG_ARGS=(segment --config "$CONFIG" --device cuda)
if [[ -n "$PMIDS_FILE" ]]; then
  SEG_ARGS+=(--pmids-file "$PMIDS_FILE" --force)
else
  SEG_ARGS+=(--limit "$LIMIT")
fi

echo "--- segment (${SEG_ARGS[*]}) ---"
neurolink "${SEG_ARGS[@]}"

echo "--- impact + embed (sample DB only) ---"
neurolink impact --config config/index/impact_sample50.yaml
neurolink embed --config config/index/embed_sample50.yaml
neurolink --db data/neurolink_sample50.db status
echo "=== Sample test done ==="

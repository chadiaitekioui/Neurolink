#!/bin/bash
#SBATCH --job-name=nl-index-llm
#SBATCH --output=logs/nl-index-llm-%j.out
#SBATCH --error=logs/nl-index-llm-%j.err
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

# Job 1 — index with LLM subject extraction (replaces PubMedBERT segmentation):
#   Mistral-7B-Instruct (4-bit) titre+abstract → 1 research direction
#   then impact → embed (MiniLM).
# Config: config/index/segment.yaml (extraction_mode: llm).
# Resumes articles missing from article_segments. Wipe segments/questions/embeddings first if re-indexing.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

echo "=== Job 1: LLM extract → impact → embed ==="
neurolink status
echo "--- LLM subject extraction (Mistral-7B-Instruct, no PubMedBERT) ---"
neurolink segment --config config/index/segment.yaml --device cuda
echo "--- Impact (rebuild questions from LLM subjects) ---"
neurolink impact --config config/index/impact.yaml
echo "--- Embed MiniLM ---"
neurolink embed --config config/index/embed.yaml
neurolink status
echo "=== Job 1 done ==="

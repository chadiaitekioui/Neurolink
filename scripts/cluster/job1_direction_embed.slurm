#!/bin/bash
#SBATCH --job-name=nl-index-llm
#SBATCH --output=logs/nl-index-llm-%j.out
#SBATCH --error=logs/nl-index-llm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=40:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Job 1 — LLM research-direction extraction → impact → embed:
#   Mistral-7B-Instruct (4-bit) title+abstract → 1 direction
#   then impact → embed (MiniLM).
# Config: config/index/direction.yaml
# Resumes articles missing from article_segments.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

echo "=== Job 1: directions → impact → embed ==="
neurolink status
echo "--- LLM direction extraction (Mistral-7B-Instruct) ---"
neurolink direction --config config/index/direction.yaml --device cuda
echo "--- Impact (rebuild questions from directions) ---"
neurolink impact --config config/index/impact.yaml
echo "--- Embed MiniLM ---"
neurolink embed --config config/index/embed.yaml
neurolink status
echo "=== Job 1 done ==="

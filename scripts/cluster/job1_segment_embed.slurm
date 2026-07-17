#!/bin/bash
#SBATCH --job-name=nl-segment
#SBATCH --output=logs/nl-segment-%j.out
#SBATCH --error=logs/nl-segment-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Job 1 — complete index on GPU: segment (resume) → impact → embed (MiniLM).
# Stages already done are skipped. Requires collect (+ optional early impact) first.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

echo "=== Job 1: segment → impact → embed (skip if already done) ==="
neurolink status
neurolink segment --config config/index/segment.yaml --device cuda
neurolink impact --config config/index/impact.yaml
neurolink embed --config config/index/embed.yaml
neurolink status
echo "=== Job 1 done ==="

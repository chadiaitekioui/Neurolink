#!/bin/bash
#SBATCH --job-name=nl-lora-base26
#SBATCH --output=logs/nl-lora-base26-%j.out
#SBATCH --error=logs/nl-lora-base26-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Separate experiment B — Train LoRA on Mistral-7B-v0.1 (base), year_max=2026.
# Uses data/neurolink.db (NOT neurolink_specific.db).
# Adapter → data/models/literature_base/year_max_2026/lora/
# Inspired by job_train_lora_base.slurm (year_max=2022).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs data/models/literature_base

echo "=== Train LoRA BASE (Mistral-7B-v0.1) year_max=2026 ==="
neurolink train-literature \
  --year-max 2026 \
  --config config/forecast/train_lora_base.yaml

echo "=== Done — adapter: data/models/literature_base/year_max_2026/lora/ ==="
ls -la data/models/literature_base/year_max_2026/lora/ | head

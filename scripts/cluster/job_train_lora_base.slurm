#!/bin/bash
#SBATCH --job-name=nl-lora-base
#SBATCH --output=logs/nl-lora-base-%j.out
#SBATCH --error=logs/nl-lora-base-%j.err
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

# Train LoRA on Mistral-7B-v0.1 (base), year_max=2022.
# Adapter → data/models/literature_base/year_max_2022/lora/
# Parallel-safe with job_train_lora_instruct.slurm (separate GPU + adapter_dir).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs data/models/literature_base

echo "=== Train LoRA BASE (Mistral-7B-v0.1) year_max=2022 ==="
neurolink train-literature \
  --year-max 2022 \
  --config config/forecast/train_lora_base.yaml

echo "=== Done — adapter: data/models/literature_base/year_max_2022/lora/ ==="
ls -la data/models/literature_base/year_max_2022/lora/ | head

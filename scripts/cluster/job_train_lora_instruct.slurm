#!/bin/bash
#SBATCH --job-name=nl-lora-inst
#SBATCH --output=logs/nl-lora-inst-%j.out
#SBATCH --error=logs/nl-lora-inst-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Train LoRA on Mistral-7B-Instruct-v0.2 with Instruct [INST] prompt, year_max=2022.
# Adapter → data/models/literature_instruct/year_max_2022/lora/
# Overwrites previous Instruct adapter (continuation-style) if present.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs data/models/literature_instruct

echo "=== Train LoRA INSTRUCT (prompt_style=instruct) year_max=2022 ==="
# Remove old checkpoints so resume does not mix continuation-trained steps.
rm -rf data/models/literature_instruct/year_max_2022/checkpoints

neurolink train-literature \
  --year-max 2022 \
  --config config/forecast/train_lora_instruct.yaml

echo "=== Done — adapter: data/models/literature_instruct/year_max_2022/lora/ ==="
ls -la data/models/literature_instruct/year_max_2022/lora/ | head

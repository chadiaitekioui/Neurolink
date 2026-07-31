#!/bin/bash
#SBATCH --job-name=nl-lora-spec22
#SBATCH --output=logs/nl-lora-spec22-%j.out
#SBATCH --error=logs/nl-lora-spec22-%j.err
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

# Specific-direction experiment A — Train LoRA-base year_max=2022 on neurolink_specific.db.
# Adapter → data/models/literature_base_specific/year_max_2022/lora/
# Does NOT overwrite data/models/literature_base/year_max_2022.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs data/models/literature_base_specific

test -f data/neurolink_specific.db || {
  echo "ERROR: missing data/neurolink_specific.db — run job_reindex_specific.slurm first"; exit 1; }

echo "=== Train LoRA BASE (specific DB) year_max=2022 ==="
neurolink train-literature \
  --year-max 2022 \
  --config config/forecast/train_lora_base_specific.yaml

echo "=== Done — adapter: data/models/literature_base_specific/year_max_2022/lora/ ==="
ls -la data/models/literature_base_specific/year_max_2022/lora/ | head

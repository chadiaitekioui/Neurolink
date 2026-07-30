#!/bin/bash
#SBATCH --job-name=nl-gen-smoke
#SBATCH --output=logs/nl-gen-smoke-%j.out
#SBATCH --error=logs/nl-gen-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Quick coherence check (2023, k=10) before full job_benchmark_2022.
# Output: eval/gen_smoke_*.json + printed directions in the .out log.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval
test -d data/models/literature_base/year_max_2022/lora || {
  echo "ERROR: missing literature_base adapter"; exit 1; }

echo "=== Generation smoke (inspect before bench22) ==="
python scripts/gen_smoke_predict.py --config config/forecast/gen_smoke.yaml
echo "=== Done — check eval/gen_smoke_*.json and this log ==="
ls -lt eval/gen_smoke_*.json | head -3

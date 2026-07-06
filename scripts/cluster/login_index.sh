#!/bin/bash
# Index stages that need HTTP — run on LOGIN node only.
# Usage: bash scripts/cluster/login_index.sh

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

export TRANSFORMERS_OFFLINE=0
unset HF_HUB_OFFLINE

echo "=== Login index: collect + impact (HTTP) ==="
neurolink init-db
neurolink collect --config config/index/collect.yaml
neurolink impact --config config/index/impact.yaml
neurolink status
echo "=== Done. Submit GPU job: sbatch scripts/cluster/job1_segment_embed.slurm ==="

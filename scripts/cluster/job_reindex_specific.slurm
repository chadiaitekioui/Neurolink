#!/bin/bash
#SBATCH --job-name=nl-reindex-spec
#SBATCH --output=logs/nl-reindex-spec-%j.out
#SBATCH --error=logs/nl-reindex-spec-%j.err
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

# Specific-direction experiment A — prepare parallel DB then re-extract:
#   1) Copy data/neurolink.db → data/neurolink_specific.db (if missing)
#   2) Clear segments / questions / predictions / evaluations (keep articles + citations)
#   3) direction --force → impact → embed (specific configs)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
NEUROLINK_ENV_QUIET=1
source scripts/cluster/env.sh

SRC_DB="data/neurolink.db"
DST_DB="data/neurolink_specific.db"

mkdir -p logs data

test -f "$SRC_DB" || {
  echo "ERROR: missing source DB $SRC_DB — run login-index first"; exit 1; }

if [[ ! -f "$DST_DB" ]]; then
  echo "=== Copy $SRC_DB → $DST_DB ==="
  cp -f "$SRC_DB" "$DST_DB"
else
  echo "=== Using existing $DST_DB ==="
fi

echo "=== Clear derived tables (keep articles + citations) ==="
python - <<'PY'
import sqlite3
from pathlib import Path

db = Path("data/neurolink_specific.db")
conn = sqlite3.connect(db)
for table in ("article_segments", "questions", "predictions", "evaluations"):
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DELETE FROM {table}")
        print(f"  cleared {table}: {n} rows")
    except sqlite3.OperationalError as exc:
        print(f"  skip {table}: {exc}")
conn.commit()
conn.close()
print("done")
PY

echo "=== LLM direction extraction (--force, specific prompt via code) ==="
neurolink direction --config config/index/direction_specific.yaml --device cuda --force
echo "=== Impact (rebuild questions) ==="
neurolink impact --config config/index/impact_specific.yaml
echo "=== Embed MiniLM ==="
neurolink embed --config config/index/embed_specific.yaml
neurolink --db data/neurolink_specific.db status
echo "=== Reindex specific done ==="

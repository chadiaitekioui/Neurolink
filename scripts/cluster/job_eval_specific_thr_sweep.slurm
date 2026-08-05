#!/bin/bash
#SBATCH --job-name=nl-eval-spec-thr
#SBATCH --output=logs/nl-eval-spec-thr-%j.out
#SBATCH --error=logs/nl-eval-spec-thr-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=tnv@v100
#SBATCH --export=ALL
#SBATCH -C v100-32g

# Specific-direction threshold sweep: τ ∈ {0.55, 0.60, 0.70}.
# Re-eval only (no compare / retrain) on data/neurolink_specific.db.
# Needs prior benchmark-specific predictions.
#
# Writes:
#   eval/summary_specific_thr55.md
#   eval/summary_specific_thr60.md
#   eval/summary_specific_thr70.md
#   eval/summary_specific_thr_sweep.md  (compact floor table)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env.sh

mkdir -p logs eval

test -f data/neurolink_specific.db || {
  echo "ERROR: missing data/neurolink_specific.db"; exit 1; }

run_thr() {
  local thr="$1"
  local cfg="config/eval/scenarios/eval_compare_2022_specific_thr${thr}.yaml"
  local out="eval/summary_specific_thr${thr}.md"
  echo "=== Eval specific τ=0.${thr} ==="
  echo "Config: ${cfg}"
  neurolink eval --config "${cfg}"
  cp -f eval/summary.md "${out}"
  echo "Wrote ${out}"
}

run_thr 55
run_thr 60
run_thr 70

# Compact extract: corpus_minilm P@50 + model P@50 per year
python - <<'PY'
from pathlib import Path
import re
from collections import defaultdict

root = Path("eval")
lines = ["# Specific threshold sweep — corpus MiniLM floor & model P@50", ""]
for thr in ("55", "60", "70"):
    path = root / f"summary_specific_thr{thr}.md"
    if not path.exists():
        lines.append(f"## τ=0.{thr} — MISSING {path}")
        continue
    by = defaultdict(list)
    run_id = None
    for line in path.read_text().splitlines():
        if run_id is None and "run_id:" in line:
            m = re.search(r"`([^`]+)`", line)
            if m:
                run_id = m.group(1)
        m = re.match(r"\| (\d+) \| (\S+) \| (\S+) \| (\S+) \| ([0-9.]+) \|", line)
        if not m:
            continue
        y, mo, me, k, v = m.groups()
        if k != "50":
            continue
        if me in ("baseline_corpus_minilm_precision", "precision@k", "beyond_conditional_corpus_minilm"):
            by[me].append((int(y), mo, float(v)))
    lines.append(f"## τ=0.{thr} (`{run_id}`)")
    lines.append("")
    # floor unique years
    seen = {}
    for y, mo, v in by["baseline_corpus_minilm_precision"]:
        seen[y] = v
    if seen:
        mean = sum(seen.values()) / len(seen)
        lines.append(
            f"- **corpus_minilm P@50**: "
            + ", ".join(f"{y}={v:.3f}" for y, v in sorted(seen.items()))
            + f" (mean={mean:.3f})"
        )
    for mo in ("literature_lora", "mistral_base", "braingpt"):
        vals = [v for y, m, v in by["precision@k"] if m == mo]
        if vals:
            lines.append(
                f"- **{mo} P@50**: "
                + ", ".join(f"{y}={v:.3f}" for y, m, v in sorted(by["precision@k"]) if m == mo)
                + f" (mean={sum(vals)/len(vals):.3f})"
            )
        bvals = [v for y, m, v in by["beyond_conditional_corpus_minilm"] if m == mo]
        if bvals:
            lines.append(
                f"- **{mo} conditional_beyond@50**: mean={sum(bvals)/len(bvals):.3f}"
            )
    lines.append("")

out = root / "summary_specific_thr_sweep.md"
out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}")
print(out.read_text())
PY

echo "=== Sweep done ==="
ls -la eval/summary_specific_thr*.md

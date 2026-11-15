#!/usr/bin/env bash
# Calibration experiments 1-5 (gnn_interp.experiments) in order, from the repo root. Each step logs to
# runs/<step>.log; if BUCKET is set, runs/ is synced to $BUCKET/runs/<hostname> after each step.
# Rerunning resumes (balance/sizes skip finished rows). Run inside tmux.
# Usage: BUCKET=gs://my-bucket bash scripts/run_experiments.sh [first_step]
set -euo pipefail
steps=(memory balance sizes gnn cv)
declare -A args=(
  [memory]="--out runs/memory"
  [balance]="--out runs/balance --n 70"
  [sizes]="--out runs/sizes --steps runs/balance --n 100"
  [gnn]="--out runs/gnn --data runs/sizes/data.pt"
  [cv]="--out runs/cv --data runs/sizes/data.pt"
)
started=0; first="${1:-memory}"
for s in "${steps[@]}"; do
  [[ "$s" == "$first" ]] && started=1; [[ $started == 1 ]] || continue
  echo "=== $s ($(date))"; mkdir -p runs
  .venv/bin/python -m gnn_interp.experiments "$s" ${args[$s]} 2>&1 | tee "runs/$s.log"
  [[ -n "${BUCKET:-}" ]] && gcloud storage rsync -r runs "$BUCKET/runs/$(hostname)" --quiet
done
echo "all done. Stop the VM when finished."

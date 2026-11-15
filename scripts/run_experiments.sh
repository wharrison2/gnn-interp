#!/usr/bin/env bash
# Calibration experiments (gnn_interp.experiments) in order, from the repo root. Each step logs to
# runs/<step>.log; if BUCKET is set, runs/ is synced to $BUCKET/runs/<hostname> after each step.
# Rerunning resumes (balance/sizes skip finished rows). Run inside tmux. DTYPE (default float32) is the
# model precision for balance/batch/sizes; switch to bfloat16 if `memory` shows float32 does not fit.
# Usage: BUCKET=gs://my-bucket bash scripts/run_experiments.sh [first_step]
set -euo pipefail
DTYPE=${DTYPE:-float32}
# step name -> experiment + args; *_it steps repeat balance/sizes/probe on the instruct model (smaller sizes run)
steps=(memory balance balance_it batch precision sizes sizes_it gnn cv probe probe_it)
declare -A cmd=(
  [memory]="memory --out runs/memory"
  [balance]="balance --out runs/balance --n 70 --dtype $DTYPE"
  [balance_it]="balance --out runs/balance_it --n 70 --dtype $DTYPE --model instruct"
  [batch]="batch --out runs/batch --steps runs/balance --n 5 --dtype $DTYPE"
  [precision]="precision --out runs/precision --steps runs/balance --n 10"
  [sizes]="sizes --out runs/sizes --steps runs/balance --n 100 --dtype $DTYPE"
  [sizes_it]="sizes --out runs/sizes_it --steps runs/balance_it --n 30 --caps 4096 --dtype $DTYPE --model instruct"
  [gnn]="gnn --out runs/gnn --data runs/sizes/data.pt"
  [cv]="cv --out runs/cv --data runs/sizes/data.pt"
  [probe]="probe --out runs/probe --steps runs/balance"
  [probe_it]="probe --out runs/probe_it --steps runs/balance_it"
)
started=0; first="${1:-memory}"
for s in "${steps[@]}"; do
  [[ "$s" == "$first" ]] && started=1; [[ $started == 1 ]] || continue
  echo "=== $s ($(date))"; mkdir -p runs
  .venv/bin/python -m gnn_interp.experiments ${cmd[$s]} 2>&1 | tee "runs/$s.log"
  if [[ -n "${BUCKET:-}" ]]; then gcloud storage rsync -r runs "$BUCKET/runs/$(hostname)" --quiet; fi
done
echo "all done. Stop the VM when finished."

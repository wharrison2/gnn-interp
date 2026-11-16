#!/usr/bin/env bash
# Calibration experiments (gnn_interp.experiments) in order, from the repo root. Each step logs to
# runs/<step>.log; if BUCKET is set, runs/ is synced to $BUCKET/runs/<short hostname> every SYNC_MINUTES
# (default 10) during a step and after it. Rerunning resumes (balance/sizes/store skip finished rows, cv
# reuses finished folds). Run inside tmux. DTYPE (default bfloat16) is the model precision.
# Preemption: runs/.resume records the current step and settings; scripts/vm_startup.sh relaunches from it
# on boot. It is removed when the run finishes or a step fails (kept if the step was killed by a signal).
# Usage: BUCKET=gs://my-bucket [SKIP="step ..."] bash scripts/run_experiments.sh [first_step]
set -euo pipefail
DTYPE=${DTYPE:-bfloat16}  # float32 does not fit an L4 (24 GB): attribution OOMs
# step name -> experiment + args. *_it steps (instruct model) are not in the default order: with the current
# prompt the instruct model ignores the step format (2026-11-16); run them explicitly if the prompt changes.
# precision (fp32 vs bf16) needs > 24 GB: fp32 OOMs even in generation on an L4; run it on a larger GPU.
steps=(memory balance batch sizes store confidence gnn cv probe)
declare -A cmd=(
  [memory]="memory --out runs/memory"
  [balance]="balance --out runs/balance --n 70 --dtype $DTYPE"
  [balance_it]="balance --out runs/balance_it --n 70 --dtype $DTYPE --model instruct"
  [batch]="batch --out runs/batch --steps runs/balance --n 5 --dtype $DTYPE"
  [precision]="precision --out runs/precision --steps runs/balance --n 10"
  [store]="store --out runs/store --steps runs/balance --keys runs/sizes/sizes.jsonl --caps 4096 --dtype $DTYPE"
  [confidence]="confidence --out runs/confidence --steps runs/balance --dtype $DTYPE"
  [sizes]="sizes --out runs/sizes --steps runs/balance --n 40 --caps 4096 2048 --dtype $DTYPE"  # ~34 s/graph at 4096
  [sizes_it]="sizes --out runs/sizes_it --steps runs/balance_it --n 30 --caps 4096 --dtype $DTYPE --model instruct"
  [gnn]="gnn --out runs/gnn --data runs/sizes/data.pt"
  [cv]="cv --out runs/cv --data runs/sizes/data.pt"
  [probe]="probe --out runs/probe --steps runs/balance"
  [probe_it]="probe --out runs/probe_it --steps runs/balance_it"
)
push() { if [[ -n "${BUCKET:-}" ]]; then gcloud storage rsync -r runs "$BUCKET/runs/$(hostname -s)" --quiet; fi; }
marker=runs/.resume
started=0; first="${1:-memory}"
for s in "${steps[@]}"; do
  [[ "$s" == "$first" ]] && started=1; [[ $started == 1 ]] || continue
  [[ " ${SKIP:-} " == *" $s "* ]] && { echo "=== skip $s"; continue; }
  echo "=== $s ($(date))"; mkdir -p runs
  printf 'STEP=%q\nBUCKET=%q\nSKIP=%q\nDTYPE=%q\n' "$s" "${BUCKET:-}" "${SKIP:-}" "$DTYPE" > "$marker"
  (while sleep $(( ${SYNC_MINUTES:-10} * 60 )); do push || true; done) & syncer=$!
  set +e; .venv/bin/python -m gnn_interp.experiments ${cmd[$s]} 2>&1 | tee "runs/$s.log"; rc=${PIPESTATUS[0]}; set -e
  kill $syncer 2>/dev/null || true
  push || true
  if (( rc != 0 )); then
    (( rc < 128 )) && rm -f "$marker"  # >= 128: killed by a signal (e.g. preemption): keep the marker
    echo "=== $s failed (exit $rc)"; exit "$rc"
  fi
done
rm -f "$marker"; push
echo "all done. Delete the VM when finished (scripts/delete_vm.sh)."

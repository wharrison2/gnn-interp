#!/usr/bin/env bash
# Restart a stopped (preempted) spot VM automatically: a Cloud Scheduler job calls the Compute Engine API's
# instances.start every 5 minutes (a no-op while the VM runs; vm_startup.sh then resumes the run).
# The job also restarts a VM stopped on purpose: pause it first, or use delete_vm.sh, which removes it.
# Usage (local): PROJECT=... [VM=gnn-vm] bash scripts/autostart.sh create ZONE | pause | resume | delete
set -euo pipefail
: "${PROJECT:?set PROJECT}"
VM=${VM:-gnn-vm}; LOCATION=${LOCATION:-us-central1}; JOB="restart-$VM"
job=(--project="$PROJECT" --location="$LOCATION")
case "${1:-}" in
  create)
    zone=${2:?usage: autostart.sh create ZONE}
    gcloud services enable cloudscheduler.googleapis.com --project="$PROJECT"
    sa="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
    gcloud scheduler jobs create http "$JOB" "${job[@]}" --schedule="*/5 * * * *" --http-method=POST \
      --uri="https://compute.googleapis.com/compute/v1/projects/$PROJECT/zones/$zone/instances/$VM/start" \
      --oauth-service-account-email="$sa" --attempt-deadline=60s ;;
  pause | resume | delete)  # retried: changes are refused ("cannot be queued") until the job's next run (<= 5 min)
    gcloud scheduler jobs describe "$JOB" "${job[@]}" > /dev/null 2>&1 || { echo "no job $JOB"; [[ $1 == delete ]]; exit; }
    for _ in $(seq 8); do gcloud scheduler jobs "$1" "$JOB" "${job[@]}" --quiet 2>/dev/null && exit 0; sleep 45; done
    gcloud scheduler jobs "$1" "$JOB" "${job[@]}" --quiet && exit 0
    exit 1 ;;
  *) echo "usage: autostart.sh create ZONE | pause | resume | delete" >&2; exit 2 ;;
esac

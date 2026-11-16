#!/usr/bin/env bash
# Delete the VM and its auto-restart job (local). Sync results to the bucket first: the disk is deleted too.
# Usage: PROJECT=... [VM=gnn-vm] bash scripts/delete_vm.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"
VM=${VM:-gnn-vm}
bash "$(dirname "$0")/autostart.sh" delete  # first, or it would restart the VM
zone=$(gcloud compute instances list --project="$PROJECT" --filter="name=$VM" --format='value(zone.basename())')
[[ -n "$zone" ]] || { echo "no VM $VM" >&2; exit 1; }
gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$zone" --quiet

#!/usr/bin/env bash
# Create a 1x L4 GPU VM (local machine), trying each zone until one has capacity.
# Usage: PROJECT=my-project bash scripts/create_vm.sh   (SPOT=1 for spot pricing; VM, ZONES, MACHINE, IMAGE_FAMILY override)
# Installs vm_startup.sh (resume after a restart). With SPOT=1 a preempted VM stops (disk kept) and an
# auto-restart job is created (autostart.sh; AUTOSTART=0 to skip). Delete with delete_vm.sh.
set -euo pipefail
: "${PROJECT:?set PROJECT}"
VM=${VM:-gnn-vm}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c"}
MACHINE=${MACHINE:-g2-standard-8}                                   # 8 vCPU, 32 GB RAM, 1x L4 (24 GB)
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2404-nvidia-580}  # Deep Learning VM, drivers preinstalled
prov=(); [[ "${SPOT:-0}" == 1 ]] && prov=(--provisioning-model=SPOT --instance-termination-action=STOP)
for z in $ZONES; do
  echo "trying $z"
  if gcloud compute instances create "$VM" --project="$PROJECT" --zone="$z" --machine-type="$MACHINE" \
      --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release \
      --boot-disk-size=200GB --boot-disk-type=pd-balanced --maintenance-policy=TERMINATE \
      --scopes=storage-full --metadata=install-nvidia-driver=True \
      --metadata-from-file=startup-script="$(dirname "$0")/vm_startup.sh" ${prov[@]+"${prov[@]}"}; then
    echo "created $VM in $z"
    if [[ "${SPOT:-0}" == 1 && "${AUTOSTART:-1}" == 1 ]]; then bash "$(dirname "$0")/autostart.sh" create "$z"; fi
    exit 0
  fi
done
echo "no zone had capacity" >&2; exit 1

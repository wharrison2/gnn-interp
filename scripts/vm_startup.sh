#!/usr/bin/env bash
# GCE startup script (runs as root on every boot; installed by create_vm.sh). After a spot preemption and
# restart: re-enables swap and, if a run was interrupted (~/gnn-interp/runs/.resume, written by
# run_experiments.sh), relaunches it from the interrupted step in tmux session "resume" as the repo's owner,
# appending to ~/resume.log.
swapon /swapfile 2>/dev/null || true
for marker in /home/*/gnn-interp/runs/.resume; do
  [[ -f "$marker" ]] || continue
  repo=${marker%/runs/.resume}; user=$(stat -c %U "$repo")
  for _ in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 5; done  # wait for the GPU driver
  sudo -u "$user" -H tmux new-session -d -s resume \
    "cd '$repo' && set -a && . runs/.resume && set +a && bash scripts/run_experiments.sh \"\$STEP\" >> ~/resume.log 2>&1"
done

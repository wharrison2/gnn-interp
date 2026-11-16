#!/usr/bin/env bash
# One-time GPU VM setup, run from the repo root: venv on the image's CUDA torch, deps, Gemma 2 2B + transcoder
# download, fast tests. Needs HF_TOKEN (gated Gemma) in the environment or in ./.env or ~/.env.
# Usage: git clone https://github.com/wharrison2/gnn-interp.git && cd gnn-interp && bash scripts/setup_vm.sh
set -euo pipefail
for f in .env ~/.env; do [[ -f $f ]] && { set -a; source "$f"; set +a; }; done
: "${HF_TOKEN:?set HF_TOKEN (environment, ./.env or ~/.env)}"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Loading the model in bf16 briefly needs > 32 GB of host RAM (g2-standard-8): add swap for the spike.
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 48G /swapfile && sudo chmod 600 /swapfile && sudo mkswap -q /swapfile && sudo swapon /swapfile
fi
grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null  # survive reboots
python3 -c "import ensurepip" 2>/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv; }
python3 -m venv --system-site-packages .venv  # reuse the image's CUDA torch
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok')"
.venv/bin/python - <<'PY'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"])
for repo in ["google/gemma-2-2b", "google/gemma-2-2b-it", "mntss/gemma-scope-transcoders"]:
    print(snapshot_download(repo))
PY
.venv/bin/python -m pytest -q -m "not slow"
echo "setup done"

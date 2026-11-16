# gnn-interp

Predict whether an arithmetic step written by Gemma 2 2B is correct, from its attribution graph
(circuit-tracer + GemmaScope transcoders), using graph neural networks.

## Modules (`gnn_interp/`)
- `gemma`: load the model, greedy generation, step localisation, transcoder error norms.
- `graph_data`: circuit-tracer graphs to PyG `Data`, features, dataset and splits.
- `models`: GNN graph classifiers.
- `baselines`: non-graph baselines for comparison.
- `train`: training and evaluation loops.
- `metrics`: classification metrics (positive class = incorrect step).
- `problems`: arithmetic problem generation, few-shot prompts, step grading (labels).
- `attribution`: build + prune attribution graphs per labelled step, write the dataset manifest.
- `experiments`: calibration runs that size the real run (memory, class balance + residual stream, attribution
  batch size, fp32 vs bf16, attribution time and graph size/quality, GNN epoch time, CV dry run, residual probe); `python -m gnn_interp.experiments --help`.

## Install
```
pip install -r requirements.txt
```

## Testing
Run from `src/` (or the parent repo: `pytest src`); config and `pythonpath` are in `pytest.ini`, no `.venv` needed.
```
python -m pytest                # everything
python -m pytest -m "not slow"  # fast subset (~1 s)
```
Layout: `tests/test_<module>.py` mirrors `gnn_interp/` (baselines split into `_text`, `_crv`, `_cv`);
`tests/synthetic.py` holds the fake-graph builders, `tests/conftest.py` the shared fixtures.
Markers: `slow` (training / cross-validation), `needs_circuit_tracer` (real `circuit_tracer`; skipped if not
installed), `gpu` (CUDA device; skipped if unavailable). Tests need no network or model downloads.
`conftest.py` sets `OMP_NUM_THREADS=1` (sklearn GBM + torch segfault workaround; harmless on Colab).

## Colab
```
!git clone https://github.com/wharrison2/gnn-interp && cd gnn-interp && pip install -r requirements.txt
```
Then run from the repo root (`from gnn_interp import ...`).

## GPU VM (Google Cloud)
`scripts/`: `create_vm.sh` (local; 1x L4 VM, needs `PROJECT`), `setup_vm.sh` (on the VM from the repo root; needs
`HF_TOKEN` in the environment or `.env`, which is gitignored), `run_experiments.sh` (all calibration experiments;
optional `BUCKET` to sync `runs/`). Usage is in each script's header.

Spot VMs (`SPOT=1 create_vm.sh`): a preempted VM stops with its disk; a Cloud Scheduler job (`autostart.sh`)
restarts it every 5 min, and `vm_startup.sh` re-enables swap and relaunches the interrupted step from
`runs/.resume`. Steps resume from their jsonl rows (truncated last lines are dropped) and CV from finished folds.
Delete with `delete_vm.sh` (removes the restart job first; otherwise it restarts a VM stopped on purpose).

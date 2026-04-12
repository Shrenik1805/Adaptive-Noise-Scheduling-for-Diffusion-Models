# Adaptive Noise Scheduling for Diffusion Models

Production-oriented research codebase for class-conditional adaptive noise scheduling on CIFAR-10.

## Abstract

Most diffusion models use a fixed noise schedule for every sample, regardless of class complexity. This project implements **Adaptive Noise Scheduling (ANS)**, where a learned `ScheduleNet` predicts class-conditional per-timestep noise levels `beta_t(x)`. The method jointly optimizes diffusion reconstruction quality, schedule efficiency, and temporal smoothness. The code includes rigorous invariants, gradient checks, end-to-end tests, DDPM/DDIM samplers, evaluation tooling (FID + timing), visualization utilities, Streamlit demo, and a NeurIPS-style paper skeleton.

## Repository Layout

```
adaptive_diffusion/
├── config.py
├── models/
├── losses/
├── data/
├── training/
├── evaluation/
├── visualization/
├── tests/
├── app/
└── paper/
```

## One-command Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r adaptive_diffusion/requirements.txt
```

Windows PowerShell setup:

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r adaptive_diffusion/requirements.txt
```

## Colab Setup Cells

```python
!pip install -q torch torchvision einops timm torch-fidelity wandb torchmetrics \
             streamlit matplotlib seaborn scipy tqdm hypothesis pytest pytest-cov \
             clean-fid accelerate safetensors
```

```python
import torch
assert torch.cuda.is_available(), "GPU required. Runtime > Change runtime type > T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

```python
import random, numpy as np, torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

## Training (Fair Comparison Protocol)

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train --schedule-mode adaptive --checkpoint-dir ./checkpoints_adaptive --sample-dir ./samples_adaptive
```

Train fixed-cosine baseline with identical architecture/hyperparameters:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train --schedule-mode fixed_cosine --checkpoint-dir ./checkpoints_fixed --sample-dir ./samples_fixed
```

Apple Silicon (M-series) training:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train --schedule-mode adaptive --device mps
source .venv/bin/activate && python -m adaptive_diffusion.train --schedule-mode fixed_cosine --device mps
```

Windows + NVIDIA GPU training:

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m adaptive_diffusion.train --schedule-mode adaptive --device cuda
python -m adaptive_diffusion.train --schedule-mode fixed_cosine --device cuda
```

Check MPS availability:

```bash
source .venv/bin/activate && python -c "import torch; print('mps_built=', torch.backends.mps.is_built(), 'mps_available=', torch.backends.mps.is_available())"
```

## Run Tests

```bash
source .venv/bin/activate && python -m pytest adaptive_diffusion/tests/ -v --tb=short
```

## Run Streamlit Demo

```bash
source .venv/bin/activate && streamlit run adaptive_diffusion/app/streamlit_app.py
```

## WandB Logging

By default, trainer uses:

- project: `adaptive-diffusion`
- mode: `online` (override with `WANDB_MODE=offline`)

Example:

```bash
export WANDB_API_KEY=<your_key>
source .venv/bin/activate && python -m adaptive_diffusion.train
```

## Evaluation and Plots

The following modules are available:

- `adaptive_diffusion/evaluation/metrics.py`: efficiency frontier, per-class metrics, schedule diversity.
- `adaptive_diffusion/visualization/schedule_viz.py`: publication-quality schedule and frontier figures.

Run full paired-checkpoint evaluation:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.evaluate \
  --adaptive-checkpoint ./checkpoints_adaptive/<best_adaptive>.pt \
  --fixed-checkpoint ./checkpoints_fixed/<best_fixed>.pt \
  --device mps \
  --num-fid-samples 10000 \
  --samples-per-class 1000 \
  --repeats 3
```

Interpretation guidance:

- At equal step count, adaptive is not expected to be faster per step.
- The core claim is frontier shift: adaptive reaches the same FID at fewer steps (or lower time at matched quality).
- Always compare separately trained checkpoints (`adaptive` vs `fixed_cosine`), not fixed-sampler counterfactuals from an adaptive-trained denoiser.

## One-command Full Pipeline

Run adaptive training + fixed baseline training + paired evaluation + summary artifacts:

```bash
chmod +x scripts/run_pair_experiment.sh
DEVICE=mps EPOCHS=100 BATCH_SIZE=128 LR=2e-4 RUN_TAG=paper_v1 scripts/run_pair_experiment.sh
```

Windows PowerShell equivalent:

```powershell
.\\.venv\\Scripts\\Activate.ps1
scripts\\run_pair_experiment.ps1 -Device cuda -Epochs 100 -BatchSize 128 -LearningRate 2e-4 -RunTag paper_v1
```

Multi-seed protocol (recommended for paper claims):

```bash
source .venv/bin/activate && python scripts/run_multi_seed_experiment.py \
  --python python \
  --device mps \
  --seeds 42 43 44 \
  --epochs 100 \
  --batch-size 128 \
  --lr 2e-4 \
  --run-prefix paper_multiseed
```

Configurable environment variables:

- `DEVICE` (default: `mps`)
- `EPOCHS` (default: `100`)
- `BATCH_SIZE` (default: `128`)
- `LR` (default: `2e-4`)
- `WANDB_MODE` (default: `online`)
- `NUM_FID_SAMPLES` (default: `10000`)
- `SAMPLES_PER_CLASS` (default: `1000`)
- `REPEATS` (default: `3`)
- `RUN_TAG` (default: `default`)

Outputs include:

- `adaptive_diffusion/analysis_<RUN_TAG>/efficiency_frontier.csv`
- `adaptive_diffusion/analysis_<RUN_TAG>/per_class_metrics.csv`
- `adaptive_diffusion/analysis_<RUN_TAG>/summary_metrics.csv`
- `adaptive_diffusion/analysis_<RUN_TAG>/summary_report.md`

## Paper + Derivation

- Paper skeleton: `adaptive_diffusion/paper/main.tex`
- Derivation notebook: `adaptive_diffusion/paper/derivation_notebook.ipynb`

## Reproducibility Notes

- All hyperparameters are centralized in `DiffusionConfig`.
- Forward-pass invariants are asserted in `ScheduleNet`.
- Gradcheck tests cover novel schedule regularizers.
- EMA checkpoints and config snapshots are saved during training.
- Checkpoint loading in evaluation restores full training config to avoid architecture mismatch.
- Data loader seeding is deterministic (`seed + worker_id`) for reproducible runs.

## GitHub CI

- Workflow: `.github/workflows/ci.yml`
- Runs on `ubuntu-latest`, `macos-latest`, and `windows-latest`
- Checks: `black --check` and full `pytest` suite

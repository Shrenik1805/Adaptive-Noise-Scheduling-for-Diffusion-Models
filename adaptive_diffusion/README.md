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

## One-command Training

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train
```

Useful overrides:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train --epochs 100 --batch-size 128 --lr 2e-4 --device cuda
```

Apple Silicon (M-series) training:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.train --device mps
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

Run full evaluation pipeline:

```bash
source .venv/bin/activate && python -m adaptive_diffusion.evaluate --checkpoint ./checkpoints/<best>.pt --device mps
```

## Paper + Derivation

- Paper skeleton: `adaptive_diffusion/paper/main.tex`
- Derivation notebook: `adaptive_diffusion/paper/derivation_notebook.ipynb`

## Reproducibility Notes

- All hyperparameters are centralized in `DiffusionConfig`.
- Forward-pass invariants are asserted in `ScheduleNet`.
- Gradcheck tests cover novel schedule regularizers.
- EMA checkpoints and config snapshots are saved during training.

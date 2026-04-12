# Quality Check Report

Date: 2026-04-11  
Project: `adaptive_diffusion`

## 1. Build and Test Status

- `black --check adaptive_diffusion/`: **PASS**
- `pytest adaptive_diffusion/tests/ -v --tb=short`: **PASS** (`25 passed`)
- `pytest --cov=adaptive_diffusion`: **PASS** (tests pass; line coverage `52%`)
- Python bytecode compile (`py_compile` on entrypoints/modules): **PASS**

## 2. Mathematical and Runtime Invariants

Validated:

- Schedule output bounded in `[0, 0.02]`: **PASS**
- `alpha_bar` strictly monotonic decreasing: **PASS**
- Boundary check at `T=1000`:  
  - `alpha_bar[0] > 0.99`: **PASS** (`0.999189`)  
  - `alpha_bar[-1] < 0.01`: **PASS** (`0.000131`)
- `ScheduleNet` / loss gradcheck tests: **PASS**
- 500-step stress run (synthetic batches) finite loss throughout: **PASS**

## 3. Methodology Integrity (Upgraded)

Validated methodological fixes:

- Fair baseline enabled via separate training modes:
  - `schedule_mode=adaptive`
  - `schedule_mode=fixed_cosine`
- Loss regularizers are applied only to adaptive mode; fixed mode trains with pure diffusion loss.
- Paired-checkpoint evaluation pipeline implemented for adaptive-vs-fixed comparisons.
- FID frontier reports mean/std over repeated runs.
- Per-class FID now uses class-matched real CIFAR-10 references.
- Counterfactual fixed-sampler results from adaptive weights are retained only as diagnostic, not primary baseline claims.

## 4. Performance and Sampling Sanity

Smoke benchmark (CPU tiny config):

- DDPM (200 steps): `0.559 s`
- DDIM (50 steps): `0.137 s`
- DDPM/DDIM ratio: `4.08x` (DDIM faster): **PASS**

Schedule diversity:

- Mean pairwise schedule L2 diversity: `0.03326`
- Gate `> 0.001`: **PASS**

## 5. End-to-End Components Present

Implemented modules:

- Config + core models + adaptive losses
- CIFAR dataloader
- Trainer (EMA, warmup-cosine scheduler, gradient clipping, checkpointing)
- Evaluation (FID wrapper, timing, metrics)
- Visualization (schedule grid, frontier, per-class speedup)
- Streamlit app
- Paper skeleton + derivation notebook
- CLI scripts:
  - `python -m adaptive_diffusion.train`
  - `python -m adaptive_diffusion.evaluate`

## 6. Environment-Specific Limitations

Current sandbox constraints prevented full final artifact generation:

- `torch.backends.mps.is_available()` is `False` in this environment (Apple GPU unavailable here).
- `wandb` service sockets are restricted in this sandbox.
- `pdflatex` is not installed here (paper compile not executed).

Because of these constraints, the following are **not finalized in this sandbox run**:

- 100-epoch full training on MPS
- Public wandb dashboard link
- Final FID tables with real trained checkpoint values
- TeX PDF compile check

## 7. Ready-to-Run Commands on Your MacBook Pro (M4 Pro)

```bash
source .venv/bin/activate
python -c "import torch; print('mps built=', torch.backends.mps.is_built(), 'mps available=', torch.backends.mps.is_available())"
export WANDB_API_KEY=<your_key>
python -m adaptive_diffusion.train --schedule-mode adaptive --device mps --checkpoint-dir ./checkpoints_adaptive --sample-dir ./samples_adaptive
python -m adaptive_diffusion.train --schedule-mode fixed_cosine --device mps --checkpoint-dir ./checkpoints_fixed --sample-dir ./samples_fixed
python -m adaptive_diffusion.evaluate --adaptive-checkpoint ./checkpoints_adaptive/<best>.pt --fixed-checkpoint ./checkpoints_fixed/<best>.pt --device mps --num-fid-samples 10000 --samples-per-class 1000 --repeats 3
streamlit run adaptive_diffusion/app/streamlit_app.py
```

One-command orchestration is now available:

```bash
chmod +x scripts/run_pair_experiment.sh
DEVICE=mps EPOCHS=100 BATCH_SIZE=128 LR=2e-4 RUN_TAG=paper_v1 scripts/run_pair_experiment.sh
```

## 8. Overall Assessment

- Codebase implementation quality: **High**
- Test reliability on implemented core math/model paths: **High**
- Production run completeness in this sandbox: **Partial (environment-limited)**
- Production run completeness on your local M4 Pro with MPS + wandb login: **Ready**

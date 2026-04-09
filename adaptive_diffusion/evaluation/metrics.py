"""Evaluation metrics for adaptive diffusion."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from adaptive_diffusion.evaluation.fid import FIDCalculator
from adaptive_diffusion.evaluation.sampling import sample_with_timing
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel
from adaptive_diffusion.visualization.schedule_viz import CIFAR10_CLASSES


@torch.no_grad()
def _cifar_reference(
    num_images: int,
    data_root: str = "./data",
    batch_size: int = 256,
) -> Tensor:
    """Load normalized CIFAR-10 test images for FID reference."""
    Path(data_root).mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        ),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    out: list[Tensor] = []
    for images, _ in loader:
        out.append(images)
        if sum(t.shape[0] for t in out) >= num_images:
            break
    return torch.cat(out, dim=0)[:num_images]


@torch.no_grad()
def compute_efficiency_frontier(
    model: AdaptiveDiffusionModel,
    num_step_values: list[int] | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Compute FID/time tradeoff table across sampling step counts."""
    if num_step_values is None:
        num_step_values = [10, 25, 50, 100, 200, 500, 1000]
    device = next(model.parameters()).device
    labels = torch.arange(batch_size, device=device) % model.config.num_classes
    real_images = _cifar_reference(num_images=batch_size).to(device)
    fid_calc = FIDCalculator(device=device)

    rows: list[dict[str, float | int]] = []
    for steps in num_step_values:
        adaptive_images, (adaptive_mean, _) = sample_with_timing(
            model=model,
            class_labels=labels,
            method="ddim_adaptive",
            num_steps=steps,
        )
        fixed_images, (fixed_mean, _) = sample_with_timing(
            model=model,
            class_labels=labels,
            method="ddim_fixed",
            num_steps=steps,
        )
        rows.append(
            {
                "num_steps": steps,
                "fid_adaptive": fid_calc.compute(
                    adaptive_images, real_images, cache_key="cifar_ref"
                ),
                "fid_fixed": fid_calc.compute(
                    fixed_images, real_images, cache_key="cifar_ref"
                ),
                "time_adaptive": adaptive_mean,
                "time_fixed": fixed_mean,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_per_class_metrics(
    model: AdaptiveDiffusionModel, samples_per_class: int = 128
) -> pd.DataFrame:
    """Compute adaptive-vs-fixed metrics for each CIFAR-10 class."""
    device = next(model.parameters()).device
    fid_calc = FIDCalculator(device=device)
    real_images = _cifar_reference(num_images=samples_per_class).to(device)
    rows: list[dict[str, float | str]] = []

    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        labels = torch.full(
            (samples_per_class,), class_idx, device=device, dtype=torch.long
        )
        adaptive_images, (adaptive_time, _) = sample_with_timing(
            model=model,
            class_labels=labels,
            method="ddim_adaptive",
            num_steps=model.config.num_sample_steps_ddim,
        )
        fixed_images, (fixed_time, _) = sample_with_timing(
            model=model,
            class_labels=labels,
            method="ddim_fixed",
            num_steps=model.config.num_sample_steps_ddim,
        )
        rows.append(
            {
                "class": class_name,
                "fid_adaptive": fid_calc.compute(
                    adaptive_images, real_images, cache_key=f"class_{class_idx}"
                ),
                "fid_fixed": fid_calc.compute(
                    fixed_images, real_images, cache_key=f"class_{class_idx}"
                ),
                "time_adaptive": adaptive_time,
                "time_fixed": fixed_time,
                "speedup_ratio": fixed_time / max(adaptive_time, 1e-9),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_schedule_diversity(model: AdaptiveDiffusionModel) -> float:
    """Compute mean pairwise L2 distance of class-specific beta schedules."""
    device = next(model.parameters()).device
    labels = torch.arange(model.config.num_classes, device=device)
    class_emb = model.schedule_class_embedding(labels)
    schedules = model.schedule_net.get_full_schedule(class_emb)  # (C, T)
    if schedules.ndim != 2:
        raise ValueError("Expected schedules with shape (num_classes, num_timesteps).")

    distances: list[Tensor] = []
    for i, j in combinations(range(schedules.shape[0]), 2):
        distances.append(torch.norm(schedules[i] - schedules[j], p=2))
    if not distances:
        return 0.0
    return float(torch.stack(distances).mean().item())

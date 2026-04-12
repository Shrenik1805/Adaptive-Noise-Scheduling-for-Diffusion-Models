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


def _cifar_test_loader(data_root: str = "./data", batch_size: int = 512) -> DataLoader:
    """Create CIFAR-10 test dataloader with normalized images."""
    Path(data_root).mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        ),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def _cifar_reference(
    num_images: int,
    class_idx: int | None = None,
    data_root: str = "./data",
    batch_size: int = 512,
) -> Tensor:
    """Load CIFAR-10 real images, optionally filtered by class."""
    loader = _cifar_test_loader(data_root=data_root, batch_size=batch_size)
    collected: list[Tensor] = []
    for images, labels in loader:
        if class_idx is not None:
            mask = labels == class_idx
            if mask.any():
                collected.append(images[mask])
        else:
            collected.append(images)
        if sum(batch.shape[0] for batch in collected) >= num_images:
            break
    if not collected:
        raise ValueError(
            f"No CIFAR-10 images collected for class_idx={class_idx} and num_images={num_images}."
        )
    return torch.cat(collected, dim=0)[:num_images]


def _ensure_comparable_models(
    adaptive_model: AdaptiveDiffusionModel,
    fixed_model: AdaptiveDiffusionModel,
) -> None:
    """Validate paired-model comparison assumptions."""
    if adaptive_model.config.num_timesteps != fixed_model.config.num_timesteps:
        raise ValueError("Adaptive and fixed models must use the same num_timesteps.")
    if adaptive_model.config.num_classes != fixed_model.config.num_classes:
        raise ValueError("Adaptive and fixed models must use the same num_classes.")
    if adaptive_model.config.image_size != fixed_model.config.image_size:
        raise ValueError("Adaptive and fixed models must use the same image_size.")
    if not adaptive_model.is_adaptive:
        raise ValueError(
            "adaptive_model must be trained with schedule_mode='adaptive'."
        )
    if fixed_model.is_adaptive:
        raise ValueError(
            "fixed_model must be trained with schedule_mode='fixed_cosine'."
        )


def _bootstrap_mean_ci(
    values: Tensor,
    num_bootstrap: int = 200,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return bootstrap confidence interval for the sample mean."""
    if values.numel() <= 1:
        scalar = float(values.mean().item())
        return scalar, scalar
    n = values.numel()
    generator = torch.Generator(device=values.device)
    generator.manual_seed(1234)
    indices = torch.randint(
        low=0,
        high=n,
        size=(num_bootstrap, n),
        generator=generator,
        device=values.device,
    )
    sampled = values[indices].mean(dim=1)
    alpha = (1.0 - confidence) * 0.5
    lower = float(torch.quantile(sampled, alpha).item())
    upper = float(torch.quantile(sampled, 1.0 - alpha).item())
    return lower, upper


@torch.no_grad()
def compute_efficiency_frontier(
    adaptive_model: AdaptiveDiffusionModel,
    fixed_model: AdaptiveDiffusionModel,
    num_step_values: list[int] | None = None,
    num_images: int = 10000,
    repeats: int = 3,
) -> pd.DataFrame:
    """Compute quality-time frontier using fair paired checkpoints.

    Notes
    -----
    Both models are sampled with DDIM at the same step count, then compared against
    a shared real CIFAR-10 reference set.
    """
    _ensure_comparable_models(adaptive_model=adaptive_model, fixed_model=fixed_model)
    if num_step_values is None:
        num_step_values = [10, 25, 50, 100, 200, 500, 1000]
    if num_images <= 0 or repeats <= 0:
        raise ValueError("num_images and repeats must be positive.")

    device = next(adaptive_model.parameters()).device
    fid_calc = FIDCalculator(device=device)
    real_images = _cifar_reference(num_images=num_images).to(device)
    labels = torch.arange(num_images, device=device) % adaptive_model.config.num_classes

    rows: list[dict[str, float | int]] = []
    for steps in num_step_values:
        fid_adaptive_runs: list[float] = []
        fid_fixed_runs: list[float] = []
        time_adaptive_runs: list[float] = []
        time_fixed_runs: list[float] = []
        for run_idx in range(repeats):
            adaptive_images, (adaptive_time_mean, _) = sample_with_timing(
                model=adaptive_model,
                class_labels=labels,
                method="ddim",
                num_steps=steps,
            )
            fixed_images, (fixed_time_mean, _) = sample_with_timing(
                model=fixed_model,
                class_labels=labels,
                method="ddim",
                num_steps=steps,
            )
            fid_adaptive_runs.append(
                fid_calc.compute(
                    adaptive_images,
                    real_images,
                    cache_key=f"cifar_ref_frontier_{num_images}",
                )
            )
            fid_fixed_runs.append(
                fid_calc.compute(
                    fixed_images,
                    real_images,
                    cache_key=f"cifar_ref_frontier_{num_images}",
                )
            )
            time_adaptive_runs.append(adaptive_time_mean)
            time_fixed_runs.append(fixed_time_mean)

        adaptive_fid_tensor = torch.tensor(fid_adaptive_runs)
        fixed_fid_tensor = torch.tensor(fid_fixed_runs)
        adaptive_time_tensor = torch.tensor(time_adaptive_runs)
        fixed_time_tensor = torch.tensor(time_fixed_runs)
        fid_delta = adaptive_fid_tensor - fixed_fid_tensor
        time_delta = adaptive_time_tensor - fixed_time_tensor
        fid_delta_ci_low, fid_delta_ci_high = _bootstrap_mean_ci(fid_delta)
        time_delta_ci_low, time_delta_ci_high = _bootstrap_mean_ci(time_delta)
        paired_win_rate = float(
            ((fid_delta <= 0.0) & (time_delta <= 0.0)).float().mean().item()
        )

        rows.append(
            {
                "num_steps": int(steps),
                "fid_adaptive_mean": float(adaptive_fid_tensor.mean().item()),
                "fid_adaptive_std": float(
                    adaptive_fid_tensor.std(unbiased=False).item()
                ),
                "fid_fixed_mean": float(fixed_fid_tensor.mean().item()),
                "fid_fixed_std": float(fixed_fid_tensor.std(unbiased=False).item()),
                "time_adaptive_mean": float(adaptive_time_tensor.mean().item()),
                "time_adaptive_std": float(
                    adaptive_time_tensor.std(unbiased=False).item()
                ),
                "time_fixed_mean": float(fixed_time_tensor.mean().item()),
                "time_fixed_std": float(fixed_time_tensor.std(unbiased=False).item()),
                "fid_delta_adaptive_minus_fixed_mean": float(fid_delta.mean().item()),
                "fid_delta_ci_low": fid_delta_ci_low,
                "fid_delta_ci_high": fid_delta_ci_high,
                "time_delta_adaptive_minus_fixed_mean": float(time_delta.mean().item()),
                "time_delta_ci_low": time_delta_ci_low,
                "time_delta_ci_high": time_delta_ci_high,
                "paired_joint_win_rate": paired_win_rate,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_per_class_metrics(
    adaptive_model: AdaptiveDiffusionModel,
    fixed_model: AdaptiveDiffusionModel,
    samples_per_class: int = 1000,
    repeats: int = 3,
) -> pd.DataFrame:
    """Compute class-conditional adaptive-vs-fixed metrics with matched real sets."""
    _ensure_comparable_models(adaptive_model=adaptive_model, fixed_model=fixed_model)
    if samples_per_class <= 0 or repeats <= 0:
        raise ValueError("samples_per_class and repeats must be positive.")

    device = next(adaptive_model.parameters()).device
    fid_calc = FIDCalculator(device=device)
    rows: list[dict[str, float | str]] = []

    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        labels = torch.full(
            (samples_per_class,), class_idx, device=device, dtype=torch.long
        )
        real_images = _cifar_reference(
            num_images=samples_per_class, class_idx=class_idx
        ).to(device)

        fid_adaptive_runs: list[float] = []
        fid_fixed_runs: list[float] = []
        time_adaptive_runs: list[float] = []
        time_fixed_runs: list[float] = []
        for _ in range(repeats):
            adaptive_images, (adaptive_time_mean, _) = sample_with_timing(
                model=adaptive_model,
                class_labels=labels,
                method="ddim",
                num_steps=adaptive_model.config.num_sample_steps_ddim,
            )
            fixed_images, (fixed_time_mean, _) = sample_with_timing(
                model=fixed_model,
                class_labels=labels,
                method="ddim",
                num_steps=fixed_model.config.num_sample_steps_ddim,
            )
            fid_adaptive_runs.append(
                fid_calc.compute(
                    adaptive_images,
                    real_images,
                    cache_key=f"class_real_{class_idx}_{samples_per_class}",
                )
            )
            fid_fixed_runs.append(
                fid_calc.compute(
                    fixed_images,
                    real_images,
                    cache_key=f"class_real_{class_idx}_{samples_per_class}",
                )
            )
            time_adaptive_runs.append(adaptive_time_mean)
            time_fixed_runs.append(fixed_time_mean)

        adaptive_fid_tensor = torch.tensor(fid_adaptive_runs)
        fixed_fid_tensor = torch.tensor(fid_fixed_runs)
        adaptive_time_tensor = torch.tensor(time_adaptive_runs)
        fixed_time_tensor = torch.tensor(time_fixed_runs)
        fid_delta = adaptive_fid_tensor - fixed_fid_tensor
        time_delta = adaptive_time_tensor - fixed_time_tensor
        fid_delta_ci_low, fid_delta_ci_high = _bootstrap_mean_ci(fid_delta)
        time_delta_ci_low, time_delta_ci_high = _bootstrap_mean_ci(time_delta)
        rows.append(
            {
                "class": class_name,
                "fid_adaptive_mean": float(adaptive_fid_tensor.mean().item()),
                "fid_adaptive_std": float(
                    adaptive_fid_tensor.std(unbiased=False).item()
                ),
                "fid_fixed_mean": float(fixed_fid_tensor.mean().item()),
                "fid_fixed_std": float(fixed_fid_tensor.std(unbiased=False).item()),
                "time_adaptive_mean": float(adaptive_time_tensor.mean().item()),
                "time_adaptive_std": float(
                    adaptive_time_tensor.std(unbiased=False).item()
                ),
                "time_fixed_mean": float(fixed_time_tensor.mean().item()),
                "time_fixed_std": float(fixed_time_tensor.std(unbiased=False).item()),
                "speedup_ratio": float(
                    fixed_time_tensor.mean().item()
                    / max(adaptive_time_tensor.mean().item(), 1e-9)
                ),
                "fid_delta_adaptive_minus_fixed_mean": float(fid_delta.mean().item()),
                "fid_delta_ci_low": fid_delta_ci_low,
                "fid_delta_ci_high": fid_delta_ci_high,
                "time_delta_adaptive_minus_fixed_mean": float(time_delta.mean().item()),
                "time_delta_ci_low": time_delta_ci_low,
                "time_delta_ci_high": time_delta_ci_high,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_schedule_diversity(model: AdaptiveDiffusionModel) -> float:
    """Compute mean pairwise L2 distance of class-specific beta schedules."""
    if not model.is_adaptive:
        return 0.0
    if model.schedule_net is None or model.schedule_class_embedding is None:
        raise ValueError("Adaptive schedule diversity requires schedule modules.")
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

"""Sampling benchmark utilities."""

from __future__ import annotations

import statistics
import time

import torch
from torch import Tensor

from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel
from adaptive_diffusion.utils.device import synchronize


@torch.no_grad()
def sample_with_timing(
    model: AdaptiveDiffusionModel,
    class_labels: Tensor,
    method: str,
    num_steps: int,
) -> tuple[Tensor, tuple[float, float]]:
    """Sample images and return timing mean/std over repeated runs.

    Parameters
    ----------
    model : AdaptiveDiffusionModel
        Diffusion model.
    class_labels : Tensor
        Class labels for generation.
    method : str
        One of ``{"ddpm", "ddim", "ddim_adaptive", "ddim_fixed"}``.
    num_steps : int
        Number of reverse steps.

    Returns
    -------
    tuple[Tensor, tuple[float, float]]
        Generated images and tuple of ``(mean_seconds, std_seconds)``.
    """
    valid_methods = {"ddpm", "ddim", "ddim_adaptive", "ddim_fixed"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {sorted(valid_methods)}."
        )

    device = class_labels.device
    synchronize(device)

    # GPU warmup
    for _ in range(3):
        if method == "ddpm":
            _ = model.ddpm_sample(class_labels=class_labels, num_steps=num_steps)
        elif method in {"ddim", "ddim_adaptive"}:
            _ = model.ddim_sample(class_labels=class_labels, num_steps=num_steps)
        else:
            _ = model.fixed_schedule_sample(
                class_labels=class_labels, num_steps=num_steps
            )
        synchronize(device)

    times: list[float] = []
    images: Tensor | None = None
    for _ in range(5):
        start = time.perf_counter()
        if method == "ddpm":
            images, _ = model.ddpm_sample(
                class_labels=class_labels, num_steps=num_steps
            )
        elif method in {"ddim", "ddim_adaptive"}:
            images, _ = model.ddim_sample(
                class_labels=class_labels, num_steps=num_steps
            )
        else:
            images = model.fixed_schedule_sample(
                class_labels=class_labels, num_steps=num_steps
            )
        synchronize(device)
        times.append(time.perf_counter() - start)

    if images is None:
        raise RuntimeError("Sampling failed to produce images.")
    return images, (statistics.mean(times), statistics.pstdev(times))

"""Learning rate scheduler utilities."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.0,
) -> LambdaLR:
    """Create cosine scheduler with linear warmup.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer instance.
    warmup_steps : int
        Number of warmup steps.
    total_steps : int
        Total training steps.
    min_lr_scale : float, optional
        Final learning rate scale relative to base LR.

    Returns
    -------
    LambdaLR
        Scheduler with warmup + cosine annealing.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

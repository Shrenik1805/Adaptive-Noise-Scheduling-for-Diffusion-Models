"""Adaptive multi-objective loss for class-conditional diffusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.models.schedule_net import ScheduleNet


class AdaptiveLoss(nn.Module):
    """Multi-objective loss combining diffusion, efficiency, and smoothness terms."""

    def __init__(
        self, schedule_net: ScheduleNet | None, config: DiffusionConfig
    ) -> None:
        """Initialize loss module.

        Parameters
        ----------
        schedule_net : ScheduleNet
            Adaptive beta schedule model.
        config : DiffusionConfig
            Global hyperparameter configuration.
        """
        super().__init__()
        self.schedule_net = schedule_net
        self.lambda_efficiency = config.lambda_efficiency
        self.lambda_smoothness = config.lambda_smoothness
        self.use_schedule_regularizers = (
            config.schedule_mode == "adaptive" and schedule_net is not None
        )
        self._last_components: dict[str, float] = {}

    def diffusion_loss(self, noise_pred: Tensor, noise_target: Tensor) -> Tensor:
        """Compute epsilon prediction loss."""
        return F.mse_loss(noise_pred, noise_target, reduction="mean")

    def efficiency_loss(self, schedules: Tensor) -> Tensor:
        """Compute mean beta level across batch and time."""
        if schedules.ndim == 1:
            schedules = schedules.unsqueeze(0)
        return schedules.mean()

    def smoothness_loss(self, schedules: Tensor) -> Tensor:
        """Compute temporal smoothness penalty over beta schedule."""
        if schedules.ndim == 1:
            schedules = schedules.unsqueeze(0)
        if schedules.shape[-1] < 2:
            raise ValueError("Schedule length must be >= 2 to compute smoothness loss.")
        deltas = torch.diff(schedules, dim=-1)
        return (deltas**2).mean()

    def forward(
        self,
        noise_pred: Tensor,
        noise_target: Tensor,
        class_embeddings: Tensor | None,
    ) -> dict[str, Tensor]:
        """Compute full adaptive objective.

        Parameters
        ----------
        noise_pred : Tensor
            Predicted noise ``epsilon_theta``.
        noise_target : Tensor
            Ground-truth Gaussian noise ``epsilon``.
        class_embeddings : Tensor
            Class embedding tensor of shape ``(B, feature_dim)``.

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing all objective components and total scalar loss.
        """
        loss_diffusion = self.diffusion_loss(
            noise_pred=noise_pred, noise_target=noise_target
        )
        if self.use_schedule_regularizers:
            if class_embeddings is None or class_embeddings.ndim != 2:
                raise ValueError(
                    "Adaptive regularization requires class_embeddings with shape "
                    f"(B, feature_dim). Received {None if class_embeddings is None else tuple(class_embeddings.shape)}."
                )
            schedules = self.schedule_net.get_full_schedule(class_embeddings)
            if schedules.ndim != 2:
                raise ValueError(
                    f"Expected schedule tensor with shape (B, T), got {tuple(schedules.shape)}."
                )
            loss_efficiency = self.efficiency_loss(schedules=schedules)
            loss_smoothness = self.smoothness_loss(schedules=schedules)
        else:
            loss_efficiency = torch.zeros_like(loss_diffusion)
            loss_smoothness = torch.zeros_like(loss_diffusion)

        total = (
            loss_diffusion
            + self.lambda_efficiency * loss_efficiency
            + self.lambda_smoothness * loss_smoothness
        )

        self._last_components = {
            "loss_diffusion": float(loss_diffusion.detach().cpu()),
            "loss_efficiency_weighted": float(
                (self.lambda_efficiency * loss_efficiency).detach().cpu()
            ),
            "loss_smoothness_weighted": float(
                (self.lambda_smoothness * loss_smoothness).detach().cpu()
            ),
            "total_loss": float(total.detach().cpu()),
        }
        return {
            "loss": total,
            "loss_diffusion": loss_diffusion,
            "loss_efficiency": loss_efficiency,
            "loss_smoothness": loss_smoothness,
            "lambda_efficiency": torch.tensor(
                self.lambda_efficiency,
                device=total.device,
                dtype=total.dtype,
            ),
            "lambda_smoothness": torch.tensor(
                self.lambda_smoothness,
                device=total.device,
                dtype=total.dtype,
            ),
        }

    def loss_weights_summary(self) -> dict[str, Any]:
        """Return relative contribution of each objective to total loss.

        Returns
        -------
        dict[str, Any]
            Ratios and weighted components from the latest forward pass.
        """
        if not self._last_components:
            return {
                "ready": False,
                "message": "Call forward() at least once before requesting summary.",
            }
        total = max(self._last_components["total_loss"], 1e-12)
        return {
            "ready": True,
            "diffusion_ratio": self._last_components["loss_diffusion"] / total,
            "efficiency_ratio": self._last_components["loss_efficiency_weighted"]
            / total,
            "smoothness_ratio": self._last_components["loss_smoothness_weighted"]
            / total,
            **self._last_components,
        }

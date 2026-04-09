"""Gradient and gradcheck tests."""

from __future__ import annotations

from dataclasses import replace

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.losses.adaptive_loss import AdaptiveLoss
from adaptive_diffusion.models.schedule_net import ScheduleNet


def _grad_config() -> DiffusionConfig:
    return DiffusionConfig(
        num_timesteps=16,
        feature_dim=8,
        schedule_hidden_dim=16,
        schedule_num_layers=2,
        unet_base_channels=16,
        unet_channel_multipliers=(1, 2),
    )


def test_schedule_net_gradcheck() -> None:
    config = _grad_config()
    model = ScheduleNet(config).double()
    model.eval()
    class_emb = torch.randn(
        2, config.feature_dim, dtype=torch.double, requires_grad=True
    )
    t = torch.rand(2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        model, (class_emb, t), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_efficiency_loss_gradcheck() -> None:
    config = _grad_config()
    loss_module = AdaptiveLoss(
        schedule_net=ScheduleNet(config).double(), config=replace(config)
    )
    schedules = (
        torch.rand(2, config.num_timesteps, dtype=torch.double, requires_grad=True)
        * 0.01
    )
    assert torch.autograd.gradcheck(
        loss_module.efficiency_loss, (schedules,), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_smoothness_loss_gradcheck() -> None:
    config = _grad_config()
    loss_module = AdaptiveLoss(
        schedule_net=ScheduleNet(config).double(), config=replace(config)
    )
    schedules = (
        torch.rand(2, config.num_timesteps, dtype=torch.double, requires_grad=True)
        * 0.01
    )
    assert torch.autograd.gradcheck(
        loss_module.smoothness_loss, (schedules,), eps=1e-6, atol=1e-4, rtol=1e-3
    )

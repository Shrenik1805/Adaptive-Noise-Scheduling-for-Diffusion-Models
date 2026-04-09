"""Unit tests for adaptive multi-objective loss."""

from __future__ import annotations

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.losses.adaptive_loss import AdaptiveLoss
from adaptive_diffusion.models.schedule_net import ScheduleNet


def _loss_setup(
    lambda_efficiency: float = 0.01,
) -> tuple[DiffusionConfig, AdaptiveLoss]:
    config = DiffusionConfig(
        num_timesteps=64,
        feature_dim=16,
        schedule_hidden_dim=32,
        lambda_efficiency=lambda_efficiency,
        unet_base_channels=16,
        unet_channel_multipliers=(1, 2),
    )
    schedule_net = ScheduleNet(config)
    loss_module = AdaptiveLoss(schedule_net=schedule_net, config=config)
    return config, loss_module


def test_loss_dict_keys() -> None:
    config, loss_module = _loss_setup()
    pred = torch.randn(4, 3, 16, 16, requires_grad=True)
    target = torch.randn_like(pred)
    class_emb = torch.randn(4, config.feature_dim)
    out = loss_module(pred, target, class_emb)
    expected = {
        "loss",
        "loss_diffusion",
        "loss_efficiency",
        "loss_smoothness",
        "lambda_efficiency",
        "lambda_smoothness",
    }
    assert expected == set(out.keys())


def test_efficiency_loss_positive() -> None:
    config, loss_module = _loss_setup()
    class_emb = torch.randn(4, config.feature_dim)
    schedules = loss_module.schedule_net.get_full_schedule(class_emb)
    assert float(loss_module.efficiency_loss(schedules).detach()) >= 0.0


def test_smoothness_loss_positive() -> None:
    config, loss_module = _loss_setup()
    class_emb = torch.randn(4, config.feature_dim)
    schedules = loss_module.schedule_net.get_full_schedule(class_emb)
    assert float(loss_module.smoothness_loss(schedules).detach()) >= 0.0


def test_total_loss_backward() -> None:
    config, loss_module = _loss_setup()
    pred = torch.randn(4, 3, 16, 16, requires_grad=True)
    target = torch.randn_like(pred)
    class_emb = torch.randn(4, config.feature_dim)
    out = loss_module(pred, target, class_emb)
    out["loss"].backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(out["loss"]).item()


def test_lambda_scaling() -> None:
    config_a, loss_a = _loss_setup(lambda_efficiency=0.01)
    class_emb = torch.randn(4, config_a.feature_dim)
    schedules = loss_a.schedule_net.get_full_schedule(class_emb)
    eff = loss_a.efficiency_loss(schedules).detach()
    contrib_a = float(config_a.lambda_efficiency * eff)
    contrib_b = float((2.0 * config_a.lambda_efficiency) * eff)
    assert contrib_b > contrib_a
    assert abs(contrib_b / max(contrib_a, 1e-8) - 2.0) < 1e-6

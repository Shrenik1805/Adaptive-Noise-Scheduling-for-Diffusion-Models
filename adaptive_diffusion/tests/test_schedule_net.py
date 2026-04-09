"""Unit tests for ScheduleNet."""

from __future__ import annotations

from dataclasses import replace

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.models.schedule_net import ScheduleNet


def _config() -> DiffusionConfig:
    return DiffusionConfig(num_timesteps=1000, feature_dim=32, schedule_hidden_dim=64)


def test_output_shape() -> None:
    config = _config()
    model = ScheduleNet(config)
    class_emb = torch.randn(4, config.feature_dim)
    t = torch.rand(4)
    out = model(class_emb, t)
    assert out.shape == (4,)


def test_output_range() -> None:
    config = _config()
    model = ScheduleNet(config)
    class_emb = torch.randn(8, config.feature_dim)
    t = torch.rand(8)
    out = model(class_emb, t)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 0.02)


def test_full_schedule_shape() -> None:
    config = _config()
    model = ScheduleNet(config)
    class_emb = torch.randn(config.feature_dim)
    schedule = model.get_full_schedule(class_emb)
    assert schedule.shape == (config.num_timesteps,)


def test_alpha_bar_monotonic() -> None:
    config = _config()
    model = ScheduleNet(config)
    class_emb = torch.randn(config.feature_dim)
    alpha_bar = model.get_alpha_bar(class_emb)
    assert torch.all(torch.diff(alpha_bar) < 0)


def test_alpha_bar_boundary() -> None:
    config = _config()
    model = ScheduleNet(config)
    class_emb = torch.randn(config.feature_dim)
    alpha_bar = model.get_alpha_bar(class_emb)
    assert float(alpha_bar[0].detach()) > 0.99
    assert float(alpha_bar[-1].detach()) < 0.01


def test_different_classes_different_schedules() -> None:
    config = _config()
    model = ScheduleNet(config)
    emb_1 = torch.randn(config.feature_dim)
    emb_2 = torch.randn(config.feature_dim) + 1.5
    sched_1 = model.get_full_schedule(emb_1)
    sched_2 = model.get_full_schedule(emb_2)
    assert torch.mean(torch.abs(sched_1 - sched_2)).item() > 1e-4


def test_gradient_flow() -> None:
    config = replace(_config(), num_timesteps=64)
    model = ScheduleNet(config)
    class_emb = torch.randn(4, config.feature_dim, requires_grad=True)
    t = torch.rand(4, requires_grad=True)
    loss = model(class_emb, t).mean()
    loss.backward()
    has_grad = [
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ]
    assert all(has_grad)

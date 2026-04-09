"""Unit tests for adaptive diffusion model wrappers."""

from __future__ import annotations

from dataclasses import replace

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel


def _tiny_config() -> DiffusionConfig:
    return DiffusionConfig(
        image_size=16,
        feature_dim=16,
        schedule_hidden_dim=32,
        unet_base_channels=16,
        unet_channel_multipliers=(1, 2),
        unet_attention_resolutions=(8,),
        num_timesteps=200,
        batch_size=4,
    )


def test_q_sample_shape() -> None:
    config = _tiny_config()
    model = AdaptiveDiffusionModel(config)
    x0 = torch.randn(3, config.in_channels, config.image_size, config.image_size)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    t = torch.tensor([0, 10, config.num_timesteps - 1], dtype=torch.long)
    xt, noise = model.q_sample(x0=x0, t=t, class_labels=labels)
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape


def test_q_sample_noise_levels() -> None:
    config = _tiny_config()
    model = AdaptiveDiffusionModel(config)
    x0 = torch.randn(2, config.in_channels, config.image_size, config.image_size)
    labels = torch.tensor([1, 1], dtype=torch.long)

    t0 = torch.zeros(2, dtype=torch.long)
    xt0, noise0 = model.q_sample(x0=x0, t=t0, class_labels=labels)
    assert torch.mean((xt0 - x0) ** 2) < torch.mean((noise0 - x0) ** 2)

    tlast = torch.full((2,), config.num_timesteps - 1, dtype=torch.long)
    xt_last, noise_last = model.q_sample(x0=x0, t=tlast, class_labels=labels)
    assert torch.mean((xt_last - noise_last) ** 2) < torch.mean((xt_last - x0) ** 2)


def test_p_losses_returns_dict() -> None:
    config = _tiny_config()
    model = AdaptiveDiffusionModel(config)
    x0 = torch.randn(
        config.batch_size, config.in_channels, config.image_size, config.image_size
    )
    labels = torch.randint(0, config.num_classes, (config.batch_size,))
    loss_dict = model.p_losses(x0=x0, class_labels=labels)
    expected = {
        "loss",
        "loss_diffusion",
        "loss_efficiency",
        "loss_smoothness",
        "lambda_efficiency",
        "lambda_smoothness",
    }
    assert expected.issubset(loss_dict.keys())


def test_ddim_faster_than_ddpm() -> None:
    config = replace(_tiny_config(), num_timesteps=300)
    model = AdaptiveDiffusionModel(config)
    labels = torch.tensor([0], dtype=torch.long)
    _, ddpm_time = model.ddpm_sample(
        class_labels=labels, num_steps=config.num_timesteps
    )
    _, ddim_time = model.ddim_sample(class_labels=labels, num_steps=50)
    assert ddim_time < ddpm_time

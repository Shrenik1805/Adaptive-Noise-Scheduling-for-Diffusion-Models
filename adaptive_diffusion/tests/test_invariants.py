"""Mathematical invariant tests."""

from __future__ import annotations

import pytest
import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel
from adaptive_diffusion.models.schedule_net import ScheduleNet
from adaptive_diffusion.models.unet import UNet


def _tiny_config() -> DiffusionConfig:
    return DiffusionConfig(
        image_size=16,
        feature_dim=16,
        schedule_hidden_dim=32,
        unet_base_channels=16,
        unet_channel_multipliers=(1, 2),
        unet_attention_resolutions=(8,),
        num_timesteps=128,
        batch_size=64,
    )


def test_forward_diffusion_mean_variance() -> None:
    config = _tiny_config()
    model = AdaptiveDiffusionModel(config)

    batch = 512
    x0_single = torch.randn(1, config.in_channels, config.image_size, config.image_size)
    x0 = x0_single.expand(batch, -1, -1, -1).contiguous()
    labels = torch.zeros(batch, dtype=torch.long)
    t_value = config.num_timesteps // 2
    t = torch.full((batch,), t_value, dtype=torch.long)
    noise = torch.randn_like(x0)

    xt, _ = model.q_sample(x0=x0, t=t, class_labels=labels, noise=noise)
    class_emb = model.schedule_class_embedding(labels)
    alpha_bar = model.schedule_net.get_alpha_bar(class_emb)[0, t_value]

    expected_mean = torch.sqrt(alpha_bar) * x0_single
    empirical_mean = xt.mean(dim=0, keepdim=True)
    mean_abs_error = torch.mean(torch.abs(empirical_mean - expected_mean))
    assert mean_abs_error.item() < 0.03

    centered = xt - expected_mean
    empirical_var = centered.pow(2).mean()
    expected_var = (1.0 - alpha_bar).item()
    assert abs(empirical_var.item() - expected_var) < 0.08


def test_schedule_monotonicity_enforced() -> None:
    config = DiffusionConfig(num_timesteps=64, feature_dim=16, schedule_hidden_dim=32)
    net = ScheduleNet(config)

    def _bad_schedule(class_emb: torch.Tensor) -> torch.Tensor:
        _ = class_emb
        return torch.full((config.num_timesteps,), -0.5)

    net.get_full_schedule = _bad_schedule  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="strictly decreasing"):
        _ = net.get_alpha_bar(torch.randn(config.feature_dim))


def test_noise_prediction_unbiased() -> None:
    config = _tiny_config()
    unet = UNet(config)
    x = torch.randn(32, config.in_channels, config.image_size, config.image_size)
    t = torch.randint(0, config.num_timesteps, (32,))
    labels = torch.randint(0, config.num_classes, (32,))
    pred = unet(x, t, labels)
    assert abs(float(pred.mean().detach())) < 0.1

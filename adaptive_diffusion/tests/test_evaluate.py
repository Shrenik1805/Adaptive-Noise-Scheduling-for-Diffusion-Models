"""Tests for checkpoint loading behavior in evaluation CLI."""

from __future__ import annotations

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.evaluate import _load_model_from_checkpoint
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel


def test_load_model_restores_checkpoint_config(tmp_path) -> None:
    """Evaluation loader should restore architecture config from checkpoint payload."""
    train_config = DiffusionConfig(
        schedule_mode="adaptive",
        device="cpu",
        unet_base_channels=32,
        schedule_hidden_dim=64,
    )
    trained_model = AdaptiveDiffusionModel(config=train_config)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": trained_model.state_dict(),
            "config": train_config.to_dict(),
        },
        ckpt_path,
    )

    loaded = _load_model_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        schedule_mode="adaptive",
        device=torch.device("cpu"),
    )

    assert loaded.config.unet_base_channels == train_config.unet_base_channels
    assert loaded.config.schedule_hidden_dim == train_config.schedule_hidden_dim
    assert loaded.config.schedule_mode == "adaptive"
    assert loaded.config.device == "cpu"

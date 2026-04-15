"""Project-wide configuration for Adaptive Diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DiffusionConfig:
    """Immutable configuration for Adaptive Diffusion training and evaluation.

    Notes
    -----
    All hyperparameters are centralized here to avoid magic numbers across modules.
    """

    image_size: int = 32
    in_channels: int = 3
    num_classes: int = 10
    num_timesteps: int = 1000
    beta_min: float = 0.0001
    beta_max: float = 0.02
    feature_dim: int = 64
    schedule_hidden_dim: int = 128
    schedule_num_layers: int = 3
    unet_base_channels: int = 64
    unet_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    unet_attention_resolutions: tuple[int, ...] = (8, 4)
    batch_size: int = 128
    learning_rate: float = 2e-4
    ema_decay: float = 0.9999
    gradient_clip_norm: float = 1.0
    num_epochs: int = 100
    warmup_steps: int = 500
    lambda_efficiency: float = 0.01
    lambda_smoothness: float = 0.1
    wandb_project: str = "adaptive-diffusion"
    seed: int = 42
    device: str = "cuda"
    schedule_mode: str = "adaptive"
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    num_sample_steps_ddpm: int = 1000
    num_sample_steps_ddim: int = 50
    num_fid_samples: int = 10000
    num_eval_repeats: int = 3
    data_root: str = "./data"
    num_workers: int = 2
    validate_every_epochs: int = 5
    validate_num_batches: int = 50

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be at least 2.")
        if self.num_timesteps <= 1:
            raise ValueError("num_timesteps must be at least 2.")
        if not (0.0 < self.beta_min < self.beta_max <= 0.02):
            raise ValueError(
                "Expected 0 < beta_min < beta_max <= 0.02. "
                f"Got beta_min={self.beta_min}, beta_max={self.beta_max}."
            )
        if self.schedule_hidden_dim <= 0 or self.schedule_num_layers <= 0:
            raise ValueError("ScheduleNet dimensions and depth must be positive.")
        if self.unet_base_channels <= 0:
            raise ValueError("unet_base_channels must be positive.")
        if len(self.unet_channel_multipliers) == 0:
            raise ValueError("unet_channel_multipliers cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.num_sample_steps_ddpm <= 0 or self.num_sample_steps_ddim <= 0:
            raise ValueError("Sampling step counts must be positive.")
        if self.schedule_mode not in {"adaptive", "fixed_cosine"}:
            raise ValueError(
                "schedule_mode must be one of {'adaptive', 'fixed_cosine'}."
            )
        if self.num_fid_samples <= 0:
            raise ValueError("num_fid_samples must be positive.")
        if self.num_eval_repeats <= 0:
            raise ValueError("num_eval_repeats must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.validate_every_epochs <= 0:
            raise ValueError("validate_every_epochs must be positive.")
        if self.validate_num_batches <= 0:
            raise ValueError("validate_num_batches must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation of config.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for checkpoint metadata and wandb.
        """
        return asdict(self)

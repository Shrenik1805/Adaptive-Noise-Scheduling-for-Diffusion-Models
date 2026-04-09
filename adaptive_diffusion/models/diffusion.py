"""Adaptive Diffusion model: ScheduleNet + U-Net + objectives."""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.losses.adaptive_loss import AdaptiveLoss
from adaptive_diffusion.models.schedule_net import ScheduleNet
from adaptive_diffusion.models.unet import UNet


def cosine_beta_schedule(
    num_timesteps: int, s: float = 0.008, max_beta: float = 0.02
) -> Tensor:
    """Create cosine beta schedule from Nichol & Dhariwal.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion steps.
    s : float, optional
        Offset in cosine schedule, by default 0.008.
    max_beta : float, optional
        Upper beta clamp, by default 0.02.

    Returns
    -------
    Tensor
        Beta schedule of shape ``(num_timesteps,)``.
    """
    t = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float32)
    f_t = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, max_beta)


class AdaptiveDiffusionModel(nn.Module):
    """Class-conditional adaptive diffusion model."""

    def __init__(self, config: DiffusionConfig) -> None:
        """Initialize adaptive diffusion model stack."""
        super().__init__()
        self.config = config
        self.schedule_mode = config.schedule_mode
        self.schedule_net = (
            ScheduleNet(config) if self.schedule_mode == "adaptive" else None
        )
        self.unet = UNet(config)
        self.schedule_class_embedding = (
            nn.Embedding(config.num_classes, config.feature_dim)
            if self.schedule_mode == "adaptive"
            else None
        )
        self.loss_fn = AdaptiveLoss(schedule_net=self.schedule_net, config=config)

        fixed_betas = cosine_beta_schedule(
            num_timesteps=config.num_timesteps, max_beta=config.beta_max
        )
        fixed_alphas = 1.0 - fixed_betas
        fixed_alpha_bar = torch.cumprod(fixed_alphas, dim=0)
        self.register_buffer("fixed_betas", fixed_betas, persistent=True)
        self.register_buffer("fixed_alpha_bar", fixed_alpha_bar, persistent=True)

    def _extract(
        self, values: Tensor, timesteps: Tensor, x_shape: torch.Size
    ) -> Tensor:
        """Extract per-sample timestep coefficients and reshape for image ops."""
        batch_indices = torch.arange(timesteps.shape[0], device=timesteps.device)
        extracted = values[batch_indices, timesteps]
        return extracted[:, None, None, None].expand(x_shape[0], 1, 1, 1)

    @property
    def is_adaptive(self) -> bool:
        """Return whether the model uses a learned adaptive schedule."""
        return self.schedule_mode == "adaptive"

    def _schedule_for_batch(self, class_labels: Tensor) -> tuple[Tensor, Tensor]:
        """Get beta and alpha_bar schedules for each batch element."""
        batch = class_labels.shape[0]
        if self.is_adaptive:
            if self.schedule_net is None or self.schedule_class_embedding is None:
                raise ValueError(
                    "Adaptive schedule mode requires ScheduleNet and class embedding."
                )
            class_emb = self.schedule_class_embedding(class_labels)
            betas = self.schedule_net.get_full_schedule(class_emb)
            if betas.ndim != 2:
                raise ValueError(
                    "Expected adaptive beta schedule with shape "
                    f"(B, T), got {tuple(betas.shape)}."
                )
            alpha_bar = torch.cumprod(1.0 - betas, dim=-1)
        else:
            betas = self.fixed_betas.unsqueeze(0).expand(batch, -1)
            alpha_bar = self.fixed_alpha_bar.unsqueeze(0).expand(batch, -1)

        if torch.any(torch.diff(alpha_bar, dim=-1) >= 0):
            raise ValueError("alpha_bar must be strictly decreasing.")
        return betas, alpha_bar

    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        class_labels: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample from forward diffusion process ``q(x_t | x_0)``.

        Parameters
        ----------
        x0 : Tensor
            Clean input image tensor ``(B, C, H, W)``.
        t : Tensor
            Integer timesteps tensor ``(B,)``.
        class_labels : Tensor
            Class labels tensor ``(B,)``.
        noise : Tensor | None, optional
            Optional externally supplied Gaussian noise.

        Returns
        -------
        tuple[Tensor, Tensor]
            Noisy sample ``x_t`` and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        _, alpha_bar = self._schedule_for_batch(class_labels=class_labels)
        alpha_bar_t = self._extract(values=alpha_bar, timesteps=t, x_shape=x0.shape)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        if torch.any(~torch.isfinite(xt)):
            raise ValueError("q_sample produced non-finite values.")
        return xt, noise

    def p_losses(self, x0: Tensor, class_labels: Tensor) -> dict[str, Tensor]:
        """Compute full training loss dictionary for random timesteps."""
        batch = x0.shape[0]
        t = torch.randint(
            low=0,
            high=self.config.num_timesteps,
            size=(batch,),
            device=x0.device,
            dtype=torch.long,
        )
        xt, noise = self.q_sample(x0=x0, t=t, class_labels=class_labels)
        noise_pred = self.unet(x=xt, t=t, class_labels=class_labels)
        class_emb = (
            self.schedule_class_embedding(class_labels)
            if self.schedule_class_embedding is not None
            else None
        )
        loss_dict = self.loss_fn(
            noise_pred=noise_pred,
            noise_target=noise,
            class_embeddings=class_emb,
        )
        return loss_dict

    def _step_indices(self, num_steps: int, device: torch.device) -> Tensor:
        """Generate descending timestep index sequence for accelerated sampling."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        base = torch.linspace(
            0, self.config.num_timesteps - 1, steps=num_steps, device=device
        )
        indices = torch.unique(base.round().long(), sorted=True)
        return torch.flip(indices, dims=[0])

    @torch.no_grad()
    def ddpm_sample(
        self, class_labels: Tensor, num_steps: int = 1000
    ) -> tuple[Tensor, float]:
        """Generate samples via ancestral DDPM reverse process."""
        self.eval()
        device = class_labels.device
        batch = class_labels.shape[0]
        x = torch.randn(
            batch,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
            device=device,
        )
        betas, alpha_bar = self._schedule_for_batch(class_labels=class_labels)
        indices = self._step_indices(num_steps=num_steps, device=device)

        start = time.perf_counter()
        for i, idx in enumerate(indices):
            t = torch.full((batch,), int(idx.item()), device=device, dtype=torch.long)
            eps_pred = self.unet(x=x, t=t, class_labels=class_labels)

            beta_t = betas[:, idx][:, None, None, None]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = alpha_bar[:, idx][:, None, None, None]
            if i == len(indices) - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            else:
                prev_idx = indices[i + 1]
                alpha_bar_prev = alpha_bar[:, prev_idx][:, None, None, None]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
            )
            if i < len(indices) - 1:
                posterior_var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var.clamp(min=1e-12)) * noise
            else:
                x = mean
        elapsed = time.perf_counter() - start
        return x.clamp(-1.0, 1.0), elapsed

    @torch.no_grad()
    def ddim_sample(
        self, class_labels: Tensor, num_steps: int = 50
    ) -> tuple[Tensor, float]:
        """Generate samples via deterministic DDIM updates."""
        self.eval()
        device = class_labels.device
        batch = class_labels.shape[0]
        x = torch.randn(
            batch,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
            device=device,
        )
        _, alpha_bar = self._schedule_for_batch(class_labels=class_labels)
        indices = self._step_indices(num_steps=num_steps, device=device)

        start = time.perf_counter()
        for i, idx in enumerate(indices):
            t = torch.full((batch,), int(idx.item()), device=device, dtype=torch.long)
            eps_pred = self.unet(x=x, t=t, class_labels=class_labels)

            alpha_bar_t = alpha_bar[:, idx][:, None, None, None]
            if i == len(indices) - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            else:
                prev_idx = indices[i + 1]
                alpha_bar_prev = alpha_bar[:, prev_idx][:, None, None, None]

            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(
                alpha_bar_t
            )
            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
            )
        elapsed = time.perf_counter() - start
        return x.clamp(-1.0, 1.0), elapsed

    @torch.no_grad()
    def fixed_schedule_sample(
        self, class_labels: Tensor, num_steps: int = 50
    ) -> Tensor:
        """Generate counterfactual fixed-schedule samples using the current denoiser."""
        self.eval()
        device = class_labels.device
        batch = class_labels.shape[0]
        x = torch.randn(
            batch,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
            device=device,
        )
        alpha_bar = self.fixed_alpha_bar.to(device)
        indices = self._step_indices(num_steps=num_steps, device=device)
        for i, idx in enumerate(indices):
            t = torch.full((batch,), int(idx.item()), device=device, dtype=torch.long)
            eps_pred = self.unet(x=x, t=t, class_labels=class_labels)
            alpha_bar_t = alpha_bar[idx].view(1, 1, 1, 1)
            if i == len(indices) - 1:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            else:
                prev_idx = indices[i + 1]
                alpha_bar_prev = alpha_bar[prev_idx].view(1, 1, 1, 1)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(
                alpha_bar_t
            )
            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
            )
        return x.clamp(-1.0, 1.0)

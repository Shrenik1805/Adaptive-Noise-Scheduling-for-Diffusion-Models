"""Adaptive noise schedule network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from adaptive_diffusion.config import DiffusionConfig


def sinusoidal_timestep_embedding(
    timesteps: Tensor,
    dim: int,
    max_period: int = 10000,
) -> Tensor:
    """Create sinusoidal timestep embeddings from DDPM.

    Parameters
    ----------
    timesteps : Tensor
        Tensor of shape ``(batch,)`` containing (possibly fractional) timesteps.
    dim : int
        Embedding dimension.
    max_period : int, optional
        Maximum sinusoidal period, by default 10000.

    Returns
    -------
    Tensor
        Sinusoidal embedding tensor of shape ``(batch, dim)``.
    """
    half = dim // 2
    device = timesteps.device
    dtype = timesteps.dtype
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=device, dtype=dtype))
        * torch.arange(half, device=device, dtype=dtype)
        / half
    )
    args = timesteps[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with pre-norm."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual feed-forward transformation."""
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


@dataclass(frozen=True)
class ScheduleBounds:
    """Container for resolved beta bounds."""

    beta_min: Tensor
    beta_max: Tensor


class ScheduleNet(nn.Module):
    """Class-conditional adaptive beta schedule model."""

    hard_beta_cap: float = 0.02
    min_bound_eps: float = 1e-6

    def __init__(self, config: DiffusionConfig) -> None:
        """Initialize ScheduleNet.

        Parameters
        ----------
        config : DiffusionConfig
            Global diffusion configuration.
        """
        super().__init__()
        if config.feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if config.schedule_hidden_dim <= 0:
            raise ValueError("schedule_hidden_dim must be positive.")
        if config.schedule_num_layers <= 0:
            raise ValueError("schedule_num_layers must be positive.")
        if config.num_timesteps <= 1:
            raise ValueError("num_timesteps must be greater than one.")
        if not (0.0 < config.beta_min < config.beta_max <= self.hard_beta_cap):
            raise ValueError(
                "Invalid beta bounds in config. "
                f"Expected 0 < beta_min < beta_max <= {self.hard_beta_cap}."
            )

        self.config = config
        self.num_timesteps = config.num_timesteps
        self.timestep_dim = config.schedule_hidden_dim

        self.beta_min_param = nn.Parameter(torch.tensor(config.beta_min))
        self.beta_max_param = nn.Parameter(torch.tensor(config.beta_max))

        input_dim = config.feature_dim + config.schedule_hidden_dim
        self.input_proj = nn.Linear(input_dim, config.schedule_hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(config.schedule_hidden_dim)
                for _ in range(config.schedule_num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.schedule_hidden_dim)
        self.output_head = nn.Linear(config.schedule_hidden_dim, 1)

        # Bias term encourages small early betas and larger late betas.
        nn.init.constant_(self.output_head.bias, 0.0)

    @property
    def beta_bounds_valid(self) -> bool:
        """Return whether resolved beta bounds satisfy strict ordering."""
        bounds = self._resolved_bounds()
        return bool((bounds.beta_min < bounds.beta_max).item())

    def _resolved_bounds(self) -> ScheduleBounds:
        """Resolve and clamp learnable bounds safely."""
        beta_min = self.beta_min_param.clamp(
            self.min_bound_eps, self.hard_beta_cap - self.min_bound_eps
        )
        upper_cap = torch.tensor(
            self.hard_beta_cap, device=beta_min.device, dtype=beta_min.dtype
        )
        beta_max = torch.clamp(
            self.beta_max_param, min=beta_min + self.min_bound_eps, max=upper_cap
        )
        if torch.any(beta_min >= beta_max):
            raise ValueError(
                "ScheduleNet invariant violated: beta_min must be strictly less than beta_max. "
                f"Resolved values: beta_min={beta_min.item():.6f}, beta_max={beta_max.item():.6f}"
            )
        return ScheduleBounds(beta_min=beta_min, beta_max=beta_max)

    def forward(self, class_embedding: Tensor, t_normalized: Tensor) -> Tensor:
        """Predict beta for each sample/class at normalized timestep.

        Parameters
        ----------
        class_embedding : Tensor
            Class embedding tensor of shape ``(batch, feature_dim)``.
        t_normalized : Tensor
            Normalized timestep values in ``[0, 1]``, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Predicted beta values, shape ``(batch,)``.
        """
        if class_embedding.ndim != 2:
            raise ValueError(
                "class_embedding must have shape (batch, feature_dim). "
                f"Received shape {tuple(class_embedding.shape)}."
            )
        if t_normalized.ndim != 1:
            raise ValueError(
                "t_normalized must have shape (batch,). "
                f"Received shape {tuple(t_normalized.shape)}."
            )
        if class_embedding.shape[0] != t_normalized.shape[0]:
            raise ValueError(
                "Batch size mismatch between class_embedding and t_normalized: "
                f"{class_embedding.shape[0]} vs {t_normalized.shape[0]}."
            )

        bounds = self._resolved_bounds()
        if not self.beta_bounds_valid:
            raise ValueError("ScheduleNet invariant violated: beta bounds are invalid.")

        t_scaled = t_normalized * (self.num_timesteps - 1)
        t_emb = sinusoidal_timestep_embedding(
            timesteps=t_scaled,
            dim=self.timestep_dim,
            max_period=10000,
        )
        x = torch.cat([class_embedding, t_emb], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        logits = self.output_head(x).squeeze(-1)

        # Monotone prior over time makes alpha_bar behavior stable at initialization.
        monotone_prior = 6.0 * (t_normalized - 0.5)
        logits = logits + monotone_prior

        beta_raw = bounds.beta_min + (
            bounds.beta_max - bounds.beta_min
        ) * torch.sigmoid(logits)
        # Guarantee small early noise so alpha_bar[0] remains above 0.99 by construction.
        early_cap = torch.tensor(0.009, device=beta_raw.device, dtype=beta_raw.dtype)
        beta_cap_t = early_cap + (bounds.beta_max - early_cap) * t_normalized
        beta_cap_t = beta_cap_t.clamp(min=bounds.beta_min, max=bounds.beta_max)
        beta = torch.minimum(beta_raw, beta_cap_t)

        if torch.any(~torch.isfinite(beta)):
            raise ValueError("ScheduleNet produced non-finite beta values.")
        if torch.any((beta < 0.0) | (beta > self.hard_beta_cap)):
            beta_min_value = float(beta.min().detach().cpu())
            beta_max_value = float(beta.max().detach().cpu())
            raise ValueError(
                "ScheduleNet beta range violation: expected beta in [0, 0.02], "
                f"got min={beta_min_value:.6f}, max={beta_max_value:.6f}."
            )
        return beta

    def get_full_schedule(self, class_emb: Tensor) -> Tensor:
        """Get differentiable full beta schedule for one or more class embeddings.

        Parameters
        ----------
        class_emb : Tensor
            Class embedding of shape ``(feature_dim,)`` or ``(batch, feature_dim)``.

        Returns
        -------
        Tensor
            If input is 1D, returns ``(T,)``; otherwise returns ``(batch, T)``.
        """
        squeeze_output = False
        if class_emb.ndim == 1:
            class_emb = class_emb.unsqueeze(0)
            squeeze_output = True
        if class_emb.ndim != 2:
            raise ValueError(
                "class_emb must have shape (feature_dim,) or (batch, feature_dim). "
                f"Received shape {tuple(class_emb.shape)}."
            )

        batch = class_emb.shape[0]
        t = torch.linspace(
            0.0,
            1.0,
            self.num_timesteps,
            device=class_emb.device,
            dtype=class_emb.dtype,
        )
        t_batch = t.unsqueeze(0).expand(batch, -1)
        emb_batch = class_emb.unsqueeze(1).expand(-1, self.num_timesteps, -1)
        betas = self.forward(
            class_embedding=emb_batch.contiguous().flatten(0, 1),
            t_normalized=t_batch.contiguous().flatten(0, 1),
        )
        betas = betas.unflatten(0, (batch, self.num_timesteps))
        return betas.squeeze(0) if squeeze_output else betas

    def get_alpha_bar(self, class_emb: Tensor) -> Tensor:
        """Compute cumulative product ``alpha_bar_t = prod_{i<=t}(1 - beta_i)``.

        Parameters
        ----------
        class_emb : Tensor
            Class embedding of shape ``(feature_dim,)`` or ``(batch, feature_dim)``.

        Returns
        -------
        Tensor
            Alpha-bar schedule with shape ``(T,)`` or ``(batch, T)``.
        """
        betas = self.get_full_schedule(class_emb=class_emb)
        squeeze_output = betas.ndim == 1
        if squeeze_output:
            betas = betas.unsqueeze(0)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=-1)

        diffs = torch.diff(alpha_bar, dim=-1)
        if torch.any(diffs >= 0):
            raise ValueError(
                "alpha_bar invariant violated: alpha_bar must be strictly decreasing."
            )
        if torch.any(alpha_bar[:, 0] <= 0.99):
            raise ValueError(
                "alpha_bar invariant violated: alpha_bar[0] must be > 0.99. "
                f"Observed min(alpha_bar[0])={float(alpha_bar[:, 0].min().detach().cpu()):.6f}"
            )
        if self.num_timesteps >= 1000 and torch.any(alpha_bar[:, -1] >= 0.01):
            raise ValueError(
                "alpha_bar invariant violated for T=1000: alpha_bar[-1] must be < 0.01. "
                f"Observed max(alpha_bar[-1])={float(alpha_bar[:, -1].max().detach().cpu()):.6f}"
            )
        return alpha_bar.squeeze(0) if squeeze_output else alpha_bar

"""Class-conditional U-Net backbone for diffusion noise prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.models.schedule_net import sinusoidal_timestep_embedding


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm while adapting group count to channel divisibility."""
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class FiLMResBlock(nn.Module):
    """Residual block with timestep/class FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        """Apply conditioned residual block.

        Parameters
        ----------
        x : Tensor
            Input feature map of shape ``(B, C, H, W)``.
        cond_emb : Tensor
            Conditioning embedding of shape ``(B, D)``.

        Returns
        -------
        Tensor
            Output feature map of shape ``(B, C_out, H, W)``.
        """
        residual = self.skip(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.emb_proj(F.silu(cond_emb))
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = rearrange(scale, "b c -> b c 1 1")
        shift = rearrange(shift, "b c -> b c 1 1")

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + residual


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}."
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _group_norm(channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention in ``(H*W)`` token space."""
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv,
            "b (three heads dim) h w -> three b heads (h w) dim",
            three=3,
            heads=self.num_heads,
            dim=self.head_dim,
        )
        scale = self.head_dim**-0.5
        attn = torch.softmax(torch.einsum("bhid,bhjd->bhij", q * scale, k), dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b heads (h w) dim -> b (heads dim) h w", h=h, w=w)
        out = self.proj(out)
        return out + residual


@dataclass
class _UNetLevel:
    """Container for per-resolution modules."""

    block1: FiLMResBlock
    block2: FiLMResBlock
    attn: nn.Module
    sample: nn.Module


class UNet(nn.Module):
    """Diffusion U-Net with timestep and class conditioning."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.image_size = config.image_size
        self.base_channels = config.unet_base_channels
        self.channel_multipliers = config.unet_channel_multipliers
        self.attn_resolutions = set(config.unet_attention_resolutions)
        self.num_timesteps = config.num_timesteps

        self.time_embed_dim = self.base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.base_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.class_embedding = nn.Embedding(config.num_classes, self.time_embed_dim)

        self.input_conv = nn.Conv2d(
            self.in_channels, self.base_channels, kernel_size=3, padding=1
        )

        down_levels: list[_UNetLevel] = []
        down_skip_channels: list[int] = []
        in_ch = self.base_channels
        resolution = self.image_size

        for level_idx, mult in enumerate(self.channel_multipliers):
            out_ch = self.base_channels * mult
            block1 = FiLMResBlock(in_ch, out_ch, self.time_embed_dim)
            block2 = FiLMResBlock(out_ch, out_ch, self.time_embed_dim)
            attn: nn.Module = (
                SelfAttention2d(out_ch, num_heads=4)
                if resolution in self.attn_resolutions
                else nn.Identity()
            )
            is_last_level = level_idx == len(self.channel_multipliers) - 1
            sample: nn.Module = (
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
                if not is_last_level
                else nn.Identity()
            )
            down_levels.append(
                _UNetLevel(block1=block1, block2=block2, attn=attn, sample=sample)
            )
            down_skip_channels.extend([out_ch, out_ch])
            in_ch = out_ch
            if not is_last_level:
                resolution //= 2

        self.down_levels = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "block1": level.block1,
                        "block2": level.block2,
                        "attn": level.attn,
                        "sample": level.sample,
                    }
                )
                for level in down_levels
            ]
        )

        self.mid_block1 = FiLMResBlock(in_ch, in_ch, self.time_embed_dim)
        self.mid_attn = SelfAttention2d(in_ch, num_heads=4)
        self.mid_block2 = FiLMResBlock(in_ch, in_ch, self.time_embed_dim)

        up_levels: list[_UNetLevel] = []
        resolution = self.image_size // (2 ** (len(self.channel_multipliers) - 1))
        skip_channels = down_skip_channels.copy()
        for level_idx, mult in enumerate(reversed(self.channel_multipliers)):
            out_ch = self.base_channels * mult
            skip1 = skip_channels.pop()
            skip2 = skip_channels.pop()
            block1 = FiLMResBlock(in_ch + skip1, out_ch, self.time_embed_dim)
            block2 = FiLMResBlock(out_ch + skip2, out_ch, self.time_embed_dim)
            attn = (
                SelfAttention2d(out_ch, num_heads=4)
                if resolution in self.attn_resolutions
                else nn.Identity()
            )
            is_last_level = level_idx == len(self.channel_multipliers) - 1
            sample = (
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                if not is_last_level
                else nn.Identity()
            )
            up_levels.append(
                _UNetLevel(block1=block1, block2=block2, attn=attn, sample=sample)
            )
            in_ch = out_ch
            if not is_last_level:
                resolution *= 2

        self.up_levels = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "block1": level.block1,
                        "block2": level.block2,
                        "attn": level.attn,
                        "sample": level.sample,
                    }
                )
                for level in up_levels
            ]
        )

        self.out_norm = _group_norm(self.base_channels)
        self.out_conv = nn.Conv2d(
            self.base_channels, self.in_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _compute_conditioning(self, t: Tensor, class_labels: Tensor) -> Tensor:
        """Create combined timestep + class conditioning embedding."""
        t = t.float()
        t_emb = sinusoidal_timestep_embedding(
            timesteps=t, dim=self.base_channels, max_period=10000
        )
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_embedding(class_labels)
        return t_emb + c_emb

    def forward(self, x: Tensor, t: Tensor, class_labels: Tensor) -> Tensor:
        """Predict diffusion noise.

        Parameters
        ----------
        x : Tensor
            Noisy input image ``(B, C, H, W)``.
        t : Tensor
            Integer timestep tensor ``(B,)``.
        class_labels : Tensor
            Class label tensor ``(B,)``.

        Returns
        -------
        Tensor
            Predicted noise with same shape as ``x``.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have rank 4, got shape {tuple(x.shape)}.")
        if t.ndim != 1 or class_labels.ndim != 1:
            raise ValueError("Expected t and class_labels to both have shape (B,).")
        if not (x.shape[0] == t.shape[0] == class_labels.shape[0]):
            raise ValueError(
                "Batch dimensions must match across x, t, and class_labels."
            )

        emb = self._compute_conditioning(t=t, class_labels=class_labels)

        x = self.input_conv(x)
        skips: list[Tensor] = []

        for level in self.down_levels:
            x = level["block1"](x, emb)
            skips.append(x)
            x = level["block2"](x, emb)
            x = level["attn"](x)
            skips.append(x)
            x = level["sample"](x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for level in self.up_levels:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = level["block1"](x, emb)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = level["block2"](x, emb)
            x = level["attn"](x)
            x = level["sample"](x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"UNet output channels mismatch: expected {self.in_channels}, got {x.shape[1]}."
            )
        return x

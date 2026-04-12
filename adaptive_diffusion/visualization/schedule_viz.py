"""Visualization utilities for adaptive schedules and efficiency tradeoffs."""

from __future__ import annotations

import os
from pathlib import Path

# Force a writable matplotlib config dir in restricted environments.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("./adaptive_diffusion/.mplconfig").resolve())
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from adaptive_diffusion.models.diffusion import (
    AdaptiveDiffusionModel,
    cosine_beta_schedule,
)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _set_pub_style() -> None:
    """Set publication-quality plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


@torch.no_grad()
def plot_schedule_grid(
    model: AdaptiveDiffusionModel,
    save_path: str,
    mc_samples: int = 8,
    embedding_noise_std: float = 0.01,
) -> plt.Figure:
    """Plot class-wise adaptive schedules against fixed cosine schedule."""
    if not model.is_adaptive:
        raise ValueError(
            "plot_schedule_grid requires an adaptive model with ScheduleNet."
        )
    if model.schedule_net is None or model.schedule_class_embedding is None:
        raise ValueError(
            "Adaptive model is missing schedule modules required for plotting."
        )
    _set_pub_style()
    device = next(model.parameters()).device
    fixed = (
        cosine_beta_schedule(model.config.num_timesteps, max_beta=model.config.beta_max)
        .cpu()
        .numpy()
    )
    t = np.arange(model.config.num_timesteps)

    fig, axes = plt.subplots(2, 5, figsize=(14, 5), sharex=True, sharey=True)
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        ax = axes[class_idx // 5, class_idx % 5]
        label = torch.tensor([class_idx], device=device, dtype=torch.long)
        base_emb = model.schedule_class_embedding(label).squeeze(0)

        schedules = []
        for _ in range(mc_samples):
            noisy_emb = base_emb + embedding_noise_std * torch.randn_like(base_emb)
            beta = (
                model.schedule_net.get_full_schedule(noisy_emb).detach().cpu().numpy()
            )
            schedules.append(beta)
        schedule_arr = np.stack(schedules, axis=0)
        mean = schedule_arr.mean(axis=0)
        std = schedule_arr.std(axis=0)

        ax.plot(t, mean, color="tab:blue", linewidth=1.2, label="Adaptive")
        ax.fill_between(
            t, mean - std, mean + std, color="tab:blue", alpha=0.2, linewidth=0
        )
        ax.plot(
            t,
            fixed,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="Fixed cosine",
        )
        ax.set_title(class_name)
        ax.set_xlabel("timestep")
        ax.set_ylabel(r"$\beta_t$")
        if class_idx == 0:
            ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


def plot_efficiency_frontier(metrics_df: pd.DataFrame, save_path: str) -> plt.Figure:
    """Plot adaptive vs fixed Pareto frontier in time-FID space."""
    _set_pub_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        metrics_df["time_adaptive_mean"],
        metrics_df["fid_adaptive_mean"],
        color="tab:blue",
        marker="o",
        linestyle="-",
        label="Adaptive",
    )
    ax.plot(
        metrics_df["time_fixed_mean"],
        metrics_df["fid_fixed_mean"],
        color="tab:red",
        marker="o",
        linestyle="--",
        label="Fixed",
    )

    for _, row in metrics_df.iterrows():
        ax.annotate(
            str(int(row["num_steps"])),
            (row["time_adaptive_mean"], row["fid_adaptive_mean"]),
            fontsize=7,
        )
        ax.annotate(
            str(int(row["num_steps"])),
            (row["time_fixed_mean"], row["fid_fixed_mean"]),
            fontsize=7,
        )

    improvement_x = np.minimum(
        metrics_df["time_adaptive_mean"].to_numpy(),
        metrics_df["time_fixed_mean"].to_numpy(),
    )
    improvement_y = np.minimum(
        metrics_df["fid_adaptive_mean"].to_numpy(),
        metrics_df["fid_fixed_mean"].to_numpy(),
    )
    ax.fill_between(
        np.sort(improvement_x),
        np.interp(
            np.sort(improvement_x), np.sort(improvement_x), np.sort(improvement_y)
        ),
        y2=max(
            metrics_df["fid_fixed_mean"].max(), metrics_df["fid_adaptive_mean"].max()
        ),
        color="tab:green",
        alpha=0.08,
        label="Improvement region",
    )

    ax.set_xlabel("Sampling time (seconds)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("Efficiency Frontier: Adaptive vs Fixed")
    ax.legend()
    fig.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


def plot_per_class_speedup(per_class_df: pd.DataFrame, save_path: str) -> plt.Figure:
    """Plot per-class adaptive speedup ratios."""
    _set_pub_style()
    sorted_df = per_class_df.sort_values("speedup_ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    norm = plt.Normalize(
        sorted_df["speedup_ratio"].min(), sorted_df["speedup_ratio"].max()
    )
    cmap = plt.cm.viridis
    colors = cmap(norm(sorted_df["speedup_ratio"].to_numpy()))

    ax.barh(sorted_df["class"], sorted_df["speedup_ratio"], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Speedup ratio (fixed/adaptive)")
    ax.set_ylabel("Class")
    ax.set_title("Per-class Sampling Speedup")
    fig.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig

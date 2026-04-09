"""Streamlit demo for Adaptive Noise Scheduling Diffusion."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("./adaptive_diffusion/.mplconfig").resolve())
)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torchvision.utils import make_grid

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.evaluation.metrics import (
    compute_efficiency_frontier,
    compute_per_class_metrics,
)
from adaptive_diffusion.models.diffusion import (
    AdaptiveDiffusionModel,
    cosine_beta_schedule,
)
from adaptive_diffusion.utils.device import resolve_device, synchronize
from adaptive_diffusion.visualization.schedule_viz import (
    CIFAR10_CLASSES,
    plot_efficiency_frontier,
    plot_per_class_speedup,
)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str | None = None) -> AdaptiveDiffusionModel:
    """Load model from checkpoint if available."""
    config = DiffusionConfig()
    model = AdaptiveDiffusionModel(config=config)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["model_state"], strict=True)
    device = resolve_device("auto")
    model.to(device).eval()
    return model


def _plot_schedule(
    model: AdaptiveDiffusionModel, class_idx: int, show_fixed: bool, num_timesteps: int
) -> plt.Figure:
    device = next(model.parameters()).device
    label = torch.tensor([class_idx], device=device, dtype=torch.long)
    class_emb = model.schedule_class_embedding(label).squeeze(0)
    beta = (
        model.schedule_net.get_full_schedule(class_emb)[:num_timesteps]
        .detach()
        .cpu()
        .numpy()
    )
    alpha_bar = torch.cumprod(1.0 - torch.from_numpy(beta), dim=0).numpy()
    fixed = cosine_beta_schedule(
        model.config.num_timesteps, max_beta=model.config.beta_max
    )[:num_timesteps].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    t = np.arange(num_timesteps)
    axes[0].plot(t, beta, color="tab:blue", linewidth=2, label="Adaptive")
    if show_fixed:
        axes[0].plot(
            t, fixed, color="tab:red", linestyle="--", linewidth=2, label="Fixed cosine"
        )
    axes[0].set_title(f"Beta Schedule: {CIFAR10_CLASSES[class_idx]}")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel(r"$\beta_t$")
    axes[0].legend()

    axes[1].plot(t, alpha_bar, color="tab:green", linewidth=2)
    axes[1].set_title(r"Cumulative $\bar{\alpha}_t = \prod_{i \le t}(1-\beta_i)$")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel(r"$\bar{\alpha}_t$")
    fig.tight_layout()
    return fig


def _to_image_grid(samples: torch.Tensor, nrow: int = 4) -> np.ndarray:
    grid = make_grid((samples.clamp(-1.0, 1.0) + 1.0) * 0.5, nrow=nrow)
    grid = grid.detach().cpu().permute(1, 2, 0).numpy()
    return (grid * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    """Run Streamlit UI."""
    st.set_page_config(page_title="Adaptive Diffusion", layout="wide")

    st.title("Adaptive Noise Scheduling for Diffusion Models")
    st.write(
        "This demo shows a class-conditional diffusion model that learns a custom noise schedule "
        "for each semantic class instead of using a single fixed schedule. The learned schedule "
        "aims to preserve sample quality while reducing denoising compute."
    )
    st.markdown("[arXiv paper](https://arxiv.org/abs/0000.00000)")

    checkpoint_path = st.sidebar.text_input("Checkpoint path (optional)", value="")
    model = load_model(checkpoint_path if checkpoint_path else None)
    device = next(model.parameters()).device

    st.header("Section 2 — Schedule Explorer")
    class_idx = st.sidebar.selectbox(
        "Class selector",
        options=list(range(len(CIFAR10_CLASSES))),
        format_func=lambda i: CIFAR10_CLASSES[i],
    )
    show_fixed = st.sidebar.checkbox("Show fixed baseline", value=True)
    num_timesteps = st.sidebar.slider(
        "Num timesteps",
        min_value=10,
        max_value=model.config.num_timesteps,
        value=200,
        step=10,
    )
    fig_schedule = _plot_schedule(
        model=model,
        class_idx=class_idx,
        show_fixed=show_fixed,
        num_timesteps=num_timesteps,
    )
    st.pyplot(fig_schedule, clear_figure=True)

    st.header("Section 3 — Sample Generator")
    gen_class = st.sidebar.selectbox(
        "Generator class",
        options=list(range(len(CIFAR10_CLASSES))),
        format_func=lambda i: CIFAR10_CLASSES[i],
        key="gen_class",
    )
    num_steps = st.sidebar.slider(
        "Generation steps",
        min_value=10,
        max_value=model.config.num_timesteps,
        value=model.config.num_sample_steps_ddim,
        step=5,
    )
    method = st.sidebar.selectbox(
        "Method",
        options=["DDIM adaptive", "DDIM fixed", "DDPM adaptive"],
    )
    if st.button("Generate 16 samples"):
        labels = torch.full((16,), gen_class, dtype=torch.long, device=device)
        with st.spinner("Generating samples..."):
            with torch.inference_mode():
                if method == "DDIM adaptive":
                    samples, elapsed = model.ddim_sample(
                        class_labels=labels, num_steps=num_steps
                    )
                elif method == "DDPM adaptive":
                    samples, elapsed = model.ddpm_sample(
                        class_labels=labels, num_steps=num_steps
                    )
                else:
                    start = time.perf_counter()
                    samples = model.fixed_schedule_sample(
                        class_labels=labels, num_steps=num_steps
                    )
                    synchronize(device)
                    elapsed = time.perf_counter() - start
        st.image(_to_image_grid(samples, nrow=4), caption=f"{method} ({elapsed:.3f}s)")
        st.write(f"Generation time: `{elapsed:.3f}` seconds")
        st.write("FID estimate: computed in Sections 4 and 5 using CIFAR-10 reference.")

    st.header("Section 4 — Efficiency Frontier")
    if st.button("Compute efficiency frontier (takes ~2 min)"):
        with st.spinner("Computing efficiency frontier..."):
            with torch.inference_mode():
                metrics_df = compute_efficiency_frontier(model=model)
                fig = plot_efficiency_frontier(
                    metrics_df,
                    save_path="adaptive_diffusion/samples/efficiency_frontier.png",
                )
        st.pyplot(fig, clear_figure=True)
        st.dataframe(metrics_df)

    st.header("Section 5 — Per-class Analysis")
    if st.button("Compute per-class metrics"):
        with st.spinner("Computing per-class metrics..."):
            with torch.inference_mode():
                per_class_df = compute_per_class_metrics(model=model)
                fig = plot_per_class_speedup(
                    per_class_df,
                    save_path="adaptive_diffusion/samples/per_class_speedup.png",
                )
        st.pyplot(fig, clear_figure=True)
        st.dataframe(per_class_df)


if __name__ == "__main__":
    main()

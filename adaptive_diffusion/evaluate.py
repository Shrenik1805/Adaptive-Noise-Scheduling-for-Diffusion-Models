"""CLI entrypoint for fair paired-checkpoint evaluation and figure generation."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.evaluation.metrics import (
    compute_efficiency_frontier,
    compute_per_class_metrics,
    compute_schedule_diversity,
)
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel
from adaptive_diffusion.utils.device import resolve_device
from adaptive_diffusion.visualization.schedule_viz import (
    plot_efficiency_frontier,
    plot_per_class_speedup,
    plot_schedule_grid,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive and fixed checkpoints with fair paired metrics."
    )
    parser.add_argument(
        "--adaptive-checkpoint",
        type=str,
        required=True,
        help="Path to adaptive schedule checkpoint (.pt).",
    )
    parser.add_argument(
        "--fixed-checkpoint",
        type=str,
        required=True,
        help="Path to fixed cosine schedule checkpoint (.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override (auto, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./adaptive_diffusion/analysis",
        help="Directory to store generated CSVs and plots.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="Generated and real samples per class for per-class metrics.",
    )
    parser.add_argument(
        "--num-fid-samples",
        type=int,
        default=10000,
        help="Number of generated/real samples for efficiency frontier FID.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent sampling repeats per measurement for mean/std reporting.",
    )
    return parser.parse_args()


def _config_from_payload(
    payload: dict[str, Any],
    schedule_mode: str,
    device: torch.device,
) -> DiffusionConfig:
    """Construct DiffusionConfig from checkpoint payload with safe filtering.

    Parameters
    ----------
    payload : dict[str, Any]
        Loaded checkpoint dictionary.
    schedule_mode : str
        Explicit mode expected for this checkpoint load path.
    device : torch.device
        Runtime device override.

    Returns
    -------
    DiffusionConfig
        Restored config aligned with checkpoint metadata.
    """
    raw = payload.get("config", {})
    valid_keys = {f.name for f in fields(DiffusionConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    filtered["schedule_mode"] = schedule_mode
    filtered["device"] = device.type
    return DiffusionConfig(**filtered)


def _load_model_from_checkpoint(
    checkpoint_path: str,
    schedule_mode: str,
    device: torch.device,
) -> AdaptiveDiffusionModel:
    """Load model checkpoint with exact architecture config restoration."""
    payload = torch.load(checkpoint_path, map_location=device)
    config = _config_from_payload(
        payload=payload,
        schedule_mode=schedule_mode,
        device=device,
    )
    model = AdaptiveDiffusionModel(config=config).to(device).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def main() -> None:
    """Run full evaluation workflow and persist outputs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    adaptive_model = _load_model_from_checkpoint(
        checkpoint_path=args.adaptive_checkpoint,
        schedule_mode="adaptive",
        device=device,
    )
    fixed_model = _load_model_from_checkpoint(
        checkpoint_path=args.fixed_checkpoint,
        schedule_mode="fixed_cosine",
        device=device,
    )

    schedule_diversity = compute_schedule_diversity(adaptive_model)
    frontier_df = compute_efficiency_frontier(
        adaptive_model=adaptive_model,
        fixed_model=fixed_model,
        num_images=args.num_fid_samples,
        repeats=args.repeats,
    )
    per_class_df = compute_per_class_metrics(
        adaptive_model=adaptive_model,
        fixed_model=fixed_model,
        samples_per_class=args.samples_per_class,
        repeats=args.repeats,
    )

    frontier_csv = output_dir / "efficiency_frontier.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    frontier_df.to_csv(frontier_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    plot_schedule_grid(
        model=adaptive_model,
        save_path=str(output_dir / "schedule_grid.png"),
    )
    plot_efficiency_frontier(
        metrics_df=frontier_df,
        save_path=str(output_dir / "efficiency_frontier.png"),
    )
    plot_per_class_speedup(
        per_class_df=per_class_df,
        save_path=str(output_dir / "per_class_speedup.png"),
    )

    print(f"Device: {device.type}")
    print(f"Loaded adaptive checkpoint: {args.adaptive_checkpoint}")
    print(f"Loaded fixed checkpoint: {args.fixed_checkpoint}")
    print(f"Schedule diversity (adaptive): {schedule_diversity:.6f}")
    print(f"Saved: {frontier_csv}")
    print(f"Saved: {per_class_csv}")


if __name__ == "__main__":
    main()

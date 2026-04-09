"""CLI entrypoint for post-training evaluation and figure generation."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        description="Evaluate adaptive diffusion checkpoint and generate plots/tables."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint .pt file. If empty, evaluates random initialization.",
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
        default=128,
        help="Samples per class for per-class metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run full evaluation workflow and persist outputs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DiffusionConfig(device=args.device)
    model = AdaptiveDiffusionModel(config=config)
    device = resolve_device(args.device)
    model.to(device).eval()

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"], strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided; evaluating current model initialization.")

    schedule_diversity = compute_schedule_diversity(model)
    frontier_df = compute_efficiency_frontier(model)
    per_class_df = compute_per_class_metrics(
        model, samples_per_class=args.samples_per_class
    )

    frontier_csv = output_dir / "efficiency_frontier.csv"
    per_class_csv = output_dir / "per_class_metrics.csv"
    frontier_df.to_csv(frontier_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    plot_schedule_grid(
        model=model,
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
    print(f"Schedule diversity: {schedule_diversity:.6f}")
    print(f"Saved: {frontier_csv}")
    print(f"Saved: {per_class_csv}")
    print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()

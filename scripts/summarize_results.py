"""Summarize paired evaluation outputs into concise paper-ready artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize paired adaptive-vs-fixed evaluation results."
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help="Directory containing efficiency_frontier.csv and per_class_metrics.csv.",
    )
    parser.add_argument(
        "--adaptive-checkpoint",
        type=str,
        required=True,
        help="Path to selected adaptive checkpoint.",
    )
    parser.add_argument(
        "--fixed-checkpoint",
        type=str,
        required=True,
        help="Path to selected fixed checkpoint.",
    )
    return parser.parse_args()


def _best_by_fid(df: pd.DataFrame, fid_col: str) -> pd.Series:
    """Return row with lowest mean FID for a model."""
    return df.loc[df[fid_col].idxmin()]


def _ci_excludes_zero(low: float, high: float) -> bool:
    """Return whether confidence interval excludes zero."""
    return (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)


def main() -> None:
    """Create summary CSV and markdown report."""
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    frontier_path = analysis_dir / "efficiency_frontier.csv"
    per_class_path = analysis_dir / "per_class_metrics.csv"

    if not frontier_path.exists():
        raise FileNotFoundError(f"Missing frontier file: {frontier_path}")
    if not per_class_path.exists():
        raise FileNotFoundError(f"Missing per-class file: {per_class_path}")

    frontier_df = pd.read_csv(frontier_path)
    per_class_df = pd.read_csv(per_class_path)

    best_adaptive = _best_by_fid(frontier_df, "fid_adaptive_mean")
    best_fixed = _best_by_fid(frontier_df, "fid_fixed_mean")
    best_delta = frontier_df.loc[
        frontier_df["fid_delta_adaptive_minus_fixed_mean"].idxmin()
    ]

    median_speedup = float(per_class_df["speedup_ratio"].median())
    mean_speedup = float(per_class_df["speedup_ratio"].mean())
    adaptive_better_classes = int(
        (per_class_df["fid_adaptive_mean"] <= per_class_df["fid_fixed_mean"]).sum()
    )

    summary_df = pd.DataFrame(
        [
            {
                "adaptive_checkpoint": args.adaptive_checkpoint,
                "fixed_checkpoint": args.fixed_checkpoint,
                "best_adaptive_steps": int(best_adaptive["num_steps"]),
                "best_adaptive_fid_mean": float(best_adaptive["fid_adaptive_mean"]),
                "best_adaptive_time_mean": float(best_adaptive["time_adaptive_mean"]),
                "best_fixed_steps": int(best_fixed["num_steps"]),
                "best_fixed_fid_mean": float(best_fixed["fid_fixed_mean"]),
                "best_fixed_time_mean": float(best_fixed["time_fixed_mean"]),
                "best_delta_steps": int(best_delta["num_steps"]),
                "best_delta_fid_adaptive_minus_fixed": float(
                    best_delta["fid_delta_adaptive_minus_fixed_mean"]
                ),
                "best_delta_fid_ci_low": float(best_delta["fid_delta_ci_low"]),
                "best_delta_fid_ci_high": float(best_delta["fid_delta_ci_high"]),
                "best_delta_time_adaptive_minus_fixed": float(
                    best_delta["time_delta_adaptive_minus_fixed_mean"]
                ),
                "best_delta_time_ci_low": float(best_delta["time_delta_ci_low"]),
                "best_delta_time_ci_high": float(best_delta["time_delta_ci_high"]),
                "best_delta_joint_win_rate": float(best_delta["paired_joint_win_rate"]),
                "mean_speedup_ratio": mean_speedup,
                "median_speedup_ratio": median_speedup,
                "num_classes_adaptive_better_or_equal": adaptive_better_classes,
            }
        ]
    )

    summary_csv = analysis_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    classes_ci_better = int(
        (
            (per_class_df["fid_delta_ci_high"] < 0.0)
            & (per_class_df["time_delta_ci_high"] < 0.0)
        ).sum()
    )
    strongest_claim_valid = _ci_excludes_zero(
        float(best_delta["fid_delta_ci_low"]),
        float(best_delta["fid_delta_ci_high"]),
    )

    summary_md = analysis_dir / "summary_report.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Paired Evaluation Summary",
                "",
                f"- Adaptive checkpoint: `{args.adaptive_checkpoint}`",
                f"- Fixed checkpoint: `{args.fixed_checkpoint}`",
                "",
                "## Frontier",
                f"- Best adaptive (lowest FID): steps={int(best_adaptive['num_steps'])}, "
                f"FID={float(best_adaptive['fid_adaptive_mean']):.4f}, "
                f"time={float(best_adaptive['time_adaptive_mean']):.4f}s",
                f"- Best fixed (lowest FID): steps={int(best_fixed['num_steps'])}, "
                f"FID={float(best_fixed['fid_fixed_mean']):.4f}, "
                f"time={float(best_fixed['time_fixed_mean']):.4f}s",
                (
                    "- Strongest adaptive-vs-fixed delta: "
                    f"steps={int(best_delta['num_steps'])}, "
                    f"FID_delta={float(best_delta['fid_delta_adaptive_minus_fixed_mean']):.4f} "
                    f"[{float(best_delta['fid_delta_ci_low']):.4f}, "
                    f"{float(best_delta['fid_delta_ci_high']):.4f}], "
                    f"time_delta={float(best_delta['time_delta_adaptive_minus_fixed_mean']):.4f}s "
                    f"[{float(best_delta['time_delta_ci_low']):.4f}, "
                    f"{float(best_delta['time_delta_ci_high']):.4f}], "
                    f"paired_joint_win_rate={float(best_delta['paired_joint_win_rate']):.2f}"
                ),
                "",
                "## Per-Class",
                f"- Mean speedup ratio (fixed/adaptive): {mean_speedup:.4f}",
                f"- Median speedup ratio (fixed/adaptive): {median_speedup:.4f}",
                f"- Classes where adaptive FID <= fixed FID: {adaptive_better_classes}/10",
                f"- Classes with both FID/time CI upper bounds < 0 (adaptive statistically better): {classes_ci_better}/10",
                "",
                "## Validity Checks",
                (
                    "- Best-step FID delta CI excludes zero: "
                    f"{'yes' if strongest_claim_valid else 'no'}"
                ),
                "- Interpret adaptive as a frontier-shift method, not same-step runtime acceleration.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary report: {summary_md}")


if __name__ == "__main__":
    main()

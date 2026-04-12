"""Run paired adaptive-vs-fixed experiments across multiple seeds and aggregate results."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI options for multi-seed orchestration."""
    parser = argparse.ArgumentParser(
        description="Run repeated paired experiments and aggregate summary metrics."
    )
    parser.add_argument(
        "--python", type=str, default="python", help="Python executable."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="List of seeds to run.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device target.")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch count.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Dataloader workers passed to training runs. "
            "Use 0 on Python 3.14/macOS to avoid multiprocessing pickling issues."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode.",
    )
    parser.add_argument(
        "--num-fid-samples",
        type=int,
        default=10000,
        help="FID sample count for frontier evaluation.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="Per-class evaluation sample count.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Sampling repeats per measurement.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="multiseed",
        help="Prefix used to name per-seed output directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./adaptive_diffusion/analysis_multiseed",
        help="Directory for aggregated artifacts.",
    )
    return parser.parse_args()


def _run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run subprocess command and raise on failure."""
    print("Running:", " ".join(command))
    merged_env = None
    if env is not None:
        merged_env = os.environ.copy()
        merged_env.update(env)
    subprocess.run(command, check=True, env=merged_env)


def _aggregate_seed_summaries(summary_paths: list[Path], output_dir: Path) -> None:
    """Aggregate per-seed summaries with mean/std and confidence intervals."""
    frames = [pd.read_csv(path) for path in summary_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_dir / "seed_level_summary.csv", index=False)

    numeric = merged.select_dtypes(include=["number"])
    stats = pd.DataFrame(
        {
            "metric": numeric.columns,
            "mean": [float(numeric[col].mean()) for col in numeric.columns],
            "std": [float(numeric[col].std(ddof=0)) for col in numeric.columns],
            "min": [float(numeric[col].min()) for col in numeric.columns],
            "max": [float(numeric[col].max()) for col in numeric.columns],
        }
    )
    stats.to_csv(output_dir / "aggregate_summary.csv", index=False)


def main() -> None:
    """Execute multi-seed experiment schedule and aggregate outputs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths: list[Path] = []

    for seed in args.seeds:
        run_tag = f"{args.run_prefix}_seed{seed}"
        adaptive_ckpt_dir = f"./checkpoints_adaptive_{run_tag}"
        fixed_ckpt_dir = f"./checkpoints_fixed_{run_tag}"
        adaptive_sample_dir = f"./samples_adaptive_{run_tag}"
        fixed_sample_dir = f"./samples_fixed_{run_tag}"
        analysis_dir = f"./adaptive_diffusion/analysis_{run_tag}"

        common_train = [
            args.python,
            "-m",
            "adaptive_diffusion.train",
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(seed),
            "--num-workers",
            str(args.num_workers),
        ]
        train_env = {"WANDB_MODE": args.wandb_mode}

        _run_command(
            common_train
            + [
                "--schedule-mode",
                "adaptive",
                "--checkpoint-dir",
                adaptive_ckpt_dir,
                "--sample-dir",
                adaptive_sample_dir,
            ],
            env=train_env,
        )
        _run_command(
            common_train
            + [
                "--schedule-mode",
                "fixed_cosine",
                "--checkpoint-dir",
                fixed_ckpt_dir,
                "--sample-dir",
                fixed_sample_dir,
            ],
            env=train_env,
        )

        find_best_cmd = [
            args.python,
            "-c",
            (
                "import re,sys;"
                "from pathlib import Path;"
                "d=Path(sys.argv[1]);"
                "p=re.compile(r'epoch_\\d+_fid_([0-9]+(?:\\.[0-9]+)?)\\.pt$');"
                "c=[(float(m.group(1)),str(x)) for x in d.glob('epoch_*_fid_*.pt') "
                "if (m:=p.search(x.name)) is not None];"
                "print(min(c)[1] if c else '')"
            ),
        ]

        adaptive_best = subprocess.check_output(
            find_best_cmd + [adaptive_ckpt_dir], text=True
        ).strip()
        fixed_best = subprocess.check_output(
            find_best_cmd + [fixed_ckpt_dir], text=True
        ).strip()
        if not adaptive_best or not fixed_best:
            raise RuntimeError(f"Missing best checkpoints for seed={seed}.")

        _run_command(
            [
                args.python,
                "-m",
                "adaptive_diffusion.evaluate",
                "--adaptive-checkpoint",
                adaptive_best,
                "--fixed-checkpoint",
                fixed_best,
                "--device",
                args.device,
                "--output-dir",
                analysis_dir,
                "--num-fid-samples",
                str(args.num_fid_samples),
                "--samples-per-class",
                str(args.samples_per_class),
                "--repeats",
                str(args.repeats),
            ]
        )

        _run_command(
            [
                args.python,
                "scripts/summarize_results.py",
                "--analysis-dir",
                analysis_dir,
                "--adaptive-checkpoint",
                adaptive_best,
                "--fixed-checkpoint",
                fixed_best,
            ]
        )

        summary_paths.append(Path(analysis_dir) / "summary_metrics.csv")

    _aggregate_seed_summaries(summary_paths=summary_paths, output_dir=output_dir)
    print(f"Saved multi-seed aggregate in: {output_dir}")


if __name__ == "__main__":
    main()

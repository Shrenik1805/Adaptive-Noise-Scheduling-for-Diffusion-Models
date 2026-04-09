"""CLI entrypoint for training Adaptive Diffusion on CIFAR-10."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.data import get_cifar10_dataloaders
from adaptive_diffusion.models import AdaptiveDiffusionModel
from adaptive_diffusion.training import Trainer
from adaptive_diffusion.utils.device import resolve_device


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive Diffusion on CIFAR-10."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (auto, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="wandb project name override."
    )
    parser.add_argument(
        "--schedule-mode",
        type=str,
        default=None,
        help="Schedule mode override (adaptive or fixed_cosine).",
    )
    return parser.parse_args()


def main() -> None:
    """Launch training workflow."""
    args = parse_args()
    base = DiffusionConfig()
    config = DiffusionConfig(
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        learning_rate=args.lr if args.lr is not None else base.learning_rate,
        num_epochs=args.epochs if args.epochs is not None else base.num_epochs,
        device=args.device if args.device is not None else base.device,
        schedule_mode=(
            args.schedule_mode
            if args.schedule_mode is not None
            else base.schedule_mode
        ),
        wandb_project=(
            args.wandb_project if args.wandb_project is not None else base.wandb_project
        ),
    )
    set_seed(config.seed)

    runtime_device = resolve_device(config.device)
    print(f"Requested device: {config.device}; runtime device: {runtime_device.type}")

    if runtime_device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif runtime_device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS) backend.")

    train_loader, val_loader = get_cifar10_dataloaders(config=config)
    model = AdaptiveDiffusionModel(config=config)
    trainer = Trainer(
        model=model, config=config, train_loader=train_loader, val_loader=val_loader
    )
    trainer.train(num_epochs=config.num_epochs)


if __name__ == "__main__":
    main()

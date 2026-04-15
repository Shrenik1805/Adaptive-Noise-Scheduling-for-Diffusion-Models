"""Training loop for adaptive diffusion."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("./adaptive_diffusion/.mplconfig").resolve())
)

import torch
import wandb
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from adaptive_diffusion.config import DiffusionConfig
from adaptive_diffusion.evaluation.fid import FIDCalculator
from adaptive_diffusion.models.diffusion import AdaptiveDiffusionModel
from adaptive_diffusion.training.scheduler import build_cosine_warmup_scheduler
from adaptive_diffusion.utils.device import resolve_device
from adaptive_diffusion.visualization.schedule_viz import plot_schedule_grid


class EMA:
    """Exponential moving average helper for model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        """Update EMA shadow weights from live model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model weights with EMA weights."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        """Restore live model weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> dict[str, Tensor]:
        """Return state dictionary for checkpointing."""
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load EMA state."""
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class Trainer:
    """Production-style trainer with EMA, warmup-cosine LR, and wandb logging."""

    def __init__(
        self,
        model: AdaptiveDiffusionModel,
        config: DiffusionConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        total_steps = max(1, len(train_loader) * config.num_epochs)
        self.lr_scheduler = build_cosine_warmup_scheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
        )
        self.ema = EMA(model=self.model, decay=config.ema_decay)
        self.fid = FIDCalculator(device=self.device)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.sample_dir = Path(config.sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_ckpts: list[tuple[float, Path]] = []

        self.wandb_run = self._init_wandb()

    def _init_wandb(self) -> wandb.sdk.wandb_run.Run | None:
        """Initialize Weights & Biases with robust fallback modes."""
        modes = [os.getenv("WANDB_MODE", "online"), "offline", "disabled"]
        attempted: list[str] = []
        for mode in modes:
            if mode in attempted:
                continue
            attempted.append(mode)
            if mode == "disabled":
                return None
            try:
                return wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                    mode=mode,
                    reinit=True,
                )
            except Exception as exc:  # pragma: no cover
                print(f"wandb.init failed in mode='{mode}': {exc}")
        return None

    def _wandb_log(self, data: dict[str, object], step: int) -> None:
        """Log to wandb when active."""
        if self.wandb_run is None:
            return
        wandb.log(data, step=step)

    @contextmanager
    def _ema_scope(self) -> Iterator[None]:
        """Temporarily evaluate model using EMA weights."""
        self.ema.apply_shadow(self.model)
        try:
            yield
        finally:
            self.ema.restore(self.model)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, float]:
        """Run one optimization step.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            Batch of images and labels.

        Returns
        -------
        dict[str, float]
            Scalar metrics for logging.
        """
        self.model.train()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self.model.p_losses(x0=images, class_labels=labels)
        loss = loss_dict["loss"]
        loss.backward()
        clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.ema.update(self.model)
        self.global_step += 1

        out = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        }
        out["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return out

    @torch.no_grad()
    def validate(self, num_batches: int = 50) -> dict[str, float]:
        """Run validation with EMA weights, including sample FID estimates."""
        self.model.eval()
        val_losses: list[float] = []
        real_batches: list[Tensor] = []
        with self._ema_scope():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                if batch_idx >= num_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                loss_dict = self.model.p_losses(x0=images, class_labels=labels)
                val_losses.append(float(loss_dict["loss"].detach().cpu()))
                real_batches.append(images.detach())

            class_labels = (
                torch.arange(64, device=self.device) % self.config.num_classes
            )

            samples_adaptive, adaptive_time = self.model.ddim_sample(
                class_labels=class_labels,
                num_steps=self.config.num_sample_steps_ddim,
            )
            real_images = torch.cat(real_batches, dim=0)[: samples_adaptive.shape[0]]
            fid_current = self.fid.compute(
                fake_images=samples_adaptive,
                real_images=real_images,
                cache_key=f"val_real_{self.config.schedule_mode}",
            )
            metrics = {
                "val_loss": float(sum(val_losses) / max(1, len(val_losses))),
                "fid_current_schedule": float(fid_current),
                "sample_time_current": float(adaptive_time),
            }
            if self.model.is_adaptive:
                start = time.perf_counter()
                counterfactual_samples = self.model.fixed_schedule_sample(
                    class_labels=class_labels,
                    num_steps=self.config.num_sample_steps_ddim,
                )
                counterfactual_time = time.perf_counter() - start
                counterfactual_fid = self.fid.compute(
                    fake_images=counterfactual_samples,
                    real_images=real_images,
                    cache_key="val_real_counterfactual_fixed",
                )
                metrics.update(
                    {
                        "fid_counterfactual_fixed_sampler": float(counterfactual_fid),
                        "sample_time_counterfactual_fixed": float(counterfactual_time),
                    }
                )

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save model/optimizer/EMA state and retain top-3 by current-schedule FID."""
        ckpt_path = (
            self.checkpoint_dir
            / f"epoch_{epoch:04d}_fid_{metrics['fid_current_schedule']:.4f}.pt"
        )
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "ema_state": self.ema.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
        }
        torch.save(payload, ckpt_path)

        config_path = ckpt_path.with_suffix(".json")
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        self.best_ckpts.append((metrics["fid_current_schedule"], ckpt_path))
        self.best_ckpts.sort(key=lambda item: item[0])
        while len(self.best_ckpts) > 3:
            _, stale_path = self.best_ckpts.pop(-1)
            if stale_path.exists():
                stale_path.unlink()
            stale_json = stale_path.with_suffix(".json")
            if stale_json.exists():
                stale_json.unlink()

    def train(self, num_epochs: int | None = None) -> None:
        """Run full training loop with periodic logging/validation."""
        target_epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        for epoch in range(1, target_epochs + 1):
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{target_epochs}")
            for batch in progress:
                metrics = self.train_step(batch=batch)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    diff=f"{metrics['loss_diffusion']:.4f}",
                    eff=f"{metrics['loss_efficiency']:.4f}",
                    smooth=f"{metrics['loss_smoothness']:.4f}",
                )
                if self.global_step % 100 == 0:
                    self._wandb_log(metrics, step=self.global_step)

                if self.global_step % 1000 == 0:
                    # Runtime schedule invariant assertion.
                    if self.model.is_adaptive:
                        labels = torch.arange(
                            self.config.num_classes, device=self.device
                        )
                        if (
                            self.model.schedule_class_embedding is None
                            or self.model.schedule_net is None
                        ):
                            raise ValueError(
                                "Adaptive schedule invariant check requires "
                                "schedule_class_embedding and schedule_net."
                            )
                        class_emb = self.model.schedule_class_embedding(labels)
                        alpha_bar = self.model.schedule_net.get_alpha_bar(class_emb)
                        if torch.any(torch.diff(alpha_bar, dim=-1) >= 0):
                            raise ValueError(
                                "alpha_bar monotonicity invariant failed during training."
                            )

            with self._ema_scope():
                demo_labels = (
                    torch.arange(64, device=self.device) % self.config.num_classes
                )
                samples, _ = self.model.ddim_sample(
                    class_labels=demo_labels,
                    num_steps=self.config.num_sample_steps_ddim,
                )
            try:
                grid = make_grid(((samples + 1.0) * 0.5).clamp(0.0, 1.0), nrow=8)
                self._wandb_log({"samples": wandb.Image(grid)}, step=self.global_step)
            except Exception as exc:  # pragma: no cover
                print(f"Skipping sample grid logging due to error: {exc}")

            if self.model.is_adaptive:
                try:
                    schedule_fig = plot_schedule_grid(
                        model=self.model,
                        save_path=str(
                            self.sample_dir / f"schedules_epoch_{epoch:04d}.png"
                        ),
                    )
                    self._wandb_log(
                        {"schedule_grid": wandb.Image(schedule_fig)},
                        step=self.global_step,
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"Skipping schedule visualization due to error: {exc}")

            do_validate = (epoch % self.config.validate_every_epochs == 0) or (
                epoch == target_epochs
            )
            if do_validate:
                val_metrics = self.validate(
                    num_batches=self.config.validate_num_batches
                )
                self._wandb_log(val_metrics, step=self.global_step)
                self.save_checkpoint(epoch=epoch, metrics=val_metrics)

        if self.wandb_run is not None:
            self.wandb_run.finish()

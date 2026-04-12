"""FID computation utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms


def _to_uint8(images: Tensor) -> Tensor:
    """Convert images from [-1, 1] float range to uint8 [0, 255]."""
    images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return images.to(torch.uint8)


class FIDCalculator:
    """Frechet Inception Distance helper with cached real feature statistics."""

    def __init__(
        self, device: torch.device | str = "cpu", feature_dim: int = 2048
    ) -> None:
        requested = torch.device(device)
        # TorchMetrics FID keeps float64 buffers internally; MPS does not support float64.
        # Run FID on CPU when MPS is requested to keep training/eval portable on Apple Silicon.
        self.device = torch.device("cpu") if requested.type == "mps" else requested
        self.feature_dim = feature_dim
        self._metrics: dict[str, FrechetInceptionDistance] = {}
        self._real_loaded: set[str] = set()

    def _metric(self, cache_key: str) -> FrechetInceptionDistance:
        if cache_key not in self._metrics:
            metric = FrechetInceptionDistance(
                feature=self.feature_dim,
                normalize=False,
                reset_real_features=False,
            ).to(self.device)
            self._metrics[cache_key] = metric
        return self._metrics[cache_key]

    @torch.no_grad()
    def compute(
        self, fake_images: Tensor, real_images: Tensor, cache_key: str = "default"
    ) -> float:
        """Compute FID between generated and real image batches."""
        metric = self._metric(cache_key=cache_key)
        fake = _to_uint8(fake_images.detach().to(self.device))
        real = _to_uint8(real_images.detach().to(self.device))
        if cache_key not in self._real_loaded:
            metric.update(real, real=True)
            self._real_loaded.add(cache_key)
        metric.update(fake, real=False)
        fid = float(metric.compute().item())
        metric.reset()
        return fid


@torch.no_grad()
def compute_fid_against_cifar10(
    fake_images: Tensor,
    data_root: str = "./data",
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> float:
    """Compute FID between generated samples and CIFAR-10 test split."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        ),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    calculator = FIDCalculator(device=device)
    real_batches: list[Tensor] = []
    needed = fake_images.shape[0]
    for images, _ in loader:
        real_batches.append(images)
        if sum(batch.shape[0] for batch in real_batches) >= needed:
            break
    real_images = torch.cat(real_batches, dim=0)[:needed]
    return calculator.compute(
        fake_images=fake_images, real_images=real_images, cache_key="cifar10_test"
    )

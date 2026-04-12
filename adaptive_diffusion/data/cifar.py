"""CIFAR-10 data utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from adaptive_diffusion.config import DiffusionConfig


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader worker process deterministically.

    Notes
    -----
    This function must stay module-level (not nested) so it is picklable
    under Python 3.14 spawn/forkserver multiprocessing start methods.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    seed = getattr(dataset, "_seed", None)
    if seed is None:
        return
    torch.manual_seed(int(seed) + worker_id)


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Construct train/eval transforms for CIFAR-10."""
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


def get_cifar10_dataloaders(
    config: DiffusionConfig,
    root: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train/test dataloaders.

    Parameters
    ----------
    config : DiffusionConfig
        Training configuration.
    root : str, optional
        Dataset root path, by default ``"./data"``.
    num_workers : int, optional
        Number of dataloader workers.
    pin_memory : bool, optional
        Whether to pin host memory.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    Path(root).mkdir(parents=True, exist_ok=True)
    train_transform, eval_transform = _build_transforms(config.image_size)

    train_set = datasets.CIFAR10(
        root=root, train=True, transform=train_transform, download=True
    )
    val_set = datasets.CIFAR10(
        root=root, train=False, transform=eval_transform, download=True
    )

    use_cuda = torch.cuda.is_available() and config.device.startswith("cuda")
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    setattr(train_set, "_seed", int(config.seed))
    setattr(val_set, "_seed", int(config.seed))

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and use_cuda,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
        drop_last=False,
    )
    return train_loader, val_loader

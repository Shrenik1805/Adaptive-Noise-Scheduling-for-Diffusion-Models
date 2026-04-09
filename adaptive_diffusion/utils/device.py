"""Device resolution helpers for CUDA / MPS / CPU."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve runtime device with graceful fallback.

    Parameters
    ----------
    requested : str | None, optional
        Requested device string. Examples: ``"cuda"``, ``"mps"``, ``"cpu"``.
        If ``None`` or ``"auto"``, best available accelerator is selected.

    Returns
    -------
    torch.device
        Selected device object.
    """
    req = "auto" if requested is None else requested.lower()
    if req in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if req in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    if req not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device specifier: {requested}")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Synchronize async kernels for deterministic timing.

    Parameters
    ----------
    device : torch.device
        Device to synchronize.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()  # type: ignore[attr-defined]

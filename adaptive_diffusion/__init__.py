"""Adaptive diffusion package."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("./adaptive_diffusion/.mplconfig").resolve())
)

from adaptive_diffusion.config import DiffusionConfig

__all__ = ["DiffusionConfig"]

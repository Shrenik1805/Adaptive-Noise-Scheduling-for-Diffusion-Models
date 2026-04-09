"""Visualization package."""

from adaptive_diffusion.visualization.schedule_viz import (
    CIFAR10_CLASSES,
    plot_efficiency_frontier,
    plot_per_class_speedup,
    plot_schedule_grid,
)

__all__ = [
    "CIFAR10_CLASSES",
    "plot_schedule_grid",
    "plot_efficiency_frontier",
    "plot_per_class_speedup",
]

"""Local backlight dimming algorithms for LED-LCD displays."""

from .display_model import DisplayModel
from .algorithms import (
    average_dimming,
    maximum_dimming,
    sqrt_dimming,
    cho_dimming,
    nam_dimming,
    proposed_histogram_dimming,
)
from .compensation import compensate_lcd

__all__ = [
    "DisplayModel",
    "average_dimming",
    "maximum_dimming",
    "sqrt_dimming",
    "cho_dimming",
    "nam_dimming",
    "proposed_histogram_dimming",
    "compensate_lcd",
]

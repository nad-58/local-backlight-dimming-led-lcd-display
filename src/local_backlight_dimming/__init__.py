"""Local backlight dimming algorithms for LED-LCD displays."""

from .display_model import DisplayModel
from .algorithms import (
    average_dimming,
    average_plus_dimming,
    chen_dimming,
    cho_dimming,
    conventional_dimming,
    local_compensation_cho,
    local_compensation_nam,
    matlab_ehsan_variant,
    maximum_dimming,
    nam_dimming,
    proposed_histogram_dimming,
    sqrt_dimming,
)
from .compensation import compensate_lcd
from .optimization import OptimizationParams, clipper_free_dimming, fast_refinement, new_dimming
from .result import RenderResult, render_result

__all__ = [
    "DisplayModel",
    "RenderResult",
    "OptimizationParams",
    "average_dimming",
    "average_plus_dimming",
    "chen_dimming",
    "cho_dimming",
    "conventional_dimming",
    "local_compensation_cho",
    "local_compensation_nam",
    "matlab_ehsan_variant",
    "maximum_dimming",
    "nam_dimming",
    "proposed_histogram_dimming",
    "sqrt_dimming",
    "clipper_free_dimming",
    "fast_refinement",
    "new_dimming",
    "compensate_lcd",
    "render_result",
]

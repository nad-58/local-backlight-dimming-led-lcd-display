from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compensation import compensate_lcd
from .display_model import DisplayModel
from .metrics import mae, mse, psnr, relative_backlight_power, ssim_global

Array = np.ndarray


@dataclass
class RenderResult:
    led_values: Array
    backlight: Array
    compensated: Array
    physical: Array
    perceived: Array
    leakage: Array
    clipper: Array
    leaking_pixels: int
    clipped_pixels: int
    max_leakage: float
    max_clipper: float
    avg_leakage: float
    avg_clipper: float
    mae: float
    mse_total: float
    psnr: float
    mssim: float
    contrast: float
    power: float


def render_result(display: DisplayModel, image: Array, led_values: Array, quantize_led: bool = True) -> RenderResult:
    """Python conversion of `renderResult.m`.

    It quantizes LED values, simulates the backlight, applies leakage-aware LCD
    compensation, computes the physical/perceived output and reports quality and
    leakage/clipping measurements.
    """
    led = np.asarray(led_values, dtype=float).ravel()
    if quantize_led:
        led = np.round(255.0 * led) / 255.0
    led = np.clip(led, 0.0, 1.0)
    backlight = display.simulate_backlight(led)
    compensated, perceived = compensate_lcd(image, backlight, leakage=display.leakage, gamma=display.gamma)
    physical = perceived ** display.gamma
    image_linear = np.asarray(image, dtype=float)
    if image_linear.max(initial=0) > 1.0:
        image_linear = image_linear / 255.0
    image_linear = np.clip(image_linear, 0.0, 1.0) ** display.gamma

    clipping_mask = (compensated >= 1.0).astype(float)
    leakage_mask = (compensated <= 0.0).astype(float)
    clipper = clipping_mask * np.maximum(image_linear - physical, 0.0)
    leakage = leakage_mask * np.maximum(physical - image_linear, 0.0)
    threshold = (1.0 / 255.0) ** display.gamma
    clipper[(clipper > 0) & (clipper < 1e-5)] = 0.0
    leakage[(leakage > 0) & (leakage < threshold)] = 0.0

    leak_values = leakage[leakage > 0]
    clip_values = clipper[clipper > 0]
    return RenderResult(
        led_values=led,
        backlight=backlight,
        compensated=compensated,
        physical=physical,
        perceived=perceived,
        leakage=leakage,
        clipper=clipper,
        leaking_pixels=int(leak_values.size),
        clipped_pixels=int(clip_values.size),
        max_leakage=float(np.max(leak_values)) if leak_values.size else 0.0,
        max_clipper=float(np.max(clip_values)) if clip_values.size else 0.0,
        avg_leakage=float(np.mean(leak_values)) if leak_values.size else 0.0,
        avg_clipper=float(np.mean(clip_values)) if clip_values.size else 0.0,
        mae=mae(image, perceived),
        mse_total=mse(image, perceived),
        psnr=psnr(image, perceived),
        mssim=ssim_global(image, perceived),
        contrast=float(np.max(physical) - np.min(physical)),
        power=relative_backlight_power(led),
    )


def compare_solutions(solution1: RenderResult, solution2: RenderResult) -> dict[str, tuple[float, float]]:
    """Converted from `compareSolutions.m`; returns metric pairs."""
    fields = [
        "leaking_pixels",
        "clipped_pixels",
        "max_leakage",
        "max_clipper",
        "mae",
        "mse_total",
        "psnr",
        "mssim",
        "contrast",
        "power",
    ]
    return {name: (getattr(solution1, name), getattr(solution2, name)) for name in fields}

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_float_image(image: Array) -> Array:
    img = np.asarray(image, dtype=float)
    if img.max(initial=0) > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def compensate_lcd(
    target: Array,
    backlight: Array,
    leakage: float = 0.001,
    gamma: float = 2.2,
) -> tuple[Array, Array]:
    """Compensate LCD transmittance for dimmed backlight.

    The model follows the thesis equations for luminance, leakage and brightness
    compensation. The target is interpreted as normalized sRGB and is converted to
    a linear-light approximation before compensation.

    Returns
    -------
    compensated_transmittance:
        LCD transmittance values clipped to [0, 1].
    reproduced:
        Simulated reproduced image after backlight and leakage.
    """
    image = _as_float_image(target)
    bl = np.asarray(backlight, dtype=float)
    if bl.ndim == 2 and image.ndim == 3:
        bl = bl[..., None]
    bl = np.clip(bl, 1e-6, 1.0)

    target_linear = np.clip(image, 0.0, 1.0) ** gamma
    ideal_transmittance = target_linear / bl
    compensated = (ideal_transmittance - leakage) / max(1.0 - leakage, 1e-9)
    compensated = np.clip(compensated, 0.0, 1.0)

    observed_transmittance = (1.0 - leakage) * compensated + leakage
    reproduced_linear = bl * observed_transmittance
    reproduced = np.clip(reproduced_linear, 0.0, 1.0) ** (1.0 / gamma)
    return compensated, reproduced


def hard_clip_compensation(target: Array, backlight: Array, gamma: float = 2.2) -> Array:
    """Simplified hard-clipping compensation used by many dimming simulations."""
    image = _as_float_image(target)
    bl = np.asarray(backlight, dtype=float)
    if bl.ndim == 2 and image.ndim == 3:
        bl = bl[..., None]
    target_linear = image**gamma
    return np.clip(target_linear / np.clip(bl, 1e-6, 1.0), 0.0, 1.0)

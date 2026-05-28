from __future__ import annotations

import numpy as np

Array = np.ndarray


def initialize_alpha_to_epsilon(center_epsilon: float, mode: str = "linear") -> Array:
    """Converted from the LUT helper inside `calculateEpsilon.m`."""
    alpha_eps = np.array(
        [0.097, 0.107, 0.123, 0.13, 0.147, 0.172, 0.178, 0.197, 0.224, 0.233, 0.237, 0.24, 0.239, 0.248, 0.236],
        dtype=float,
    )
    alpha_eps = center_epsilon * (alpha_eps / alpha_eps[0])
    lut = np.zeros(90, dtype=float)
    if mode == "quantized":
        for i in range(14):
            lut[i * 5 : (i + 1) * 5] = alpha_eps[i]
        lut[70:] = alpha_eps[-1]
    elif mode == "linear":
        for i in range(14):
            j = i * 5
            lut[j + 0] = alpha_eps[i]
            lut[j + 1] = 0.8 * alpha_eps[i] + 0.2 * alpha_eps[i + 1]
            lut[j + 2] = 0.6 * alpha_eps[i] + 0.4 * alpha_eps[i + 1]
            lut[j + 3] = 0.4 * alpha_eps[i] + 0.6 * alpha_eps[i + 1]
            lut[j + 4] = 0.2 * alpha_eps[i] + 0.8 * alpha_eps[i + 1]
        lut[70:] = alpha_eps[-1]
    else:
        raise ValueError("mode must be 'linear' or 'quantized'")
    return lut


def calculate_epsilon(
    height: int,
    width: int,
    center_epsilon: float,
    distance: float = 1080.0,
    vx: float | None = None,
    vy: float | None = None,
    mode: str = "constant",
) -> Array:
    """Converted from `calculateEpsilon.m`.

    Modes are `constant`, `horVariation`, and `horVerVariation`.
    """
    if vx is None:
        vx = width / 2.0
    if vy is None:
        vy = height / 2.0
    if mode == "constant":
        return np.full((height, width), center_epsilon, dtype=float)
    lut = initialize_alpha_to_epsilon(center_epsilon, "linear")
    eps = np.zeros((height, width), dtype=float)
    xs = np.arange(width)
    if mode == "horVariation":
        alpha = np.abs(np.round(np.degrees(np.arctan((vx - xs) / distance)))).astype(int)
        alpha = np.clip(alpha, 0, len(lut) - 1)
        eps[:, :] = lut[alpha][None, :]
        return eps
    if mode == "horVerVariation":
        yy, xx = np.mgrid[0:height, 0:width]
        dist = np.sqrt((vx - xx) ** 2 + (vy - yy) ** 2)
        alpha = np.abs(np.round(np.degrees(np.arctan(dist / distance)))).astype(int)
        alpha = np.clip(alpha, 0, len(lut) - 1)
        return lut[alpha]
    raise ValueError("mode must be 'constant', 'horVariation', or 'horVerVariation'")

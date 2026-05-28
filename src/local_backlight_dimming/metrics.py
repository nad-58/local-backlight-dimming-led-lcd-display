from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_float(arr: Array) -> Array:
    out = np.asarray(arr, dtype=float)
    if out.max(initial=0) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def mse(reference: Array, test: Array) -> float:
    ref = _as_float(reference)
    tst = _as_float(test)
    return float(np.mean((ref - tst) ** 2))


def mae(reference: Array, test: Array) -> float:
    ref = _as_float(reference)
    tst = _as_float(test)
    return float(np.mean(np.abs(ref - tst)))


def psnr(reference: Array, test: Array, max_value: float = 1.0) -> float:
    err = mse(reference, test)
    if err <= 0:
        return float("inf")
    return float(10.0 * np.log10((max_value * max_value) / err))


def ssim_global(reference: Array, test: Array, max_value: float = 1.0) -> float:
    """Small global SSIM-like metric without external dependencies.

    This is not a windowed replacement for scikit-image SSIM, but it is useful for
    lightweight regression tests and algorithm comparisons.
    """
    x = _as_float(reference).ravel()
    y = _as_float(test).ravel()
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    mux, muy = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    numerator = (2 * mux * muy + c1) * (2 * cov + c2)
    denominator = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def relative_backlight_power(led_values: Array) -> float:
    """Mean LED drive level, used as a relative backlight power estimate."""
    led = np.clip(np.asarray(led_values, dtype=float), 0.0, 1.0)
    return float(np.mean(led))


def temporal_ssd(sequence: Array) -> float:
    """Average sum of squared frame-to-frame differences.

    Lower values indicate less temporal variation and usually lower flicker risk.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.shape[0] < 2:
        return 0.0
    diffs = np.diff(seq, axis=0)
    return float(np.mean(np.sum(diffs * diffs, axis=tuple(range(1, diffs.ndim)))))

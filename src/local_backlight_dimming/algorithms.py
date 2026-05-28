from __future__ import annotations

import numpy as np

from .display_model import DisplayModel

Array = np.ndarray


def _as_float_image(image: Array) -> Array:
    img = np.asarray(image, dtype=float)
    if img.max(initial=0) > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _gray_max_rgb(patch: Array) -> Array:
    if patch.ndim == 2:
        return patch
    return np.max(patch, axis=2)


def _segment_feature(display: DisplayModel, image: Array, reducer) -> Array:
    img = _as_float_image(image)
    values = []
    for _, _, patch in display.iter_segments(img):
        values.append(float(reducer(patch)))
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def average_dimming(display: DisplayModel, image: Array) -> Array:
    """Average-value baseline converted from `algAvg.m`."""
    return _segment_feature(display, image, lambda patch: np.mean(patch))


def maximum_dimming(display: DisplayModel, image: Array) -> Array:
    """Maximum-value baseline converted from `algMax.m`."""
    return _segment_feature(display, image, lambda patch: np.max(patch))


def sqrt_dimming(display: DisplayModel, image: Array) -> Array:
    """Square-root baseline, a common power/quality compromise."""
    avg = average_dimming(display, image)
    return np.sqrt(np.clip(avg, 0.0, 1.0))


def cho_dimming(display: DisplayModel, image: Array) -> Array:
    """Cho-style average plus correction term.

    This follows the MATLAB `algCho.m` structure and thesis Eq. 3.1-3.2:
    r = average + 0.5 * (diff + diff^2 / 255), where diff=max-average.
    """
    img = _as_float_image(image)
    values = []
    for _, _, patch in display.iter_segments(img):
        p = _gray_max_rgb(patch) * 255.0
        avg = float(np.mean(p))
        diff = float(np.max(p) - avg)
        value = avg + 0.5 * (diff + (diff * diff) / 255.0)
        values.append(value / 255.0)
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def nam_dimming(display: DisplayModel, image: Array, gamma: float = 1.0) -> Array:
    """Nam-style local dimming baseline converted from `algNam.m`.

    The implementation keeps the global/local max-average decision logic while
    using safe numerical guards for modern NumPy use.
    """
    img = _as_float_image(image)
    gray = _gray_max_rgb(img) * 255.0
    full_max = float(np.max(gray))
    full_avg = float(np.mean(gray))
    full_m = (full_max + full_avg) / 2.0
    full_r = full_avg

    values = []
    for _, _, patch in display.iter_segments(img):
        p = _gray_max_rgb(patch) * 255.0
        g_max = float(np.max(p))
        g_avg = float(np.mean(p))
        g_m = (g_max + g_avg) / 2.0
        if g_m > full_m:
            led = (full_max / 255.0) ** gamma
        elif g_max <= full_r:
            led = (g_max / 255.0) ** gamma
        else:
            denom = max(full_max - full_r, 1e-9)
            b = 255.0 * (1.0 - ((1.0 - full_r / max(full_m, 1e-9)) / denom) * (g_max - full_r))
            led = (full_r / max(b, 1e-9)) ** gamma
        values.append(led)
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def proposed_histogram_dimming(
    display: DisplayModel,
    image: Array,
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.8,
    low_percentile: float = 60.0,
    medium_percentile: float = 65.0,
    high_percentile: float = 70.0,
) -> Array:
    """Thesis-inspired local histogram and image-feature algorithm.

    This implements the method described in Chapter 3, Section 3.1.4 and Appendix
    A.4 of the thesis. For each segment:

    1. Compute 8-bit local intensity g=max(R,G,B).
    2. Classify the segment by average intensity: low, medium or high.
    3. Use a histogram percentile as the initial backlight level.
    4. Add an adaptive feature term based on average, square-root average and
       coefficient of variation.

    Returns normalized LED values in [0, 1].
    """
    img = _as_float_image(image)
    values = []

    for _, _, patch in display.iter_segments(img):
        g = _gray_max_rgb(patch) * 255.0
        avg = float(np.mean(g))
        mx = float(np.max(g))
        var = float(np.var(g))
        sqrt_avg = float(np.sqrt(max(avg, 0.0) / 255.0) * 255.0)
        cv = var / max(avg, 1e-6)
        diff = mx - avg

        if avg <= 31.0:
            percentile = low_percentile
            feature = avg
        elif avg <= 95.0:
            percentile = medium_percentile
            feature = sqrt_avg
        else:
            percentile = high_percentile
            feature = sqrt_avg + cv

        ibl = float(np.percentile(g, percentile))
        if diff <= 31.0:
            led = ibl + alpha * feature
        elif diff <= 95.0:
            led = ibl + beta * feature
        else:
            led = ibl + gamma * feature
        values.append(led / 255.0)

    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def matlab_ehsan_variant(display: DisplayModel, image: Array) -> Array:
    """Cleaned variant inspired by the available `algEhsan.m` MATLAB file.

    The uploaded MATLAB file contained merge-conflict markers. This function keeps
    the recoverable core idea: mean plus Cho-like global/local correction plus a
    variance-over-mean local-detail term.
    """
    img = _as_float_image(image)
    gray_full = _gray_max_rgb(img) * 255.0
    global_mean = float(np.mean(gray_full))
    values = []
    for _, _, patch in display.iter_segments(img):
        p = _gray_max_rgb(patch) * 255.0
        mean = float(np.mean(p))
        diff = abs(global_mean - float(np.max(p)))
        correction = 0.5 * (diff + diff * diff / 256.0)
        detail = float(np.std(p)) / (mean + 0.01)
        values.append((mean + correction + detail) / 255.0)
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)

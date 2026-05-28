from __future__ import annotations

from typing import Callable
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


def _segment_feature(display: DisplayModel, image: Array, reducer: Callable[[Array], float]) -> Array:
    img = _as_float_image(image)
    values = [float(reducer(patch)) for _, _, patch in display.iter_segments(img)]
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def conventional_dimming(display: DisplayModel) -> Array:
    """Conventional full-backlight mode converted from `algConv.m`."""
    return np.ones(display.segment_count, dtype=float)


def average_dimming(display: DisplayModel, image: Array) -> Array:
    """Average-value baseline converted from `algAvg.m`."""
    return _segment_feature(display, image, lambda patch: np.mean(patch))


def maximum_dimming(display: DisplayModel, image: Array) -> Array:
    """Maximum-value baseline converted from `algMax.m`."""
    return _segment_feature(display, image, lambda patch: np.max(patch))


def sqrt_dimming(display: DisplayModel, image: Array) -> Array:
    """Square-root baseline, a common power/quality compromise."""
    return np.sqrt(np.clip(average_dimming(display, image), 0.0, 1.0))


def average_plus_dimming(display: DisplayModel, image: Array, percentile: float = 99.0) -> Array:
    """Improved average algorithm converted from `algAvgPlus.m`.

    The MATLAB version first computes average LED values, simulates the backlight,
    then scales the LED vector so that a high percentile of pixels can be shown.
    """
    img = _as_float_image(image)
    led = average_dimming(display, img)
    backlight = np.clip(display.simulate_backlight(led), 1e-6, 1.0)
    if img.ndim == 3:
        backlight = backlight[..., None]
    required = np.sort((img / backlight).ravel())
    idx = int(np.ceil((percentile / 100.0) * required.size)) - 1
    idx = int(np.clip(idx, 0, required.size - 1))
    return np.clip(led * required[idx], 0.0, 1.0)


def cho_dimming(display: DisplayModel, image: Array) -> Array:
    """Cho-style average-plus-correction algorithm converted from `algCho.m`."""
    img = _as_float_image(image)
    values = []
    for _, _, patch in display.iter_segments(img):
        p = _gray_max_rgb(patch) * 255.0
        avg = float(np.mean(p))
        diff = float(np.max(p) - avg)
        values.append((avg + 0.5 * (diff + diff * diff / 255.0)) / 255.0)
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)


def nam_dimming(display: DisplayModel, image: Array, gamma: float = 1.0) -> Array:
    """Nam-style local dimming baseline converted from `algNam.m`."""
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


def _box_filter2d(image: Array, radius: int = 1) -> Array:
    if radius <= 0:
        return image.copy()
    pad = radius
    padded = np.pad(image, [(pad, pad), (pad, pad)] + ([(0, 0)] if image.ndim == 3 else []), mode="edge")
    out = np.zeros_like(image, dtype=float)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = np.mean(padded[y : y + 2 * radius + 1, x : x + 2 * radius + 1], axis=(0, 1))
    return out


def chen_dimming(display: DisplayModel, image: Array, temporal_alpha: float = 0.125) -> Array:
    """Chen/Ehsan histogram-based dimming converted from `algChen.m`.

    The MATLAB code smooths the image, forms local histograms, calculates a
    histogram-weighted LED intensity, applies dark-scene enhancement and a simple
    temporal filter. This is a clean single-frame Python conversion.
    """
    img = _as_float_image(image)
    gray = _gray_max_rgb(_box_filter2d(img, radius=1))
    img8 = np.floor(255.0 * gray).astype(int)
    initial = []
    for _, _, patch in display.iter_segments(img8):
        hist = np.bincount(patch.ravel(), minlength=256).astype(float)
        hist /= max(hist.sum(), 1.0)
        initial.append(min(255.0, float(np.sum(hist * np.arange(1, 257)))) - 1.0)
    initial = np.asarray(initial, dtype=float)
    mean_initial = float(np.mean(initial))
    enhanced = initial.copy()
    mask = np.logical_and(initial >= mean_initial, mean_initial <= 5.0)
    enhanced[mask] = np.minimum(255.0, initial[mask] + 3.0 * (initial[mask] - mean_initial))
    normalized_mean = float(np.mean(img8) / 255.0)
    r = min(1.0, temporal_alpha + abs(normalized_mean))
    filtered = r * enhanced + (1.0 - r) * 255.0
    return np.clip(filtered / 255.0, 0.0, 1.0)


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

    For each segment: compute g=max(R,G,B), classify brightness, choose a local
    histogram percentile, and add an adaptive image-feature correction.
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
            percentile, feature = low_percentile, avg
        elif avg <= 95.0:
            percentile, feature = medium_percentile, sqrt_avg
        else:
            percentile, feature = high_percentile, sqrt_avg + cv
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
    """Cleaned variant inspired by the available `algEhsan.m` revisions."""
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


def local_compensation_cho(target: Array, backlight: Array) -> Array:
    """Pixel compensation converted from `algComCho.m`."""
    img = _as_float_image(target)
    bl = np.clip(np.asarray(backlight, dtype=float), 1e-6, 1.0)
    if img.ndim == 3 and bl.ndim == 2:
        bl = bl[..., None]
    image8 = np.round(255.0 * img)
    bl8 = 255.0 * bl
    max_in = float(np.max(image8))
    x1 = 2.0 * bl8 - max_in
    bl_com = np.where(image8 > x1, bl8, (max_in + image8) / 2.0)
    return np.clip(image8 * 255.0 / (bl_com + 1e-4) / 255.0, 0.0, 1.0)


def local_compensation_nam(display: DisplayModel, target: Array) -> Array:
    """Approximate pixel-compensation stage converted from `algComNam.m`."""
    img = _gray_max_rgb(_as_float_image(target))
    img8 = np.floor(255.0 * img)
    full_max = float(np.max(img8))
    full_avg = float(np.mean(img8))
    full_m = (full_max + full_avg) / 2.0
    full_r = full_avg
    out = np.zeros_like(img8, dtype=float)
    weight = np.zeros_like(img8, dtype=float)
    for r_i in range(display.rows):
        for c_i in range(display.cols):
            ys, xs = display.segment_slices(r_i, c_i)
            patch = img8[ys, xs]
            seg_max = float(np.max(patch))
            seg_avg = float(np.mean(patch))
            seg_m = (seg_max + seg_avg) / 2.0
            if seg_m > full_m:
                low = patch <= full_r
                comp = np.empty_like(patch, dtype=float)
                comp[low] = (255.0 / max(full_m, 1e-9)) * patch[low]
                x = (255.0 - 255.0 * full_r / max(full_m, 1e-9)) / max(full_max - full_r, 1e-9)
                comp[~low] = x * (patch[~low] - full_r) + 255.0 / max(full_m, 1e-9) * full_r
            elif seg_max <= full_r:
                comp = 255.0 / max(seg_max, 1e-9) * patch
            else:
                b = 255.0 * (1.0 - ((1.0 - full_r / max(full_m, 1e-9)) / max(full_max - full_r, 1e-9)) * (seg_max - full_r))
                low = patch <= full_r
                comp = np.empty_like(patch, dtype=float)
                comp[low] = (b / max(full_r, 1e-9)) * patch[low]
                x = (255.0 - 255.0 * full_r / max(full_m, 1e-9)) / max(full_max - full_r, 1e-9)
                comp[~low] = b + (patch[~low] - full_r) * x
            out[ys, xs] += comp
            weight[ys, xs] += 1.0
    return np.clip(out / np.maximum(weight, 1.0) / 255.0, 0.0, 1.0)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .display_model import DisplayModel

Array = np.ndarray


@dataclass
class OptimizationParams:
    """Parameters converted from the MATLAB `inputParams` structure."""

    target_epsilon: float = 0.0
    norm: int = 1
    hf_enhance: float = 0.0
    use_gamma_perceptual_optimization: bool = True
    power_weight: float = 0.001
    gamma: float = 2.2
    psf_reflections: int = 0
    iterations: int = 250
    step_size: float = 0.2


def laplacian_of_gaussian_activity(image: Array) -> Array:
    """Lightweight replacement for MATLAB `imfilter(y, -fspecial('log'))`."""
    y = np.asarray(image, dtype=float)
    if y.ndim == 3:
        y = np.max(y, axis=2)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    padded = np.pad(y, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(y, dtype=float)
    for row in range(y.shape[0]):
        for col in range(y.shape[1]):
            out[row, col] = -np.sum(padded[row : row + 3, col : col + 3] * kernel)
    return np.maximum(out, 0.0)


def calculate_weights(target: Array, gamma: float = 2.2, toe: float = 0.18) -> Array:
    """Converted from the MATLAB `calculateWeights` helper."""
    target = np.clip(np.asarray(target, dtype=float), 1e-6, 1.0)
    slopes = gamma * target ** (gamma - 1.0)
    return np.maximum(slopes, toe)


def calculate_influence_matrix(display: DisplayModel) -> Array:
    """Build the pixel-by-LED influence matrix used by optimisation algorithms.

    This is the Python equivalent of `calculateInfluenceMatrix` in `algNew.m`
    and `algFast.m`. It is intended for low-resolution experiments because the
    dense matrix can be large.
    """
    columns = []
    for index in range(display.segment_count):
        led = np.zeros(display.segment_count, dtype=float)
        led[index] = 1.0
        columns.append(display.simulate_backlight(led).ravel())
    return np.stack(columns, axis=1)


def optimize_led_projected_gradient(
    target: Array,
    influence: Array,
    params: OptimizationParams,
    epsilon: Array | float = 0.001,
    activity: Array | None = None,
) -> Array:
    """Numerical Python replacement for the CVX optimisation in `algNew.m`.

    The MATLAB version solves a convex problem with CVX. To avoid a heavy solver
    dependency, this function uses projected gradient descent on a smooth
    surrogate of the same constraints: clipping, leakage and average LED power.
    """
    y = np.asarray(target, dtype=float)
    if y.ndim == 3:
        y = np.max(y, axis=2)
    y = np.clip(y, 0.0, 1.0)
    y = y * (1.0 - params.target_epsilon) + params.target_epsilon
    w = y.ravel()

    z = np.zeros_like(w) if activity is None else np.asarray(activity, dtype=float).ravel()
    texture_weight = 1.0 + params.hf_enhance * z

    if params.use_gamma_perceptual_optimization:
        perceptual_weight = calculate_weights(w, params.gamma).ravel()
    else:
        perceptual_weight = np.ones_like(w)

    eps = np.asarray(epsilon, dtype=float)
    eps_vec = np.full_like(w, float(eps)) if eps.ndim == 0 else eps.ravel()

    h = np.asarray(influence, dtype=float)
    r = np.clip(np.linalg.lstsq(h, w, rcond=None)[0], 0.0, 1.0)
    lr = params.step_size

    for _ in range(params.iterations):
        pred = h @ r
        clipping_residual = np.maximum(w - pred, 0.0) * texture_weight / perceptual_weight
        leakage_residual = np.maximum(eps_vec * pred - w, 0.0) / perceptual_weight
        grad = -(h.T @ clipping_residual) + h.T @ (eps_vec * leakage_residual)
        grad += params.power_weight / max(r.size, 1)
        r = np.clip(r - lr * grad / max(h.shape[0], 1), 0.0, 1.0)
    return r


def new_dimming(display: DisplayModel, target: Array, params: OptimizationParams | None = None) -> Array:
    """Converted version of `algNew.m` using projected-gradient optimisation."""
    params = params or OptimizationParams()
    activity = laplacian_of_gaussian_activity(target)
    influence = calculate_influence_matrix(display)
    eps = display.leakage
    return optimize_led_projected_gradient(target, influence, params, epsilon=eps, activity=activity)


def clipper_free_dimming(display: DisplayModel, target: Array, power_weight: float = 1e-6) -> Array:
    """Approximate conversion of `algClipperFreeXiao.m` without CVX.

    The original CVX formulation minimises power while enforcing `H r >= y`. This
    implementation solves a projected least-squares surrogate and then increases
    LEDs until sampled pixels satisfy the clipping-free constraint.
    """
    y = np.asarray(target, dtype=float)
    if y.ndim == 3:
        y = np.max(y, axis=2)
    y = np.clip(y, 0.0, 1.0).ravel()
    h = calculate_influence_matrix(display)
    r = np.clip(np.linalg.lstsq(h + power_weight, y, rcond=None)[0], 0.0, 1.0)
    for _ in range(100):
        pred = h @ r
        residual = y - pred
        if np.max(residual) <= 1e-4:
            break
        worst = np.argmax(residual)
        influence = h[worst]
        if np.max(influence) <= 1e-12:
            break
        r += (residual[worst] / max(np.sum(influence), 1e-12)) * influence
        r = np.clip(r, 0.0, 1.0)
    return r


def fast_refinement(
    display: DisplayModel,
    target: Array,
    starting_led: Array,
    damaged_mask: Array | None = None,
    max_iter: int = 15,
    params: OptimizationParams | None = None,
) -> Array:
    """Compact Python counterpart of `algFast.m`.

    The MATLAB method repeatedly re-optimises leakage/clipping-damaged pixels.
    This implementation accepts an optional mask and refines LED values on that
    mask; if no mask is supplied it falls back to `new_dimming`.
    """
    params = params or OptimizationParams(iterations=120)
    if damaged_mask is None or not np.any(damaged_mask):
        return new_dimming(display, target, params)
    target_gray = np.asarray(target, dtype=float)
    if target_gray.ndim == 3:
        target_gray = np.max(target_gray, axis=2)
    influence = calculate_influence_matrix(display)
    mask = damaged_mask.astype(bool).ravel()
    h_masked = influence[mask]
    y_masked = np.clip(target_gray.ravel()[mask], 0.0, 1.0)
    r = np.asarray(starting_led, dtype=float).copy().ravel()
    for _ in range(max_iter):
        correction = optimize_led_projected_gradient(
            y_masked.reshape(-1, 1),
            h_masked,
            params,
            epsilon=display.leakage,
            activity=None,
        )
        if np.linalg.norm(correction - r) < 1e-4:
            break
        r = correction
    return np.clip(r, 0.0, 1.0)

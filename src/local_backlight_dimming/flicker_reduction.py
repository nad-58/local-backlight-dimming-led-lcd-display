from __future__ import annotations

import numpy as np

Array = np.ndarray


def exponential_smoothing(led_sequence: Array, alpha: float = 0.25) -> Array:
    """First-order temporal smoothing for LED values.

    Parameters
    ----------
    led_sequence:
        Sequence shaped (frames, segments) or (frames, rows, cols).
    alpha:
        Update coefficient. Smaller values produce stronger flicker suppression.
    """
    seq = np.asarray(led_sequence, dtype=float)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if seq.shape[0] == 0:
        return seq.copy()
    out = np.empty_like(seq, dtype=float)
    out[0] = seq[0]
    for t in range(1, seq.shape[0]):
        out[t] = alpha * seq[t] + (1.0 - alpha) * out[t - 1]
    return np.clip(out, 0.0, 1.0)


def adaptive_iir_filter(
    led_sequence: Array,
    activity_sequence: Array | None = None,
    base_alpha: float = 0.125,
    max_alpha: float = 1.0,
) -> Array:
    """Adaptive temporal IIR filter inspired by the thesis flicker-reduction step.

    The thesis proposes adaptive filtering to reduce abrupt LED changes while still
    allowing larger updates when image content changes significantly. This function
    uses the mean LED change, optionally blended with an external activity signal,
    to select the update coefficient per frame.
    """
    seq = np.asarray(led_sequence, dtype=float)
    if seq.shape[0] == 0:
        return seq.copy()
    out = np.empty_like(seq, dtype=float)
    out[0] = seq[0]
    activity = None if activity_sequence is None else np.asarray(activity_sequence, dtype=float)

    for t in range(1, seq.shape[0]):
        delta = float(np.mean(np.abs(seq[t] - seq[t - 1])))
        if activity is not None:
            delta = 0.5 * delta + 0.5 * float(np.clip(activity[t], 0.0, 1.0))
        alpha = float(np.clip(base_alpha + delta, base_alpha, max_alpha))
        out[t] = alpha * seq[t] + (1.0 - alpha) * out[t - 1]
    return np.clip(out, 0.0, 1.0)


def second_order_iir(led_sequence: Array, a1: float = 0.6, a2: float = 0.2) -> Array:
    """Simple second-order IIR smoother for LED sequences.

    y[t] = (1-a1-a2) x[t] + a1 y[t-1] + a2 y[t-2]
    """
    seq = np.asarray(led_sequence, dtype=float)
    if seq.shape[0] == 0:
        return seq.copy()
    if a1 < 0 or a2 < 0 or a1 + a2 >= 1.0:
        raise ValueError("expected a1>=0, a2>=0 and a1+a2<1")
    out = np.empty_like(seq, dtype=float)
    out[0] = seq[0]
    if seq.shape[0] > 1:
        out[1] = (1.0 - a1) * seq[1] + a1 * out[0]
    for t in range(2, seq.shape[0]):
        out[t] = (1.0 - a1 - a2) * seq[t] + a1 * out[t - 1] + a2 * out[t - 2]
    return np.clip(out, 0.0, 1.0)

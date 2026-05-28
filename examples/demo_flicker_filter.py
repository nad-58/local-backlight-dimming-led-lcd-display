from __future__ import annotations

import numpy as np

from local_backlight_dimming.flicker_reduction import adaptive_iir_filter, second_order_iir
from local_backlight_dimming.metrics import temporal_ssd


def main() -> None:
    rng = np.random.default_rng(7)
    frames, segments = 24, 16
    base = np.linspace(0.2, 0.8, frames)[:, None]
    noise = 0.18 * rng.normal(size=(frames, segments))
    led_sequence = np.clip(base + noise, 0.0, 1.0)

    adaptive = adaptive_iir_filter(led_sequence)
    second_order = second_order_iir(led_sequence)

    print(f"Raw temporal SSD:          {temporal_ssd(led_sequence):.4f}")
    print(f"Adaptive IIR temporal SSD: {temporal_ssd(adaptive):.4f}")
    print(f"Second-order temporal SSD: {temporal_ssd(second_order):.4f}")


if __name__ == "__main__":
    main()

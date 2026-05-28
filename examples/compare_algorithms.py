from __future__ import annotations

from local_backlight_dimming.algorithms import (
    average_dimming,
    cho_dimming,
    maximum_dimming,
    nam_dimming,
    proposed_histogram_dimming,
    sqrt_dimming,
)
from local_backlight_dimming.compensation import compensate_lcd
from local_backlight_dimming.display_model import DisplayModel
from local_backlight_dimming.image_utils import synthetic_hdr_like_image
from local_backlight_dimming.metrics import psnr, relative_backlight_power


def main() -> None:
    image = synthetic_hdr_like_image(180, 320)
    display = DisplayModel.from_grid(height=180, width=320, rows=6, cols=10)
    algorithms = {
        "avg": average_dimming,
        "max": maximum_dimming,
        "sqrt": sqrt_dimming,
        "cho": cho_dimming,
        "nam": nam_dimming,
        "proposed": proposed_histogram_dimming,
    }

    print("algorithm,power,psnr_db")
    for name, fn in algorithms.items():
        led = fn(display, image)
        backlight = display.simulate_backlight(led)
        _, reproduced = compensate_lcd(image, backlight, leakage=display.leakage, gamma=display.gamma)
        print(f"{name},{relative_backlight_power(led):.4f},{psnr(image, reproduced):.3f}")


if __name__ == "__main__":
    main()

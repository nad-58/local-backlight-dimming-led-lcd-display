"""Lightweight end-to-end example for continuous integration."""
import numpy as np

from local_backlight_dimming.algorithms import proposed_histogram_dimming
from local_backlight_dimming.compensation import compensate_lcd
from local_backlight_dimming.display_model import DisplayModel
from local_backlight_dimming.metrics import psnr, relative_backlight_power


def main() -> None:
    image = np.full((24, 32, 3), 0.5, dtype=float)
    display = DisplayModel.from_grid(24, 32, 3, 4)
    led = proposed_histogram_dimming(display, image)
    backlight = display.simulate_backlight(led)
    _, reproduced = compensate_lcd(image, backlight)
    print(f"segments={display.segment_count}")
    print(f"power={relative_backlight_power(led):.4f}")
    print(f"psnr={psnr(image, reproduced):.3f}")


if __name__ == "__main__":
    main()

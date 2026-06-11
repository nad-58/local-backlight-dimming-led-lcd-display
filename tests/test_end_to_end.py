import numpy as np

from local_backlight_dimming.algorithms import proposed_histogram_dimming
from local_backlight_dimming.compensation import compensate_lcd
from local_backlight_dimming.display_model import DisplayModel


def test_end_to_end_pipeline():
    image = np.full((24, 32, 3), 0.5, dtype=float)
    display = DisplayModel.from_grid(24, 32, 3, 4)
    led = proposed_histogram_dimming(display, image)
    backlight = display.simulate_backlight(led)
    compensated, reproduced = compensate_lcd(image, backlight)
    assert led.shape == (12,)
    assert backlight.shape == (24, 32)
    assert compensated.shape == image.shape
    assert reproduced.shape == image.shape
    assert np.isfinite(reproduced).all()

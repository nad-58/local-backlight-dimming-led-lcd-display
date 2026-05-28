import numpy as np

from local_backlight_dimming.algorithms import (
    average_dimming,
    cho_dimming,
    maximum_dimming,
    proposed_histogram_dimming,
    sqrt_dimming,
)
from local_backlight_dimming.display_model import DisplayModel


def test_algorithms_return_valid_led_values():
    image = np.ones((40, 60, 3)) * 0.5
    display = DisplayModel.from_grid(40, 60, 4, 6)
    for fn in [average_dimming, maximum_dimming, sqrt_dimming, cho_dimming, proposed_histogram_dimming]:
        led = fn(display, image)
        assert led.shape == (display.segment_count,)
        assert np.all(led >= 0.0)
        assert np.all(led <= 1.0)


def test_maximum_is_not_below_average_on_same_image():
    rng = np.random.default_rng(3)
    image = rng.random((40, 60, 3))
    display = DisplayModel.from_grid(40, 60, 4, 6)
    assert np.all(maximum_dimming(display, image) >= average_dimming(display, image))

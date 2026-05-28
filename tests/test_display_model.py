import numpy as np

from local_backlight_dimming.display_model import DisplayModel


def test_backlight_shape_and_range():
    display = DisplayModel.from_grid(40, 60, 4, 6)
    led = np.ones(display.segment_count)
    backlight = display.simulate_backlight(led)
    assert backlight.shape == (40, 60)
    assert np.min(backlight) >= 0.0
    assert np.max(backlight) <= 1.0


def test_iter_segments_covers_all_segments():
    display = DisplayModel.from_grid(40, 60, 4, 6)
    image = np.zeros((40, 60, 3))
    assert sum(1 for _ in display.iter_segments(image)) == display.segment_count

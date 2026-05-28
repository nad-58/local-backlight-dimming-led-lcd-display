import numpy as np

from local_backlight_dimming.metrics import mse, psnr, relative_backlight_power, temporal_ssd


def test_mse_and_psnr_identity():
    image = np.ones((8, 8)) * 0.4
    assert mse(image, image) == 0.0
    assert psnr(image, image) == float("inf")


def test_power_metric():
    led = np.array([0.0, 0.5, 1.0])
    assert relative_backlight_power(led) == 0.5


def test_temporal_ssd_zero_for_static_sequence():
    seq = np.ones((3, 4)) * 0.2
    assert temporal_ssd(seq) == 0.0

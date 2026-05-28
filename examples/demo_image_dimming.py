from __future__ import annotations

from local_backlight_dimming import DisplayModel, compensate_lcd, proposed_histogram_dimming
from local_backlight_dimming.image_utils import synthetic_hdr_like_image
from local_backlight_dimming.metrics import psnr, relative_backlight_power, ssim_global


def main() -> None:
    image = synthetic_hdr_like_image(180, 320)
    display = DisplayModel.from_grid(height=180, width=320, rows=6, cols=10)

    led = proposed_histogram_dimming(display, image)
    backlight = display.simulate_backlight(led)
    _, reproduced = compensate_lcd(image, backlight, leakage=display.leakage, gamma=display.gamma)

    print(f"LED segments: {display.segment_count}")
    print(f"Relative backlight power: {relative_backlight_power(led):.3f}")
    print(f"PSNR: {psnr(image, reproduced):.2f} dB")
    print(f"Global SSIM-like score: {ssim_global(image, reproduced):.4f}")


if __name__ == "__main__":
    main()

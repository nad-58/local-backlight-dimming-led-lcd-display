from __future__ import annotations

import argparse
from pathlib import Path

from backlight import (
    InputParams,
    alg_max,
    alg_new,
    img_init,
    lcd_init,
    read_image,
    render_result,
    save_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backlight dimming optimization Python port.")
    parser.add_argument("--image", default="images/City1.png", help="Input image path.")
    parser.add_argument("--output-dir", default="results", help="Directory for output images.")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--epsilon-modeling", default="constant", choices=["constant", "horVariation", "horVerVariation"])
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--downscaling-factor", type=int, default=10)
    parser.add_argument("--backlight-rows", type=int, default=13)
    parser.add_argument("--backlight-columns", type=int, default=17)
    parser.add_argument("--psf-reflections", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--target-epsilon", type=float, default=0.0)
    parser.add_argument("--norm", type=int, default=1, choices=[1, 2])
    parser.add_argument("--hf-enhance", type=float, default=0.0)
    parser.add_argument("--power-weight", type=float, default=0.001)
    parser.add_argument("--skip-new", action="store_true", help="Run only the Max algorithm.")
    parser.add_argument("--solver", default=None, help="Optional CVXPY solver name, e.g. CLARABEL, SCS, ECOS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = InputParams(
        image_filenames=Path(args.image).name,
        epsilons=args.epsilon,
        epsilon_modeling=args.epsilon_modeling,
        gamma=args.gamma,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        downscaling_factor=args.downscaling_factor,
        backlight_rows=args.backlight_rows,
        backlight_columns=args.backlight_columns,
        psf_reflections=args.psf_reflections,
        target_epsilon=args.target_epsilon,
        norm=args.norm,
        hf_enhance=args.hf_enhance,
        power_weight=args.power_weight,
    )

    high_display, low_display = lcd_init(params)
    input_img = read_image(args.image)
    high_image, low_image = img_init(input_img, params, high_display, low_display)

    max_leds = alg_max(high_display, high_image.target)
    max_result = render_result(params, high_display, high_image, max_leds)
    save_result(max_result, args.output_dir, "max")
    print_metrics("Max", max_result)

    if not args.skip_new:
        new_leds = alg_new(low_image.target, low_display, params, solver=args.solver)
        new_result = render_result(params, high_display, high_image, new_leds)
        save_result(new_result, args.output_dir, "new")
        print_metrics("New", new_result)


def print_metrics(name, result) -> None:
    print(
        f"{name}: mae={result.mae:.6g}, mse={result.mse_total:.6g}, "
        f"psnr={result.psnr:.3f}, ssim={result.mssim:.6f}, power={result.power:.6f}, "
        f"leaking={result.leaking_pixels}, clipped={result.clipped_pixels}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
from PIL import Image
from scipy import ndimage, signal, sparse
from skimage.metrics import structural_similarity
from skimage.transform import resize


@dataclass
class InputParams:
    image_filenames: str = "City1.png"
    epsilons: float = 0.001
    epsilon_modeling: str = "constant"
    gamma: float = 2.2
    screen_peak_white: float = 1000.0
    screen_width: int = 1920
    screen_height: int = 1080
    downscaling_factor: int = 10
    use_sim2_backlight: bool = False
    backlight_rows: int = 13
    backlight_columns: int = 17
    psf_reflections: int = 1
    target_peak_white: float = 1000.0
    use_color_input: bool = False
    target_epsilon: float = 0.0
    norm: int = 1
    hf_enhance: float = 0.0
    use_gamma_perceptual_optimization: bool = True
    power_weight: float = 0.001


@dataclass
class Display:
    downscaling_factor: int
    screen_height: int
    screen_width: int
    backlight_rows: int
    backlight_columns: int
    backlight_segments_number: int
    vertical_led_distance: int
    horizontal_led_distance: int
    y_led_positions: np.ndarray
    x_led_positions: np.ndarray
    epsilon: np.ndarray
    peak_white: float
    gamma: float
    point_spreading_function: np.ndarray
    full_backlight: np.ndarray
    backlight_scale: float = 0.0


@dataclass
class ImageData:
    input: np.ndarray
    gray: np.ndarray
    input_max_value: float
    input_transmittances: np.ndarray
    input_components: int
    luminance: np.ndarray
    target: np.ndarray


@dataclass
class Result:
    led_values: np.ndarray
    backlight: np.ndarray
    compensated: np.ndarray
    physical: np.ndarray
    perceived: np.ndarray
    leakage: np.ndarray
    clipper: np.ndarray
    leaking_pixels: int
    clipped_pixels: int
    max_leakage: float
    max_clipper: float
    avg_leakage: float
    avg_clipper: float
    mae: float
    mse_total: float
    mse_leakage: float
    mse_clipper: float
    mse_leakage_pc: float
    mse_clipper_pc: float
    psnr: float
    mssim: float
    contrast: float
    power: float


def gaussian_kernel(shape: tuple[int, int], sigma: float) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[:h, :w]
    y = y - (h - 1) / 2
    x = x - (w - 1) / 2
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return k / np.sum(k)


def calculate_epsilon(
    display: Display | object,
    epsilon: float,
    reference_height: int,
    center_x: float,
    center_y: float,
    modeling: str,
) -> np.ndarray:
    if modeling == "constant":
        return np.full((display.screen_height, display.screen_width), epsilon, dtype=float)

    x = np.arange(display.screen_width, dtype=float)[None, :]
    y = np.arange(display.screen_height, dtype=float)[:, None]
    if modeling == "horVariation":
        return epsilon * (1 + np.abs(x - center_x) / reference_height)
    if modeling == "horVerVariation":
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return epsilon * (1 + dist / reference_height)
    raise ValueError(f"Unsupported epsilon modeling: {modeling}")


def simulate_backlight(display: Display, led_values: np.ndarray) -> np.ndarray:
    canvas = np.zeros((display.screen_height, display.screen_width), dtype=float)
    y = np.clip(display.y_led_positions.astype(int), 0, display.screen_height - 1)
    x = np.clip(display.x_led_positions.astype(int), 0, display.screen_width - 1)
    canvas[y, x] = np.asarray(led_values, dtype=float).reshape(-1)
    return signal.convolve2d(canvas, display.point_spreading_function, mode="same")


def simulate_backlight_reflections(
    params: InputParams, display: Display, led_values: np.ndarray
) -> tuple[np.ndarray, float]:
    offset_y, offset_x = np.floor(np.array(display.point_spreading_function.shape) / 2).astype(int)
    if params.use_sim2_backlight:
        gap_x_left = int(np.floor(offset_x * 0.9768))
        gap_x_right = int(np.floor(offset_x * 0.9826))
        gap_y_up = int(np.floor(offset_y * 0.9970))
        gap_y_down = int(np.floor(offset_y * 1.0030))
    else:
        gap_x_left = gap_x_right = int(np.floor(offset_x * 0.75))
        gap_y_up = gap_y_down = int(np.floor(offset_y * 0.75))

    attenuation = 0.98
    canvas = np.zeros((display.screen_height, display.screen_width), dtype=float)
    y = np.clip(display.y_led_positions.astype(int), 0, display.screen_height - 1)
    x = np.clip(display.x_led_positions.astype(int), 0, display.screen_width - 1)
    canvas[y, x] = np.asarray(led_values, dtype=float).reshape(-1)
    full = signal.convolve2d(canvas, display.point_spreading_function, mode="full")

    if gap_x_left > 0:
        full[:, gap_x_left : 2 * gap_x_left] += full[:, gap_x_left - 1 :: -1][:, :gap_x_left] * attenuation
    if gap_x_right > 0:
        full[:, -2 * gap_x_right : -gap_x_right] += full[:, :-gap_x_right - 1 : -1][:, :gap_x_right] * attenuation
    if gap_y_up > 0:
        full[gap_y_up : 2 * gap_y_up, :] += full[gap_y_up - 1 :: -1, :][:gap_y_up, :] * attenuation
    if gap_y_down > 0:
        full[-2 * gap_y_down : -gap_y_down, :] += full[:-gap_y_down - 1 : -1, :][:gap_y_down, :] * attenuation

    out = full[offset_y : offset_y + display.screen_height, offset_x : offset_x + display.screen_width]
    scale = display.backlight_scale or 1.0 / max(np.min(out), np.finfo(float).eps)
    return np.minimum(out * scale, 1.0), scale


def lcd_init(params: InputParams) -> tuple[Display, Display]:
    high = _lcd_init_regular(params, downscaling_factor=1, source_display=None)
    low = _lcd_init_regular(params, downscaling_factor=params.downscaling_factor, source_display=high)
    return high, low


def _lcd_init_regular(
    params: InputParams, downscaling_factor: int, source_display: Optional[Display]
) -> Display:
    height = int(np.ceil(params.screen_height / downscaling_factor))
    width = int(np.ceil(params.screen_width / downscaling_factor))
    rows = params.backlight_rows
    cols = params.backlight_columns
    segments = rows * cols

    if source_display is None:
        y_grid, x_grid = np.meshgrid(
            np.round(np.linspace(0, height - 1, rows)).astype(int),
            np.round(np.linspace(0, width - 1, cols)).astype(int),
            indexing="ij",
        )
        vdist = int(np.floor((height - 1) / (rows - 1)))
        hdist = int(np.floor((width - 1) / (cols - 1)))
    else:
        y_grid = np.ceil(source_display.y_led_positions / downscaling_factor).astype(int)
        x_grid = np.ceil(source_display.x_led_positions / downscaling_factor).astype(int)
        vdist = int(np.floor((params.screen_height - 1) / (rows - 1) / downscaling_factor))
        hdist = int(np.floor((params.screen_width - 1) / (cols - 1) / downscaling_factor))

    shell = type("DisplayShell", (), {"screen_height": height, "screen_width": width})()
    epsilon = calculate_epsilon(shell, params.epsilons, 1080, width / 2, height / 2, params.epsilon_modeling)

    if source_display is None:
        psf_shape = (max(1, vdist * 4), max(1, hdist * 4))
        psf = gaussian_kernel(psf_shape, max(vdist, hdist) / 2.5)
        psf = ndimage.convolve(psf, np.ones((max(1, vdist), max(1, hdist))), mode="constant")
    else:
        psf = resize(
            source_display.point_spreading_function,
            (
                max(1, int(round(source_display.point_spreading_function.shape[0] / downscaling_factor))),
                max(1, int(round(source_display.point_spreading_function.shape[1] / downscaling_factor))),
            ),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )

    display = Display(
        downscaling_factor=downscaling_factor,
        screen_height=height,
        screen_width=width,
        backlight_rows=rows,
        backlight_columns=cols,
        backlight_segments_number=segments,
        vertical_led_distance=max(vdist, 1),
        horizontal_led_distance=max(hdist, 1),
        y_led_positions=y_grid.reshape(-1),
        x_led_positions=x_grid.reshape(-1),
        epsilon=epsilon,
        peak_white=params.screen_peak_white,
        gamma=params.gamma,
        point_spreading_function=psf,
        full_backlight=np.ones((height, width), dtype=float),
    )

    full = simulate_backlight(display, np.ones(segments))
    scale = max(np.max(full), np.finfo(float).eps)
    display.point_spreading_function = display.point_spreading_function / scale
    display.full_backlight = full / scale
    _, display.backlight_scale = simulate_backlight_reflections(params, display, np.ones(segments))
    return display


def read_image(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _fit_to_screen(img: np.ndarray, height: int, width: int) -> np.ndarray:
    y, x = img.shape[:2]
    if y / x >= height / width:
        new_h = height
        new_w = int(round(height * x / y))
    else:
        new_h = int(round(width * y / x))
        new_w = width
    resized = resize(img, (new_h, new_w), order=1, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    out = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


def rgb2gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def srgb2xyz(srgb: np.ndarray) -> np.ndarray:
    s = srgb.astype(float) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    if lin.ndim == 2:
        return np.dstack((0.9505 * lin, lin, 1.0889 * lin))
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    return np.einsum("...c,kc->...k", lin, matrix)


def img_init(
    input_img: np.ndarray, params: InputParams, high_display: Display, low_display: Display
) -> tuple[ImageData, ImageData]:
    high_input = _fit_to_screen(input_img, high_display.screen_height, high_display.screen_width)
    low_input = _fit_to_screen(input_img, low_display.screen_height, low_display.screen_width)
    return (
        _make_image_data(high_input, params, high_display),
        _make_image_data(low_input, params, low_display),
    )


def _make_image_data(input_img: np.ndarray, params: InputParams, display: Display) -> ImageData:
    gray = rgb2gray_uint8(input_img)
    selected = input_img if params.use_color_input else gray
    selected_float = selected.astype(float) / 255.0
    components = 1 if selected.ndim == 2 else selected.shape[2]
    xyz = srgb2xyz(selected)
    luminance = xyz[..., 1]
    return ImageData(
        input=selected,
        gray=gray,
        input_max_value=255.0,
        input_transmittances=selected_float ** (1 / display.gamma),
        input_components=components,
        luminance=luminance,
        target=luminance,
    )


def _segment_bounds(display: Display, i: int, factor: float = 0.7) -> tuple[slice, slice]:
    d_y = int(np.floor(display.vertical_led_distance * factor))
    d_x = int(np.floor(display.horizontal_led_distance * factor))
    y = int(display.y_led_positions[i])
    x = int(display.x_led_positions[i])
    return (
        slice(max(y - d_y, 0), min(y + d_y + 1, display.screen_height)),
        slice(max(x - d_x, 0), min(x + d_x + 1, display.screen_width)),
    )


def alg_conv(display: Display) -> np.ndarray:
    return np.ones(display.backlight_segments_number, dtype=float)


def alg_avg(display: Display, target: np.ndarray) -> np.ndarray:
    r = np.zeros(display.backlight_segments_number, dtype=float)
    for i in range(display.backlight_segments_number):
        ys, xs = _segment_bounds(display, i)
        r[i] = np.mean(target[ys, xs])
    return np.clip(r, 0, 1)


def alg_avg_plus(display: Display, target: np.ndarray) -> np.ndarray:
    r = alg_avg(display, target)
    backlight = np.maximum(simulate_backlight(display, r), np.finfo(float).eps)
    scale = np.quantile((target / backlight).reshape(-1), 0.99)
    return np.clip(r * scale, 0, 1)


def alg_max(display: Display, target: np.ndarray) -> np.ndarray:
    r = np.zeros(display.backlight_segments_number, dtype=float)
    for i in range(display.backlight_segments_number):
        ys, xs = _segment_bounds(display, i)
        r[i] = np.max(target[ys, xs])
    return np.clip(r, 0, 1)


def alg_cho(display: Display, target: np.ndarray) -> np.ndarray:
    img = 255 * target
    r = np.zeros(display.backlight_segments_number, dtype=float)
    for i in range(display.backlight_segments_number):
        ys, xs = _segment_bounds(display, i)
        p = img[ys, xs]
        diff = np.max(p) - np.mean(p)
        r[i] = np.mean(p) + 0.5 * (diff + diff * diff / 255.0)
    return np.clip(r / 255.0, 0, 1)


def calculate_influence_matrix(params: InputParams, display: Display) -> sparse.csr_matrix:
    rows = display.screen_height * display.screen_width
    cols = display.backlight_segments_number
    matrix = sparse.lil_matrix((rows, cols), dtype=float)
    led_values = np.zeros(cols, dtype=float)
    for i in range(cols):
        led_values[i] = 1.0
        if params.psf_reflections == 2:
            c, _ = simulate_backlight_reflections(params, display, led_values)
        else:
            c = simulate_backlight(display, led_values)
        matrix[:, i] = c.reshape(-1, 1)
        led_values[i] = 0.0
    return matrix.tocsr()


def calculate_weights(target: np.ndarray, gamma: float) -> np.ndarray:
    slopes = gamma * np.maximum(target, np.finfo(float).eps) ** (gamma - 1)
    return np.maximum(slopes, 0.18)


def alg_new(target: np.ndarray, display: Display, params: InputParams, solver: Optional[str] = None) -> np.ndarray:
    z = np.maximum(0, ndimage.gaussian_laplace(target, sigma=1.0)).reshape(-1)
    y = (target * (1 - params.target_epsilon) + params.target_epsilon).reshape(-1)
    h = calculate_influence_matrix(params, display)

    if params.psf_reflections == 1:
        w = y * display.full_backlight.reshape(-1)
    else:
        w = y

    weight = calculate_weights(y, params.gamma) if params.use_gamma_perceptual_optimization else np.ones_like(w)
    e = display.epsilon.reshape(-1)
    n = y.size

    r = cp.Variable(display.backlight_segments_number)
    l = cp.Variable(n)
    objective_error = cp.sum(cp.multiply(1.0 / weight, l)) if params.norm == 1 else cp.norm(cp.multiply(1.0 / weight, l), 2)
    objective = cp.Minimize(objective_error + params.power_weight * cp.sum(r) / display.backlight_segments_number)
    constraints = [
        l >= cp.multiply(e, h @ r) - w,
        l >= cp.multiply(1 + params.hf_enhance * z, w - h @ r),
        l >= 0,
        r >= 0,
        r <= 1,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)
    if r.value is None:
        raise RuntimeError(f"CVXPY failed to solve alg_new; status={problem.status}")
    return np.clip(np.asarray(r.value).reshape(-1), 0, 1)


def brightness_compensation(backlight: np.ndarray, image: ImageData, display: Display) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_backlight = np.maximum(backlight, np.finfo(float).eps)
    factor = 1.0 / safe_backlight
    if image.input_components > 1:
        factor = factor[..., None]
        epsilon = display.epsilon[..., None]
    else:
        epsilon = display.epsilon
    tr_g = (image.input.astype(float) / 255.0) ** display.gamma
    tr_lin = (factor * tr_g - epsilon) / (1 - epsilon)
    tr_lin = np.maximum(tr_lin, 0)
    compensated_float = 255.0 * tr_lin ** (1 / display.gamma)
    clipping_mask = (compensated_float > 255).astype(float)
    compensated = np.clip(compensated_float, 0, 255).astype(np.uint8)
    compensated_transmittance = (compensated.astype(float) / 255.0) ** display.gamma
    return compensated, compensated_transmittance, clipping_mask


def analyze_clipper(clipping_mask: np.ndarray, physical: np.ndarray, display: Display) -> np.ndarray:
    clipping = clipping_mask * (1 - physical)
    threshold = (1 / 255.0) ** display.gamma
    clipping[(clipping > 0) & (clipping < threshold)] = 0
    return clipping


def analyze_leakage(compensated: np.ndarray, physical: np.ndarray, display: Display) -> tuple[np.ndarray, np.ndarray]:
    mask = (compensated == 0).astype(float)
    leakage = mask * physical
    threshold = (1 / 255.0) ** display.gamma
    leakage[(leakage > 0) & (leakage < threshold)] = 0
    return leakage, mask


def render_result(params: InputParams, display: Display, image: ImageData, led_values: np.ndarray) -> Result:
    led_values = np.round(255 * np.asarray(led_values, dtype=float)) / 255.0
    if params.psf_reflections == 0:
        backlight = simulate_backlight(display, led_values)
    elif params.psf_reflections == 1:
        backlight = simulate_backlight(display, led_values) / np.maximum(display.full_backlight, np.finfo(float).eps)
    elif params.psf_reflections == 2:
        backlight, _ = simulate_backlight_reflections(params, display, led_values)
    else:
        raise ValueError(f"Unsupported PSF reflection mode: {params.psf_reflections}")

    compensated, compensated_transmittance, clipping_mask = brightness_compensation(backlight, image, display)
    bl = backlight[..., None] if image.input_components > 1 else backlight
    epsilon = display.epsilon[..., None] if image.input_components > 1 else display.epsilon
    physical = bl * (epsilon + (1 - epsilon) * compensated_transmittance)
    perceived = physical ** (1 / display.gamma)
    leakage, leakage_mask = analyze_leakage(compensated, physical, display)
    clipper = analyze_clipper(clipping_mask, physical, display)

    error = image.input.astype(float).reshape(-1) / 255.0 - perceived.reshape(-1)
    clipping_error = error * clipping_mask.reshape(-1)
    leakage_error = error * leakage_mask.reshape(-1)
    mse_total = float(np.mean(error**2))
    mse_clipping = float(np.sum(clipping_error**2) / error.size)
    mse_leakage = float(np.sum(leakage_error**2) / error.size)
    psnr = float("inf") if mse_total == 0 else float(10 * np.log10(1 / mse_total))

    input_for_ssim = image.input
    perceived_uint8 = np.clip(np.round(perceived * 255), 0, 255).astype(np.uint8)
    channel_axis = -1 if input_for_ssim.ndim == 3 else None
    mssim = float(structural_similarity(input_for_ssim, perceived_uint8, data_range=255, channel_axis=channel_axis))

    leakage_positive = leakage[leakage > 0]
    clipper_positive = clipper[clipper > 0]
    return Result(
        led_values=led_values,
        backlight=backlight,
        compensated=compensated,
        physical=physical,
        perceived=perceived,
        leakage=leakage,
        clipper=clipper,
        leaking_pixels=int(leakage_positive.size),
        clipped_pixels=int(clipper_positive.size),
        max_leakage=float(np.max(leakage)),
        max_clipper=float(np.max(clipper)),
        avg_leakage=float(np.mean(leakage_positive)) if leakage_positive.size else 0.0,
        avg_clipper=float(np.mean(clipper_positive)) if clipper_positive.size else 0.0,
        mae=float(np.mean(np.abs(error))),
        mse_total=mse_total,
        mse_leakage=mse_leakage,
        mse_clipper=mse_clipping,
        mse_leakage_pc=float(mse_leakage / mse_total) if mse_total else 0.0,
        mse_clipper_pc=float(mse_clipping / mse_total) if mse_total else 0.0,
        psnr=psnr,
        mssim=mssim,
        contrast=float(np.max(physical) - np.min(physical)),
        power=float(np.mean(led_values)),
    )


def save_result(result: Result, directory: str | Path, prefix: str) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    _save_float_image(result.backlight, out / f"{prefix}_backlight.png")
    Image.fromarray(result.compensated).save(out / f"{prefix}_compensated.png")
    _save_float_image(result.perceived, out / f"{prefix}_perceived.png")
    _save_float_image(result.leakage, out / f"{prefix}_leakage.png")
    _save_float_image(result.clipper, out / f"{prefix}_clipper.png")


def _save_float_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(np.round(arr * 255), 0, 255).astype(np.uint8)).save(path)

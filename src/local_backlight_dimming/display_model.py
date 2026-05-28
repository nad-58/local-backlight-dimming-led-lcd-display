from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


Array = np.ndarray


def _gaussian_kernel(size: int, sigma: float) -> Array:
    """Create a normalized 2-D Gaussian point-spread function."""
    if size % 2 == 0:
        size += 1
    radius = size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel /= np.max(kernel)
    return kernel.astype(float)


def _convolve2d_same(image: Array, kernel: Array) -> Array:
    """Small dependency-free 2-D convolution with same-size output."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(image, dtype=float)
    flipped = kernel[::-1, ::-1]
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(region * flipped)
    return out


@dataclass(frozen=True)
class DisplayModel:
    """Simple LED-LCD display model.

    The model follows the same high-level idea as the MATLAB code: LED values are
    placed on a grid and convolved with a point-spread function to approximate the
    light diffusion across the LCD panel.
    """

    height: int
    width: int
    rows: int
    cols: int
    gamma: float = 2.2
    leakage: float = 0.001
    psf: Array | None = None

    @classmethod
    def from_grid(
        cls,
        height: int,
        width: int,
        rows: int = 8,
        cols: int = 16,
        gamma: float = 2.2,
        leakage: float = 0.001,
        psf_size: int | None = None,
        psf_sigma: float | None = None,
    ) -> "DisplayModel":
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        if height < rows or width < cols:
            raise ValueError("display resolution must be at least as large as the LED grid")

        if psf_size is None:
            psf_size = max(9, int(round(min(height / rows, width / cols) * 2.5)))
        if psf_sigma is None:
            psf_sigma = max(1.0, psf_size / 5.0)
        return cls(
            height=height,
            width=width,
            rows=rows,
            cols=cols,
            gamma=gamma,
            leakage=leakage,
            psf=_gaussian_kernel(psf_size, psf_sigma),
        )

    @property
    def segment_count(self) -> int:
        return self.rows * self.cols

    @property
    def led_positions(self) -> Array:
        ys = np.linspace(0, self.height - 1, self.rows).round().astype(int)
        xs = np.linspace(0, self.width - 1, self.cols).round().astype(int)
        return np.array([(y, x) for y in ys for x in xs], dtype=int)

    @property
    def vertical_led_distance(self) -> float:
        return self.height / max(self.rows, 1)

    @property
    def horizontal_led_distance(self) -> float:
        return self.width / max(self.cols, 1)

    def segment_slices(self, row: int, col: int, overlap: float = 0.7) -> Tuple[slice, slice]:
        """Return image slices for the local area associated with one LED segment."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("segment index out of range")
        idx = row * self.cols + col
        cy, cx = self.led_positions[idx]
        half_h = max(1, int(round(self.vertical_led_distance * overlap)))
        half_w = max(1, int(round(self.horizontal_led_distance * overlap)))
        y0 = max(0, cy - half_h)
        y1 = min(self.height, cy + half_h + 1)
        x0 = max(0, cx - half_w)
        x1 = min(self.width, cx + half_w + 1)
        return slice(y0, y1), slice(x0, x1)

    def iter_segments(self, image: Array, overlap: float = 0.7) -> Iterator[Tuple[int, int, Array]]:
        """Yield row, column and corresponding image patch for each LED zone."""
        if image.shape[0] != self.height or image.shape[1] != self.width:
            raise ValueError("image size does not match display model")
        for r in range(self.rows):
            for c in range(self.cols):
                ys, xs = self.segment_slices(r, c, overlap=overlap)
                yield r, c, image[ys, xs]

    def simulate_backlight(self, led_values: Array, normalize: bool = True) -> Array:
        """Simulate spatial backlight diffusion from LED segment values.

        Parameters
        ----------
        led_values:
            A vector of length rows*cols or an array shaped (rows, cols), with values
            in [0, 1].
        normalize:
            When true, normalize so that all LEDs at full scale produce peak one.
        """
        led = np.asarray(led_values, dtype=float).reshape(self.rows, self.cols)
        led = np.clip(led, 0.0, 1.0)

        impulses = np.zeros((self.height, self.width), dtype=float)
        for value, (y, x) in zip(led.ravel(), self.led_positions):
            impulses[y, x] = value

        kernel = self.psf if self.psf is not None else _gaussian_kernel(31, 6.0)
        backlight = _convolve2d_same(impulses, kernel)

        if normalize:
            full = _convolve2d_same((impulses * 0.0) + self._full_impulse_grid(), kernel)
            scale = np.max(full)
            if scale > 0:
                backlight = backlight / scale
        return np.clip(backlight, 0.0, 1.0)

    def _full_impulse_grid(self) -> Array:
        impulses = np.zeros((self.height, self.width), dtype=float)
        for y, x in self.led_positions:
            impulses[y, x] = 1.0
        return impulses

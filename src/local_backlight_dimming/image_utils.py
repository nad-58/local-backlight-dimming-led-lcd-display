from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Array = np.ndarray


def load_image(path: str | Path, size: tuple[int, int] | None = None) -> Array:
    """Load an RGB image as float array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=float) / 255.0


def save_image(path: str | Path, image: Array) -> None:
    """Save a float image in [0, 1]."""
    arr = np.clip(np.asarray(image), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def synthetic_hdr_like_image(height: int = 360, width: int = 640) -> Array:
    """Create a synthetic image with dark, mid-tone and bright regions."""
    y = np.linspace(0.0, 1.0, height)[:, None]
    x = np.linspace(0.0, 1.0, width)[None, :]
    base = 0.08 + 0.35 * x + 0.25 * y
    image = np.dstack([base * 0.8, base, base * 1.2])

    yy, xx = np.mgrid[0:height, 0:width]
    spot = ((yy - height * 0.35) ** 2 + (xx - width * 0.72) ** 2) < (min(height, width) * 0.12) ** 2
    image[spot] = [1.0, 0.92, 0.65]

    dark = ((yy - height * 0.65) ** 2 + (xx - width * 0.25) ** 2) < (min(height, width) * 0.18) ** 2
    image[dark] *= 0.18
    return np.clip(image, 0.0, 1.0)

from __future__ import annotations

import numpy as np

Array = np.ndarray


def srgb_to_linear(srgb: Array) -> Array:
    srgb = np.asarray(srgb, dtype=float)
    if srgb.max(initial=0) > 1.0:
        srgb = srgb / 255.0
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_rgb_to_xyz(linear_rgb: Array) -> Array:
    """Converted from the MATLAB `lin2xyz` helper."""
    lin = np.asarray(linear_rgb, dtype=float)
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
        dtype=float,
    )
    if lin.ndim == 2:
        xyz = np.zeros(lin.shape + (3,), dtype=float)
        xyz[..., 0] = np.sum(matrix[0]) * lin
        xyz[..., 1] = np.sum(matrix[1]) * lin
        xyz[..., 2] = np.sum(matrix[2]) * lin
        return xyz
    return np.tensordot(lin, matrix.T, axes=1)


def srgb_to_xyz(srgb: Array) -> Array:
    """Converted from `srgb2xyz.m`."""
    return linear_rgb_to_xyz(srgb_to_linear(srgb))


def _lab_function(t: Array) -> Array:
    delta = 6.0 / 29.0
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)


def xyz_to_lab(xyz: Array, xyz_white: Array | None = None) -> Array:
    if xyz_white is None:
        xyz_white = srgb_to_xyz(np.ones((1, 1, 3), dtype=float))[0, 0]
    xyz = np.asarray(xyz, dtype=float)
    white = np.asarray(xyz_white, dtype=float)
    ratio = xyz / np.maximum(white, 1e-12)
    lab = np.zeros_like(xyz, dtype=float)
    lab[..., 0] = 116.0 * _lab_function(ratio[..., 1]) - 16.0
    lab[..., 1] = 500.0 * (_lab_function(ratio[..., 0]) - _lab_function(ratio[..., 1]))
    lab[..., 2] = 200.0 * (_lab_function(ratio[..., 1]) - _lab_function(ratio[..., 2]))
    return lab


def compute_color_distortion(srgb_reference: Array, physical_test_linear: Array) -> Array:
    """Converted from `computeColorDistortion.m`.

    Returns per-pixel Delta-E style Euclidean distance in Lab space.
    """
    xyz_ref = srgb_to_xyz(srgb_reference)
    xyz_test = linear_rgb_to_xyz(physical_test_linear)
    white = srgb_to_xyz(np.ones((1, 1, 3), dtype=float))[0, 0]
    lab_ref = xyz_to_lab(xyz_ref, white)
    lab_test = xyz_to_lab(xyz_test, white)
    return np.sqrt(np.sum((lab_ref - lab_test) ** 2, axis=-1))

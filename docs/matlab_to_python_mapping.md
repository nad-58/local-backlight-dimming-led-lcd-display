# MATLAB to Python mapping

The uploaded MATLAB archive was reviewed and cleaned before conversion. Legacy source-control and editor artefacts were intentionally excluded:

- `.svn/` folders,
- `.asv` autosave files,
- merge-conflict files such as `.mine`, `.r102`, `.r113`.

## Converted files

| MATLAB file | Python location | Notes |
|---|---|---|
| `algAvg.m` | `algorithms.average_dimming` | Mean local intensity per LED segment. |
| `algMax.m` | `algorithms.maximum_dimming` | Maximum local intensity per LED segment. |
| `algCho.m` | `algorithms.cho_dimming` | Average plus correction using local max-average difference. |
| `algNam.m` | `algorithms.nam_dimming` | Global/local max-average decision logic. |
| `algEhsan.m` | `algorithms.matlab_ehsan_variant` | Cleaned recoverable logic from conflicted MATLAB file. |
| Thesis Section 3.1.4 | `algorithms.proposed_histogram_dimming` | Main thesis-inspired local histogram / feature algorithm. |
| `simulateBacklight.m` | `display_model.DisplayModel.simulate_backlight` | PSF-style spatial backlight diffusion. |
| `renderResult.m` | `compensation.compensate_lcd` | LCD transmittance and reproduced image model. |
| `ssim.m` | `metrics.ssim_global` | Lightweight global SSIM-like metric. |
| `calculateEpsilon.m` | `DisplayModel.leakage` and compensation arguments | Simplified constant leakage model. |

## Intentional changes

The Python version prioritises readability and reproducibility over line-by-line MATLAB equivalence. In particular:

- all image values are normalised to `[0, 1]`,
- LED outputs are returned as vectors in `[0, 1]`,
- display geometry is represented by a `DisplayModel` dataclass,
- dependencies are kept lightweight,
- proprietary or calibration-specific display files are not required.

## Known limitation

The original MATLAB project includes screen-specific calibration concepts such as measured PSFs and display leakage variation. This repository provides a general research model. For exact replication of a real display, measured PSF and leakage maps should be loaded into `DisplayModel` and `compensate_lcd`.

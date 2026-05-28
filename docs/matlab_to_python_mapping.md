# MATLAB to Python mapping

The uploaded MATLAB source files were reviewed and converted into Python modules. The original MATLAB files are intentionally **not** committed to this repository; only the cleaned Python implementation is included.

Legacy source-control and editor artefacts were excluded:

- `.svn/` folders,
- `.asv` autosave files,
- merge-conflict files such as `.mine`, `.r102`, `.r113`.

## Converted algorithm files

| MATLAB file | Python location | Notes |
|---|---|---|
| `algConv.m` | `algorithms.conventional_dimming` | Full-backlight conventional baseline. |
| `algAvg.m` | `algorithms.average_dimming` | Mean local intensity per LED segment. |
| `algAvgPlus.m` | `algorithms.average_plus_dimming` | Average dimming followed by percentile-based scale correction. |
| `algMax.m` | `algorithms.maximum_dimming` | Maximum local intensity per LED segment. |
| `algCho.m` | `algorithms.cho_dimming` | Average plus correction using local max-average difference. |
| `algNam.m` | `algorithms.nam_dimming` | Global/local max-average decision logic. |
| `algChen.m` | `algorithms.chen_dimming` | Histogram-weighted LED estimate, enhancement and temporal filtering. |
| `algEhsan.m` | `algorithms.matlab_ehsan_variant` | Cleaned recoverable logic from the available revisions. |
| Thesis Section 3.1.4 | `algorithms.proposed_histogram_dimming` | Main thesis-inspired local histogram / image-feature algorithm. |
| `algComCho.m` | `algorithms.local_compensation_cho` | Cho-style pixel compensation. |
| `algComNam.m` | `algorithms.local_compensation_nam` | Nam-style pixel compensation. |
| `algClipperFreeXiao.m` | `optimization.clipper_free_dimming` | CVX formulation converted to a projected numerical approximation. |
| `algNew.m` | `optimization.new_dimming` | CVX optimisation converted to projected-gradient optimisation. |
| `algFast.m` | `optimization.fast_refinement` | Iterative masked re-optimisation converted to Python. |

## Converted support files

| MATLAB file | Python location | Notes |
|---|---|---|
| `simulateBacklight.m` | `display_model.DisplayModel.simulate_backlight` | PSF-style spatial backlight diffusion. |
| `simulateBacklightReflections.m` | `display_model` conceptually | Reflection modelling summarised; current model uses edge padding and normalisation. |
| `renderResult.m` | `result.render_result` | LED quantisation, backlight simulation, compensation and quality analysis. |
| `calculateEpsilon.m` | `epsilon.calculate_epsilon` | Constant, horizontal and horizontal-vertical leakage variation. |
| `computeColorDistortion.m` | `color.compute_color_distortion` | sRGB/XYZ/Lab conversion and colour-distance calculation. |
| `srgb2xyz.m` | `color.srgb_to_xyz` | sRGB linearisation and CIE XYZ conversion. |
| `compareSolutions.m` | `result.compare_solutions` | Metric-pair comparison helper. |
| `imgInit.m` | `image_utils.load_image` and `image_utils.synthetic_hdr_like_image` | Image loading, normalisation and resizing support. |
| `lcdInit.m` / `lcdInitHex.m` | `display_model.DisplayModel.from_grid` | Square-grid display model; SIM2-specific hex calibration is not included. |
| `saveSolution.m` | examples / user scripts | Saving can be done with `image_utils.save_image`. |
| `solutions2excel.m` | pandas/export outside core package | Not included as a core dependency. |
| `generateSIM2frame.m` | not included | Hardware-frame packing is SIM2-specific and outside the general display model. |

## Important implementation choices

The Python version prioritises readability, reproducibility and public portfolio use over line-by-line MATLAB equivalence. In particular:

- all image values are normalised to `[0, 1]`,
- LED outputs are returned as vectors in `[0, 1]`,
- display geometry is represented by a `DisplayModel` dataclass,
- the original CVX optimisation is converted to a lightweight projected-gradient approximation,
- MATLAB source files are not committed,
- vendor-specific or hardware-specific calibration files are not required.

## Known limitation

The original MATLAB project includes screen-specific calibration concepts such as measured PSFs, SIM2 hexagonal LED layout and leakage variation. This repository provides a general research model. For exact hardware replication, measured PSF and leakage maps should be loaded into `DisplayModel` and `compensate_lcd`.

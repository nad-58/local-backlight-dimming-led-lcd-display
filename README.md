# Local Backlight Dimming of LED-LCD Display

Python implementation of local backlight dimming algorithms for LED-LCD displays, based on PhD research on image-quality preservation, power reduction, brightness compensation, and flicker reduction.

This repository converts and modernises MATLAB source code associated with Chapter 3 of the thesis:

> E. Nadernejad, *Processing Decoded Video for LCD-LED Backlight Display: Post processing of decoded video and local backlight dimming for LCD technology with LED-based backlight*, Technical University of Denmark, 2013.

Official public thesis PDF: https://orbit.dtu.dk/files/87595827/thesisMain.pdf

## Scope

The project focuses on local dimming for LED-backlit LCD displays. It includes:

- local backlight zone modelling,
- average, maximum, square-root, Cho-style and Nam-style dimming baselines,
- Chen-style and Ehsan/proposed histogram-feature dimming algorithms,
- CVX-free Python approximations of optimisation-based dimming algorithms,
- backlight diffusion simulation using a point-spread function approximation,
- LCD brightness compensation with leakage modelling,
- colour-space and colour-distortion utilities,
- quality and power metrics,
- temporal flicker reduction using adaptive IIR filtering,
- examples and tests for reproducible experimentation.

## Background

LCD panels use liquid crystals as light filters and require a backlight. With LED backlights, the backlight can be split into independently controlled zones. Local dimming reduces power consumption and can improve contrast, but it may introduce clipping, haloing, leakage errors, or temporal flicker if not controlled carefully.

The thesis method uses local image statistics to determine LED values. Each local segment is analysed using features such as average luminance, maximum value, local histogram percentile, square-root average and coefficient of variation. This produces a practical trade-off between image fidelity and energy saving.

## Repository structure

```text
src/local_backlight_dimming/
  algorithms.py             # baseline, Chen, Nam, Cho and proposed dimming algorithms
  optimization.py           # optimisation-based dimming approximations
  compensation.py           # LCD transmittance and brightness compensation
  display_model.py          # LED grid, PSF and backlight simulation
  epsilon.py                # leakage / epsilon modelling
  color.py                  # sRGB, XYZ, Lab and colour distortion utilities
  result.py                 # render-result and comparison analysis
  flicker_reduction.py      # temporal smoothing / adaptive IIR filtering
  image_utils.py            # image preparation utilities
  metrics.py                # MSE, PSNR, SSIM-like and power metrics

examples/
  demo_image_dimming.py     # run one image through local dimming
  compare_algorithms.py     # compare several algorithms on synthetic input
  demo_flicker_filter.py    # demonstrate temporal LED smoothing

docs/
  thesis_method_summary.md
  matlab_to_python_mapping.md
  algorithm_notes.md

research/
  README.md                 # official public thesis link

tests/
  test_algorithms.py
  test_display_model.py
  test_metrics.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Quick start

Run a synthetic local-dimming example:

```bash
python examples/demo_image_dimming.py
```

Compare baseline algorithms:

```bash
python examples/compare_algorithms.py
```

Run tests:

```bash
pytest
```

## Minimal Python usage

```python
import numpy as np
from local_backlight_dimming.display_model import DisplayModel
from local_backlight_dimming.algorithms import proposed_histogram_dimming
from local_backlight_dimming.compensation import compensate_lcd
from local_backlight_dimming.metrics import relative_backlight_power, psnr

image = np.random.random((360, 640, 3))
display = DisplayModel.from_grid(height=360, width=640, rows=8, cols=16)
led = proposed_histogram_dimming(display, image)
backlight = display.simulate_backlight(led)
compensated, reproduced = compensate_lcd(image, backlight)

print("relative power:", relative_backlight_power(led))
print("PSNR:", psnr(image, reproduced))
```

## MATLAB conversion notes

The uploaded MATLAB source files were reviewed and converted into Python. The original MATLAB source files are intentionally not committed. The Python version is structured, typed and documented for maintainability.

Important MATLAB files mapped into this Python package include:

- `algConv.m` -> `conventional_dimming`
- `algAvg.m` -> `average_dimming`
- `algAvgPlus.m` -> `average_plus_dimming`
- `algMax.m` -> `maximum_dimming`
- `algCho.m` -> `cho_dimming`
- `algNam.m` -> `nam_dimming`
- `algChen.m` -> `chen_dimming`
- `algEhsan.m` and thesis Section 3.1.4 -> `proposed_histogram_dimming` / `matlab_ehsan_variant`
- `algClipperFreeXiao.m` -> `clipper_free_dimming`
- `algNew.m` -> `new_dimming`
- `algFast.m` -> `fast_refinement`
- `simulateBacklight.m` -> `DisplayModel.simulate_backlight`
- `renderResult.m` -> `render_result`
- `calculateEpsilon.m` -> `calculate_epsilon`
- `computeColorDistortion.m` and `srgb2xyz.m` -> colour utilities in `color.py`

## Status

This is a clean research and portfolio implementation. It is not a vendor display driver and does not include confidential data or proprietary display calibration files.

## License

MIT License.

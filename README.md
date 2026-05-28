# Local Backlight Dimming of LED-LCD Display

Python implementation of local backlight dimming algorithms for LED-LCD displays, based on PhD research on image-quality preservation, power reduction, brightness compensation, and flicker reduction.

This repository converts and modernises MATLAB source code associated with Chapter 3 of the thesis:

> E. Nadernejad, *Processing Decoded Video for LCD-LED Backlight Display: Post processing of decoded video and local backlight dimming for LCD technology with LED-based backlight*, Technical University of Denmark, 2013.

## Scope

The project focuses on local dimming for LED-backlit LCD displays. It includes:

- local backlight zone modelling,
- average, maximum, square-root, Cho-style and Nam-style dimming baselines,
- the thesis-inspired local histogram / image-feature algorithm,
- backlight diffusion simulation using a point-spread function approximation,
- LCD brightness compensation with leakage modelling,
- quality and power metrics,
- temporal flicker reduction using adaptive IIR filtering,
- examples and tests for reproducible experimentation.

## Background

LCD panels use liquid crystals as light filters and require a backlight. With LED backlights, the backlight can be split into independently controlled zones. Local dimming reduces power consumption and can improve contrast, but it may introduce clipping, haloing, leakage errors, or temporal flicker if not controlled carefully.

The thesis method uses local image statistics to determine LED values. Each local segment is analysed using features such as average luminance, maximum value, local histogram percentile, square-root average and coefficient of variation. This produces a practical trade-off between image fidelity and energy saving.

## Repository structure

```text
src/local_backlight_dimming/
  algorithms.py             # baseline and thesis-inspired dimming algorithms
  compensation.py           # LCD transmittance and brightness compensation
  display_model.py          # LED grid, PSF and backlight simulation
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

The uploaded MATLAB archive contained useful algorithm files and legacy artefacts such as `.svn` folders, `.asv` autosaves and merge-conflict variants. Only the clean algorithmic content has been converted. The Python version is structured, typed and documented for maintainability.

Important MATLAB files mapped into this Python package include:

- `algAvg.m` -> `average_dimming`
- `algMax.m` -> `maximum_dimming`
- `algCho.m` -> `cho_dimming`
- `algNam.m` -> `nam_dimming`
- `algEhsan.m` and thesis Section 3.1.4 -> `proposed_histogram_dimming`
- `simulateBacklight.m` -> `DisplayModel.simulate_backlight`
- `renderResult.m` -> `compensate_lcd`
- `ssim.m` -> `metrics.ssim_global`

## Status

This is a clean research and portfolio implementation. It is not a vendor display driver and does not include confidential data or proprietary display calibration files.

## License

MIT License.

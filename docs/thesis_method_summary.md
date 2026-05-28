# Thesis method summary

This repository implements the local backlight dimming part of Chapter 3 of the PhD thesis *Processing Decoded Video for LCD-LED Backlight Display*.

## Problem

LED-LCD displays use an LED backlight behind a liquid-crystal panel. Local dimming controls different backlight zones independently. The aim is to reduce backlight power and improve perceived contrast while preserving image quality.

Main risks of local dimming are:

- clipping of bright pixels when the local backlight is too low,
- light leakage through liquid crystals,
- spatial halo artefacts caused by backlight diffusion,
- temporal flickering when LED values vary abruptly across frames.

## Display model

The simplified model in this repository follows the same concepts as the thesis:

```text
output luminance = backlight intensity × observed LCD transmittance
```

Observed transmittance includes leakage:

```text
t_observed = (1 - epsilon) * t + epsilon
```

Backlight diffusion is approximated by placing LED values on a grid and convolving them with a point-spread function.

## Proposed feature/histogram algorithm

The thesis-inspired algorithm analyses every LED segment independently:

1. Convert a local image segment into intensity `g = max(R, G, B)`.
2. Compute local average, maximum, variance and square-root average.
3. Classify the segment as low, medium or high luminance.
4. Use a local histogram percentile to set the initial backlight level.
5. Add an adaptive feature correction term.

The implementation is available as:

```python
from local_backlight_dimming.algorithms import proposed_histogram_dimming
```

## Flicker reduction

The thesis discusses temporal flicker caused by abrupt LED variations. This repository provides practical smoothing filters:

- first-order exponential smoothing,
- adaptive IIR smoothing,
- second-order IIR smoothing.

These are available in:

```python
from local_backlight_dimming.flicker_reduction import adaptive_iir_filter, second_order_iir
```

## Notes

The implementation is a clean Python research version. It is not intended to reproduce a particular commercial display exactly because real products require measured PSFs, calibrated leakage parameters and panel-specific electro-optical transfer functions.

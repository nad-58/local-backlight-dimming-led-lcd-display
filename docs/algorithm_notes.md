# Algorithm notes

## Baseline algorithms

### Average dimming

Each LED segment is driven by the average intensity of its associated image region. This gives strong power saving in dark scenes but can clip bright details.

### Maximum dimming

Each LED segment is driven by the maximum local intensity. This preserves local bright details but usually saves less power.

### Square-root dimming

The square root of the average local intensity is used as a compromise between average and maximum dimming.

### Cho-style dimming

The local average is corrected using the gap between local maximum and local average. This increases the backlight level when the segment contains bright pixels over a darker background.

### Nam-style dimming

The local decision is influenced by global image statistics, using the relationship between local and global maximum/average intensity.

## Proposed histogram-feature algorithm

The proposed implementation is the main portfolio algorithm in this repository. It combines:

- local histogram percentile initialisation,
- brightness class selection,
- average feature for low-brightness regions,
- square-root feature for medium-brightness regions,
- square-root plus coefficient-of-variation correction for high-brightness regions,
- adaptive correction strength based on the local maximum-average difference.

This captures the main idea of the thesis method: balancing energy saving with preservation of bright image detail.

## Display model

The display model is intentionally simple:

1. LED values are placed on a 2-D grid.
2. A Gaussian PSF approximates backlight spread.
3. Full-backlight response is used for normalisation.
4. LCD compensation is simulated with a leakage-aware transmittance equation.

For real hardware, replace the Gaussian PSF with a measured point-spread function and provide panel-specific leakage data.

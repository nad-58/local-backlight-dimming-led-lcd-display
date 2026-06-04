# Backlight Dimming Optimization Python Port

Python translation of the MATLAB backlight dimming simulation.

The default CLI mirrors `lcdscript.m`: it initializes the LCD/display model,
loads an image, runs the Max and New algorithms, and writes rendered outputs
and metrics.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python lcdscript.py --image ../backlight_dimming_optimization_matlab/images/City1.png
```

The `New` algorithm uses CVXPY as the Python equivalent of MATLAB CVX.
For large resolutions, keep `--downscaling-factor` high enough that the
optimization matrix remains tractable.

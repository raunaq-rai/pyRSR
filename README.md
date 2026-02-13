# PyRSR

**High-redshift galaxy analysis tools for JWST/NIRSpec data.**

`PyRSR` provides a suite of tools for processing and analyzing spectroscopic data, with a focus on emission line fitting, continuum subtraction, and robust flux measurements for high-redshift galaxies.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Features

- **Emission Line Fitting**: Fit single or multiple emission lines with Gaussian profiles (`excels_fit_poly`).
- **Broad Component Analysis**: Detect and fit broad Balmer lines (Hα, Hβ, Hδ) using BIC-based model selection (`single_broad_fit`).
- **Bootstrap Uncertainty**: Robust uncertainty estimation via bootstrap resampling (`broad_fit`, `bootstrap_excels_fit`).
- **Continuum Subtraction**: Polynomial or Moving Average continuum fitting with options for automatic masking around Lyman-α.
- **Support for NIRSpec Gratings**: Tuned constraints and presets for PRISM, MEDIUM, and HIGH resolution gratings.
- **Photometry & Stacking**: Tools for photometry handling and spectral stacking (in `photometry.py` and `stacking.py`).
- **High-Performance Version (PyRSRX)**: A Cythonized version (`PyRSRX`) is available for computationally intensive bootstrapping, offering ~1.5x speedup.

## PyRSR vs PyRSRX

This repository contains two versions of the package:

| Package | Description | Best For | Installation |
|---|---|---|---|
| **PyRSR** | Pure Python implementation. | General usage, detailed debugging, environments without C compilers. | `pip install .` |
| **PyRSRX** | Cython-optimized implementation. | Heavy bootstrapping (`broad_fit`), production pipelines. | `pip install -e .` (requires C compiler) |

**Walkthrough: Switching to PyRSRX**

Both packages share the exact same API. To use the faster version, simply change your import:

```python
# Standard version
from PyRSR.broad_line_fit import broad_fit

# Fast Cython version
from PyRSRX.broad_line_fit import broad_fit
```

The `PyRSRX` module uses compiled C extensions for the Gaussian model generation and integration steps, which are the bottleneck in iterative fitting. This results in a significant speedup (approx 1.5x faster per fit) which adds up when running 1000+ bootstrap iterations.

## Installation

### Using Conda (Recommended)

You can easily set up the environment and install the package using Conda:

```bash
# 1. Clone the repository
git clone https://github.com/raunaq-rai/pyRSR.git
cd PyRSR

# 2. Create the environment
conda env create -f environment.yaml

# 3. Activate the environment
conda activate pyrsr

# 4. Install the package in editable mode
pip install -e .
```

### Using pip only

If you prefer to use your own environment:

```bash
git clone https://github.com/raunaq-rai/pyRSR.git
cd PyRSR
pip install -e .
```

### Dependencies

- Python >= 3.9
- `numpy`, `matplotlib`, `scipy`, `astropy`, `uncertainties`

## Usage

### 1. Basic Line Fitting

Fit emission lines in a spectrum with a polynomial continuum.

```python
from PyRSR.line_fit import excels_fit_poly

# Load your data (e.g., from dictionary or FITS)
# source = {"lam": lam_um, "flux": flux_uJy, "err": err_uJy}
# or
# source = "path/to/spectrum.fits"

results = excels_fit_poly(
    source=source,
    z=2.5,                  # Redshift
    grating="PRISM",        # "PRISM", "MEDIUM", or "HIGH"
    lines_to_use=["H⍺", "NII_6585"],
    deg=2,                  # Continuum polynomial degree
    plot=True
)

print(results["per_line"]["H⍺"])
print(results["per_line"]["H⍺"])
```

### 2. Advanced Continuum Fitting

PyRSR supports two methods for continuum fitting: **Polynomial** (default) and **Moving Average** (robust).

#### Polynomial Fit (Default)
Standard finding of the best-fit polynomial of degree `deg`.

```python
results = broad_fit(..., continuum_fit="polyfit", deg=2)
```

#### Moving Average Fit
A robust, sigma-clipped moving average that can handle complex continuum shapes and the Lyman break better than high-degree polynomials.

**Key Arguments:**
*   `continuum_fit="moving_average"`: Enable this mode.
*   `continuum_movavg_window_um`: Window width in microns (default `0.05`).
*   `continuum_exclude_regions`: List of `(min, max)` tuples in microns to **manually mask** artifacts or broad emission lines that standard masking misses.
*   `continuum_lyman_window_um`: A smaller window size (e.g., `0.01`) to use specifically near the Lyman-alpha break to preserve the sharp step function.
*   `continuum_lya_mask_A`: Buffer in rest-frame Angstroms around LyA where the smaller window is applied (default `200.0`).

**Example:**
```python
results = broad_fit(
    source=source,
    z=6.0,
    continuum_fit="moving_average",
    continuum_movavg_window_um=0.05,       # Global window: 500A (obs) at z=0, 0.05um
    continuum_exclude_regions=[(1.48, 1.52)], # Manually mask a glitch at 1.5um
    continuum_lyman_window_um=0.01,        # Use 100A window near Lyman break for sharpness
)
```

### 3. Broad Line Search

Fit a spectrum and automatically determine if a broad component (e.g., for AGN) is supported by the data.

```python
from PyRSR.broad_line_fit import single_broad_fit

results = single_broad_fit(
    source=source,
    z=2.5,
    grating="PRISM",
    lines_to_use=["H⍺"],       # Focus on H-alpha
    broad_mode="auto",         # Use BIC to select best model (narrow vs broad)
    plot=True
)

print(f"Selected Model: {results.get('broad_choice_HA', 'narrow')}")
```

### 3. Bootstrap Analysis

Run a bootstrap analysis to get robust error estimates for line fluxes and widths.

```python
from PyRSR.broad_line_fit import broad_fit

# formatting options for output table can be handled by
# print_bootstrap_line_table_broad(res)

res = broad_fit(
    source=source,
    z=2.5,
    grating="PRISM",
    lines_to_use=["H⍺", "HBETA", "OIII_5007"],
    n_boot=1000,
    broad_mode="auto",
    plot=True,
    save_path="./output_plots"
)

# Access summary statistics
print(res["summary"]["H⍺"]["F_line"])
```

### 5. Using PyRSRX

To use the high-performance Cython version, simply change the import:

```python
from PyRSRX.broad_line_fit import broad_fit
```

Both `PyRSR` and `PyRSRX` share the exact same API and support all features including robust continuum fitting.

## Contributing

This project is under constant development.

## License

MIT License

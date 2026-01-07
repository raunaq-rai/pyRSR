# PyRSR

**High-redshift galaxy analysis tools for JWST/NIRSpec data.**

`PyRSR` provides a suite of tools for processing and analyzing spectroscopic data, with a focus on emission line fitting, continuum subtraction, and robust flux measurements for high-redshift galaxies.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Features

- **Emission Line Fitting**: Fit single or multiple emission lines with Gaussian profiles (`excels_fit_poly`).
- **Broad Component Analysis**: Detect and fit broad Balmer lines (Hα, Hβ, Hδ) using BIC-based model selection (`single_broad_fit`).
- **Bootstrap Uncertainty**: Robust uncertainty estimation via bootstrap resampling (`broad_fit`, `bootstrap_excels_fit`).
- **Continuum Subtraction**: Polynomial continuum fitting with options for automatic masking around Lyman-α.
- **Support for NIRSpec Gratings**: Tuned constraints and presets for PRISM, MEDIUM, and HIGH resolution gratings.
- **Photometry & Stacking**: Tools for photometry handling and spectral stacking (in `photometry.py` and `stacking.py`).

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
```

### 2. Broad Line Search

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

## Contributing

This project is under constant development.

## License

MIT License

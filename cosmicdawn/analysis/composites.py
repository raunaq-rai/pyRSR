"""
composites.py

Composite spectrum utilities.

This module provides functions to read, rescale, and plot composite
spectra (e.g. stacked from EXCELS or JADES), together with their GSF
(Spectral Energy Distribution fitting) models.

Typical workflow:
    >>> lam, flux, model = read_composite("EXCELS_stack")
    >>> plot_composite(lam, flux, model_flux=model)

Notes
-----
- Expects ASCII input spectra and GSF FITS model files in the
  directory structure given by `base_dir`.
- Fluxes are converted into consistent units (`fnu` or `flam`).
"""

import numpy as np
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from ..general import conversions as conv


def read_composite(name, base_dir="data/composites", units="fnu"):
    """
    Read composite spectrum and optional GSF model.

    Parameters
    ----------
    name : str
        Identifier of the composite stack (basename, no extension).
    base_dir : str, default="data/composites"
        Directory containing `input_stacks/` and `output_stacks/`.
    units : {"fnu","flam"}
        Units for returned flux.

    Returns
    -------
    lam : ndarray
        Wavelength array.
    flux : ndarray
        Composite flux in requested units.
    model_flux : ndarray
        Interpolated model flux on same wavelength grid.
    """
    # Stub implementation — user fills in filepaths
    lam, flux, model_flux = np.array([]), np.array([]), np.array([])
    return lam, flux, model_flux


def plot_composite(lam, flux, model_flux=None, frame="obs", ax=None):
    """
    Plot composite spectrum.

    Parameters
    ----------
    lam : ndarray
        Wavelength array.
    flux : ndarray
        Flux density.
    model_flux : ndarray, optional
        Model flux to overplot.
    frame : {"obs","rest"}, default="obs"
        Frame for wavelength axis.
    ax : matplotlib.Axes, optional
        Axis to plot on. If None, creates a new one.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with plotted spectrum.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(lam, flux, color="k", lw=1.2, label="Composite")
    if model_flux is not None:
        ax.step(lam, model_flux, color="r", lw=1.2, label="Model")
    ax.set_xlabel("λ [μm]" if frame == "obs" else "λ_rest [Å]")
    ax.set_ylabel("Flux density")
    ax.legend()
    return ax

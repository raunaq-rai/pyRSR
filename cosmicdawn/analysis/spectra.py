"""
spectra.py

Utilities for handling and plotting spectroscopic data.

This module provides functions to:
    - Load prism and grating spectra (from dicts, arrays, or msaexp-like formats)
    - Plot spectra with consistent styling
    - Wrap msaexp spectrum plotting for quick-look analysis

Intended for general use; does not depend on specific catalogs.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..general.plotting import style_axes



# --------------------------
# Spectrum loading
# --------------------------

def get_prism_spectrum(spectra_dict, msaid, units="msaexp"):
    """
    Retrieve prism spectrum for a given source ID.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary with spectra. Expected key structure:
        spectra_dict[msaid]["prism-clear"] = {"lam", "flux", "err"}.
    msaid : str or int
        Source ID (must match dict key).
    units : {"msaexp", "flam"}, default="msaexp"
        Return units. "msaexp" = raw fν-like units.
        "flam" = erg/s/cm²/Å (requires λ in Å).

    Returns
    -------
    lam, flux, err : ndarray
        Wavelength, flux, and error arrays.
    """
    d = spectra_dict[msaid]["prism-clear"]
    lam, flux, err = np.array(d["lam"]), np.array(d["flux"]), np.array(d["err"])

    if units.lower() == "flam":
        from ..general.conversions import fnu_to_flam
        lam_AA = lam * 1e4  # microns → Å if needed
        flux = fnu_to_flam(lam_AA, flux)
        err = fnu_to_flam(lam_AA, err)

    return lam, flux, err


def get_grating_spectrum(spectra_dict, msaid, units="msaexp"):
    """
    Retrieve combined grating spectrum for a source ID.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary with spectra. Keys may include "g235m", "g395m", etc.
    msaid : str or int
        Source ID.
    units : {"msaexp", "flam"}, default="msaexp"
        Return units.

    Returns
    -------
    lam, flux, err : ndarray
        Wavelength, flux, and error arrays (sorted by λ).
    """
    lam, flux, err = [], [], []

    for R, spec in spectra_dict[msaid].items():
        if "prism" in R.lower():
            continue
        lam.extend(spec["lam"])
        flux.extend(spec["flux"])
        err.extend(spec["err"])

    lam, flux, err = map(np.array, (lam, flux, err))
    order = np.argsort(lam)
    lam, flux, err = lam[order], flux[order], err[order]

    if units.lower() == "flam":
        from .conversions import fnu_to_flam
        lam_AA = lam * 1e4
        flux = fnu_to_flam(lam_AA, flux)
        err = fnu_to_flam(lam_AA, err)

    return lam, flux, err


# --------------------------
# Spectrum plotting
# --------------------------

def plot_prism_spectrum(spectra_dict, msaid, ax=None, **kwargs):
    """
    Quick-look plot of prism spectrum.

    Parameters
    ----------
    spectra_dict : dict
        Spectra dictionary with prism entry.
    msaid : str or int
        Source ID.
    ax : matplotlib.Axes, optional
        Axis to plot on.
    kwargs : dict
        Passed to matplotlib.plot().
    """
    lam, flux, err = get_prism_spectrum(spectra_dict, msaid)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lam, flux, **kwargs)
    ax.fill_between(lam, flux - err, flux + err, alpha=0.3, color=kwargs.get("color", "k"))
    style_axes(ax, "Wavelength [µm]", "Flux")
    return ax


def plot_grating_spectrum(spectra_dict, msaid, ax=None, **kwargs):
    """
    Quick-look plot of grating spectrum.

    Parameters
    ----------
    spectra_dict : dict
        Spectra dictionary with grating entries.
    msaid : str or int
        Source ID.
    ax : matplotlib.Axes, optional
        Axis to plot on.
    kwargs : dict
        Passed to matplotlib.plot().
    """
    lam, flux, err = get_grating_spectrum(spectra_dict, msaid)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lam, flux, **kwargs)
    ax.fill_between(lam, flux - err, flux + err, alpha=0.3, color=kwargs.get("color", "k"))
    style_axes(ax, "Wavelength [µm]", "Flux")
    return ax


def plot_msaexp_spectrum(spectra_dict, msaid, which="prism", ax=None, **kwargs):
    """
    General wrapper to plot msaexp-format spectra.

    Parameters
    ----------
    spectra_dict : dict
        Dictionary with msaexp spectra.
    msaid : str or int
        Source ID.
    which : {"prism","grating"}, default="prism"
        Which spectrum to plot.
    ax : matplotlib.Axes, optional
        Axis to plot on.
    kwargs : dict
        Passed to matplotlib.plot().
    """
    if which == "prism":
        return plot_prism_spectrum(spectra_dict, msaid, ax=ax, **kwargs)
    elif which == "grating":
        return plot_grating_spectrum(spectra_dict, msaid, ax=ax, **kwargs)
    else:
        raise ValueError(f"Unknown spectrum type: {which}")

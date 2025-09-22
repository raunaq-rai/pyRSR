"""
plotting.py

General plotting utilities for high-redshift galaxy analysis.

This module provides helper functions for:
    - Styling axes consistently
    - Plotting spectra and filters
    - Converting between flux/magnitude and plotting SEDs
    - Making cutouts from FITS images

Heavy catalog-specific functions (JEWELS, Astrodeep, etc.) have been omitted
to keep this package portable. Users should extend those externally.

Dependencies:
    numpy, matplotlib, astropy, uncertainties
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u
from uncertainties import unumpy as unp
from uncertainties import ufloat
from .utils import find_nearest


# ----------------------
# Axis styling utilities
# ----------------------

def style_axes(ax, xlabel=None, ylabel=None, fontsize=16, labelsize=14, linewidth=1.5):
    """
    Apply consistent axis styling to matplotlib axes.
    """
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=labelsize)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_linewidth(linewidth)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    return ax


# ----------------------
# Spectrum-related plots
# ----------------------

def plot_spectrum(lam, flux, err=None, ax=None, label=None, color="black", alpha=0.8):
    """
    Plot a 1D spectrum with optional error shading.

    Parameters
    ----------
    lam : array_like
        Wavelength array.
    flux : array_like
        Flux array.
    err : array_like, optional
        Error array for shading.
    ax : matplotlib.Axes, optional
        Existing axes to plot on.
    label : str, optional
        Label for the spectrum.
    color : str
        Line color.
    alpha : float
        Line/area transparency.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(lam, flux, color=color, alpha=alpha, label=label)
    if err is not None:
        ax.fill_between(lam, flux - err, flux + err, color=color, alpha=0.3)

    style_axes(ax, xlabel="Wavelength [Å]", ylabel="Flux")
    if label:
        ax.legend()
    return ax


def plot_filters(filter_files, filter_dir="data/filters/", ax=None, colors=None):
    """
    Plot filter transmission curves from ASCII files.

    Parameters
    ----------
    filter_files : list of str
        Filenames of filter transmission curves (two-column ASCII).
    filter_dir : str
        Directory where filter files are stored.
    ax : matplotlib.Axes, optional
        Axes to plot on.
    colors : list of str, optional
        Colors for plotting each filter.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for i, f in enumerate(filter_files):
        path = f"{filter_dir}/{f}"
        lam, resp = np.loadtxt(path, unpack=True)
        resp /= resp.max()
        ax.plot(lam, resp, label=f, color=None if colors is None else colors[i])

    style_axes(ax, xlabel="Wavelength [Å]", ylabel="Transmission")
    ax.legend(fontsize=8)
    return ax


# ----------------------
# Imaging utilities
# ----------------------

def make_cutout(input_fits, output_fits, ra, dec, size_arcsec, plot_cutout=False, clim=None):
    """
    Make a FITS cutout centered on (ra, dec).

    Parameters
    ----------
    input_fits : str
        Input FITS filename.
    output_fits : str
        Output FITS filename.
    ra, dec : float
        Center coordinates in degrees.
    size_arcsec : float
        Cutout size in arcsec.
    plot_cutout : bool
        If True, display the cutout with matplotlib.
    clim : tuple, optional
        vmin, vmax for imshow.
    """
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

        size = (size_arcsec * u.arcsec).to(u.deg)
        center = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5")

        cutout = Cutout2D(data, position=center, size=size, wcs=wcs, copy=True)

        new_header = cutout.wcs.to_header()
        for key in new_header:
            header[key] = new_header[key]

        hdul[0].data = cutout.data
        hdul[0].header = header
        hdul.writeto(output_fits, overwrite=True)

    if plot_cutout:
        img = fits.getdata(output_fits)
        plt.figure(figsize=(5, 5))
        plt.imshow(img, origin="lower", cmap="gray", clim=None if clim is None else clim)
        plt.axis("off")
        plt.title("FITS Cutout")
        plt.show()

    return output_fits

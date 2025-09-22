"""
plotting.py

General plotting utilities for high-redshift galaxy analysis.

This module provides helper functions for:
    - Styling axes consistently
    - Plotting spectra and filters
    - Making cutouts from FITS images

Dependencies:
    numpy, matplotlib, astropy
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u


# ----------------------
# Axis styling utilities
# ----------------------

def style_axes(ax, xlabel=None, ylabel=None, fontsize=16, labelsize=14, linewidth=1.5):
    """Apply consistent axis styling to matplotlib axes."""
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


def plot_filters(filter_files=None, filter_dir="data/filters/", ax=None, colors=None, filter_arrays=None):
    """
    Plot filter transmission curves.

    Parameters
    ----------
    filter_files : list of str, optional
        Filenames of filter transmission curves.
    filter_dir : str
        Directory where filter files are stored.
    filter_arrays : list of tuple, optional
        Directly provide [(lam, resp), ...] arrays instead of files.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if filter_files:
        for i, f in enumerate(filter_files):
            path = f"{filter_dir}/{f}"
            lam, resp = np.loadtxt(path, unpack=True)
            resp /= resp.max()
            ax.plot(lam, resp, label=f, color=None if colors is None else colors[i])

    if filter_arrays:
        for i, (lam, resp) in enumerate(filter_arrays):
            resp /= resp.max()
            ax.plot(lam, resp, label=f"array{i+1}", color=None if colors is None else colors[i])

    style_axes(ax, xlabel="Wavelength [Å]", ylabel="Transmission")
    ax.legend(fontsize=8)
    return ax



# ----------------------
# Imaging utilities
# ----------------------

def make_cutout(input_fits, output_fits, ra, dec, size_arcsec, plot_cutout=False, clim=None):
    """
    Make a FITS cutout centered on (ra, dec).
    """
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

        size = (size_arcsec * u.arcsec).to(u.deg)
        center = SkyCoord(ra=ra, dec=dec, unit="deg", frame="fk5")

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
        plt.imshow(img, origin="lower", cmap="gray", clim=clim)
        plt.axis("off")
        plt.title("FITS Cutout")
        plt.show()

    return output_fits

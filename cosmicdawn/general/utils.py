"""
utils.py

General-purpose utilities for analysis, plotting, coordinates, and file I/O.

Functions
---------
find_nearest(array, value, return_index=True)
    Return nearest value (or its index) in array.

style_axes(ax, xlabel=None, ylabel=None, fontsize=18, labelsize=18, linewidth=1.25)
    Apply consistent axis styling to matplotlib axes.

lighten_color(color, amount=1.0)
    Lighten an input matplotlib color string, hex, or RGB tuple.

get_continuous_cmap(hex_list, float_list=None)
    Build a continuous matplotlib colormap from a list of hex colors.

make_cutout(input_fits, output_fits, ra, dec, size_arcsec, plot_cutout=False, clim=None)
    Extract a FITS cutout around target coordinates.

dja_cutout(ra, dec, boxsize=1.5, filters=None, output=None, asinh=True)
    Download RGB cutouts from grizli-cutout service for given coords.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import os


# --------------------------
# Array & math helpers
# --------------------------

def find_nearest(array, value, return_index=True):
    """
    Find nearest value in array.

    Parameters
    ----------
    array : array-like
        Input array.
    value : float
        Value to find nearest to.
    return_index : bool
        If True, return index; else return value.

    Returns
    -------
    idx or val : int or float
        Index of nearest value, or the value itself.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if return_index else array[idx]


# --------------------------
# Plotting helpers
# --------------------------

def style_axes(ax, xlabel=None, ylabel=None,
               fontsize=18, labelsize=18, linewidth=1.25,
               ticks={'bottom': True, 'top': True, 'left': True, 'right': True}):
    """
    Apply consistent styling to axes.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to style.
    xlabel, ylabel : str, optional
        Axis labels.
    fontsize : float
        Font size for labels.
    labelsize : float
        Size of tick labels.
    linewidth : float
        Width of axis spines.
    ticks : dict
        Which spines to draw ticks on.
    """
    ax.tick_params(axis='both', which='both', direction='in',
                   bottom=ticks['bottom'], top=ticks['top'],
                   left=ticks['left'], right=ticks['right'],
                   labelsize=labelsize)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(linewidth)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)


def lighten_color(color, amount=1.0):
    """
    Lighten given color.

    Parameters
    ----------
    color : str or tuple
        Matplotlib color string, hex string, or RGB tuple.
    amount : float
        Factor (0=black, 1=unchanged).

    Returns
    -------
    rgb : tuple
        Lightened RGB tuple.
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_continuous_cmap(hex_list, float_list=None):
    """
    Build continuous colormap from hex colors.

    Parameters
    ----------
    hex_list : list
        List of hex color codes.
    float_list : list, optional
        Floats between 0 and 1 for placement. Defaults to linear spacing.

    Returns
    -------
    cmp : matplotlib colormap
        Continuous colormap object.
    """
    def hex_to_rgb(value):
        value = value.strip("#")
        lv = len(value)
        return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

    def rgb_to_dec(value):
        return [v / 256 for v in value]

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list is None:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = {col: [[float_list[i], rgb_list[i][j], rgb_list[i][j]]
                   for i in range(len(float_list))]
             for j, col in enumerate(['red', 'green', 'blue'])}
    return mc.LinearSegmentedColormap('custom_cmap', segmentdata=cdict, N=256)


# --------------------------
# FITS / cutouts
# --------------------------

def make_cutout(input_fits, output_fits, ra, dec, size_arcsec,
                plot_cutout=False, clim=None):
    """
    Create a cutout from FITS file around given RA/Dec.

    Parameters
    ----------
    input_fits : str
        Input FITS filename.
    output_fits : str
        Output FITS filename.
    ra, dec : float
        Target coordinates [deg].
    size_arcsec : float
        Cutout size [arcsec].
    plot_cutout : bool
        If True, show plot.
    clim : tuple, optional
        Color limits for plotting.
    """
    with pyfits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

        size = (size_arcsec * u.arcsec).to(u.deg)
        center = SkyCoord(ra=ra, dec=dec, unit='deg', frame='fk5')
        cutout = Cutout2D(data, position=center, size=size, wcs=wcs, copy=True)

        new_header = cutout.wcs.to_header()
        for key in new_header:
            header[key] = new_header[key]

        hdul[0].data = cutout.data
        hdul[0].header = header
        hdul.writeto(output_fits, overwrite=True)

    if plot_cutout:
        img = pyfits.getdata(output_fits)
        plt.figure(figsize=(5, 5))
        plt.imshow(img, origin='lower', clim=clim)
        plt.axis('off')
        plt.show()


def dja_cutout(ra, dec, boxsize=1.5, filters=None, output=None, asinh=True):
    """
    Download RGB cutouts from grizli-cutout service.

    Parameters
    ----------
    ra, dec : float
        Coordinates [deg].
    boxsize : float
        Cutout size [arcsec].
    filters : list of str
        Filters to request.
    output : str
        Output prefix for files.
    asinh : bool
        Use asinh scaling.
    """
    assert filters, "Must supply at least one filter."
    for filt in filters:
        cmd = (
            f"wget 'https://grizli-cutout.herokuapp.com/thumb"
            f"?all_filters=True&size={boxsize}&scl=1.0&asinh={asinh}"
            f"&filters={filt}&ra={ra}&dec={dec}&output=fits_weight' "
            f"-O '{output}_{filt}.fits'"
        )
        os.system(cmd)

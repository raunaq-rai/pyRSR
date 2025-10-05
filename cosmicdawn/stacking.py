"""
stacking.py

Functions for stacking, binning, and masking spectroscopic data.
Useful for combining multiple spectra, down-binning data, and
creating emission-line masks for composite analysis.
"""

import numpy as np
from astropy.stats import sigma_clip
from cosmicdawn import utils  


# --------------------------
# Stacking functions
# --------------------------

def stack_spectra(stack_flux, stack_err=None, op="median", clip=False, sigma_sig=3.0):
    """
    Create a composite spectrum from multiple aligned spectra.

    Parameters
    ----------
    stack_flux : ndarray (N_spectra, N_pixels)
        2D array of flux values in the same wavelength frame.
    stack_err : ndarray (N_spectra, N_pixels), optional
        2D array of uncertainties associated with flux values.
    op : {"median", "mean"}, default="median"
        Operation for stacking.
    clip : bool, default=False
        Apply sigma-clipping to reject outliers.
    sigma_sig : float, default=3.0
        Sigma threshold for clipping if `clip=True`.

    Returns
    -------
    flux_stack : ndarray (N_pixels,)
        Stacked flux values.
    std_stack : ndarray (N_pixels,)
        Standard deviation at each pixel.
    err_stack : ndarray (N_pixels,), optional
        Combined error, if input errors provided.
    n_stack : ndarray (N_pixels,)
        Number of spectra contributing to each pixel.
    """
    stack_flux = np.array(stack_flux)
    if stack_err is not None:
        stack_err = np.array(stack_err)

    flux_stack, std_stack, err_stack, n_stack = [], [], [], []

    for i in range(stack_flux.shape[1]):
        vals = stack_flux[:, i]
        vals = vals[np.isfinite(vals)]

        if clip:
            vals = sigma_clip(vals, sigma=sigma_sig, maxiters=5, masked=False)

        if op == "median":
            flux_stack.append(np.nanmedian(vals))
        elif op == "mean":
            flux_stack.append(np.nanmean(vals))
        else:
            raise ValueError(f"Unknown op '{op}'")

        std_stack.append(np.nanstd(vals))
        n_stack.append(len(vals))

        if stack_err is not None:
            errs = stack_err[:, i]
            errs = errs[np.isfinite(errs)]
            err_stack.append(np.sqrt(np.nanmean(errs**2)))

    flux_stack = np.array(flux_stack)
    std_stack = np.array(std_stack)
    n_stack = np.array(n_stack)

    if stack_err is not None:
        err_stack = np.array(err_stack)
        return flux_stack, std_stack, err_stack, n_stack
    else:
        return flux_stack, std_stack, n_stack


# --------------------------
# Binning functions
# --------------------------

def bin_1d_spec(lam, flux, err=None, factor=1, method="median"):
    """
    Down-bin a 1D spectrum in wavelength space.

    Parameters
    ----------
    lam : ndarray
        Wavelength array.
    flux : ndarray
        Flux array.
    err : ndarray, optional
        Error array.
    factor : int, default=1
        Number of pixels per new bin.
    method : {"median","mean","sum"}, default="median"
        Combination method.

    Returns
    -------
    lam_b, flux_b : ndarray
        Binned wavelength and flux arrays.
    err_b : ndarray, optional
        Binned errors, if `err` provided.
    """
    lam, flux = np.array(lam), np.array(flux)
    n = len(lam)

    lam_b, flux_b, err_b = [], [], []

    for i in range(0, n, factor):
        lam_bin = lam[i : i + factor]
        flux_bin = flux[i : i + factor]

        lam_b.append(np.nanmedian(lam_bin))

        if method == "median":
            flux_b.append(np.nanmedian(flux_bin))
        elif method == "mean":
            flux_b.append(np.nanmean(flux_bin))
        elif method == "sum":
            flux_b.append(np.nansum(flux_bin))
        else:
            raise ValueError(f"Unknown method '{method}'")

        if err is not None:
            err_bin = err[i : i + factor]
            err_b.append(np.sqrt(np.nanmean(err_bin**2)))

    lam_b = np.array(lam_b)
    flux_b = np.array(flux_b)

    if err is not None:
        return lam_b, flux_b, np.array(err_b)
    return lam_b, flux_b


def bin_2d_spec(spec2d, factor=1, axis="lam"):
    """
    Down-bin a 2D spectrum (e.g., wavelength × spatial).

    Parameters
    ----------
    spec2d : ndarray (N_y, N_x)
        Input 2D spectral image.
    factor : int, default=1
        Binning factor.
    axis : {"lam","spatial","all"}, default="lam"
        Which axis to bin:
        - "lam" = spectral axis (columns),
        - "spatial" = spatial axis (rows),
        - "all" = both axes.

    Returns
    -------
    spec2d_b : ndarray
        Binned 2D spectrum.
    """
    m, n = spec2d.shape

    if axis == "lam":
        new_n = n // factor
        spec2d_b = spec2d[:, : new_n * factor].reshape(m, new_n, factor).sum(axis=2)
    elif axis == "spatial":
        new_m = m // factor
        spec2d_b = spec2d[: new_m * factor, :].reshape(new_m, factor, n).sum(axis=1)
    elif axis == "all":
        new_m, new_n = m // factor, n // factor
        spec2d_b = spec2d[: new_m * factor, : new_n * factor].reshape(
            new_m, factor, new_n, factor
        ).sum(axis=(1, 3))
    else:
        raise ValueError(f"Unknown axis '{axis}'")

    return spec2d_b


# --------------------------
# Diagnostics
# --------------------------

def custom_sigma_clip(array, low=3.0, high=3.0, op="median"):
    """
    Custom sigma-clipping for 1D arrays.

    Parameters
    ----------
    array : ndarray
        Input array.
    low, high : float, default=3.0
        Lower/upper sigma thresholds.
    op : {"median","mean"}, default="median"
        Central value definition.

    Returns
    -------
    clipped : ndarray
        Values within thresholds.
    low_val, high_val : float
        Threshold values.
    idx : ndarray
        Indices of surviving elements.
    """
    arr = np.array(array)
    if op == "mean":
        mu = np.nanmean(arr)
    elif op == "median":
        mu = np.nanmedian(arr)
    else:
        raise ValueError(f"Unknown op '{op}'")

    sig = np.nanstd(arr)
    low_val = mu - low * sig
    high_val = mu + high * sig

    idx = np.where((arr >= low_val) & (arr <= high_val))[0]
    return arr[idx], low_val, high_val, idx


def generate_line_mask(lam_AA, z_spec=0.0, dv=2000):
    """
    Generate a mask array covering known emission lines at a given redshift.

    Parameters
    ----------
    lam_AA : ndarray
        Wavelength array in Angstroms.
    z_spec : float, default=0.0
        Redshift of the object.
    dv : float, default=2000
        Velocity window (km/s) around each line.

    Returns
    -------
    mask : ndarray (int)
        Array of 0 (outside lines) and 1 (inside line windows).
    """
    c_kms = 299792.458
    mask = np.zeros_like(lam_AA, dtype=int)

    lines = {
        "Lyα": [1215.67],
        "OII": [3727.092, 3729.875],
        "NeIII": [3968.59, 3869.86],
        "Hδ": [4102.8922],
        "OIII4364": [4364.436],
        "Hβ": [4862.6830],
        "OIII": [4960.295, 5008.240],
        "HeI": [5877.252],
        "Hα": [6564.608],
        "HeI_2": [7065.196],
    }

    for _, rest_waves in lines.items():
        for lam0 in rest_waves:
            zlam = lam0 * (1.0 + z_spec)
            dlam = (dv / c_kms) * zlam
            lo, hi = zlam - dlam, zlam + dlam
            mask[(lam_AA >= lo) & (lam_AA <= hi)] = 1

    return mask

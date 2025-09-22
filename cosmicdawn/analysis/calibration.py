"""
calibration.py

Spectroscopic calibration utilities.

This module provides helper functions for:
    - Correcting slit losses based on aperture size, seeing, and object profile
    - Applying a simple flux calibration from an exposure-time calculator (ETC)

Notes
-----
These functions are deliberately simplified:
    - `correct_slitloss` assumes a Gaussian profile for both PSF and source.
    - `calibrate_etc_spec` rescales synthetic spectra based on ETC predictions.
Users should replace the toy models with instrument-specific data when available.
"""

import numpy as np
from scipy.special import erf



# --------------------------
# Slit loss correction
# --------------------------

def correct_slitloss(wave, flux, slit_width, seeing, source_size=0.0):
    """
    Apply a simple slit-loss correction to a spectrum.

    Assumes both PSF and source profiles are Gaussian and computes
    the fraction of light entering a rectangular slit.

    Parameters
    ----------
    wave : array_like
        Wavelength array [Å].
    flux : array_like
        Observed flux density [erg/s/cm²/Å].
    slit_width : float
        Projected slit width on sky [arcsec].
    seeing : float
        FWHM of the PSF [arcsec].
    source_size : float, default=0.0
        Intrinsic source FWHM [arcsec]. Set 0 for point source.

    Returns
    -------
    flux_corr : ndarray
        Slit-loss corrected flux density [erg/s/cm²/Å].
    slit_fraction : float
        Average fraction of light that entered the slit.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    # Effective FWHM (quadrature sum of seeing and source size)
    fwhm_eff = np.sqrt(seeing**2 + source_size**2)

    # Approximate slit throughput = erf(slit_width / (sqrt(2)*FWHM))
    sigma = fwhm_eff / 2.355
    slit_fraction = erf(slit_width / (2 * np.sqrt(2) * sigma))


    flux_corr = flux / slit_fraction
    return flux_corr, slit_fraction


# --------------------------
# Flux calibration
# --------------------------

def calibrate_etc_spec(wave, flux, etc_snr, exptime, target_snr, new_exptime=None):
    """
    Rescale a synthetic spectrum based on ETC S/N predictions.

    Parameters
    ----------
    wave : array_like
        Wavelength array [Å].
    flux : array_like
        Input (uncalibrated) flux density [arbitrary units].
    etc_snr : float
        Predicted S/N from the exposure time calculator at `exptime`.
    exptime : float
        Original exposure time [s].
    target_snr : float
        Desired S/N level to calibrate to.
    new_exptime : float, optional
        If provided, scale S/N according to new exposure time.

    Returns
    -------
    flux_cal : ndarray
        Calibrated flux density [relative scaling applied].
    snr_scaled : float
        Scaled S/N at the chosen setup.
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    # Scale factor from ETC to desired target SNR
    snr_ratio = target_snr / etc_snr

    if new_exptime is not None:
        snr_scaled = etc_snr * np.sqrt(new_exptime / exptime)
        scale = target_snr / snr_scaled
    else:
        snr_scaled = target_snr
        scale = snr_ratio

    flux_cal = flux * scale
    return flux_cal, snr_scaled

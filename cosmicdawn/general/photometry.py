"""
photometry.py

Synthetic photometry and filter-related utilities.

Functions
---------
photo_from_filter(wave, flux, filt_name, wave_unit='AA', filter_dir=None)
    Compute synthetic photometry by convolving a spectrum with a filter.

get_filter_info(filt_name, filter_dir=None, output_unit='micron')
    Return central wavelength, error, and limits for a given filter curve.

convolve_spectrum(lam, flux, filt_name, filter_dir=None)
    Convolve spectrum with filter response to get bandpass-averaged flux.

calc_IRAC_color(mAB, EW, z, lam_eff, filter_name)
    Compute IRAC color excess given continuum magnitude + EW.

calc_IRAC_EW(mAB_cont, color, z, lam_eff, filter_name)
    Estimate rest-frame EW from IRAC color excess.

emission_line(xvals, x0, A, width, FWHM=False)
    Generate Gaussian emission line profile.
"""

import numpy as np
from astropy.io import ascii
from uncertainties import unumpy as unp
from .conversions import AB_to_flux, fnu_to_flam


# --------------------------
# Filter + synthetic photometry
# --------------------------

def photo_from_filter(wave, flux, filt_name, wave_unit='AA', filter_dir=None):
    """
    Convolve a spectrum with a filter transmission curve.

    Parameters
    ----------
    wave : array
        Wavelength array (Angstroms by default).
    flux : array
        Flux density array (same units throughout).
    filt_name : str
        Filter curve file name (two-column: wavelength [Å], transmission).
    wave_unit : str
        'AA' (default) or 'micron'.
    filter_dir : str, optional
        Directory with filter files. If None, expects `filt_name` is a path.

    Returns
    -------
    lam_cent : float
        Effective central wavelength of filter [Å].
    flux_eff : float
        Bandpass-averaged flux [same units as input flux].
    """
    if wave_unit == 'micron':
        wave = wave * 1e4

    # Load filter curve
    if filter_dir is None:
        filt_lam, filt_resp = np.genfromtxt(filt_name, unpack=True)
    else:
        filt_lam, filt_resp = np.genfromtxt(f"{filter_dir}/{filt_name}", unpack=True)

    # Interpolate to spectrum grid
    resp = np.interp(wave, filt_lam, filt_resp, left=0., right=0.)

    flux_eff = np.trapz(flux * resp, wave) / np.trapz(resp, wave)
    lam_cent = np.median(wave[(resp / np.nanmax(resp)) > 0.5])
    return lam_cent, flux_eff


def get_filter_info(filt_name, filter_dir=None, output_unit='micron'):
    """
    Get filter central wavelength, error, and range.

    Parameters
    ----------
    filt_name : str
        Filter curve file (two-column: wavelength [Å], response).
    filter_dir : str, optional
        Directory where filter file lives.
    output_unit : str, {'micron','AA'}
        Unit for returned wavelengths.

    Returns
    -------
    lam_cent : float
        Central wavelength.
    lam_err : list
        Asymmetric errors [[low],[high]].
    lam_range : list
        [min, max] effective wavelength.
    """
    if filter_dir is None:
        f = ascii.read(filt_name)
    else:
        f = ascii.read(f"{filter_dir}/{filt_name}")

    f['col2'] = f['col2'] / np.nanmax(f['col2'])
    lam = f['col1'].data

    if output_unit == 'micron':
        lam = lam / 1e4

    idx = np.where(f['col2'] > 0.5)[0]
    lam_cent = np.median(lam[idx])
    lam_min, lam_max = np.min(lam[idx]), np.max(lam[idx])
    lam_err_low = lam_cent - lam_min
    lam_err_high = lam_max - lam_cent

    return lam_cent, [[lam_err_low], [lam_err_high]], [lam_min, lam_max]


def convolve_spectrum(lam, flux, filt_name, filter_dir=None):
    """
    Convolve spectrum with a filter response curve.

    Parameters
    ----------
    lam : array
        Wavelength array [Å].
    flux : array
        Flux density array.
    filt_name : str
        Filter curve filename.
    filter_dir : str, optional
        Directory where filter curves live.

    Returns
    -------
    lam_cent : float
        Central wavelength of filter [Å].
    flux_eff : float
        Filter-convolved flux [same units as flux].
    """
    if filter_dir is None:
        filt_lam, filt_resp = np.genfromtxt(filt_name, unpack=True)
    else:
        filt_lam, filt_resp = np.genfromtxt(f"{filter_dir}/{filt_name}", unpack=True)

    resp = np.interp(lam, filt_lam, filt_resp, left=0, right=0)
    flux_eff = np.trapz(flux * resp, lam) / np.trapz(resp, lam)
    lam_cent = np.median(lam[(resp / np.nanmax(resp)) > 0.5])
    return lam_cent, flux_eff


# --------------------------
# IRAC color / EW relations
# --------------------------

def calc_IRAC_color(mAB, EW, z, lam_eff, filter_name):
    """
    Compute IRAC color excess from line contribution.

    Parameters
    ----------
    mAB : float
        Continuum apparent AB magnitude.
    EW : float
        Rest-frame equivalent width of line(s) [Å].
    z : float
        Redshift.
    lam_eff : float
        Effective wavelength of filter [Å].
    filter_name : str
        Filter file with transmission curve.

    Returns
    -------
    color : float
        Color excess [mag].
    """
    flux_fnu = unp.nominal_values(AB_to_flux(mAB, 0., output_unit='fnu'))
    flux_flam = fnu_to_flam(lam_eff, flux_fnu)

    EW_obs = EW * (1. + z)
    lineflux = EW_obs * flux_flam

    filt_lam, filt_resp = np.genfromtxt(filter_name, unpack=True)
    idx = np.where(filt_resp > 0.5)
    filter_width = max(filt_lam[idx]) - min(filt_lam[idx])

    lineflux_filter = lineflux / filter_width
    color = -2.5 * np.log10(flux_flam / lineflux_filter)
    return color


def calc_IRAC_EW(mAB_cont, color, z, lam_eff, filter_name):
    """
    Estimate equivalent width from IRAC color.

    Parameters
    ----------
    mAB_cont : float
        Continuum apparent AB magnitude.
    color : float
        [3.6]-[4.5] color [mag].
    z : float
        Redshift.
    lam_eff : float
        Effective wavelength of filter [Å].
    filter_name : str
        Filter file with transmission curve.

    Returns
    -------
    EW0 : float
        Rest-frame equivalent width [Å].
    EW_obs : float
        Observed equivalent width [Å].
    """
    line_mag = mAB_cont - color
    ratio = 10 ** ((mAB_cont - line_mag) / -2.5)

    flux_fnu_cont = unp.nominal_values(AB_to_flux(mAB_cont, 0., output_unit='fnu'))
    flux_flam_cont = fnu_to_flam(lam_eff, flux_fnu_cont)

    lineflux_filter = flux_flam_cont / ratio

    filt_lam, filt_resp = np.genfromtxt(filter_name, unpack=True)
    idx = np.where(filt_resp > 0.5)
    filter_width = max(filt_lam[idx]) - min(filt_lam[idx])

    lineflux = lineflux_filter * filter_width
    EW_obs = lineflux / flux_flam_cont
    EW0 = EW_obs / (1. + z)
    return EW0, EW_obs


# --------------------------
# Emission line helper
# --------------------------

def emission_line(xvals, x0, A, width, FWHM=False):
    """
    Generate Gaussian emission line profile.

    Parameters
    ----------
    xvals : array
        Wavelength or frequency grid.
    x0 : float
        Line center.
    A : float
        Amplitude.
    width : float
        Line sigma or FWHM (see FWHM flag).
    FWHM : bool
        If True, interpret `width` as FWHM.

    Returns
    -------
    gauss : array
        Gaussian line profile.
    """
    sigma = width / 2. / np.sqrt(2. * np.log(2.)) if FWHM else width
    gauss = A * np.exp(-((xvals - x0) ** 2.) / (2. * sigma ** 2.))
    return gauss

"""
conversions.py

Utilities for converting between different flux, magnitude, and luminosity units.

Functions
---------
flux_to_AB(flux, flux_err=None, unit='jy')
    Convert flux density (Jy or fν) to AB magnitude.

AB_to_flux(mag, mag_err=None, output_unit='jy')
    Convert AB magnitude to flux density (Jy or fν).

fnu_to_flam(lam, fnu, fnu_err=None)
    Convert flux density from fν [erg/s/cm²/Hz] to fλ [erg/s/cm²/Å].

flam_to_fnu(lam, flam, flam_err=None)
    Convert flux density from fλ [erg/s/cm²/Å] to fν [erg/s/cm²/Hz].

M_from_m(mab, z=7.0, beta=-2)
    Convert apparent AB magnitude to absolute magnitude, applying a K-correction.
"""

import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy.constants import c
from astropy.cosmology import Planck15


def flux_to_AB(flux, flux_err=None, unit='jy'):
    """
    Convert flux density to AB magnitude.

    Parameters
    ----------
    flux : float or array
        Flux density in Jy or fν (erg/s/cm²/Hz).
    flux_err : float or array, optional
        Uncertainty on flux density.
    unit : str, {'jy','fnu'}
        Unit of input flux. 'jy' for Jansky, 'fnu' for erg/s/cm²/Hz.

    Returns
    -------
    mag : float, array, or uncertainties.ufloat/unumpy.uarray
        AB magnitude with optional uncertainties.
    """
    assert unit in ['fnu', 'jy'], "'unit' must be 'fnu' or 'jy'"

    if flux_err is not None:
        if np.size(flux) == 1:
            fluxpoint = ufloat(flux, flux_err)
        else:
            fluxpoint = unp.uarray(flux, flux_err)
    else:
        fluxpoint = flux

    if unit == 'jy':
        if flux_err is not None:
            mag = 2.5 * (23.0 - unp.log10(fluxpoint)) - 48.6
        else:
            mag = 2.5 * (23.0 - np.log10(fluxpoint)) - 48.6
    elif unit == 'fnu':
        if flux_err is not None:
            mag = 2.5 * (23.0 - unp.log10(fluxpoint / 1e-23)) - 48.6
        else:
            mag = 2.5 * (23.0 - np.log10(fluxpoint / 1e-23)) - 48.6
    return mag


def AB_to_flux(mag, mag_err=None, output_unit='jy'):
    """
    Convert AB magnitude to flux density.

    Parameters
    ----------
    mag : float or array
        AB magnitude(s).
    mag_err : float or array, optional
        Magnitude uncertainty. If 0 or None, returns plain float/array.
    output_unit : {'fnu','jy'}
        Desired flux unit.

    Returns
    -------
    flux : float, array, or uncertainties object
        Flux density in requested units.
    """
    assert output_unit in ['fnu', 'jy'], "'output_unit' must be 'fnu' or 'jy'"

    if mag_err is not None and np.any(np.asarray(mag_err) > 0):
        if np.size(mag) == 1:
            magpoint = ufloat(mag, mag_err)
        else:
            magpoint = unp.uarray(mag, mag_err)
    else:
        magpoint = mag  # just use raw float/array, no uncertainties

    # AB definition
    jy = 10**(23. - (magpoint + 48.6) / 2.5)
    fnu = jy * 1e-23

    return fnu if output_unit == 'fnu' else jy



def fnu_to_flam(lam, fnu, fnu_err=None):
    """
    Convert flux density from fν to fλ.

    Parameters
    ----------
    lam : float or array
        Wavelength in Å.
    fnu : float, array, or unumpy.uarray
        Flux density in erg/s/cm²/Hz.
    fnu_err : float or array, optional
        Uncertainty on fν.

    Returns
    -------
    flam : float, array, or unumpy.uarray
        Flux density in erg/s/cm²/Å.
    """
    if fnu_err is not None:
        spec = unp.uarray(fnu, fnu_err)
    else:
        spec = fnu

    c_ang = c.to('AA/s').value
    flam = (c_ang / (lam ** 2.0)) * spec
    return flam


def flam_to_fnu(lam, flam, flam_err=None):
    """
    Convert flux density from fλ to fν.

    Parameters
    ----------
    lam : float or array
        Wavelength in Å.
    flam : float, array, or unumpy.uarray
        Flux density in erg/s/cm²/Å.
    flam_err : float or array, optional
        Uncertainty on fλ.

    Returns
    -------
    fnu : float, array, or unumpy.uarray
        Flux density in erg/s/cm²/Hz.
    """
    if flam_err is not None:
        spec = unp.uarray(flam, flam_err)
    else:
        spec = flam

    c_ang = c.to('AA/s').value
    fnu = ((lam ** 2.0) / c_ang) * spec
    return fnu


def M_from_m(mab, z=7.0, beta=-2.0):
    """
    Convert apparent AB magnitude to absolute magnitude.

    Applies a distance modulus (Planck15 cosmology) and a simple
    power-law K-correction with slope beta.

    Parameters
    ----------
    mab : float or array
        Apparent AB magnitude.
    z : float
        Redshift.
    beta : float
        UV continuum slope (default = -2).

    Returns
    -------
    MAB : float or array
        Absolute AB magnitude.
    """
    DL = Planck15.luminosity_distance(z)  # in Mpc
    K_corr = (-beta - 1) * 2.5 * np.log10(1.0 + z)
    MAB = mab - 5.0 * (np.log10(DL.value * 1e6) - 1.0) + K_corr
    return MAB

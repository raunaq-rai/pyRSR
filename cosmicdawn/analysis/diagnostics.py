"""
diagnostics.py

Physical diagnostics from spectra and photometry.

This module provides estimators for:
    - UV absolute magnitudes (MUV) from spectrum or photometry
    - Line ratio diagnostics (e.g., R_Sanders = ([OIII]+[OII])/Hβ)
    - Dust reddening via the Balmer decrement E(B–V)

References
----------
- Kennicutt (1998): SFR calibrations
- Calzetti et al. (2000): Attenuation curves
- Sanders et al. (2016): Strong-line metallicity diagnostics
"""

import numpy as np
from astropy.constants import c
from astropy.cosmology import Planck15
from ..general.conversions import AB_to_flux, flux_to_AB


# --------------------------
# UV absolute magnitudes
# --------------------------

def MUV_from_spec(lam, flux, z, lam_ref=1500.0, width=100.0):
    """
    Estimate rest-frame UV absolute magnitude from a spectrum.

    Parameters
    ----------
    lam : ndarray
        Wavelength array [Å].
    flux : ndarray
        Flux density [erg/s/cm²/Å].
    z : float
        Redshift of the source.
    lam_ref : float, default=1500.0
        Reference rest-frame wavelength [Å].
    width : float, default=100.0
        Width of window [Å] to average around lam_ref.

    Returns
    -------
    MUV : float
        Absolute AB magnitude at lam_ref.
    """
    lam_rest = lam / (1.0 + z)
    mask = (lam_rest > lam_ref - width / 2) & (lam_rest < lam_ref + width / 2)
    if not np.any(mask):
        raise ValueError("No spectral coverage around reference wavelength.")

    lam_eff = np.mean(lam[mask])
    flux_eff = np.mean(flux[mask])

    # Convert fλ → fν
    fnu = (lam_eff**2 / c.to("AA/s").value) * flux_eff
    mag = flux_to_AB(fnu, unit="fnu")
    DL = Planck15.luminosity_distance(z).to("pc").value
    MUV = mag - 5 * (np.log10(DL) - 1)
    return MUV


def MUV_from_photo(mAB, z):
    """
    Convert apparent AB magnitude to absolute MUV.

    Parameters
    ----------
    mAB : float
        Apparent AB magnitude.
    z : float
        Redshift.

    Returns
    -------
    MUV : float
        Absolute AB magnitude.
    """
    DL = Planck15.luminosity_distance(z).to("pc").value
    return mAB - 5 * (np.log10(DL) - 1)


# --------------------------
# Line diagnostics
# --------------------------

def R_Sanders(OII_flux, OIII_flux, Hb_flux):
    """
    Compute R_Sanders metallicity diagnostic.

    Defined as:
        R = ( [OIII]5007 + [OIII]4959 + [OII]3727 ) / Hβ

    Parameters
    ----------
    OII_flux : float
        [OII] λ3727 flux [erg/s/cm²].
    OIII_flux : float
        Combined [OIII] λλ4959,5007 flux [erg/s/cm²].
    Hb_flux : float
        Hβ λ4861 flux [erg/s/cm²].

    Returns
    -------
    R : float
        Line ratio diagnostic.
    """
    if Hb_flux <= 0:
        raise ValueError("Hβ flux must be > 0 for diagnostic.")
    return (OII_flux + OIII_flux) / Hb_flux


# --------------------------
# Dust attenuation
# --------------------------

def calc_ebv(Ha_flux, Hb_flux, k_Ha=3.33, k_Hb=4.60, intrinsic_ratio=2.86):
    """
    Compute E(B–V) from the Balmer decrement.

    Parameters
    ----------
    Ha_flux : float
        Observed Hα flux [erg/s/cm²].
    Hb_flux : float
        Observed Hβ flux [erg/s/cm²].
    k_Ha : float, default=3.33
        Extinction curve value at Hα (Calzetti 2000).
    k_Hb : float, default=4.60
        Extinction curve value at Hβ.
    intrinsic_ratio : float, default=2.86
        Case B recombination ratio (T=10⁴ K, ne=10² cm⁻³).

    Returns
    -------
    ebv : float
        Color excess E(B–V).
    """
    if Hb_flux <= 0 or Ha_flux <= 0:
        raise ValueError("Fluxes must be > 0.")

    observed_ratio = Ha_flux / Hb_flux
    if observed_ratio <= 0:
        return 0.0

    ebv = (2.5 / (k_Hb - k_Ha)) * np.log10(observed_ratio / intrinsic_ratio)
    return max(ebv, 0.0)

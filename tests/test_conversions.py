import numpy as np
import pytest
from uncertainties import ufloat
from highzpy.general import conversions as conv  # adjust import to your package


def test_flux_to_AB_and_back_jy():
    """Test Jy ↔ AB magnitude round-trip."""
    flux_jy = 1.0  # Jy
    mag = conv.flux_to_AB(flux_jy, unit="jy")
    flux_back = conv.AB_to_flux(mag, output_unit="jy")
    assert np.isclose(flux_back, flux_jy, rtol=1e-10)


def test_flux_to_AB_and_back_fnu():
    """Test fnu ↔ AB magnitude round-trip."""
    flux_fnu = 1e-23  # erg/s/cm²/Hz ( = 1 Jy)
    mag = conv.flux_to_AB(flux_fnu, unit="fnu")
    flux_back = conv.AB_to_flux(mag, output_unit="fnu")
    assert np.isclose(flux_back, flux_fnu, rtol=1e-10)


def test_flux_to_AB_with_uncertainties():
    """Test propagation of uncertainties in flux_to_AB."""
    flux = ufloat(1.0, 0.1)  # Jy ± 0.1
    mag = conv.flux_to_AB(flux.nominal_value, flux.std_dev, unit="jy")
    assert hasattr(mag, "nominal_value")
    assert hasattr(mag, "std_dev")


def test_AB_to_flux_with_uncertainties():
    """Test propagation of uncertainties in AB_to_flux."""
    mag = ufloat(25.0, 0.1)
    flux = conv.AB_to_flux(mag.nominal_value, mag.std_dev, output_unit="jy")
    assert hasattr(flux, "nominal_value")
    assert hasattr(flux, "std_dev")


def test_fnu_flam_roundtrip():
    """Check that fnu ↔ flam round-trip is consistent."""
    lam = 1500.0  # Å
    fnu = 1e-29   # erg/s/cm²/Hz
    flam = conv.fnu_to_flam(lam, fnu)
    fnu_back = conv.flam_to_fnu(lam, flam)
    assert np.isclose(fnu, fnu_back, rtol=1e-10)


def test_M_from_m_known_value():
    """Check absolute magnitude calculation roughly matches expectations."""
    mab = 27.0
    z = 7.0
    MAB = conv.M_from_m(mab, z=z, beta=-2.0)
    # For z=7, DL ~ 68500 Mpc, expect M_AB ~ -20
    assert -25 < MAB < -15

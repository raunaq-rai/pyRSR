import numpy as np
import pytest
from uncertainties import ufloat
from cosmicdawn import conversions as conv
from astropy.cosmology import Planck15


@pytest.mark.parametrize("flux_jy, expected_mag", [
    (1.0, 8.90),     # 1 Jy → ~8.90 mag
    (3631.0, 0.0),   # 3631 Jy → 0 mag by definition
])
def test_flux_to_AB_jy_known_values(flux_jy, expected_mag):
    mag = conv.flux_to_AB(flux_jy, unit="jy")
    assert np.isclose(mag, expected_mag, atol=1e-2)


def test_AB_to_flux_jy_known_values():
    mag = 0.0
    flux = conv.AB_to_flux(mag, output_unit="jy")
    assert np.isclose(flux, 3631.0, rtol=1e-4)  # 0 mag = 3631 Jy


@pytest.mark.parametrize("flux_fnu", [1e-23, 5e-30])
def test_flux_to_AB_and_back_fnu(flux_fnu):
    mag = conv.flux_to_AB(flux_fnu, unit="fnu")
    flux_back = conv.AB_to_flux(mag, output_unit="fnu")
    assert np.isclose(flux_back, flux_fnu, rtol=1e-12)


def test_flux_to_AB_with_uncertainties():
    flux = ufloat(1.0, 0.1)  # Jy ± 0.1
    mag = conv.flux_to_AB(flux.nominal_value, flux.std_dev, unit="jy")
    assert np.isclose(mag.nominal_value, 8.90, atol=0.1)
    assert mag.std_dev > 0


def test_AB_to_flux_with_uncertainties():
    mag = ufloat(25.0, 0.1)
    flux = conv.AB_to_flux(mag.nominal_value, mag.std_dev, output_unit="jy")
    assert flux.nominal_value > 0
    assert flux.std_dev > 0


def test_fnu_flam_roundtrip():
    lam = 1500.0  # Å
    fnu = 1e-29
    flam = conv.fnu_to_flam(lam, fnu)
    fnu_back = conv.flam_to_fnu(lam, flam)
    assert np.isclose(fnu, fnu_back, rtol=1e-12)


def test_M_from_m_matches_distance_modulus():
    mab = 27.0
    z = 7.0
    beta = -2.0

    # direct calculation: M = m - DM + Kcorr
    DL = Planck15.luminosity_distance(z).value  # Mpc
    DM = 5 * (np.log10(DL * 1e6) - 1)           # distance modulus
    Kcorr = (-beta - 1) * 2.5 * np.log10(1 + z)
    expected_M = mab - DM + Kcorr

    MAB = conv.M_from_m(mab, z=z, beta=beta)
    assert np.isclose(MAB, expected_M, atol=1e-6)

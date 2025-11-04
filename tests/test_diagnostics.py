# tests/test_diagnostics.py
import numpy as np
import pytest
import PyRSR
from PyRSR import diagnostics


def test_MUV_from_spec_valid():
    """MUV_from_spec returns a finite value when spectrum covers 1500 Å rest-frame."""
    z = 6.0
    lam = np.linspace(1000, 20000, 5000)  # observed Å
    flux = np.full_like(lam, 1e-19)       # flat spectrum
    MUV = diagnostics.MUV_from_spec(lam, flux, z, lam_ref=1500.0, width=100.0)
    assert np.isfinite(MUV)
    assert -30 < MUV < -10  # plausible astrophysical range


def test_MUV_from_spec_out_of_range():
    """Raises if reference window not covered."""
    lam = np.linspace(5000, 6000, 1000)   # observed Å, but no 1500 Å rest-frame coverage
    flux = np.ones_like(lam) * 1e-19
    with pytest.raises(ValueError):
        diagnostics.MUV_from_spec(lam, flux, z=0.1, lam_ref=1500.0)


def test_MUV_from_photo_consistency():
    """Check MUV_from_photo scales correctly with distance."""
    mAB = 25.0
    z = 6.0
    MUV = diagnostics.MUV_from_photo(mAB, z)
    assert np.isfinite(MUV)
    # Adding distance modulus back should approximately return mAB
    DL = diagnostics.Planck15.luminosity_distance(z).to("pc").value
    mAB_back = MUV + 5 * (np.log10(DL) - 1)
    assert np.isclose(mAB_back, mAB, rtol=1e-6)


def test_R_Sanders_valid():
    """R_Sanders should compute (OII+OIII)/Hb correctly."""
    OII, OIII, Hb = 1.0, 2.0, 1.0
    R = diagnostics.R_Sanders(OII, OIII, Hb)
    assert np.isclose(R, 3.0)


def test_R_Sanders_invalid_Hb():
    """Raises if Hβ <= 0."""
    with pytest.raises(ValueError):
        diagnostics.R_Sanders(1.0, 1.0, 0.0)


@pytest.mark.parametrize("Ha,Hb,expected_positive", [
    (3.0, 1.0, True),   # ratio > intrinsic → positive E(B–V)
    (2.0, 1.0, False),  # ratio < intrinsic → E(B–V)=0
    (2.86, 1.0, False), # ratio == intrinsic → E(B–V)=0
])
def test_calc_ebv_behavior(Ha, Hb, expected_positive):
    ebv = diagnostics.calc_ebv(Ha, Hb)
    if expected_positive:
        assert ebv > 0
    else:
        assert ebv == 0.0



def test_calc_ebv_invalid_fluxes():
    """Raises if input fluxes are <= 0."""
    with pytest.raises(ValueError):
        diagnostics.calc_ebv(-1.0, 1.0)
    with pytest.raises(ValueError):
        diagnostics.calc_ebv(1.0, 0.0)


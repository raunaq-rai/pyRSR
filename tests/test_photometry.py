import numpy as np
import pytest
from cosmicdawn.general import photometry as phot

# --------------------------
# Filter + synthetic photometry
# --------------------------
def test_photo_from_filter_flat_spectrum(make_boxcar_filter):
    filt_path = make_boxcar_filter()
    wave = np.linspace(3500, 8500, 200)
    flux = np.ones_like(wave) * 1e-19

    lam_cent, flux_eff = phot.photo_from_filter(wave, flux, filt_path, wave_unit="AA")
    assert 5000 < lam_cent < 7000
    assert np.isclose(flux_eff, 1e-19, rtol=1e-6)


def test_get_filter_info_boxcar(make_boxcar_filter):
    filt_path = make_boxcar_filter(wmin=4000, wmax=8000)
    lam_cent, lam_err, lam_range = phot.get_filter_info(filt_path, output_unit="AA")

    assert np.isclose(lam_cent, 6000, atol=100)
    assert np.isclose(lam_range[0], 4000, atol=10)
    assert np.isclose(lam_range[1], 8000, atol=10)


def test_convolve_spectrum_boxcar(make_boxcar_filter):
    filt_path = make_boxcar_filter(wmin=4000, wmax=8000)
    lam = np.linspace(3000, 9000, 200)
    flux = lam * 0 + 2.0  # constant flux

    lam_cent, flux_eff = phot.convolve_spectrum(lam, flux, filt_path)
    assert np.isclose(flux_eff, 2.0, rtol=1e-6)


def test_calc_IRAC_color_and_EW_roundtrip(make_boxcar_filter):
    lam_eff = 45000.0  # Å
    filt_path = make_boxcar_filter(wmin=40000, wmax=50000)

    mAB = 25.0
    EW = 100.0
    z = 7.0

    color = phot.calc_IRAC_color(mAB, EW, z, lam_eff, filt_path)
    EW0, EW_obs = phot.calc_IRAC_EW(mAB, color, z, lam_eff, filt_path)

    assert np.isclose(EW0, EW, rtol=0.3)

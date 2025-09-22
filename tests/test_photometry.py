import numpy as np
import pytest
import tempfile
import os

from cosmicdawn.general import photometry as phot


def make_boxcar_filter(wmin=4000, wmax=8000, npts=2000):
    """
    Create a simple boxcar filter file for testing.
    Transmission = 1 between wmin–wmax Å, else 0.
    Returns the path to a temporary file.
    """
    lam = np.linspace(3000, 9000, npts)
    resp = np.where((lam >= wmin) & (lam <= wmax), 1.0, 0.0)
    data = np.column_stack([lam, resp])
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    np.savetxt(path, data)
    return path


# --------------------------
# Filter + synthetic photometry
# --------------------------

def test_photo_from_filter_flat_spectrum():
    filt_path = make_boxcar_filter()
    wave = np.linspace(3500, 8500, 200)
    flux = np.ones_like(wave) * 1e-19

    lam_cent, flux_eff = phot.photo_from_filter(wave, flux, filt_path, wave_unit="AA")
    assert 5000 < lam_cent < 7000  # central λ should be inside the box
    assert np.isclose(flux_eff, 1e-19, rtol=1e-6)  # flat spectrum returns same flux

    os.remove(filt_path)


def test_get_filter_info_boxcar():
    filt_path = make_boxcar_filter(4000, 8000)
    lam_cent, lam_err, lam_range = phot.get_filter_info(filt_path, output_unit="AA")

    # For symmetric boxcar, λ cent ~ midpoint
    assert np.isclose(lam_cent, 6000, atol=100)
    assert np.isclose(lam_range[0], 4000, atol=10)
    assert np.isclose(lam_range[1], 8000, atol=10)

    os.remove(filt_path)


def test_convolve_spectrum_boxcar():
    filt_path = make_boxcar_filter(4000, 8000)
    lam = np.linspace(3000, 9000, 200)
    flux = lam * 0 + 2.0  # constant flux

    lam_cent, flux_eff = phot.convolve_spectrum(lam, flux, filt_path)
    assert np.isclose(flux_eff, 2.0, rtol=1e-6)

    os.remove(filt_path)


# --------------------------
# Emission line
# --------------------------

def test_emission_line_peak_and_width():
    x = np.linspace(4950, 5050, 500)
    gauss = phot.emission_line(x, x0=5000, A=1.0, width=5.0, FWHM=False)

    # Peak should be near A
    assert np.isclose(gauss.max(), 1.0, rtol=1e-2)
    # Symmetry: mean weighted by flux ~ x0
    mean = np.sum(x * gauss) / np.sum(gauss)
    assert np.isclose(mean, 5000, atol=0.5)


# --------------------------
# IRAC color/EW (synthetic sanity checks)
# --------------------------

def test_calc_IRAC_color_and_EW_roundtrip():
    # Create dummy filter ~ flat transmission around lam_eff
    lam_eff = 45000.0  # Å
    filt_path = make_boxcar_filter(40000, 50000)

    mAB = 25.0
    EW = 100.0
    z = 7.0

    color = phot.calc_IRAC_color(mAB, EW, z, lam_eff, filt_path)
    EW0, EW_obs = phot.calc_IRAC_EW(mAB, color, z, lam_eff, filt_path)

    # Roundtrip should give back similar EW0
    assert np.isclose(EW0, EW, rtol=0.3)  # allow slack for simplifications

    os.remove(filt_path)

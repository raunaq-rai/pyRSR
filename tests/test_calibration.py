import numpy as np
import pytest
from cosmicdawn.analysis import calibration

# --------------------------
# Slit loss correction tests
# --------------------------

def test_correct_slitloss_point_source():
    """For a point source in poor seeing, slit fraction should be <1."""
    wave = np.linspace(4000, 5000, 10)
    flux = np.ones_like(wave)
    slit_width = 1.0   # arcsec
    seeing = 1.0       # arcsec

    flux_corr, slit_fraction = calibration.correct_slitloss(wave, flux, slit_width, seeing)
    assert flux_corr.shape == flux.shape
    assert 0 < slit_fraction < 1   # should lose some flux
    # Corrected flux should be higher than input
    assert np.all(flux_corr > flux)


def test_correct_slitloss_extended_source():
    """Adding source size increases losses (fraction smaller)."""
    wave = np.linspace(4000, 5000, 5)
    flux = np.full_like(wave, 2.0)
    slit_width = 1.0
    seeing = 0.5
    flux_corr_pt, frac_pt = calibration.correct_slitloss(wave, flux, slit_width, seeing, source_size=0.0)
    flux_corr_ext, frac_ext = calibration.correct_slitloss(wave, flux, slit_width, seeing, source_size=1.0)
    assert frac_ext < frac_pt
    assert np.all(flux_corr_ext > flux_corr_pt)


def test_correct_slitloss_wide_slit():
    """With a very wide slit, fraction should →1 and flux_corr≈flux."""
    wave = np.linspace(5000, 6000, 3)
    flux = np.linspace(1.0, 3.0, 3)
    slit_width = 100.0
    seeing = 1.0
    flux_corr, slit_fraction = calibration.correct_slitloss(wave, flux, slit_width, seeing)
    assert np.isclose(slit_fraction, 1.0, atol=1e-3)
    assert np.allclose(flux_corr, flux, rtol=1e-3)


# --------------------------
# Flux calibration tests
# --------------------------

def test_calibrate_etc_spec_basic_scaling():
    """Flux should scale linearly with target/ETC SNR ratio if no new exptime."""
    wave = np.linspace(4000, 5000, 4)
    flux = np.ones_like(wave)
    etc_snr, exptime, target_snr = 10.0, 1000.0, 20.0
    flux_cal, snr_scaled = calibration.calibrate_etc_spec(wave, flux, etc_snr, exptime, target_snr)
    assert np.allclose(flux_cal, flux * 2.0)
    assert snr_scaled == target_snr


def test_calibrate_etc_spec_with_new_exptime():
    """SNR should scale as sqrt(t) when new_exptime provided."""
    wave = np.array([4000, 5000])
    flux = np.array([1.0, 2.0])
    etc_snr, exptime = 10.0, 1000.0
    target_snr = 20.0
    new_exptime = 4000.0  # 4× longer → 2× SNR
    flux_cal, snr_scaled = calibration.calibrate_etc_spec(
        wave, flux, etc_snr, exptime, target_snr, new_exptime=new_exptime
    )
    assert np.isclose(snr_scaled, 20.0, atol=1e-6)
    # Scale factor should bring flux back near original
    assert np.allclose(flux_cal, flux, rtol=1e-6)


def test_calibrate_etc_spec_increasing_target_snr():
    """Flux scaling should increase with higher target SNR."""
    wave = np.linspace(4000, 5000, 2)
    flux = np.array([1.0, 1.0])
    f_low, _ = calibration.calibrate_etc_spec(wave, flux, 10, 1000, target_snr=5)
    f_high, _ = calibration.calibrate_etc_spec(wave, flux, 10, 1000, target_snr=20)
    assert np.all(f_high > f_low)


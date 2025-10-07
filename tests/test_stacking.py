# tests/test_stacking.py
import numpy as np
import pytest
from pyRSR import stacking


def test_stack_spectra_mean_and_median():
    # Fake data: 3 spectra × 5 pixels
    flux = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
    ])
    err = np.ones_like(flux) * 0.1

    # Median stack
    f_med, std_med, err_med, n_med = stacking.stack_spectra(flux, err, op="median")
    assert f_med.shape == (5,)
    assert np.allclose(f_med, [2, 3, 4, 5, 6])
    assert np.all(n_med == 3)

    # Mean stack
    f_mean, std_mean, err_mean, n_mean = stacking.stack_spectra(flux, err, op="mean")
    assert np.allclose(f_mean, [2, 3, 4, 5, 6])
    assert np.all(n_mean == 3)

    # Invalid op raises error
    with pytest.raises(ValueError):
        stacking.stack_spectra(flux, op="invalid")


def test_bin_1d_spec_mean_and_sum():
    lam = np.arange(10)
    flux = np.arange(10).astype(float)
    err = np.ones_like(flux) * 0.5

    # Factor 2 binning, mean
    lam_b, flux_b, err_b = stacking.bin_1d_spec(lam, flux, err=err, factor=2, method="mean")
    assert lam_b.shape[0] == 5
    assert np.allclose(flux_b, [0.5, 2.5, 4.5, 6.5, 8.5])
    assert np.allclose(err_b, 0.5)

    # Factor 5 binning, sum
    lam_b, flux_b = stacking.bin_1d_spec(lam, flux, factor=5, method="sum")
    assert lam_b.shape[0] == 2
    assert np.allclose(flux_b, [10, 35])

    # Invalid method raises error
    with pytest.raises(ValueError):
        stacking.bin_1d_spec(lam, flux, factor=2, method="bad")


def test_bin_2d_spec_modes():
    spec2d = np.arange(16).reshape(4, 4)

    # Bin along lam (columns)
    binned_lam = stacking.bin_2d_spec(spec2d, factor=2, axis="lam")
    assert binned_lam.shape == (4, 2)

    # Bin along spatial (rows)
    binned_spatial = stacking.bin_2d_spec(spec2d, factor=2, axis="spatial")
    assert binned_spatial.shape == (2, 4)

    # Bin along both
    binned_all = stacking.bin_2d_spec(spec2d, factor=2, axis="all")
    assert binned_all.shape == (2, 2)

    # Invalid axis raises error
    with pytest.raises(ValueError):
        stacking.bin_2d_spec(spec2d, factor=2, axis="wrong")


def test_custom_sigma_clip():
    arr = np.array([1, 2, 3, 100])  # outlier at 100
    clipped, low, high, idx = stacking.custom_sigma_clip(arr, low=2, high=2)
    assert np.all(clipped < 100)
    assert 1 in clipped and 3 in clipped
    assert isinstance(low, float) and isinstance(high, float)
    assert np.all(idx < len(arr))

    # Invalid op raises error
    with pytest.raises(ValueError):
        stacking.custom_sigma_clip(arr, op="bad")


def test_generate_line_mask_basic():
    lam = np.linspace(1000, 7000, 1000)
    mask = stacking.generate_line_mask(lam, z_spec=0.0, dv=2000)
    assert mask.shape == lam.shape
    # At least some points should be masked near Hα ~6565 Å
    assert mask[lam.searchsorted(6565)] == 1
    # Values outside line windows should be 0
    assert np.all((mask == 0) | (mask == 1))


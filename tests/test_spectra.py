import numpy as np
import matplotlib
matplotlib.use("Agg")  # use headless backend
import matplotlib.pyplot as plt

import pytest

from cosmicdawn.general import conversions
from cosmicdawn.analysis import spectra


@pytest.fixture
def fake_spectra_dict():
    """Synthetic msaexp-style spectra for testing."""
    return {
        "123": {
            "prism-clear": {
                "lam": np.linspace(0.8, 2.0, 20),       # microns
                "flux": np.linspace(1.0, 2.0, 20) * 1e-3,
                "err": np.ones(20) * 1e-4,
            },
            "g235m": {
                "lam": np.linspace(1.7, 2.3, 15),
                "flux": np.linspace(2.0, 3.0, 15) * 1e-3,
                "err": np.ones(15) * 1e-4,
            },
            "g395m": {
                "lam": np.linspace(2.8, 3.2, 10),
                "flux": np.linspace(3.0, 4.0, 10) * 1e-3,
                "err": np.ones(10) * 1e-4,
            },
        }
    }


# -------------------
# Spectrum retrieval
# -------------------

def test_get_prism_spectrum_units(fake_spectra_dict):
    lam, flux, err = spectra.get_prism_spectrum(fake_spectra_dict, "123")
    assert lam.shape == flux.shape == err.shape
    assert lam.min() >= 0.8 and lam.max() <= 2.0

    # Check conversion to flam
    lam_flam, flux_flam, err_flam = spectra.get_prism_spectrum(fake_spectra_dict, "123", units="flam")
    assert np.allclose(lam, lam_flam)  # wavelength unchanged
    assert np.all(flux_flam > 0)       # fλ should be positive


def test_get_grating_spectrum(fake_spectra_dict):
    lam, flux, err = spectra.get_grating_spectrum(fake_spectra_dict, "123")
    assert lam.shape == flux.shape == err.shape
    # Ensure prism not included
    assert lam.min() > 1.6
    # Sorted
    assert np.all(np.diff(lam) > 0)


# -------------------
# Spectrum plotting
# -------------------

def test_plot_prism_spectrum(fake_spectra_dict):
    ax = spectra.plot_prism_spectrum(fake_spectra_dict, "123", color="blue")
    assert isinstance(ax, plt.Axes)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_color() == "blue"


def test_plot_grating_spectrum(fake_spectra_dict):
    ax = spectra.plot_grating_spectrum(fake_spectra_dict, "123", color="red")
    assert isinstance(ax, plt.Axes)
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_color() == "red"


def test_plot_msaexp_dispatch(fake_spectra_dict):
    ax1 = spectra.plot_msaexp_spectrum(fake_spectra_dict, "123", which="prism")
    ax2 = spectra.plot_msaexp_spectrum(fake_spectra_dict, "123", which="grating")
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)

    with pytest.raises(ValueError):
        spectra.plot_msaexp_spectrum(fake_spectra_dict, "123", which="nonsense")

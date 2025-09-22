import numpy as np
import matplotlib
matplotlib.use("Agg")  # prevent GUI popup
import matplotlib.pyplot as plt

import pytest

from cosmicdawn.analysis import composites


def test_read_composite_stub_returns_empty():
    lam, flux, model = composites.read_composite("demo")
    # Stub should return empty arrays for now
    assert isinstance(lam, np.ndarray)
    assert isinstance(flux, np.ndarray)
    assert isinstance(model, np.ndarray)
    assert lam.size == flux.size == model.size == 0


def test_plot_composite_basic():
    # Fake wavelength grid (obs frame in microns)
    lam = np.linspace(0.8, 2.5, 50)
    flux = np.sin(lam * 5) * 1e-3  # arbitrary
    model_flux = flux * 0.9

    ax = composites.plot_composite(lam, flux, model_flux=model_flux, frame="obs")
    assert isinstance(ax, plt.Axes)
    # Check labels
    assert "λ" in ax.get_xlabel()
    assert "Flux" in ax.get_ylabel()
    # Legend should contain both labels
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Composite" in labels
    assert "Model" in labels


def test_plot_composite_rest_frame():
    lam = np.linspace(1000, 3000, 30)  # rest-frame Å
    flux = np.ones_like(lam) * 1e-2

    ax = composites.plot_composite(lam, flux, frame="rest")
    assert ax.get_xlabel().startswith("λ_rest")


def test_plot_composite_no_model(monkeypatch):
    lam = np.linspace(1.0, 2.0, 20)
    flux = np.linspace(0.1, 0.2, 20)

    ax = composites.plot_composite(lam, flux, model_flux=None)
    lines = ax.get_lines()
    assert len(lines) == 1  # only composite line should be plotted


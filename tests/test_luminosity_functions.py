"""
tests/test_luminosity_functions.py

Test suite for cosmicdawn.models.luminosity_functions
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # avoid GUI in pytest
import matplotlib.pyplot as plt

from cosmicdawn.models import luminosity_functions as lf


# --------------------------
# Basic functional tests
# --------------------------

def test_schechter_behavior():
    M = np.linspace(-24, -16, 50)
    phi = lf.schechter(M, M_star=-21, phi_star=1e-3, alpha=-1.8)

    # Values should be non-negative and finite
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0)

    # Sanity check: brighter mags (M=-24) should have lower density than fainter (M=-16)
    assert phi[0] < phi[-1]

    # Check normalization around M_star: φ should be of order phi_star
    idx_star = np.argmin(np.abs(M - -21))
    assert 1e-5 < phi[idx_star] < 1e-2



def test_dpl_behavior():
    M = np.linspace(-24, -16, 50)
    phi = lf.dpl(M, M_star=-21, phi_star=1e-3, alpha=-1.8, beta=-3.5)

    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0)


@pytest.mark.parametrize("z", [6, 7, 8])
def test_mason15_and_bouwens15(z):
    M = np.linspace(-24, -16, 20)
    phi_m = lf.mason15(M, z=z)
    phi_b = lf.bouwens15(M, z=z)

    assert np.all(phi_m >= 0)
    assert np.all(phi_b >= 0)


def test_donnan_and_harikane_placeholders():
    M = np.linspace(-24, -16, 20)
    phi_d = lf.donnan23(M, z=10)
    phi_h = lf.harikane23(M, z=10)

    assert np.all(phi_d >= 0)
    assert np.all(phi_h >= 0)


# --------------------------
# Plotting test
# --------------------------

def test_plot_all_models(tmp_path):
    """
    Smoke test: generate a plot of all LFs at z~6–10.
    Saves to a temporary file so we can check plotting works.
    """
    M = np.linspace(-24, -16, 200)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(M, lf.schechter(M, -21, 1e-3, -1.8), label="Schechter")
    ax.plot(M, lf.dpl(M, -21, 1e-3, -1.8, -3.5), label="DPL")
    ax.plot(M, lf.mason15(M, z=6), label="Mason+15 (z=6)")
    ax.plot(M, lf.bouwens15(M, z=6), label="Bouwens+15 (z=6)")
    ax.plot(M, lf.donnan23(M, z=10), label="Donnan+23 (z=10)")
    ax.plot(M, lf.harikane23(M, z=10), label="Harikane+23 (z=10)")

    ax.set_yscale("log")
    ax.set_xlabel("M_UV [AB mag]")
    ax.set_ylabel("φ(M) [Mpc⁻³ mag⁻¹]")
    ax.legend(fontsize=8)

    outpath = tmp_path / "lf_test.png"
    fig.savefig(outpath)
    assert outpath.exists()


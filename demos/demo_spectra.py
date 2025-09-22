"""
Demo script for spectra.py

This script generates complicated synthetic spectra and
displays the plots interactively (with error bands).
Run it directly with:

    python demos/demo_spectra.py
"""

import numpy as np
import matplotlib.pyplot as plt
from cosmicdawn.analysis import spectra

# -----------------
# Simulated spectra
# -----------------

def make_fake_spectra():
    rng = np.random.default_rng(42)

    lam_prism = np.linspace(0.8, 2.0, 200)  # µm
    flux_prism = 1e-3 * (np.sin(5*lam_prism) + 1.5) + rng.normal(0, 1e-4, lam_prism.size)
    err_prism = np.ones_like(lam_prism) * 1e-4

    lam_g235m = np.linspace(1.7, 2.3, 150)
    flux_g235m = 1.5e-3 * np.exp(-0.5*((lam_g235m-2.0)/0.05)**2) + rng.normal(0, 5e-5, lam_g235m.size)
    err_g235m = np.ones_like(lam_g235m) * 5e-5

    lam_g395m = np.linspace(2.8, 3.2, 120)
    flux_g395m = 8e-4 + 2e-4*np.cos(15*lam_g395m) + rng.normal(0, 5e-5, lam_g395m.size)
    err_g395m = np.ones_like(lam_g395m) * 5e-5

    return {
        "demo_source": {
            "prism-clear": {"lam": lam_prism, "flux": flux_prism, "err": err_prism},
            "g235m": {"lam": lam_g235m, "flux": flux_g235m, "err": err_g235m},
            "g395m": {"lam": lam_g395m, "flux": flux_g395m, "err": err_g395m},
        }
    }


def main():
    spectra_dict = make_fake_spectra()

    # Prism
    ax1 = spectra.plot_prism_spectrum(spectra_dict, "demo_source", color="blue")
    ax1.set_title("Simulated Prism Spectrum")

    # Grating
    ax2 = spectra.plot_grating_spectrum(spectra_dict, "demo_source", color="red")
    ax2.set_title("Simulated Grating Spectrum")

    # MSAEXP wrapper
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    spectra.plot_msaexp_spectrum(spectra_dict, "demo_source", which="prism", ax=axs[0], color="green")
    axs[0].set_title("MSAEXP Wrapper – Prism")
    spectra.plot_msaexp_spectrum(spectra_dict, "demo_source", which="grating", ax=axs[1], color="orange")
    axs[1].set_title("MSAEXP Wrapper – Grating")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


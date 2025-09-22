"""
igm.py

Intergalactic medium (IGM) transmission utilities.

Implements simple models of line blanketing and Lyman-series absorption,
based on Inoue et al. (2014).

Functions
---------
get_Inoue14_trans(rest_wavs, z_obs, coef_file=None)
    Compute IGM transmission curves using Inoue+14 parameterization.

get_IGM_absorption(z_obs, lam_min=500, lam_max=3000, dlam=0.5, coef_file=None)
    Convenience wrapper to get wavelength and transmission arrays.
"""

import numpy as np
import os


# --------------------------
# Inoue+2014 model
# --------------------------

def get_Inoue14_trans(rest_wavs, z_obs, coef_file=None):
    """
    Calculate IGM transmission using Inoue et al. (2014) model.

    Parameters
    ----------
    rest_wavs : array_like
        Rest-frame wavelengths [Å].
    z_obs : float
        Observed redshift.
    coef_file : str, optional
        Path to coefficient file (Inoue+14 Table 2).
        If None, uses a bundled minimal fallback.

    Returns
    -------
    trans : array
        Transmission fraction (exp(-tau)).
    """
    rest_wavs = np.atleast_1d(rest_wavs)

    # Load coefficients
    if coef_file is None:
        # Minimal synthetic fallback coefficients:
        # col0 = line ID, col1 = wavelength, col2+ = fit parameters
        coef_file = os.path.join(os.path.dirname(__file__),
                                 "data", "inoue2014_table2.txt")
        if not os.path.exists(coef_file):
            raise FileNotFoundError(
                f"Coefficient file not found: {coef_file}\n"
                "Provide with coef_file=... or add to cosmicdawn/general/data/."
            )
    coefs = np.loadtxt(coef_file)

    # Tau arrays
    tau_LAF_LS = np.zeros((coefs.shape[0], rest_wavs.shape[0]))
    tau_DLA_LS = np.zeros_like(tau_LAF_LS)
    tau_LAF_LC = np.zeros(rest_wavs.shape[0])
    tau_DLA_LC = np.zeros(rest_wavs.shape[0])

    # Loop over Lyman-series lines (similar structure to Inoue14 Appendix)
    for j in range(coefs.shape[0]):
        lam_j = coefs[j, 1]

        if z_obs < 1.2:
            mask = ((rest_wavs * (1. + z_obs) > lam_j) &
                    (rest_wavs * (1. + z_obs) < (1 + z_obs) * lam_j))
            tau_LAF_LS[j, mask] = (coefs[j, 2] *
                                   (rest_wavs[mask] * (1. + z_obs) / lam_j) ** 1.2)

        else:
            # Simple toy extension — in practice, use full Table 2 definitions
            mask = ((rest_wavs * (1. + z_obs) > lam_j) &
                    (rest_wavs * (1. + z_obs) < (1 + z_obs) * lam_j))
            tau_LAF_LS[j, mask] = (coefs[j, 2] *
                                   (rest_wavs[mask] * (1. + z_obs) / lam_j) ** 2.0)

        # DLA component (toy)
        tau_DLA_LS[j, mask] = 0.1 * tau_LAF_LS[j, mask]

    # Lyman continuum (very simplified)
    tau_LAF_LC[rest_wavs < 912.] = (rest_wavs[rest_wavs < 912.] / 912.) ** 3
    tau_DLA_LC[rest_wavs < 912.] = 0.5 * tau_LAF_LC[rest_wavs < 912.]

    tau = np.sum(tau_LAF_LS, axis=0) + np.sum(tau_DLA_LS, axis=0) + tau_LAF_LC + tau_DLA_LC
    return np.exp(-tau)


def get_IGM_absorption(z_obs, lam_min=500, lam_max=3000, dlam=0.5, coef_file=None):
    """
    Generate IGM transmission curve over a wavelength grid.

    Parameters
    ----------
    z_obs : float
        Observed redshift.
    lam_min, lam_max : float
        Wavelength range [Å].
    dlam : float
        Step size [Å].
    coef_file : str, optional
        Path to coefficient file.

    Returns
    -------
    lam_obs : array
        Observed-frame wavelengths [Å].
    trans : array
        Transmission fraction.
    """
    rest_wavs = np.arange(lam_min, lam_max, dlam)
    trans = get_Inoue14_trans(rest_wavs, z_obs, coef_file=coef_file)
    lam_obs = rest_wavs * (1. + z_obs)
    return lam_obs, trans

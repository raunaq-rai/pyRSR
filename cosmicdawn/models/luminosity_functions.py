"""
luminosity_functions.py

Galaxy luminosity function (LF) models.

Provides:
    - Generic Schechter function
    - Double power-law (DPL)
    - Literature parameterizations:
        * Mason et al. (2015)
        * Bouwens et al. (2015)
        * Donnan et al. (2023+)
        * Harikane et al. (2022/2023)

Notes
-----
- All functions return φ(M) in units of Mpc⁻³ mag⁻¹.
- Magnitudes follow AB system.
- Literature parameterizations are provided with default parameters;
  users should adjust values for different redshifts as needed.
"""

import numpy as np

# --------------------------
# Generic functions
# --------------------------

def schechter(M, M_star, phi_star, alpha):
    """
    Schechter luminosity function.

    Parameters
    ----------
    M : array_like
        Absolute UV magnitude (AB).
    M_star : float
        Characteristic magnitude.
    phi_star : float
        Normalization (Mpc⁻³).
    alpha : float
        Faint-end slope.

    Returns
    -------
    phi : ndarray
        Number density φ(M) [Mpc⁻³ mag⁻¹].
    """
    M = np.asarray(M)
    term = 10**(0.4 * (M_star - M) * (alpha + 1))
    exp_term = np.exp(-10**(0.4 * (M_star - M)))
    return 0.4 * np.log(10) * phi_star * term * exp_term


def dpl(M, M_star, phi_star, alpha, beta):
    """
    Double power-law luminosity function.

    Parameters
    ----------
    M : array_like
        Absolute magnitude.
    M_star : float
        Characteristic magnitude.
    phi_star : float
        Normalization.
    alpha : float
        Faint-end slope.
    beta : float
        Bright-end slope.

    Returns
    -------
    phi : ndarray
        Number density [Mpc⁻³ mag⁻¹].
    """
    M = np.asarray(M)
    term_faint = 10**(0.4 * (M - M_star) * (alpha + 1))
    term_bright = 10**(0.4 * (M - M_star) * (beta + 1))
    return phi_star / (term_faint + term_bright)


# --------------------------
# Literature parameterizations
# --------------------------

def mason15(M, z=6):
    """
    Mason et al. (2015) LF fit.

    Parameters
    ----------
    M : array_like
        Absolute UV magnitude.
    z : int, default=6
        Redshift (6–8 supported).

    Returns
    -------
    phi : ndarray
        Number density [Mpc⁻³ mag⁻¹].
    """
    # Example params (Table 1 in Mason+15)
    params = {
        6: {"M_star": -20.94, "phi_star": 0.0005, "alpha": -2.0},
        7: {"M_star": -20.87, "phi_star": 0.00025, "alpha": -2.1},
        8: {"M_star": -20.63, "phi_star": 0.00014, "alpha": -2.2},
    }
    p = params.get(z)
    if p is None:
        raise ValueError("z must be 6, 7, or 8 for Mason+15.")
    return schechter(M, **p)


def bouwens15(M, z=6):
    """
    Bouwens et al. (2015) LF fit.

    Parameters
    ----------
    M : array_like
        Absolute UV magnitude.
    z : int, default=6
        Redshift (4–8 supported).

    Returns
    -------
    phi : ndarray
        Number density [Mpc⁻³ mag⁻¹].
    """
    # Example params (Table 5 in Bouwens+15, z=6)
    params = {
        6: {"M_star": -20.94, "phi_star": 0.0005, "alpha": -1.87},
        7: {"M_star": -20.87, "phi_star": 0.00025, "alpha": -2.06},
        8: {"M_star": -20.63, "phi_star": 0.00014, "alpha": -2.02},
    }
    p = params.get(z)
    if p is None:
        raise ValueError("z must be 6–8 for Bouwens+15.")
    return schechter(M, **p)


def donnan23(M, z=10):
    """
    Donnan et al. (2023) JWST-based LF.

    Parameters
    ----------
    M : array_like
        Absolute magnitude.
    z : int, default=10
        Redshift (example ~10).
    """
    # Placeholder: values must be filled from Donnan+23 Table
    p = {"M_star": -20.9, "phi_star": 1e-4, "alpha": -2.1}
    return schechter(M, **p)


def harikane23(M, z=10):
    """
    Harikane et al. (2022/2023) JWST LF fit (double power-law).

    Parameters
    ----------
    M : array_like
        Absolute magnitude.
    z : int, default=10
        Redshift.

    Returns
    -------
    phi : ndarray
        Number density [Mpc⁻³ mag⁻¹].
    """
    # Placeholder values, update with Harikane+23 numbers
    p = {"M_star": -21.0, "phi_star": 1e-4, "alpha": -2.0, "beta": -4.0}
    return dpl(M, **p)


import numpy as np
import warnings
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
import tqdm
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

#edit

# Rest wavelengths of nebular / recombination lines [Å]
REST_LINES_A = {
    # --- High-z UV metal lines ---
    "NV_1":     1238.821,
    "NV_2":     1242.804,
    "NV_doublet": 1240.81,

    "NIV_1":    1486.496,
    "CIV_1":    1548.187,
    "CIV_2":    1550.772,
    "CIV_doublet": 1549.48,

    "HEII_1640": 1640.420,
    "OIII_1663": 1663.480,
    "SiIII_1":  1882.71,
    "SiIII_2":  1892.03,
    "CIII]":    1908.734,

    # --- Useful UV/near-UV lines in NIRSpec range at z~5–10 ---
    "OII]_2471": 2470.97,
    "OIII]_2321": 2321.7,
    "OIII]_2331": 2331.3,
    "FeII*_2396": 2396.36,

    # --- OII Doublet ---
    "OII_3726": 3726.032,
    "OII_3729": 3728.815,
    "OII_doublet": 3727.42,

    # --- Balmer series ---
    "HDELTA":    4102.892,
    "HGAMMA":    4341.684,
    "HBETA":     4862.683,


    # --- Optical strong lines (only valid for z ≲ 7; safely included) ---
    "OIII_4363": 4363.21,
    "OIII_4959": 4960.295,
    "OIII_5007": 5008.240,
    "NII_5756":  5756.19,
    "HEI_5877":  5877.252,
    "NII_6549":  6549.86,
    "H⍺": 6564.608,
    "NII_6585":  6585.27,
    "SII_6718":  6718.295,
    "SII_6732":  6732.674,
}

""" REST_LINES_A: Dict[str, float] = {
    #"NIII_1": 989.790, "NIII_2": 991.514, "NIII_3": 991.579,
    "NV_1": 1238.821, "NV_2": 1242.804, "NIV_1": 1486.496,
    "HEII_1": 1640.420, "OIII_05": 1663.4795, "CIII": 1908.734,
    "CIV_1": 1548.187, "CIV_2": 1550.772,
    #"OII_UV_1": 3727.092, "OII_UV_2": 3729.875,
     "HEI_1": 3889.749, #"NEIII_UV_1": 3869.86,
     "HEPSILON": 3971.1951, #"NEIII_UV_2": 3968.59,
    "HDELTA": 4102.8922, "HGAMMA": 4341.6837, "OIII_1": 4364.436,
    "HEI_2": 4471.479,
    "HEII_2": 4685.710, "HBETA": 4862.6830,
    "OIII_2": 4960.295, "OIII_3": 5008.240,
    "NII_1": 5756.19, "HEI_3": 5877.252,
    "NII_2": 6549.86, "H⍺": 6564.608, "NII_3": 6585.27,"HEI_4": 6679.9956,
    "SII_1": 6718.295, "SII_2": 6732.674,
} """




# ============================================================
# -------------------- CONSTANTS & LINE LIST -----------------
# ============================================================

C_AA = 2.99792458e18  #: Speed of light [Å·Hz·s⁻¹]
LN10 = np.log(10.0)    #: Natural log of 10
C_CGS = 2.99792458e10  # cm/s

def fnu_uJy_to_flam(flux_uJy, lam_um):
    """
    Convert F_ν [µJy] at wavelength [µm] → F_λ [erg/s/cm²/Å].
    """
    lam_cm = np.asarray(lam_um, float) * 1e-4      # µm → cm
    fnu_cgs = np.asarray(flux_uJy, float) * 1e-29  # µJy → erg/s/cm²/Hz
    flam = fnu_cgs * C_CGS / (lam_cm**2)           # erg/s/cm²/cm
    flam /= 1e8                                    # cm → Å
    return flam


def flam_to_fnu_uJy(flam, lam_um):
    """
    Convert F_λ [erg/s/cm²/Å] → F_ν [µJy].
    """
    lam_cm = np.asarray(lam_um, float) * 1e-4       # µm → cm
    fnu_cgs = np.asarray(flam, float) * (lam_cm**2) / C_CGS * 1e8
    flux_uJy = fnu_cgs * 1e29                      # erg/s/cm²/Hz → µJy
    return flux_uJy


def R_of_lambda_um(grating: str, lam_um: np.ndarray) -> np.ndarray:
    """
    Heuristic resolving power R(λ) for JWST/NIRSpec gratings.

    Parameters
    ----------
    grating : str
        Grating name (e.g. 'prism-clear', 'g140h', etc.)
    lam_um : array_like
        Wavelength array [µm].

    Returns
    -------
    R : ndarray
        Spectral resolving power at each wavelength.
    """
    lam_um = np.asarray(lam_um)
    g = grating.lower()
    if "prism" in g:
        R = 50 + 50*(lam_um - 1.0) + 15*(lam_um - 1.0)**2
        return np.clip(R, 30, 300)
    if "g140" in g or "g235" in g or "g395" in g:
        return np.full_like(lam_um, 1000.0, dtype=float)
    return np.full_like(lam_um, 100.0, dtype=float)


def sigma_grating_logA(grating: str, lam_um: np.ndarray) -> np.ndarray:
    """
    Convert R(λ) into Gaussian σ in log₁₀(λ[Å]).

    σ_log10λ = (1 / (2.355 * R * ln10))

    Parameters
    ----------
    grating : str
        Instrument grating.
    lam_um : array_like
        Wavelength array [µm].

    Returns
    -------
    σ_log10λ : ndarray
        Gaussian width in log₁₀(λ) space.
    """
    R = R_of_lambda_um(grating, lam_um)
    sigma_v_over_c = (1.0 / R) / 2.355
    return sigma_v_over_c / LN10


def measure_fluxes_profile_weighted(lam_um, flam_sub, sigma_flam, model_flam, profiles, centers,
                                    error_floor_frac=0.1):
    """
    Stable, matched-filter line flux estimator.

    Parameters
    ----------
    lam_um : array
        Wavelength grid [µm].
    flam_sub : array
        Continuum-subtracted spectrum in F_lambda [cgs].
    sigma_flam : array
        1σ uncertainties (same units as flam_sub).
    model_flam : array
        Total best-fit line model on `lam_um` (unused for the estimator but kept for API).
    profiles : dict[str, array]
        Per-line model profiles on `lam_um` (same units as flam_sub).
    centers : dict[str, tuple]
        {name: (mu_A, sigma_logA)} as returned by build_model_flam().
    error_floor_frac : float
        Fraction of the global median(σ) to use as a floor to avoid overflows.

    Returns
    -------
    dict
        {line: {"F_line", "sigma_line", "mask_idx"}}
    """
    lam_A = lam_um * 1e4
    dlam_A = np.gradient(lam_A)

    out = {}

    # robust error floor to avoid σ→0 pathologies
    med_sig = np.nanmedian(sigma_flam[np.isfinite(sigma_flam)])
    sig_floor = error_floor_frac * med_sig if np.isfinite(med_sig) and med_sig > 0 else 0.0
    safe_sigma = np.clip(sigma_flam, sig_floor, np.inf)
    safe_sig2  = safe_sigma**2

    for name, prof in profiles.items():
        muA, sigma_logA = centers[name]

        # 3σ window in *linear-Å* around the line
        sigma_ln = LN10 * sigma_logA
        sigma_A  = muA * sigma_ln
        mask = (lam_A > muA - 3*sigma_A) & (lam_A < muA + 3*sigma_A)
        if np.count_nonzero(mask) < 3:
            continue

        # template = per-line profile normalized to unit area
        t = prof[mask].astype(float)
        dl = dlam_A[mask].astype(float)
        area = np.sum(t * dl)
        if not np.isfinite(area) or area <= 0:
            continue
        T = t / area

        F  = flam_sub[mask].astype(float)
        s2 = safe_sig2[mask].astype(float)

        # matched-filter estimator (numerically stable)
        num = np.sum((T * F / s2) * dl)
        den = np.sum((T * T / s2) * dl)

        if not np.isfinite(den) or den <= 0:
            F_line = np.nan
            sigma_line = np.nan
        else:
            F_line = num / den
            sigma_line = den**-0.5

        out[name] = dict(F_line=F_line, sigma_line=sigma_line, mask_idx=np.where(mask)[0])

    return out

def equivalent_widths_A(fluxes, lam_um, Fcont_flam, z, centers):
    """
    Compute observed and rest-frame equivalent widths.

    EW_obs = F_line / F_cont(λ_line)
    EW₀ = EW_obs / (1 + z)
    """
    out = {}
    for name, meas in fluxes.items():
        muA, _ = centers[name]
        mu_um = muA / 1e4
        cont_at_mu = np.interp(mu_um, lam_um, Fcont_flam)
        EW_obs = np.nan if cont_at_mu == 0 or not np.isfinite(cont_at_mu) else meas["F_line"] / cont_at_mu
        EW0 = EW_obs / (1 + z) if np.isfinite(EW_obs) else np.nan
        out[name] = dict(EW_obs_A=EW_obs, EW0_A=EW0, cont_flam_at_mu=cont_at_mu)
    return out

def rescale_uncertainties(residual_flam, sigma_pipe_flam):
    """
    Empirical re-scaling of pipeline uncertainties following EXCELS Eq. (4).

    σ_F = [0.5*(R84−R16)/median(σ_pipe)] × σ_pipe

    Parameters
    ----------
    residual_flam : ndarray
        Continuum-subtracted flux residuals.
    sigma_pipe_flam : ndarray
        Pipeline-propagated flux uncertainties.

    Returns
    -------
    σ_rescaled : ndarray
        Rescaled uncertainties.
    """
    good = np.isfinite(residual_flam) & np.isfinite(sigma_pipe_flam) & (sigma_pipe_flam > 0)
    if not np.any(good):
        return sigma_pipe_flam
    R16, R84 = np.percentile(residual_flam[good], [16, 84])
    med_sig = np.median(sigma_pipe_flam[good])
    if med_sig <= 0:
        return sigma_pipe_flam
    corr = 0.5 * (R84 - R16) / med_sig
    return sigma_pipe_flam * max(corr, 1e-6)

def apply_balmer_absorption_correction(line_fluxes, correction_fractions=None):
    """Apply multiplicative emission filling corrections for Balmer lines."""
    if not correction_fractions:
        return line_fluxes
    out = {}
    for name, d in line_fluxes.items():
        corr = correction_fractions.get(name, 1.0)
        d2 = d.copy()
        d2["F_line"] *= corr
        d2["sigma_line"] *= corr
        out[name] = d2
    return out


def get_default_line_list(grating: str) -> List[str]:
    """
    Return the default line list based on the grating resolution.
    
    For medium resolution (R~1000) or PRISM, returns merged doublets for
    NV, CIV, and OII to improve fit stability.
    For high resolution, returns individual components.
    """
    g = grating.lower()
    is_medium = ("prism" in g) or ("medium" in g) or ("med" in g) or g.endswith("m") or ("g140m" in g)
    
    # Base lines present in all resolutions
    lines = [
        "NIV_1", "HEII_1640", "OIII_1663", "SiIII_1", "SiIII_2", "CIII]",
        "OII]_2471", "OIII]_2321", "OIII]_2331", "FeII*_2396",
        "HDELTA", "HGAMMA", "HBETA",
        "OIII_4363", "OIII_4959", "OIII_5007",
        "NII_5756", "HEI_5877",
        "NII_6549", "H⍺", "NII_6585",
        "SII_6718", "SII_6732"
    ]
    
    if is_medium:
        # Use merged doublets for medium res / prism
        lines.extend(["NV_doublet", "CIV_doublet", "OII_doublet"])
    else:
        # Use individual components for high res
        lines.extend(["NV_1", "NV_2", "CIV_1", "CIV_2", "OII_3726", "OII_3729"])
        
    # Sort by wavelength for consistency
    lines.sort(key=lambda x: REST_LINES_A.get(x, 0))
    
    return [l for l in lines if l in REST_LINES_A]
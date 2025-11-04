"""
PyRSR line_fit — Gaussian line-fitting in linear wavelength (F_λ)
=================================================================

This module implements a *linear-wavelength, pure-Gaussian* emission–line fitter
for JWST/NIRSpec 1D spectra. It provides:

1) A σ–weighted polynomial continuum fitter with robust masking of emission lines.
2) Independent Gaussian fits for each emission line:
   - integrated flux A  [erg s⁻¹ cm⁻²]
   - Gaussian width σ_A [Å]
   - centroid μ_A       [Å]
   Each line is fitted *independently* (no tied amplitudes/ratios).
3) Pixel-integrated model evaluation (bin-averaged Gaussians on pixel edges) so
   the fitted model respects spectral sampling and does not look “knife–edged”.
4) Automatic centroid seeding from *local observed peaks* near catalog positions.
5) Flux density plots in **both** µJy and F_λ using bin-aware “stairs” rendering,
   plus zoom windows around Hβ+[O III] for high-z galaxies (in µJy and F_λ).
   Data points with ±1σ uncertainties are overplotted in grey on all panels.
6) A parametric bootstrap utility to propagate uncertainties by repeatedly
   re-fitting spectra with noise realizations (with tqdm progress bar,
   optional figure saving, and ready-to-print line summaries).
7) Grey vertical guides with non-overlapping labels for each emission line.

Units & conventions
-------------------
- Wavelength arrays in **µm**; all line centers/widths reported in **Å**.
- Flux density in **µJy** and in **F_λ** [erg s⁻¹ cm⁻² Å⁻¹].
- Integrated line fluxes in **erg s⁻¹ cm⁻²**.
- All model profiles are *area-normalized* in F_λ; the amplitude A is the
  integrated line flux.
"""

from __future__ import annotations
import os
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares
from scipy.special import erf

# tqdm (auto if available; no-op fallback)
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm isn't installed
    def tqdm(x, *args, **kwargs):
        return x

# --------------------------------------------------------------------
# Imports from line.py (do not modify that file)
# --------------------------------------------------------------------
from PyRSR.ma_line_fit import (
    fnu_uJy_to_flam, flam_to_fnu_uJy,
    sigma_grating_logA,             # σ_gr in log10(λ), λ in µm
    line_centers_obs_A,             # nominal observed centers in Å
    measure_fluxes_profile_weighted,
    equivalent_widths_A,
    rescale_uncertainties,
    _lines_in_range,
    apply_balmer_absorption_correction,
)

C_AA = 2.99792458e18  #: Speed of light [Å·Hz·s⁻¹]
LN10 = np.log(10.0)    #: Natural log of 10

#: Rest wavelengths of commonly fitted nebular and recombination lines [Å]
REST_LINES_A = {
    "NIII_1": 989.790, "NIII_2": 991.514,"NIII_3": 991.579,

    "NV_1": 1238.821, "NV_2": 1242.804, "NIV_1": 1486.496,

    "HEII_1": 1640.420,  "OIII_05": 1663.4795, "CIII": 1908.734,

    "OII_UV_1": 3727.092, "OII_UV_2": 3729.875,
    "NEIII_UV_1": 3869.86, "NEIII_UV_2": 3968.59,
    "HDELTA": 4102.8922, "HGAMMA": 4341.6837, "OIII_1": 4364.436,
    "HEI_1": 4471.479, 
    
    "HEII_2": 4685.710, "HBETA": 4862.6830,
    "OIII_2": 4960.295, "OIII_3": 5008.240, 
    
    "NII_1":5756.19,"HEI": 5877.252,
    
    "NII_2":6549.86,"HALPHA": 6564.608,"NII_3":6585.27, "SII_1": 6718.295, "SII_2": 6732.674,
}

# ============================================================
# Helpers
# ============================================================

def _lyman_cut_um(z: float, which: str | None = "lya") -> float:
    if which is None:
        return -np.inf
    lam_rest_A = 1215.67 if str(which).lower() == "lya" else 912.0
    return lam_rest_A * (1.0 + z) / 1e4  # µm

def _pixel_edges_A(lam_A: np.ndarray):
    """Pixel edges in Å from a center-grid wavelength array in Å."""
    lam_A = np.asarray(lam_A, float)
    d = np.diff(lam_A)
    left  = np.r_[lam_A[0] - 0.5*d[0], lam_A[:-1] + 0.5*d]
    right = np.r_[lam_A[:-1] + 0.5*d, lam_A[-1] + 0.5*d[-1]]
    return left, right

def _pixel_edges_um(lam_um: np.ndarray):
    """Pixel edges in µm from a center-grid wavelength array in µm."""
    lam_um = np.asarray(lam_um, float)
    d = np.diff(lam_um)
    left  = np.r_[lam_um[0] - 0.1*d[0], lam_um[:-1] + 0.1*d]
    right = np.r_[lam_um[:-1] + 0.1*d, lam_um[-1] + 0.1*d[-1]]
    return left, right

def _gauss_binavg_area_normalized_A(lam_left_A, lam_right_A, muA, sigmaA):
    """
    Mean value (per pixel) of a *unit-area* Gaussian over [left,right] bins (Å⁻¹).
    """
    if not (np.isfinite(muA) and np.isfinite(sigmaA)) or sigmaA <= 0:
        return np.zeros_like(lam_left_A)
    inv = 1.0 / (sqrt(2.0) * sigmaA)
    cdf_r = 0.5 * (1.0 + erf((lam_right_A - muA) * inv))
    cdf_l = 0.5 * (1.0 + erf((lam_left_A  - muA) * inv))
    area = cdf_r - cdf_l
    width = (lam_right_A - lam_left_A)
    width = np.where(width > 0, width, np.nan)
    mean = area / width
    mean[ ~np.isfinite(mean) ] = 0.0
    return mean

def _safe_median(x, default):
    x = np.asarray(x, float)
    v = np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan
    return v if np.isfinite(v) else default

from matplotlib.transforms import blended_transform_factory
import numpy as np

def _annotate_lines(ax, which_lines, z, per_line=None,
                    text_color='0.25',
                    levels=(0.90, 0.78, 0.66, 0.54),
                    min_dx_um=0.01,
                    fontsize=8,
                    only_within_xlim=True,
                    shorten_names=True):
    """
    Annotate emission lines as staggered text labels (no vertical guide lines).

    Parameters
    ----------
    ax : matplotlib Axes
        The axes on which to annotate.
    which_lines : list[str]
        Line names to annotate.
    z : float
        Redshift.
    per_line : dict, optional
        Mapping from line names to dicts with 'lam_obs_A' values (Å).
    text_color : str or tuple, optional
        Color of the text labels.
    levels : tuple
        Y-axis fractional heights for staggered labels.
    min_dx_um : float
        Minimum wavelength separation (µm) between labels on the same level.
    fontsize : int
        Font size of labels.
    only_within_xlim : bool
        Skip labels outside the current x-limits.
    shorten_names : bool
        Simplify names (remove brackets and underscores).
    """
    if not which_lines:
        return

    xs, labels = [], []
    for nm in which_lines:
        if (per_line is not None) and (nm in per_line) and np.isfinite(per_line[nm].get('lam_obs_A', np.nan)):
            lam_um = per_line[nm]['lam_obs_A'] / 1e4
        else:
            lam_um = REST_LINES_A[nm] * (1.0 + z) / 1e4

        lab = nm
        if shorten_names:
            lab = lab.replace('[', '').replace(']', '').replace('_', ' ')
        xs.append(float(lam_um))
        labels.append(lab)

    xs = np.asarray(xs, float)
    labels = np.asarray(labels, object)

    if only_within_xlim:
        xlo, xhi = ax.get_xlim()
        keep = (xs >= xlo) & (xs <= xhi)
        xs, labels = xs[keep], labels[keep]

    if xs.size == 0:
        return

    order = np.argsort(xs)
    xs, labels = xs[order], labels[order]

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    last_x_at_level = np.full(len(levels), -np.inf)

    for xval, lab in zip(xs, labels):
        # Removed vertical line here
        level_idx = None
        for i, x_last in enumerate(last_x_at_level):
            if (xval - x_last) >= min_dx_um:
                level_idx = i
                break
        if level_idx is None:
            level_idx = len(levels) - 1

        y_frac = levels[level_idx]
        last_x_at_level[level_idx] = xval

        ax.text(
            xval, y_frac, lab,
            rotation=90, ha='center', va='bottom',
            color=text_color, fontsize=fontsize,
            transform=trans, zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=0.5)
        )

# ============================================================
# NEW: Two-window continuum selection around observed Lyα
# ============================================================

def continuum_windows_two_sides_of_lya(
    lam_source_or_dict,
    z: float,
    buffer_rest_A: float = 200.0,  # exclude ± this many rest-Å around Lyα
    min_pts: int = 12,             # require at least this many pixels per side
    verbose: bool = False,
):
    """
    Return up to two (lo_um, hi_um) windows for continuum fitting:
      - Blueward: [λ_min,  λ_Lyα_obs - buffer_obs]
      - Redward : [λ_Lyα_obs + buffer_obs, λ_max]
    Any side with < min_pts is dropped.

    Parameters
    ----------
    lam_source_or_dict : array-like of λ(µm) OR a dict with 'lam'/'wave'
    z : float
    buffer_rest_A : float
    min_pts : int
    verbose : bool
    """
    # Accept dict or array of wavelengths
    if isinstance(lam_source_or_dict, dict):
        lam_um = np.asarray(
            lam_source_or_dict.get("lam", lam_source_or_dict.get("wave")), float
        )
    else:
        lam_um = np.asarray(lam_source_or_dict, float)

    if lam_um.size == 0:
        return []

    lam_um = lam_um[np.isfinite(lam_um)]
    if lam_um.size == 0:
        return []

    lam_min, lam_max = float(np.min(lam_um)), float(np.max(lam_um))

    # Observed Lyα and observed buffer (in µm)
    LYA_REST_A = 1215.67
    lya_obs_um = LYA_REST_A * (1.0 + z) / 1e4
    buf_obs_um = buffer_rest_A * (1.0 + z) / 1e4

    blue_lo, blue_hi = lam_min, min(lya_obs_um - buf_obs_um, lam_max)
    red_lo,  red_hi  = max(lya_obs_um + buf_obs_um, lam_min), lam_max

    windows = []

    def maybe_add(lo, hi):
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return
        sel = (lam_um >= lo) & (lam_um <= hi)
        if np.count_nonzero(sel) >= min_pts:
            windows.append((float(lo), float(hi)))

    maybe_add(blue_lo, blue_hi)
    maybe_add(red_lo,  red_hi)

    if verbose:
        print(f"Lyman-α (obs): {lya_obs_um:.5f} µm | buffer: ±{buffer_rest_A:.0f} Å (rest)")
        print(f"Continuum windows (blue/red): {windows if windows else '(none)'}")

    return windows

# ============================================================
# Continuum (σ-weighted polynomial, with line masking)
# ============================================================

def fit_continuum_polynomial(lam_um, flam, z, deg=2, windows=None,
                             lyman_cut="lya", sigma_flam=None,
                             grating="PRISM", clip_sigma=2.5, max_iter=5):
    """
    Fit σ-weighted polynomial continuum *independently in each specified window*,
    masking emission lines. Returns stitched full-spectrum continuum and coef array.
    """
    lam_um = np.asarray(lam_um, float)
    flam = np.asarray(flam, float)

    # --- Lyman cut ---
    ly_um = _lyman_cut_um(z, lyman_cut)
    global_mask = lam_um >= ly_um

    # --- Default σ if not given ---
    if sigma_flam is None:
        sigma_flam = np.full_like(flam, max(1e-30, _safe_median(flam, 0.0)) * 0.05)
    sigma_flam = np.asarray(sigma_flam, float)

    # --- Default window (full coverage) ---
    if not windows:
        windows = [(float(np.nanmin(lam_um)), float(np.nanmax(lam_um)))]

    # --- Output continuum array ---
    Fcont = np.full_like(flam, np.nan)

    for (lo, hi) in windows:
        # Limit to each window + Lyα cut
        m = (lam_um >= lo) & (lam_um <= hi) & global_mask
        if np.count_nonzero(m) < max(deg + 2, 10):
            continue

        lam_fit = lam_um[m]
        flam_fit = flam[m].copy()
        sig_fit = sigma_flam[m].copy()

        # --- Mask emission lines ---
        lam_A = lam_fit * 1e4
        for lam_rest in REST_LINES_A.values():
            muA = lam_rest * (1 + z)
            mu_um = muA / 1e4
            sig_gr_log = float(sigma_grating_logA(grating, mu_um))
            sigma_A = muA * LN10 * sig_gr_log * 1.5
            core = (lam_A > muA - 6 * sigma_A) & (lam_A < muA + 6 * sigma_A)
            flam_fit[core] = np.nan
            sig_fit[core] = np.nan

        good = np.isfinite(flam_fit) & np.isfinite(sig_fit) & (sig_fit > 0)
        if np.count_nonzero(good) < max(deg + 2, 8):
            continue

        lam_fit, flam_fit, sig_fit = lam_fit[good], flam_fit[good], sig_fit[good]
        w = 1.0 / np.clip(sig_fit, _safe_median(sig_fit, 1.0) * 1e-3, np.inf)
        w /= np.nanmax(w)

        # --- Iterative σ-clipping ---
        for _ in range(max_iter):
            p = Polynomial.fit(lam_fit, flam_fit, deg, w=w)
            m_est = p(lam_fit)
            resid = (flam_fit - m_est) / np.clip(sig_fit, 1e-30, None)
            keep = np.abs(resid) < clip_sigma
            if np.all(keep):
                break
            lam_fit, flam_fit, sig_fit = lam_fit[keep], flam_fit[keep], sig_fit[keep]
            if lam_fit.size < max(deg + 2, 8):
                break
            w = 1.0 / np.clip(sig_fit, _safe_median(sig_fit, 1.0) * 1e-3, np.inf)
            w /= np.nanmax(w)

        p_final = Polynomial.fit(lam_fit, flam_fit, deg, w=w)
        Fcont[m] = p_final(lam_um)[m]

    # --- Fill gaps linearly between windows ---
    if np.any(~np.isfinite(Fcont)):
        good = np.isfinite(Fcont)
        if np.count_nonzero(good) >= 2:
            Fcont = np.interp(lam_um, lam_um[good], Fcont[good], left=Fcont[good][0], right=Fcont[good][-1])
        else:
            Fcont[:] = _safe_median(flam, 0.0)

    # --- Compatibility: return global coefs ---
    mgood = np.isfinite(Fcont)
    if np.count_nonzero(mgood) >= max(deg + 2, 10):
        w_all = 1.0 / np.clip(sigma_flam[mgood], _safe_median(sigma_flam[mgood], 1.0) * 1e-3, np.inf)
        w_all /= np.nanmax(w_all)
        p_coefs = Polynomial.fit(lam_um[mgood], Fcont[mgood], deg, w=w_all).convert().coef
    else:
        p_coefs = np.array([_safe_median(Fcont[mgood], _safe_median(flam, 0.0))])

    return Fcont, p_coefs


# ============================================================
# Seeds: local peaks for μ
# ============================================================

def _find_local_peaks(lam_um, resid_flam, expected_um, sigma_gr_um, min_window_um=0.001):
    peaks = []
    for mu0, s_um in zip(expected_um, sigma_gr_um):
        w = max(5.0 * s_um, float(min_window_um))
        m = (lam_um > mu0 - w) & (lam_um < mu0 + w)
        if np.count_nonzero(m) > 2:
            y = resid_flam[m]
            if np.any(np.isfinite(y)):
                peaks.append(lam_um[m][np.nanargmax(y)])
            else:
                peaks.append(mu0)
        else:
            peaks.append(mu0)
    return np.array(peaks, float)

# ============================================================
# Model: sum of area-normalized Gaussians in Fλ
# ============================================================

def build_model_flam_linear(params, lam_um, z, grating, which_lines, mu_seed_um):
    nL = len(which_lines)
    A = np.array(params[0:nL], float)
    sigmaA = np.array(params[nL:2*nL], float)
    muA = np.array(params[2*nL:3*nL], float)

    lam_A = lam_um * 1e4
    left_A, right_A = _pixel_edges_A(lam_A)

    model = np.zeros_like(lam_um, float)
    profiles, centers = {}, {}

    # Lower bound to avoid sub-pixel aliasing (≤ about half a pixel)
    pix_A = np.median(np.diff(lam_A))
    min_sigma = max(0.35 * pix_A, 0.01)  # ~0.35 px preserves peak shapes

    for j, name in enumerate(which_lines):
        sj = np.clip(sigmaA[j], min_sigma, 1e6)
        profA = _gauss_binavg_area_normalized_A(left_A, right_A, muA[j], sj)
        prof_flam = A[j] * profA
        model += prof_flam
        profiles[name] = prof_flam
        sigma_logA = sj / (muA[j] * LN10) if (np.isfinite(muA[j]) and muA[j] > 0) else np.nan
        centers[name] = (muA[j], sigma_logA)

    return model, profiles, centers

# ============================================================
# Robust seeds/bounds finalization
# ============================================================

def _finalize_seeds_and_bounds(p0, lb, ub):
    p0 = np.array(p0, float); lb = np.array(lb, float); ub = np.array(ub, float)
    bad = ~np.isfinite(lb) | ~np.isfinite(ub) | (ub <= lb)
    if np.any(bad):
        lb[bad] = -1e6
        ub[bad] =  1e6
    p0_bad = ~np.isfinite(p0)
    if np.any(p0_bad):
        p0[p0_bad] = 0.5*(lb[p0_bad] + ub[p0_bad])
    eps = 1e-10
    p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)
    outside = (p0 <= lb) | (p0 >= ub)
    if np.any(outside):
        span = np.maximum(ub - lb, 1e-8)
        lb[outside] -= 0.05*span[outside]
        ub[outside] += 0.05*span[outside]
        p0[outside] = np.minimum(np.maximum(p0[outside], lb[outside] + eps), ub[outside] - eps)
    return p0, lb, ub

# ============================================================
# Main fit
# ============================================================
def excels_fit_poly(source, z, grating="PRISM", lines_to_use=None,
                    deg=2, continuum_windows=None, lyman_cut="lya",
                    fit_window_um=None, plot=True, verbose=True,
                    absorption_corrections=None):
    # --- Load spectrum ---
    if isinstance(source, dict):
        lam_um = np.asarray(source.get("lam", source.get("wave")), float)
        flux_uJy = np.asarray(source["flux"], float)
        err_uJy  = np.asarray(source["err"], float)
    else:
        hdul = source if hasattr(source, "__iter__") else fits.open(source)
        d1 = hdul["SPEC1D"].data
        lam_um = np.asarray(d1["wave"], float)
        flux_uJy = np.asarray(d1["flux"], float)
        err_uJy  = np.asarray(d1["err"], float)

    ok = np.isfinite(lam_um) & np.isfinite(flux_uJy) & np.isfinite(err_uJy) & (err_uJy > 0)
    lam_um, flux_uJy, err_uJy = lam_um[ok], flux_uJy[ok], err_uJy[ok]

    # --- AUTO continuum windows around Lyα if requested ---
    auto_windows = None
    if isinstance(continuum_windows, str) and continuum_windows.lower() == "two_sided_lya":
        auto_windows = continuum_windows_two_sides_of_lya(lam_um, z, buffer_rest_A=200.0, min_pts=12, verbose=verbose)
    elif isinstance(continuum_windows, dict) and continuum_windows.get("mode", "").lower() == "two_sided_lya":
        brA = float(continuum_windows.get("buffer_rest_A", 200.0))
        mpts = int(continuum_windows.get("min_pts", 12))
        auto_windows = continuum_windows_two_sides_of_lya(lam_um, z, buffer_rest_A=brA, min_pts=mpts, verbose=verbose)

    if auto_windows is not None:
        continuum_windows = auto_windows
        lyman_cut = None  # we already gapped Lyα with explicit windows

    # --- Convert to F_lambda ---
    flam     = fnu_uJy_to_flam(flux_uJy, lam_um)     # data in Fλ
    sig_flam = fnu_uJy_to_flam(err_uJy,  lam_um)     # 1σ in Fλ

    # --- Continuum ---
    Fcont, _ = fit_continuum_polynomial(
        lam_um, flam, z, deg=deg, windows=continuum_windows,
        lyman_cut=lyman_cut, sigma_flam=sig_flam, grating=grating
    )
    resid_full   = flam - Fcont
    sig_flam_fit = rescale_uncertainties(resid_full, sig_flam)  # used in fitting

    # --- Fit window ---
    if fit_window_um:
        lo, hi = fit_window_um
        w = (lam_um >= lo) & (lam_um <= hi)
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um[w], resid_full[w], sig_flam_fit[w], Fcont[w]
    else:
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um, resid_full, sig_flam_fit, Fcont

    # --- Lines in coverage ---
    which_lines = _lines_in_range(z, lam_fit, lines_to_use, margin_um=0.02)
    if not which_lines:
        raise ValueError("No emission lines in wavelength coverage.")

    # --- Instrument σ and centroid seeds ---
    expected_um  = np.array([REST_LINES_A[nm]*(1+z)/1e4 for nm in which_lines])
    sigma_gr_log = np.array([sigma_grating_logA(grating, mu_um) for mu_um in expected_um])  # log10
    muA_nom      = expected_um * 1e4                          # Å
    sigmaA_inst  = muA_nom * LN10 * sigma_gr_log              # Å (instrumental σ)
    sigma_um_inst = sigmaA_inst / 1e4                         # µm (for peak search)

    mu_seed_um = _find_local_peaks(lam_fit, resid_fit, expected_um, sigma_um_inst, min_window_um=0.001)
    muA_seed   = mu_seed_um * 1e4  # Å

    # ============================
    # Seeds & bounds — corrected
    # ============================
    # ============================
    # Seeds & bounds — improved
    # ============================
    nL = len(which_lines)

    # --- pixel scale in Å (for gentle floors to avoid sub-pixel collapse)
    lamA_fit = lam_fit * 1e4
    pix_A = float(np.median(np.diff(lamA_fit))) if lamA_fit.size > 1 else 1.0

    # --- quick per-line local S/N in ±5 σ_inst windows ---
    snr_loc = []
    for j, mu_um in enumerate(expected_um):
        w = np.abs(lam_fit - mu_um) < 5.0 * (sigmaA_inst[j] / 1e4)  # convert σ_A→σ_µm
        if np.any(w):
            peak = np.nanmax(resid_fit[w])
            noise = np.nanmedian(sig_fit[w]) if np.any(np.isfinite(sig_fit[w])) else np.nan
            snr_loc.append(peak / (noise + 1e-30))
        else:
            snr_loc.append(0.0)
    snr_loc = np.asarray(snr_loc, float)
    good_snr = snr_loc >= 8.0

    # --- Robust σ bounds tuned to grating resolution ---
    g = str(grating).lower()

    if "prism" in g:
        sigmaA_lo = np.maximum(0.40 * pix_A, 0.45 * sigmaA_inst)
        sigmaA_hi = np.maximum(1.50 * pix_A, 1.70 * sigmaA_inst)
        seed_factor = 0.90  # ~σ_inst

    elif any(k in g for k in ["m", "med"]):
        sigmaA_lo = np.maximum(0.12 * pix_A, 0.22 * sigmaA_inst)
        sigmaA_hi = np.maximum(0.75 * pix_A, 1.05 * sigmaA_inst)
        seed_factor = 0.60

    elif any(k in g for k in ["h", "high"]):
        sigmaA_lo = np.where(
            good_snr,
            np.maximum(0.10 * pix_A, 0.18 * sigmaA_inst),
            np.maximum(0.20 * pix_A, 0.40 * sigmaA_inst)
        )
        sigmaA_hi = np.where(
            good_snr,
            np.maximum(0.60 * pix_A, 0.95 * sigmaA_inst),
            np.maximum(0.75 * pix_A, 1.15 * sigmaA_inst)
        )
        seed_factor = np.where(good_snr, 0.50, 0.65)



    else:
        sigmaA_lo = np.maximum(0.30 * pix_A, 0.60 * sigmaA_inst)
        sigmaA_hi = np.maximum(0.70 * pix_A, 1.15 * sigmaA_inst)
        seed_factor = 0.85

    # --- σ seeds near (or slightly below) instrumental width, clipped into [lo, hi]
    sigmaA_seed = np.clip(seed_factor * sigmaA_inst, sigmaA_lo, sigmaA_hi)

    # --- Amplitude seeds in area units (erg s^-1 cm^-2), consistent with σ
    # peak_Fλ ≈ A / (sqrt(2π) σ_A)  =>  A ≈ peak_Fλ * sqrt(2π) * σ_A
    SQRT2PI = np.sqrt(2.0 * np.pi)

    A0, A_lo, A_hi = [], np.zeros(len(sigmaA_seed)), []
    rms_flam = float(_safe_median(np.abs(resid_fit), 0.0))
    rms_flam = max(rms_flam, 1e-30)

    for j, mu_um in enumerate(expected_um):
        win = (lam_fit > mu_um - 0.05) & (lam_fit < mu_um + 0.05)
        peak_flam = np.nanmax(resid_fit[win]) if np.any(win) else 3.0 * rms_flam
        peak_flam = max(peak_flam, 3.0 * rms_flam)  # ensure not tiny at low S/N

        # use the σ seed (not σ_inst) for self-consistency
        A_seed = peak_flam * SQRT2PI * max(sigmaA_seed[j], 0.9 * sigmaA_inst[j])


        # generous upper bound; scale with σ_hi to avoid runaway broad fits
        A_upper = 150.0 * max(peak_flam, 3.0 * rms_flam) * SQRT2PI * np.maximum(sigmaA_hi[j], sigmaA_seed[j])

        A0.append(A_seed)
        A_hi.append(A_upper)

    A0 = np.array(A0, float)
    A_hi = np.array(A_hi, float)



    # Centroids μ_A (Å) — allow several σ of freedom (wider for PRISM)
    if "prism" in g:
        muA_lo = muA_seed - 12.0 * np.maximum(sigmaA_inst, 1.0)
        muA_hi = muA_seed + 12.0 * np.maximum(sigmaA_inst, 1.0)
    else:
        muA_lo = muA_seed - 6.0  * np.maximum(sigmaA_inst, 1.0)
        muA_hi = muA_seed + 6.0  * np.maximum(sigmaA_inst, 1.0)

    # Pack parameters and finalize to guarantee x0 inside bounds
    p0 = np.r_[A0,           sigmaA_seed, muA_seed]
    lb = np.r_[A_lo,         sigmaA_lo,   muA_lo]
    ub = np.r_[A_hi,         sigmaA_hi,   muA_hi]
    p0, lb, ub = _finalize_seeds_and_bounds(p0, lb, ub)

    # --- χ² residuals with pixel-width weighting in Å ---
    #dlA = np.gradient(lamA_fit)
    #w_pix = np.sqrt(dlA / _safe_median(dlA, 1.0))
    # --- χ² residual weights: keep some bin-width correction, but gentle & bounded
    dlA = np.gradient(lamA_fit)
    rat = dlA / np.nanmedian(dlA)

    # 1) bound the ratio so no region is overly down/up-weighted
    rat = np.clip(rat, 0.6, 1.6)

    # 2) choose orientation: give slightly *more* weight to narrow bins (line cores)
    #    but keep it mild (exponent 0.35) to preserve stability
    w_pix = (np.nanmedian(dlA) / np.clip(dlA, 1e-12, None))**0.35

    # 3) enforce same bounds as above (defensive)
    w_pix = np.clip(w_pix, 0.8, 1.25)

    def fun(p):
        model, _, _ = build_model_flam_linear(p, lam_fit, z, grating, which_lines, mu_seed_um)
        return w_pix * (resid_fit - model) / np.clip(sig_fit, 1e-30, None)

    res = least_squares(fun, p0, bounds=(lb, ub), max_nfev=80000, xtol=1e-8, ftol=1e-8)

    # --- Best-fit model on window ---
    model_flam_win, profiles_win, centers = build_model_flam_linear(
        res.x, lam_fit, z, grating, which_lines, mu_seed_um
    )

    # --- Fluxes & EWs ---
    fluxes = measure_fluxes_profile_weighted(
        lam_fit, resid_fit, sig_fit, model_flam_win, profiles_win, centers
    )
    ews = equivalent_widths_A(fluxes, lam_fit, Fcont_fit, z, centers)
    if absorption_corrections:
        fluxes = apply_balmer_absorption_correction(fluxes, absorption_corrections)

    # --- Collect per-line outputs ---
    per_line = {}
    for j, name in enumerate(which_lines):
        F_line   = fluxes.get(name, {}).get("F_line", np.nan)
        sig_line = fluxes.get(name, {}).get("sigma_line", np.nan)
        ew_obs   = ews.get(name, {}).get("EW_obs_A", np.nan)
        ew0      = ews.get(name, {}).get("EW0_A", np.nan)
        snr      = np.nan if not np.isfinite(sig_line) or sig_line == 0 else F_line / sig_line

        muA, sigma_logA = centers[name]
        sigma_A  = sigma_logA * muA * LN10
        FWHM_A   = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_A

        per_line[name] = dict(
            F_line=F_line, sigma_line=sig_line, SNR=snr,
            EW_obs_A=ew_obs, EW0_A=ew0,
            lam_obs_A=muA, sigma_A=sigma_A, FWHM_A=FWHM_A
        )

    # --- Plot (unchanged except for your existing styling) ---
    if plot:
        if fit_window_um:
            model_full = np.zeros_like(lam_um)
            w = (lam_um >= lo) & (lam_um <= hi)  # from above
            model_full[w] = model_flam_win
            total_model_flam = Fcont + model_full
        else:
            total_model_flam = Fcont + model_flam_win

        cont_uJy  = flam_to_fnu_uJy(Fcont,            lam_um)
        model_uJy = flam_to_fnu_uJy(total_model_flam, lam_um)

        left_um, right_um = _pixel_edges_um(lam_um)
        edges_um = np.r_[left_um[0], right_um]

        o3_lo, o3_hi = 4.45, 4.70
        mZ = (lam_um > o3_lo) & (lam_um < o3_hi)

        fig, axes = plt.subplots(
            4, 1, figsize=(11, 12.2),
            gridspec_kw={"height_ratios": [1.4, 1.2, 1.0, 1.0]}
        )
        ax1, ax2, ax3, ax4 = axes

        ax1.stairs(flux_uJy, edges_um, label='Data (bins)', color='k', linewidth=0.9, alpha=0.85)
        ax1.errorbar(lam_um, flux_uJy, yerr=err_uJy, fmt='o', ms=2.5,
                     color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                     ecolor=(0,0,0,0.12), elinewidth=0.4, capsize=0, zorder=1,
                     label='Data ±1σ')
        ax1.stairs(cont_uJy, edges_um, label=f'Continuum (deg={deg})', color='b', linestyle='--', linewidth=1.0)
        ax1.stairs(model_uJy, edges_um, label='Continuum + Lines', color='r', linewidth=1.2)
        ax1.set_ylabel('Flux density [µJy]')
        ax1.legend(ncol=3, fontsize=9, frameon=False)
        _annotate_lines(ax1, which_lines, z, per_line=per_line, min_dx_um=0.01)

        data_flam = flam
        data_flam_err = sig_flam
        ax2.stairs(data_flam,          edges_um, label='Data (Fλ, bins)', color='k', linewidth=0.9, alpha=0.85)
        ax2.errorbar(lam_um, data_flam, yerr=data_flam_err, fmt='o', ms=2.5,
                     color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                     ecolor=(0,0,0,0.12), elinewidth=0.4, capsize=0, zorder=1,
                     label='Data (Fλ) ±1σ')
        ax2.stairs(Fcont,              edges_um, label='Continuum (Fλ)', color='b', linestyle='--', linewidth=1.0)
        ax2.stairs(total_model_flam,   edges_um, label='Model (Fλ)',     color='r', linewidth=1.2)
        ax2.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        ax2.legend(ncol=3, fontsize=9, frameon=False)
        _annotate_lines(ax2, which_lines, z, per_line=per_line, min_dx_um=0.01)

        if np.any(mZ):
            lam_zoom = lam_um[mZ]
            left_z, right_z = _pixel_edges_um(lam_zoom)
            edges_z = np.r_[left_z[0], right_z]

            ax3.stairs(flux_uJy[mZ],  edges_z, label='Data (bins)', color='k', linewidth=0.9, alpha=0.85)
            ax3.errorbar(lam_zoom, flux_uJy[mZ], yerr=err_uJy[mZ], fmt='o', ms=2.0,
                         color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                         ecolor=(0,0,0,0.16), elinewidth=0.45, capsize=0, zorder=1,
                         label='Data ±1σ')
            ax3.stairs(cont_uJy[mZ],  edges_z, label='Continuum', color='b', linestyle='--', linewidth=0.9)
            ax3.stairs(model_uJy[mZ], edges_z, label='Model',     color='r', linewidth=1.1)
            ax3.set_xlim(o3_lo, o3_hi)
            ax3.legend(fontsize=8, frameon=False, ncol=3)
            _annotate_lines(ax3, which_lines, z, per_line=per_line, min_dx_um=0.002, levels=(0.92,0.80,0.68))
        ax3.set_xlabel('Observed wavelength [µm]')
        ax3.set_ylabel('Flux density [µJy]')

        if np.any(mZ):
            lam_zoom = lam_um[mZ]
            left_z, right_z = _pixel_edges_um(lam_zoom)
            edges_z = np.r_[left_z[0], right_z]

            data_flam_z = data_flam[mZ]
            data_flam_err_z = data_flam_err[mZ]
            cont_flam_z = Fcont[mZ]
            model_flam_z = total_model_flam[mZ]

            ax4.stairs(data_flam_z, edges_z, label='Data (Fλ, bins)', color='k', linewidth=0.9, alpha=0.85)
            ax4.errorbar(lam_zoom, data_flam_z, yerr=data_flam_err_z, fmt='o', ms=2.0,
                         color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                         ecolor=(0,0,0,0.16), elinewidth=0.45, capsize=0, zorder=1,
                         label='Data (Fλ) ±1σ')
            ax4.stairs(cont_flam_z,  edges_z, label='Continuum (Fλ)', color='b', linestyle='--', linewidth=0.9)
            ax4.stairs(model_flam_z, edges_z, label='Model (Fλ)',     color='r', linewidth=1.1)
            ax4.set_xlim(o3_lo, o3_hi)
            ax4.legend(fontsize=8, frameon=False, ncol=3)
            _annotate_lines(ax4, which_lines, z, per_line=per_line, min_dx_um=0.002, levels=(0.92,0.80,0.68))
        ax4.set_xlabel('Observed wavelength [µm]')
        ax4.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')

        plt.tight_layout()
        plt.show()

    return dict(
        success=res.success,
        message=res.message,
        lam_fit=lam_fit,
        model_window_flam=model_flam_win,
        continuum_flam=Fcont,
        lines=per_line,
        which_lines=which_lines,
    )


# ============================================================
# Bootstrapping with µJy+Fλ stairs, zooms, data errorbars, line labels, tqdm
# ============================================================

def _o3_hb_zoom_bounds_um(z, pad_um=0.03):
    hbA, o3aA, o3bA = 4861.33, 4958.91, 5006.84
    obsA = np.array([hbA, o3aA, o3bA]) * (1.0 + z)
    return obsA.min()/1e4 - pad_um, obsA.max()/1e4 + pad_um

def _sigma_clip_mean_std(a, axis=0, sigma=3.0, min_keep=5):
    a = np.asarray(a, float)
    med = np.nanmedian(a, axis=axis)
    mad = 1.4826 * np.nanmedian(np.abs(a - np.expand_dims(med, axis=axis)), axis=axis)
    lo, hi = med - sigma*mad, med + sigma*mad
    m = (a >= np.expand_dims(lo, axis=axis)) & (a <= np.expand_dims(hi, axis=axis))
    if axis == 0:
        nfin = np.sum(m & np.isfinite(a), axis=0)
        m[:, nfin < min_keep] = False
    a = np.where(m, a, np.nan)
    return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis)

# -------------------------------
# Smart-zoom helpers (drop-in)
# -------------------------------
def _obs_um_from_rest_A(rest_A, z):
    return (np.asarray(rest_A, float) * (1.0 + z)) / 1e4

def _window_from_lines_um(rest_list, z, pad_A=250.0):
    """
    Return [lo_um, hi_um] spanning the given rest-Å lines at redshift z,
    with symmetric padding of pad_A (rest Å) on each side.
    """
    obs_um = _obs_um_from_rest_A(rest_list, z)
    lo = np.min(obs_um) - pad_A * (1 + z) / 1e4
    hi = np.max(obs_um) + pad_A * (1 + z) / 1e4
    return float(lo), float(hi)

def _has_coverage_window(lam_um, lo_um, hi_um, min_pts=5):
    m = (lam_um >= lo_um) & (lam_um <= hi_um) & np.isfinite(lam_um)
    return bool(np.count_nonzero(m) >= min_pts)

def _has_fit_in_window(base_lines, candidate_names, snr_min=1.0):
    """
    True if the fit produced at least one of the candidate lines with finite params
    and SNR above snr_min.
    """
    if not base_lines:
        return False
    for nm in candidate_names:
        d = base_lines.get(nm)
        if d is None:
            continue
        snr = d.get("SNR", np.nan)
        f   = d.get("F_line", np.nan)
        mu  = d.get("lam_obs_A", np.nan)
        if np.isfinite(snr) and (snr >= snr_min) and np.isfinite(f) and np.isfinite(mu):
            return True
    return False

def _make_zoom_panel(ax, lam_um, flux_uJy, err_uJy, cont_uJy, mu_uJy, sig_uJy,
                     which_lines, per_line, z, lo_um, hi_um, title, annotate_min_dx_um=0.002):
    """Render one zoom panel; returns True if drawn, else False (axis hidden)."""
    sel = (lam_um >= lo_um) & (lam_um <= hi_um)
    if not np.any(sel):
        ax.setVisible(False)
        return False

    left, right = _pixel_edges_um(lam_um[sel])
    edges = np.r_[left[0], right]

    ax.stairs(flux_uJy[sel], edges, label="Data (bins)", linewidth=0.9, alpha=0.9)
    ax.errorbar(lam_um[sel], flux_uJy[sel], yerr=err_uJy[sel], fmt='o', ms=2.2,
                color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                ecolor=(0,0,0,0.16), elinewidth=0.45, capsize=0, zorder=1,
                label="±1σ")
    ax.stairs(cont_uJy[sel], edges, label="Continuum", linestyle="--", linewidth=0.9)
    ax.stairs(mu_uJy[sel],   edges, label="Model mean", linewidth=1.0)
    ax.fill_between(lam_um[sel], mu_uJy[sel] - sig_uJy[sel], mu_uJy[sel] + sig_uJy[sel],
                    step='mid', alpha=0.18, linewidth=0)
    ax.set_xlim(lo_um, hi_um)
    ax.set_title(title, fontsize=10)
    _annotate_lines(ax, which_lines, z, per_line=per_line,
                    min_dx_um=annotate_min_dx_um, levels=(0.92, 0.80, 0.68))
    return True

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm.auto import tqdm

# assumes these come from your module namespace already:
# - excels_fit_poly
# - flam_to_fnu_uJy
# - _sigma_clip_mean_std
# - _window_from_lines_um, _has_coverage_window, _has_fit_in_window
# - REST_LINES_A
# - _annotate_lines

def _bin_edges_from_centers_um(lam_um: np.ndarray) -> np.ndarray:
    """
    Given wavelength centers in µm (possibly non-uniform), return bin edges in µm
    such that each center is the midpoint of its bin.
    """
    lam = np.asarray(lam_um, float)
    if lam.size < 2:
        # fabricate a tiny symmetric bin if we only have one point
        d = 1e-6
        return np.array([lam[0] - d, lam[0] + d], float)

    mid = 0.5 * (lam[1:] + lam[:-1])                 # interior edges
    left  = lam[0]  - 0.5 * (lam[1]  - lam[0])       # first edge
    right = lam[-1] + 0.5 * (lam[-1] - lam[-2])      # last edge
    return np.r_[left, mid, right]


def bootstrap_excels_fit(
    source,
    z,
    grating="PRISM",
    n_boot=200,
    source_id=None, 
    deg=2,
    continuum_windows=None,
    lyman_cut="lya",
    fit_window_um=None,
    absorption_corrections=None,
    random_state=None,
    verbose=False,
    plot=True,
    show_progress=True,
    save_path: str | None = None,
    save_dpi: int = 500,
    save_format: str = "png",
    save_transparent: bool = False,
    lines_to_use=None,
    plot_unit='both'
):
    # -------- load once --------
    if isinstance(source, dict):
        lam_um = np.asarray(source.get("lam", source.get("wave")), float)
        flux_uJy = np.asarray(source["flux"], float)
        err_uJy  = np.asarray(source["err"], float)
    else:
        hdul = source if hasattr(source, "__iter__") else fits.open(source)
        d1 = hdul["SPEC1D"].data
        lam_um = np.asarray(d1["wave"], float)
        flux_uJy = np.asarray(d1["flux"], float)
        err_uJy  = np.asarray(d1["err"], float)

    ok = np.isfinite(lam_um) & np.isfinite(flux_uJy) & np.isfinite(err_uJy) & (err_uJy > 0)
    lam_um, flux_uJy, err_uJy = lam_um[ok], flux_uJy[ok], err_uJy[ok]

    # -------- single run (anchors thresholds) --------
    base = excels_fit_poly(
        source=dict(lam=lam_um, flux=flux_uJy, err=err_uJy),
        z=z, grating=grating, deg=deg,
        continuum_windows=continuum_windows,
        lyman_cut=lyman_cut,
        fit_window_um=fit_window_um,
        plot=False, verbose=False,
        absorption_corrections=absorption_corrections,
        lines_to_use=lines_to_use  # all in coverage
    )
    which_lines = list(base.get("which_lines", []))
    if not which_lines:
        raise ValueError("No emission lines in coverage (cannot bootstrap).")

    base_lines = base["lines"]
    flux_cap  = {ln: 10.0 * abs(base_lines[ln]["F_line"])  if ln in base_lines else np.inf
                 for ln in which_lines}
    width_cap = {ln: 10.0 * abs(base_lines[ln]["sigma_A"]) if ln in base_lines else np.inf
                 for ln in which_lines}

    if fit_window_um:
        lo, hi = fit_window_um
        wfit = (lam_um >= lo) & (lam_um <= hi)
    else:
        wfit = slice(None)

    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))

    # -------- storage --------
    samples = {ln: {"F_line": [], "sigma_A": [], "lam_obs_A": [], "EW0_A": [], "SNR": []}
               for ln in which_lines}
    model_stack_flam, keep_mask = [], []

    # Progress iterator (use tqdm when available)
    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(
            range(n_boot),
            desc=f"Bootstrap ({n_boot} draws)",
            unit="draw",
            dynamic_ncols=True,
            leave=True
        )

    for _ in iterator:
        flux_uJy_b = flux_uJy + rng.normal(0.0, err_uJy)

        try:
            fb = excels_fit_poly(
                source=dict(lam=lam_um, flux=flux_uJy_b, err=err_uJy),
                z=z, grating=grating, deg=deg,
                continuum_windows=continuum_windows,
                lyman_cut=lyman_cut,
                fit_window_um=fit_window_um,
                plot=False, verbose=False,
                absorption_corrections=absorption_corrections,
                lines_to_use=lines_to_use,    # <<=== ADD THIS
            )

            ok_fit = True
        except Exception:
            ok_fit = False
            fb = {}

        keep = ok_fit and bool(fb.get("lines")) and fb.get("success", True)

        if keep:
            for ln in which_lines:
                d = fb["lines"].get(ln, None)
                if (d is None) or \
                   (np.abs(d.get("F_line", np.nan))  > flux_cap[ln]) or \
                   (np.abs(d.get("sigma_A", np.nan)) > width_cap[ln]) or \
                   (not np.isfinite(d.get("lam_obs_A", np.nan))):
                    keep = False
                    break

        if not keep:
            keep_mask.append(False)
            model_stack_flam.append(np.full_like(lam_um, np.nan))
            for ln in which_lines:
                for k in samples[ln]:
                    samples[ln][k].append(np.nan)
            continue

        # per-line metrics
        for ln in which_lines:
            d = fb["lines"][ln]
            for k in samples[ln]:
                samples[ln][k].append(d[k])

        # full F_lambda total model on lam_um (continuum + lines)
        cont = fb.get("continuum_flam", np.zeros_like(lam_um))

        # Mask continuum blueward of Lyα (avoid extrapolation entering bootstrap mean)
        cont_mask = np.ones_like(lam_um, dtype=bool)
        if (lyman_cut is not None) and (str(lyman_cut).lower() == "lya"):
            lya_edge = 0.1216 * (1.0 + z)
            disp_mask = (lam_um >= lya_edge)  # hard cut below Lyα


        # Optionally also limit to your continuum windows
        if continuum_windows and isinstance(continuum_windows, (list, tuple)):
            cw_mask = np.zeros_like(lam_um, dtype=bool)
            for lo, hi in continuum_windows:
                cw_mask |= (lam_um >= lo) & (lam_um <= hi)
            cont_mask &= cw_mask

        cont_masked = np.full_like(cont, np.nan)
        cont_masked[cont_mask] = cont[cont_mask]

        # Combine masked continuum with model in fit window
        if isinstance(wfit, slice):
            total_flam = cont_masked + fb.get("model_window_flam", np.zeros_like(lam_um))
        else:
            tmp = np.zeros_like(lam_um)
            tmp[wfit] = fb.get("model_window_flam", np.zeros_like(lam_um[wfit]))
            total_flam = cont_masked + tmp


        # basic sanity filter against wild models
        if np.nanmax(np.abs(total_flam)) > 1e-11:
            keep_mask.append(False)
            model_stack_flam.append(np.full_like(lam_um, np.nan))
            for ln in which_lines:
                for k in samples[ln]:
                    samples[ln][k].append(np.nan)
            continue

        model_stack_flam.append(total_flam)
        keep_mask.append(True)

    # arrays
    model_stack_flam = np.asarray(model_stack_flam, float)
    keep_mask = np.asarray(keep_mask, bool)
    for ln in which_lines:
        for k in samples[ln]:
            samples[ln][k] = np.asarray(samples[ln][k], float)

    # summaries (value ± error + preformatted text)
    def _ve(x):
        x = x[keep_mask]
        if x.size == 0 or np.all(~np.isfinite(x)):
            return np.nan, np.nan
        val = np.nanmean(x)
        err = np.nanstd(x)
        if np.count_nonzero(np.isfinite(x)) >= 8:
            xc = x[np.isfinite(x)]
            m, s = np.nanmean(xc), np.nanstd(xc)
            xc = xc[(xc > m - 3*s) & (xc < m + 3*s)]
            if xc.size:
                val, err = np.nanmean(xc), np.nanstd(xc)
        return val, err

    summary = {}
    for ln in which_lines:
        summary[ln] = {}
        vF, eF   = _ve(samples[ln]["F_line"])
        vEW, eEW = _ve(samples[ln]["EW0_A"])
        vS, eS   = _ve(samples[ln]["sigma_A"])
        vMu, eMu = _ve(samples[ln]["lam_obs_A"])
        vSN, eSN = _ve(samples[ln]["SNR"])
        summary[ln]["F_line"]    = {"value": vF,  "err": eF,  "text": f"{vF:.3e} ± {eF:.3e}"}
        summary[ln]["EW0_A"]     = {"value": vEW, "err": eEW, "text": f"{vEW:.2f} ± {eEW:.2f}"}
        summary[ln]["sigma_A"]   = {"value": vS,  "err": eS,  "text": f"{vS:.2f} ± {eS:.2f}"}
        summary[ln]["lam_obs_A"] = {"value": vMu, "err": eMu, "text": f"{vMu:.1f} ± {eMu:.1f}"}
        summary[ln]["SNR"]       = {"value": vSN, "err": eSN, "text": f"{vSN:.2f} ± {eSN:.2f}"}

    # -------- plotting + optional saving --------
    if plot:
        # mean ± std (sigma-clipped) of total F_lambda model
        mu_flam, sig_flam = _sigma_clip_mean_std(model_stack_flam[keep_mask], axis=0, sigma=3.0)
        cont_flam = np.asarray(base.get("continuum_flam", np.zeros_like(lam_um)))

        # also prepare µJy versions
        mu_uJy  = flam_to_fnu_uJy(mu_flam,  lam_um)
        sig_uJy = flam_to_fnu_uJy(sig_flam, lam_um)
        cont_uJy = flam_to_fnu_uJy(cont_flam, lam_um)

        # shared wavelength mask
        disp_mask = np.ones_like(lam_um, dtype=bool)
        if (lyman_cut is not None) and (str(lyman_cut).lower() == "lya"):
            lya_edge = 0.1216 * (1.0 + z)
            disp_mask &= (lam_um >= lya_edge)
        if continuum_windows and isinstance(continuum_windows, (list, tuple)):
            cw_mask = np.zeros_like(lam_um, dtype=bool)
            for lo, hi in continuum_windows:
                cw_mask |= (lam_um >= lo) & (lam_um <= hi)
            disp_mask &= cw_mask
        if fit_window_um:
            lo, hi = fit_window_um
            disp_mask &= (lam_um >= lo) & (lam_um <= hi)

        # reusable zoom definitions
        o3hb_names = ["HBETA", "OIII_2", "OIII_3"]
        aur_names  = ["HDELTA", "OIII_1"]
        oiiuv_names = ["OII_UV_1", "OII_UV_2"]
        zoom_defs = [
            dict(title="Hβ + [O III]4959,5007", names=o3hb_names),
            dict(title="Hδ + [O III]4363 (auroral)", names=aur_names),
            dict(title="[O II] UV 3727,3729", names=oiiuv_names),
        ]
        for zd in zoom_defs:
            restA = [REST_LINES_A[n] for n in zd["names"]]
            zd["lo"], zd["hi"] = _window_from_lines_um(restA, z, pad_A=100.0)
        base_lines = base.get("lines", {})
        show_flags = []
        for zd in zoom_defs:
            cov = _has_coverage_window(lam_um, zd["lo"], zd["hi"])
            fitok = _has_fit_in_window(base_lines, zd["names"], snr_min=1.0)
            show_flags.append(cov and fitok)

        # ----------------------------------------------
        # helper: single plotting routine for any unit
        # ----------------------------------------------
        def _plot_unit(unit="fnu"):
            if unit.lower() == "flam":
                flux = fnu_uJy_to_flam(flux_uJy, lam_um)
                err  = fnu_uJy_to_flam(err_uJy,  lam_um)
                cont = cont_flam
                mu   = mu_flam
                sig  = sig_flam
                ylabel = r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"
                tag = "flam"
            else:
                flux = flux_uJy
                err  = err_uJy
                cont = cont_uJy
                mu   = mu_uJy
                sig  = sig_uJy
                ylabel = r"$F_\nu$ [µJy]"
                tag = "fnu"

            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12.0, 8.2))
            gs = gridspec.GridSpec(2, 3, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.25)
            ax_full = fig.add_subplot(gs[0, :])
            ax_z = [fig.add_subplot(gs[1, i]) for i in range(3)]

            edges = _bin_edges_from_centers_um(lam_um)
            centers = 0.5 * (edges[:-1] + edges[1:])
            errbar_kwargs = dict(
                fmt='o', ms=1,
                mfc='white',
                mec=(0,0,0,0.35),   # marker edge much lighter
                mew=0.4,

                ecolor=(0,0,0,0.11),  # << reduce opacity of the *error bar*
                elinewidth=0.3,      # optionally thin the error lines
                capsize=1.0,

                alpha=0.5,            # marker transparency (still controls the dot)
                zorder=2.5
            )


            title = f"{source_id}   (z = {z:.3f})" if source_id else f"z = {z:.3f}"
            ax_full.set_title(title, fontsize=12, pad=8)
            ax_full.stairs(flux, edges, color="k", lw=1, alpha=0.9, label="Data")
            ax_full.errorbar(centers, flux, yerr=err, **errbar_kwargs)
            ax_full.stairs(cont, edges, color="b", ls="--", lw=0.7, label="Continuum")
            ax_full.stairs(mu, edges, color="r", lw=0.7, label="Model mean")
            ax_full.fill_between(centers, mu - sig, mu + sig, step='mid', color="r", alpha=0.18, lw=0)
            ax_full.set_xlabel("Observed wavelength [µm]")
            ax_full.set_ylabel(ylabel)
            ax_full.legend(ncol=3, fontsize=9, frameon=False)
            _annotate_lines(ax_full, which_lines, z, per_line=base["lines"], min_dx_um=0.01)

            # zoom panels
            for ax, zd, show in zip(ax_z, zoom_defs, show_flags):
                sel = (lam_um >= zd["lo"]) & (lam_um <= zd["hi"])
                if np.any(sel):
                    lam_z = lam_um[sel]
                    flux_z, err_z = flux[sel], err[sel]
                    cont_z, mu_z, sig_z = cont[sel], mu[sel], sig[sel]
                    edges_z = _bin_edges_from_centers_um(lam_z)
                    centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])
                    ax.stairs(flux_z, edges_z, color="k", lw=1, alpha=0.9)
                    ax.errorbar(centers_z, flux_z, yerr=err_z, **errbar_kwargs)
                    ax.stairs(cont_z, edges_z, color="b", lw=0.7, ls="--")
                    ax.stairs(mu_z, edges_z, color="r", lw=0.7)
                    ax.fill_between(centers_z, mu_z - sig_z, mu_z + sig_z,
                                    step="mid", alpha=0.18, lw=0)
                    _annotate_lines(ax, which_lines, z, per_line=base["lines"],
                                    min_dx_um=0.002, levels=(0.92, 0.80, 0.68))
                else:
                    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
                    ax.text(0.5, 0.5, "No coverage", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="0.5")
                if not show:
                    ax.text(0.5, 0.1, "No detection", transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=8, color="0.6", alpha=0.8)
                ax.set_xlim(zd["lo"], zd["hi"])
                ax.set_title(zd["title"], fontsize=10)
                ax.set_xlabel("Observed wavelength [µm]")
                ax.set_ylabel(ylabel)

            # save
            if save_path:
                root, ext = os.path.splitext(save_path)
                fname = f"{root}_{tag}.{save_format}" if ext == "" else f"{root}_{tag}{ext}"
                fig.savefig(fname, dpi=save_dpi, bbox_inches="tight", transparent=save_transparent)
                summary_txt = os.path.splitext(fname)[0] + "_summary.txt"
                print_bootstrap_line_table(dict(which_lines=which_lines, summary=summary),
                                           save_path=summary_txt)
            plt.show()
            plt.close(fig)

        # --- make plot(s) ---
        if plot_unit.lower() in ("flam", "both"):
            _plot_unit("flam")
        if plot_unit.lower() in ("fnu", "both"):
            _plot_unit("fnu")


    return {
        "samples": samples,
        "summary": summary,
        "which_lines": which_lines,
        "model_stack_flam": model_stack_flam,
        "keep_mask": keep_mask,
        "lam_um": lam_um,
        "data_flux_uJy": flux_uJy,
    }


def print_bootstrap_line_table(boot, save_path: str | None = None):
    """
    Print a formatted bootstrap summary to console.
    Optionally save the same output to a text file.
    """

    header = (
        "\n=== BOOTSTRAP SUMMARY (value ± error) ===\n"
        f"{'Line':10s} {'F_line [erg/s/cm²]':>26s} {'EW₀ [Å]':>16s} "
        f"{'σ_A [Å]':>14s} {'μ_obs [Å]':>16s} {'SNR':>12s}\n"
        + "-" * 100 + "\n"
    )

    lines = []
    for ln in boot["which_lines"]:
        s = boot["summary"][ln]
        lines.append(
            f"{ln:10s} "
            f"{s['F_line']['text']:>26s} "
            f"{s['EW0_A']['text']:>16s} "
            f"{s['sigma_A']['text']:>14s} "
            f"{s['lam_obs_A']['text']:>16s} "
            f"{s['SNR']['text']:>12s}"
        )

    table_text = header + "\n".join(lines)

    # Print to console
    print(table_text)

    # Optionally save to text file
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(table_text)
        print(f"\nSaved bootstrap summary → {save_path}")


"""
PyRSR — line_fit.py (linear-λ, pure Gaussian)
=============================================

- Independent Gaussian per line: flux A, width σ_A [Å], centroid μ_A [Å]
- σ-weighted polynomial continuum with emission-line masking
- Per-line centroid seeds from local observed peaks
- Area-normalized Gaussians in F_lambda (A equals integrated flux)
- Flux density plots (µJy) + [O III] zoom
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
from numpy.polynomial import Polynomial

# --- add near the top of line_fit.py (after imports) ---
try:
    from tqdm.auto import tqdm
except Exception:           # fallback if tqdm isn't installed
    def tqdm(x, *args, **kwargs):
        return x


# --------------------------------------------------------------------
# Imports from line.py (do not modify that file)
# --------------------------------------------------------------------
from PyRSR.line import (
    C_AA, LN10, REST_LINES_A,
    fnu_uJy_to_flam, flam_to_fnu_uJy,
    sigma_grating_logA,             # σ_gr in log10(λ), λ in µm
    line_centers_obs_A,             # nominal observed centers in Å
    measure_fluxes_profile_weighted,
    equivalent_widths_A,
    rescale_uncertainties,
    _lines_in_range,
    apply_balmer_absorption_correction
)

# ============================================================
# Helpers
# ============================================================

def _lyman_cut_um(z: float, which: str | None = "lya") -> float:
    if which is None:
        return -np.inf
    lam_rest_A = 1215.67 if str(which).lower() == "lya" else 912.0
    return lam_rest_A * (1.0 + z) / 1e4  # µm

from math import sqrt
from scipy.special import erf


def _pixel_edges_A(lam_A):
    """Compute pixel edges (Å) from center grid (Å)."""
    lam_A = np.asarray(lam_A, float)
    d = np.diff(lam_A)
    left  = np.r_[lam_A[0] - d[0]*0.5, lam_A[:-1] + 0.5*d]
    right = np.r_[lam_A[:-1] + 0.5*d, lam_A[-1] + d[-1]*0.5]
    return left, right

def _gauss_binavg_area_normalized_A(lam_left_A, lam_right_A, muA, sigmaA):
    """
    Per-pixel mean of an area-normalized Gaussian (∫ G dλ = 1) on Å edges.
    Returns array with same length as lam_left/right (number of pixels).
    """
    if not (np.isfinite(muA) and np.isfinite(sigmaA)) or sigmaA <= 0:
        return np.zeros_like(lam_left_A)
    inv = 1.0 / (sqrt(2.0) * sigmaA)
    # CDF differences over each pixel, then divide by pixel width => mean flux density
    cdf_right = 0.5 * (1.0 + erf((lam_right_A - muA) * inv))
    cdf_left  = 0.5 * (1.0 + erf((lam_left_A  - muA) * inv))
    area = cdf_right - cdf_left
    width = (lam_right_A - lam_left_A)
    width = np.where(width > 0, width, np.nan)
    mean = area / width
    mean[~np.isfinite(mean)] = 0.0
    return mean

def _safe_median(x, default):
    x = np.asarray(x, float)
    v = np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan
    return v if np.isfinite(v) else default

# ============================================================
# Continuum (σ-weighted polynomial, with line masking)
# ============================================================

def fit_continuum_polynomial(lam_um, flam, z, deg=2, windows=None,
                             lyman_cut="lya", sigma_flam=None,
                             grating="PRISM", clip_sigma=2.5, max_iter=5):
    ly_um = _lyman_cut_um(z, lyman_cut)
    mask = lam_um >= ly_um

    if windows:
        mask_win = np.zeros_like(mask, dtype=bool)
        for (lo, hi) in windows:
            mask_win |= (lam_um >= float(lo)) & (lam_um <= float(hi))
        mask &= mask_win

    lam_fit = lam_um[mask]
    flam_fit = flam[mask]
    if sigma_flam is None:
        sigma_flam = np.full_like(flam, max(1e-30, _safe_median(flam, 0.0)) * 0.05)
    sig_fit = sigma_flam[mask]

    # Mask ±4 σ_inst (×1.5) around all catalog lines
    lam_A = lam_fit * 1e4
    for lam_rest in REST_LINES_A.values():
        muA = lam_rest * (1 + z)
        mu_um = muA / 1e4
        sig_gr_log = float(sigma_grating_logA(grating, mu_um))  # log10
        sigma_A = muA * LN10 * sig_gr_log * 1.5
        core = (lam_A > muA - 4 * sigma_A) & (lam_A < muA + 4 * sigma_A)
        flam_fit[core] = np.nan
        sig_fit[core] = np.nan

    good = np.isfinite(flam_fit) & np.isfinite(sig_fit) & (sig_fit > 0)
    lam_fit, flam_fit, sig_fit = lam_fit[good], flam_fit[good], sig_fit[good]

    if lam_fit.size < max(deg + 2, 10):
        med = _safe_median(flam_fit, _safe_median(flam, 0.0))
        return np.full_like(flam, med), np.array([med])

    w = 1.0 / np.clip(sig_fit, _safe_median(sig_fit, 1.0) * 1e-3, np.inf)
    w /= np.nanmax(w)

    for _ in range(max_iter):
        p = Polynomial.fit(lam_fit, flam_fit, deg, w=w)
        m = p(lam_fit)
        resid = (flam_fit - m) / np.clip(sig_fit, 1e-30, None)
        keep = np.abs(resid) < clip_sigma
        if np.all(keep):
            break
        lam_fit, flam_fit, sig_fit = lam_fit[keep], flam_fit[keep], sig_fit[keep]
        w = 1.0 / np.clip(sig_fit, _safe_median(sig_fit, 1.0) * 1e-3, np.inf)
        w /= np.nanmax(w)

    p_final = Polynomial.fit(lam_fit, flam_fit, deg, w=w)
    return p_final(lam_um), p_final.convert().coef

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
    """
    params per line: [A_1..A_N, sigmaA_1..sigmaA_N, muA_1..muA_N]
      A_j      : integrated flux (erg s^-1 cm^-2)
      sigmaA_j : Gaussian sigma in Å
      muA_j    : centroid in Å
    """
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
        # mean flux density per pixel from unit-area Gaussian, then scale by flux A_j
        profA = _gauss_binavg_area_normalized_A(left_A, right_A, muA[j], sj)
        prof_flam = A[j] * profA
        model += prof_flam
        profiles[name] = prof_flam
        # for EW machinery: supply (mu_A, sigma_log10λ) — convert σ_A→σ_log10λ
        sigma_logA = sj / (muA[j] * LN10) if (np.isfinite(muA[j]) and muA[j] > 0) else np.nan
        centers[name] = (muA[j], sigma_logA)

    return model, profiles, centers


# ============================================================
# Robust seeds/bounds finalization (prevents x0-outside-bounds)
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

    # --- Convert to F_lambda ---
    flam = fnu_uJy_to_flam(flux_uJy, lam_um)
    sig_flam = fnu_uJy_to_flam(err_uJy, lam_um)

    # --- Continuum ---
    Fcont, _ = fit_continuum_polynomial(
        lam_um, flam, z, deg=deg, windows=continuum_windows,
        lyman_cut=lyman_cut, sigma_flam=sig_flam, grating=grating
    )
    resid_full = flam - Fcont
    sig_flam = rescale_uncertainties(resid_full, sig_flam)

    # --- Fit window ---
    if fit_window_um:
        lo, hi = fit_window_um
        w = (lam_um >= lo) & (lam_um <= hi)
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um[w], resid_full[w], sig_flam[w], Fcont[w]
    else:
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um, resid_full, sig_flam, Fcont

    # --- Lines in coverage ---
    which_lines = _lines_in_range(z, lam_fit, lines_to_use, margin_um=0.02)
    if not which_lines:
        raise ValueError("No emission lines in wavelength coverage.")

    # --- Instrument σ and centroid seeds ---
    expected_um = np.array([REST_LINES_A[nm]*(1+z)/1e4 for nm in which_lines])
    sigma_gr_log = np.array([sigma_grating_logA(grating, mu_um) for mu_um in expected_um])  # log10
    muA_nom = expected_um * 1e4                           # Å
    sigmaA_inst = muA_nom * LN10 * sigma_gr_log           # Å (instrumental σ)
    sigma_um_inst = sigmaA_inst / 1e4                     # µm (for peak search)

    mu_seed_um = _find_local_peaks(lam_fit, resid_fit, expected_um, sigma_um_inst, min_window_um=0.001)
    muA_seed = mu_seed_um * 1e4  # Å

    # --- Seeds & bounds per line ---
    nL = len(which_lines)
    rms = _safe_median(resid_fit, 0.0)

    # Amplitudes (≥0)
    A0, A_lo, A_hi = [], np.zeros(nL), []
    for mu in expected_um:
        m = (lam_fit > mu - 0.05) & (lam_fit < mu + 0.05)
        pk = np.nanmax(resid_fit[m]) if np.any(m) else 3.0*max(rms, 1e-30)
        A0.append(max(0.5*rms, 1.5*pk))
        A_hi.append(max(1000.0*rms, 10.0*pk))
    A0 = np.array(A0, float); A_hi = np.array(A_hi, float)

    # Widths σ_A (Å)
    g = str(grating).lower()
    if "prism" in g:
        sigmaA_lo = np.maximum(0.25 * sigmaA_inst, 0.02)
        sigmaA_hi = np.maximum(2.5  * sigmaA_inst, 0.05)
    else:
        sigmaA_lo = np.maximum(0.20 * sigmaA_inst, 0.02)
        sigmaA_hi = np.maximum(1.50 * sigmaA_inst, 0.05)
    sigmaA_seed = np.clip(0.8 * np.where(np.isfinite(sigmaA_inst), sigmaA_inst, 1.0),
                          sigmaA_lo, sigmaA_hi)

    # Centroids μ_A (Å)
    if "prism" in g:
        muA_lo = muA_seed - 12.0 * np.maximum(sigmaA_inst, 1.0)
        muA_hi = muA_seed + 12.0 * np.maximum(sigmaA_inst, 1.0)
    else:
        muA_lo = muA_seed - 6.0  * np.maximum(sigmaA_inst, 1.0)
        muA_hi = muA_seed + 6.0  * np.maximum(sigmaA_inst, 1.0)

    # Pack parameters and finalize to guarantee x0 inside bounds
    p0 = np.r_[A0, sigmaA_seed, muA_seed]
    lb = np.r_[A_lo, sigmaA_lo, muA_lo]
    ub = np.r_[A_hi, sigmaA_hi, muA_hi]
    p0, lb, ub = _finalize_seeds_and_bounds(p0, lb, ub)

    # --- χ² residuals with pixel-width weighting in Å ---
    lamA_fit = lam_fit * 1e4
    dlA = np.gradient(lamA_fit)
    w_pix = np.sqrt(dlA / _safe_median(dlA, 1.0))

    def fun(p):
        model, _, _ = build_model_flam_linear(p, lam_fit, z, grating, which_lines, mu_seed_um)
        return w_pix * (resid_fit - model) / np.clip(sig_fit, 1e-30, None)

    res = least_squares(fun, p0, bounds=(lb, ub), max_nfev=200000, xtol=1e-12, ftol=1e-12)

    # --- Best-fit model on window ---
    model_flam_win, profiles_win, centers = build_model_flam_linear(
        res.x, lam_fit, z, grating, which_lines, mu_seed_um
    )

    # --- Fluxes & EWs (call with POSITIONAL args; no kwargs) ---
    fluxes = measure_fluxes_profile_weighted(
        lam_fit,                # lam_um
        resid_fit,              # flam_sub (continuum-subtracted)
        sig_fit,                # sigma_flam
        model_flam_win,         # model_flam
        profiles_win,           # profiles
        centers                 # centers
    )
    ews = equivalent_widths_A(fluxes, lam_fit, Fcont_fit, z, centers)
    if absorption_corrections:
        fluxes = apply_balmer_absorption_correction(fluxes, absorption_corrections)

    # --- Collect per-line outputs (add σ_A & FWHM_A for convenience) ---
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

    # --- Plot in flux density (µJy): data & (continuum + lines) ---
    if plot:
        if fit_window_um:
            model_full = np.zeros_like(lam_um)
            model_full[w] = model_flam_win
            total_model_flam = Fcont + model_full
        else:
            total_model_flam = Fcont + model_flam_win

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False,
                                       gridspec_kw={'height_ratios': [1.6, 1.0]})

        # Full
        ax1.plot(lam_um, flux_uJy, 'k-', lw=0.8, label='Data')
        ax1.plot(lam_um, flam_to_fnu_uJy(Fcont, lam_um), 'b--', lw=1.0, label=f'Continuum (deg={deg})')
        ax1.plot(lam_um, flam_to_fnu_uJy(total_model_flam, lam_um), 'r-', lw=1.2, label='Continuum + Lines')
        ax1.set_ylabel('Flux density [µJy]')
        ax1.legend(ncol=3, fontsize=9)
        ax1.grid(alpha=0.25)

        # Zoom [O III] region (typical for z~8)
        o3_lo, o3_hi = 4.45, 4.70
        mZ = (lam_um > o3_lo) & (lam_um < o3_hi)
        if np.any(mZ):
            ax2.plot(lam_um[mZ], flux_uJy[mZ], 'k-', lw=0.9, label='Data')
            ax2.plot(lam_um[mZ], flam_to_fnu_uJy(Fcont[mZ], lam_um[mZ]), 'b--', lw=0.9, label='Continuum')
            ax2.plot(lam_um[mZ], flam_to_fnu_uJy(total_model_flam[mZ], lam_um[mZ]), 'r-', lw=1.0, label='Model')
            ax2.set_xlim(o3_lo, o3_hi)
            ax2.legend(fontsize=8)
        ax2.set_xlabel('Observed wavelength [µm]')
        ax2.set_ylabel('Flux density [µJy]')
        ax2.grid(alpha=0.25)
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
# Bootstrapping with full-spectrum & OIII/Hβ zoom plots
# ============================================================
from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def bootstrap_excels_fit(
    source,
    z,
    grating="PRISM",
    n_boot=200,
    deg=2,
    continuum_windows=None,
    lyman_cut="lya",
    fit_window_um=None,
    absorption_corrections=None,
    random_state=None,
    verbose=False,
    plot=True,
    show_progress=True,
    # ------ NEW: saving controls ------
    save_path: str | None = None,   # file path or directory; None = don't save
    save_dpi: int = 500,            # e.g. 300, 500, 600
    save_format: str = "png",       # png/pdf/svg/eps…
    save_transparent: bool = False,
):
    """
    Robust parametric bootstrap for excels_fit_poly.
    Returns per-line metrics as value ± error (mean ± std of filtered draws)
    and includes preformatted strings in summary[line][metric]["text"].

    If `save_path` is provided, the bootstrap figure is saved at `save_dpi`.
    - If `save_path` is a directory, a filename will be auto-generated.
    - If it's a filepath, that exact path (and extension) will be used.
    """
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

    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap", unit="draw")

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

        # full F_lambda model on lam_um
        cont = fb.get("continuum_flam", np.zeros_like(lam_um))
        if isinstance(wfit, slice):
            total_flam = cont + fb.get("model_window_flam", np.zeros_like(lam_um))
        else:
            tmp = np.zeros_like(lam_um)
            tmp[wfit] = fb.get("model_window_flam", np.zeros_like(lam_um[wfit]))
            total_flam = cont + tmp

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
        val = np.nanmean(x); err = np.nanstd(x)
        if np.count_nonzero(np.isfinite(x)) >= 8:
            xc = x[np.isfinite(x)]
            m, s = np.nanmean(xc), np.nanstd(xc)
            xc = xc[(xc > m - 3*s) & (xc < m + 3*s)]
            val, err = (np.nanmean(xc), np.nanstd(xc)) if xc.size else (val, err)
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
        mu_flam, sig_flam = _sigma_clip_mean_std(model_stack_flam[keep_mask], axis=0, sigma=3.0)
        mu_uJy  = flam_to_fnu_uJy(mu_flam,  lam_um)
        sig_uJy = flam_to_fnu_uJy(sig_flam, lam_um)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                       gridspec_kw={"height_ratios": [1.6, 1.0]})
        ax1.plot(lam_um, flux_uJy, "k-", lw=0.8, label="Data")
        ax1.plot(lam_um, mu_uJy,  "r-", lw=1.0, label="Model (mean, clipped)")
        ax1.fill_between(lam_um, mu_uJy - sig_uJy, mu_uJy + sig_uJy,
                         color="r", alpha=0.18, linewidth=0,
                         label="Model ±1σ (bootstrap)")
        ax1.set_ylabel("Flux density [µJy]")
        ax1.grid(alpha=0.25); ax1.legend(ncol=3, fontsize=9)

        zlo, zhi = _o3_hb_zoom_bounds_um(z, pad_um=0.03)
        mz = (lam_um >= zlo) & (lam_um <= zhi)
        if np.any(mz):
            ax2.plot(lam_um[mz], flux_uJy[mz], "k-", lw=0.9, label="Data")
            ax2.plot(lam_um[mz], mu_uJy[mz],  "r-", lw=1.0, label="Model (mean)")
            ax2.fill_between(lam_um[mz], mu_uJy[mz] - sig_uJy[mz], mu_uJy[mz] + sig_uJy[mz],
                             color="r", alpha=0.2, linewidth=0)
            ax2.set_xlim(zlo, zhi); ax2.legend(fontsize=8)
        ax2.set_xlabel("Observed wavelength [µm]")
        ax2.set_ylabel("Flux density [µJy]")
        ax2.grid(alpha=0.25)
        plt.tight_layout()

        # ---- SAVE HERE if requested ----
        if save_path:
            # If a directory was given, build a filename
            root, ext = os.path.splitext(save_path)
            if os.path.isdir(save_path):
                fname = os.path.join(save_path, f"bootstrap_{grating.lower()}_{n_boot}draws.{save_format}")
            else:
                # If no extension in provided path, add one from save_format
                if ext == "":
                    fname = f"{save_path}.{save_format}"
                else:
                    fname = save_path
            fig.savefig(fname, dpi=save_dpi, format=fname.split(".")[-1],
                        bbox_inches="tight", transparent=save_transparent)
        plt.show()
        plt.close(fig)

    return {
        "samples": samples,
        "summary": summary,
        "which_lines": which_lines,
        "model_stack_flam": model_stack_flam,
        "keep_mask": keep_mask,
        "lam_um": lam_um,
        "data_flux_uJy": flux_uJy,
    }

def print_bootstrap_line_table(boot):
    print("\n=== BOOTSTRAP SUMMARY (value ± error) ===")
    print(f"{'Line':10s} {'F_line [erg/s/cm²]':>26s} {'EW₀ [Å]':>16s} {'σ_A [Å]':>14s} "
          f"{'μ_obs [Å]':>16s} {'SNR':>12s}")
    print("-"*100)
    for ln in boot["which_lines"]:
        s = boot["summary"][ln]
        print(f"{ln:10s} "
              f"{s['F_line']['text']:>26s} "
              f"{s['EW0_A']['text']:>16s} "
              f"{s['sigma_A']['text']:>14s} "
              f"{s['lam_obs_A']['text']:>16s} "
              f"{s['SNR']['text']:>12s}")

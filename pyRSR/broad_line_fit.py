"""
centrod bounds and peak width
Broad-line emission fitting with BIC-based model selection. new

Fits emission lines using area-normalized Gaussians with optional broad
Balmer components (H-delta, H-beta, H-alpha). Uses Bayesian Information
Criterion to select between narrow-only and broad+narrow models.

Functions
---------
single_broad_fit : Fit a single spectrum with optional broad components
broad_fit : Fit with bootstrap uncertainty estimation
"""

from __future__ import annotations

import os
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares
from scipy.special import erf

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


from PyRSR.fitting_helpers import (
    fnu_uJy_to_flam, flam_to_fnu_uJy,
    sigma_grating_logA,             # σ_gr in log10(λ), λ in µm
    measure_fluxes_profile_weighted,
    equivalent_widths_A,
    rescale_uncertainties,
    apply_balmer_absorption_correction,
    REST_LINES_A,
    get_default_line_list,
)

__all__ = [
    "single_broad_fit",
    "broad_fit",
    "print_bootstrap_line_table_broad",
]


# --------------------------------------------------------------------
# Constants + rest wavelengths
# --------------------------------------------------------------------

C_AA = 2.99792458e18  # Speed of light [Å·Hz·s⁻¹]
LN10 = np.log(10.0)   # ln(10)



# Broad Balmer aliases (same rest λ, but allowed much larger σ)
REST_LINES_A["HBETA_BROAD"] = REST_LINES_A["HBETA"]
REST_LINES_A["HBETA_BROAD2"] = REST_LINES_A["HBETA"]


REST_LINES_A["H⍺_BROAD"] = REST_LINES_A["H⍺"]
# Second broad Hα component (same rest λ)
REST_LINES_A["H⍺_BROAD2"] = REST_LINES_A["H⍺"]

REST_LINES_A["HDELTA_BROAD"] = REST_LINES_A["HDELTA"]
REST_LINES_A["HDELTA_BROAD2"] = REST_LINES_A["HDELTA"]



MAX_BROAD_FWHM_UM = 0.0001
MAX_BROAD_SIGMA_A = (MAX_BROAD_FWHM_UM / 2.355) * 1e4  # ≈ 2.1e2 Å



# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _lyman_cut_um(z: float, which: str | None = "lya") -> float:
    if which is None:
        return -np.inf
    lam_rest_A = 1215.67 if str(which).lower() == "lya" else 912.0
    return lam_rest_A * (1.0 + z) / 1e4  # µm


def _pixel_edges_A(lam_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel edges in Å from wavelength centres in Å."""
    lam_A = np.asarray(lam_A, float)
    d = np.diff(lam_A)
    left  = np.r_[lam_A[0] - 0.5 * d[0], lam_A[:-1] + 0.5 * d]
    right = np.r_[lam_A[:-1] + 0.5 * d, lam_A[-1] + 0.5 * d[-1]]
    return left, right


def _pixel_edges_um(lam_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel edges in µm from wavelength centres in µm."""
    lam_um = np.asarray(lam_um, float)
    d = np.diff(lam_um)
    left  = np.r_[lam_um[0] - 0.1 * d[0], lam_um[:-1] + 0.1 * d]
    right = np.r_[lam_um[:-1] + 0.1 * d, lam_um[-1] + 0.1 * d[-1]]
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
    mean[~np.isfinite(mean)] = 0.0
    return mean


def _safe_median(x, default):
    x = np.asarray(x, float)
    v = np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan
    return v if np.isfinite(v) else default

def _annotate_lines_zoom_no_overlap(
    ax,
    lam_z,
    model_z,
    line_names,
    z,
    per_line=None,
    label_offset_frac=0.06,
    x_margin_frac=0.02,
    shorten_names=True,
):
    """
    Annotate lines in a zoom panel:

    - Horizontal text (no rotation).
    - Each label sits a fixed fraction above the local model peak.
    - Simple de-overlap: if two lines are very close in x, later labels
      are nudged further upward.

    Parameters
    ----------
    ax : matplotlib Axes
    lam_z : array
        Wavelength grid in this zoom panel [µm].
    model_z : array
        Model flux in this zoom panel (same units as y-axis).
    line_names : list of str
        Line IDs to try to annotate (e.g. ["HBETA","OIII_4959","OIII_5007"]).
    z : float
        Redshift.
    per_line : dict or None
        base["lines"] dictionary with "lam_obs_A" etc. If present, use the
        fitted centroid; otherwise fall back to rest λ × (1+z).
    label_offset_frac : float
        Fraction of total y-range to place labels above the local peak.
    x_margin_frac : float
        Fraction of x-range for defining "closeness" in x; used to
        vertically stagger labels that would otherwise overlap.
    """
    lam_z = np.asarray(lam_z, float)
    model_z = np.asarray(model_z, float)

    if lam_z.size == 0 or model_z.size == 0:
        return

    x_lo, x_hi = np.nanmin(lam_z), np.nanmax(lam_z)
    xs, y_peaks, labels = [], [], []

    for nm in line_names:
        # --- observed wavelength ---
        if (per_line is not None) and (nm in per_line) and np.isfinite(per_line[nm].get("lam_obs_A", np.nan)):
            lam_c = float(per_line[nm]["lam_obs_A"]) / 1e4
        else:
            if nm not in REST_LINES_A:
                continue
            lam_c = REST_LINES_A[nm] * (1.0 + z) / 1e4

        if not (x_lo <= lam_c <= x_hi):
            continue

        # --- local model peak around that line ---
        # (use a small window; 0.02 µm usually fine for NIRSpec)
        m = np.abs(lam_z - lam_c) <= 0.02
        if not np.any(m):
            continue

        y_loc = model_z[m]
        if not np.any(np.isfinite(y_loc)):
            continue

        y_peak = float(np.nanmax(y_loc))

        lab = nm
        if shorten_names:
            lab = lab.replace("[", "").replace("]", "").replace("_", " ")
            lab = lab.replace("HALPHA", "Hα").replace("HBETA", "Hβ")

        xs.append(lam_c)
        y_peaks.append(y_peak)
        labels.append(lab)

    if not xs:
        return

    xs = np.asarray(xs, float)
    y_peaks = np.asarray(y_peaks, float)
    labels = np.asarray(labels, dtype=object)

    # sort left→right, so we can stagger “close” neighbours
    order = np.argsort(xs)
    xs, y_peaks, labels = xs[order], y_peaks[order], labels[order]

    y_min, y_max = ax.get_ylim()
    dy = (y_max - y_min) * float(label_offset_frac)
    dx_thresh = (x_hi - x_lo) * float(x_margin_frac)

    placed_ys = []

    for i, (x, y0, lab) in enumerate(zip(xs, y_peaks, labels)):
        y = y0 + dy

        # If this line is very close in x to the previous one,
        # nudge it further up to avoid overlapping text.
        if i > 0 and abs(x - xs[i - 1]) < dx_thresh:
            y = max(y, placed_ys[-1] + dy)

        placed_ys.append(y)

        ax.text(
            x, y, lab,
            ha="center", va="bottom",
            rotation=0,
            fontsize=8,
            color="0.25",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )

def _annotate_lines_above_model(
    ax,
    lam,
    model,
    line_names,
    z,
    per_line=None,
    shorten_names=True,
    label_offset_frac=0.06,
    x_margin_frac=0.02,
    fontsize=8,
):
    """
    Annotate emission lines in data coordinates:

    - Uses the *model* (not raw data) to find the local peak.
    - Places each label a fixed fraction of the y-range above that peak.
    - Horizontal text (no rotation).
    - Simple de-overlap: if line centres are very close in x, later
      labels are nudged further upward.

    Parameters
    ----------
    ax : matplotlib Axes
    lam : array
        Wavelength grid in this panel [µm].
    model : array
        Model flux in this panel (same units as y-axis).
    line_names : list of str
        Line IDs to try to annotate (e.g. ["HBETA","OIII_4959","OIII_5007"]).
    z : float
        Redshift.
    per_line : dict or None
        base["lines"] dictionary with "lam_obs_A" etc. If present, use the
        fitted centroid; otherwise fall back to rest λ × (1+z).
    label_offset_frac : float
        Fraction of total y-range to place labels above the local peak.
    x_margin_frac : float
        Fraction of x-range used to decide when two lines are "close"
        in x and should be vertically staggered.
    """
    lam = np.asarray(lam, float)
    model = np.asarray(model, float)
    if lam.size == 0 or model.size == 0:
        return

    x_lo, x_hi = np.nanmin(lam), np.nanmax(lam)
    y_min, y_max = ax.get_ylim()
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    xs, y_peaks, labels = [], [], []

    for nm in line_names:
        # --- observed wavelength: fitted if available, else rest*(1+z) ---
        if (per_line is not None) and (nm in per_line) and np.isfinite(
            per_line[nm].get("lam_obs_A", np.nan)
        ):
            lam_c = float(per_line[nm]["lam_obs_A"]) / 1e4
        else:
            if nm not in REST_LINES_A:
                continue
            lam_c = REST_LINES_A[nm] * (1.0 + z) / 1e4

        if not (x_lo <= lam_c <= x_hi):
            continue

        # --- local model peak around that line ---
        # window size ~0.02 µm is fine for NIRSpec; tweak if you like
        win = np.abs(lam - lam_c) <= 0.02
        if not np.any(win):
            continue

        y_loc = model[win]
        if not np.any(np.isfinite(y_loc)):
            continue

        y_peak = float(np.nanmax(y_loc))

        lab = nm
        if shorten_names:
            lab = lab.replace("[", "").replace("]", "").replace("_", " ")
            lab = lab.replace("HALPHA", "Hα").replace("HBETA", "Hβ")

        xs.append(lam_c)
        y_peaks.append(y_peak)
        labels.append(lab)

    if not xs:
        return

    xs = np.asarray(xs, float)
    y_peaks = np.asarray(y_peaks, float)
    labels = np.asarray(labels, dtype=object)

    # sort left→right, then stagger vertically for close neighbours
    order = np.argsort(xs)
    xs, y_peaks, labels = xs[order], y_peaks[order], labels[order]

    dy = (y_max - y_min) * float(label_offset_frac)
    dx_thresh = (x_hi - x_lo) * float(x_margin_frac)

    placed_ys = []

    for i, (x, y0, lab) in enumerate(zip(xs, y_peaks, labels)):
        # baseline: fixed offset above local peak
        y = y0 + dy

        # nudge up if close in x to previous label
        if i > 0 and abs(x - xs[i - 1]) < dx_thresh:
            y = max(y, placed_ys[-1] + 0.8 * dy)

        # avoid going off the top
        y = min(y, y_max - 0.3 * dy)

        placed_ys.append(y)

        ax.text(
            x, y, lab,
            ha="center", va="bottom",
            rotation=0,
            fontsize=fontsize,
            color="0.25",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )



def _annotate_lines(ax, which_lines, z, per_line=None,
                    text_color='0.25',
                    levels=(0.90, 0.78, 0.66, 0.54),
                    min_dx_um=0.01,
                    fontsize=8,
                    only_within_xlim=True,
                    shorten_names=True):
    """
    Annotate emission lines as staggered text labels (no vertical guide lines).
    """
    if not which_lines:
        return

    xs, labels = [], []
    for nm in which_lines:
        if (per_line is not None) and (nm in per_line) and np.isfinite(per_line[nm].get('lam_obs_A', np.nan)):
            lam_um = per_line[nm]['lam_obs_A'] / 1e4
        else:
            if nm not in REST_LINES_A:
                continue
            lam_um = REST_LINES_A[nm] * (1.0 + z) / 1e4

        lab = nm
        if shorten_names:
            lab = lab.replace('[', '').replace(']', '').replace('_', ' ')
        xs.append(float(lam_um))
        labels.append(lab)

    if not xs:
        return

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

def _obs_um_from_rest_A(rest_A, z):
    return (np.asarray(rest_A, float) * (1.0 + z)) / 1e4

def _lines_in_range(z, lam_obs_um, lines=None, margin_um=0.02):
    """
    Return the subset of `lines` (or all REST_LINES_A if None) whose
    observed centres fall within the provided wavelength array (±margin).
    """
    lam_obs_um = np.asarray(lam_obs_um, float)
    finite = np.isfinite(lam_obs_um)

    # NEW: handle no coverage / all-NaN gracefully
    if lam_obs_um.size == 0 or not np.any(finite):
        return []

    lo, hi = np.nanmin(lam_obs_um[finite]), np.nanmax(lam_obs_um[finite])
    pool = REST_LINES_A if lines is None else {k: REST_LINES_A[k] for k in lines}

    keep = []
    for nm, lam0_A in pool.items():
        mu_um = lam0_A * (1.0 + z) / 1e4
        if (lo - margin_um) <= mu_um <= (hi + margin_um):
            keep.append(nm)
    return keep


def _build_width_tie_groups(which_lines):
    """
    Return list of groups of indices in `which_lines` whose Gaussian σ_A
    should be tied together.

    For now:
      - Tie the *narrow* Hα + [N II] λλ6549,6585 components:
        (NII_6549, HALPHA, NII_6585) share a single σ.

    We deliberately do NOT include HALPHA_BROAD here.
    """
    name_to_idx = {nm: i for i, nm in enumerate(which_lines)}
    groups = []

    ha_triplet = ["NII_6549", "H⍺", "NII_6585"]
    if all(nm in name_to_idx for nm in ha_triplet):
        groups.append([name_to_idx[nm] for nm in ha_triplet])

    # If you later want to tie broad components too, e.g.:
    # broad_grp = ["HBETA_BROAD", "HALPHA_BROAD"]
    # if all(nm in name_to_idx for nm in broad_grp):
    #     groups.append([name_to_idx[nm] for nm in broad_grp])

    return groups


# --------------------------------------------------------------------
# Continuum helpers
# --------------------------------------------------------------------

def continuum_windows_two_sides_of_lya(
    lam_source_or_dict,
    z: float,
    buffer_rest_A: float = 200.0,
    min_pts: int = 12,
    verbose: bool = False,
):
    """
    Return up to two (lo_um, hi_um) windows for continuum fitting:
      - Blueward: [λ_min,  λ_Lyα_obs - buffer_obs]
      - Redward : [λ_Lyα_obs + buffer_obs, λ_max]
    """
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

    LYA_REST_A = 1215.67
    lya_obs_um = LYA_REST_A * (1.0 + z) / 1e4
    buf_obs_um = buffer_rest_A * (1.0 + z) / 1e4

    blue_lo, blue_hi = lam_min, min(lya_obs_um - buf_obs_um, lam_max)
    red_lo,  red_hi  = max(lya_obs_um + buf_obs_um, lam_min), lam_max

    windows: List[Tuple[float, float]] = []

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


def fit_continuum_polynomial(lam_um, flam, z, deg=2, windows=None,
                             lyman_cut="lya", sigma_flam=None,
                             grating="PRISM", clip_sigma=2.5, max_iter=5):
    """
    Fit σ-weighted polynomial continuum *independently in each specified window*,
    masking emission lines. Returns stitched full-spectrum continuum and coef array.
    """
    lam_um = np.asarray(lam_um, float)
    flam = np.asarray(flam, float)

    ly_um = _lyman_cut_um(z, lyman_cut)
    global_mask = lam_um >= ly_um

    if sigma_flam is None:
        sigma_flam = np.full_like(flam, max(1e-30, _safe_median(flam, 0.0)) * 0.05)
    sigma_flam = np.asarray(sigma_flam, float)

    if not windows:
        windows = [(float(np.nanmin(lam_um)), float(np.nanmax(lam_um)))]

    Fcont = np.full_like(flam, np.nan)

    for (lo, hi) in windows:
        m = (lam_um >= lo) & (lam_um <= hi) & global_mask
        if np.count_nonzero(m) < max(deg + 2, 10):
            continue

        lam_fit = lam_um[m]
        flam_fit = flam[m].copy()
        sig_fit = sigma_flam[m].copy()

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

    if np.any(~np.isfinite(Fcont)):
        good = np.isfinite(Fcont)
        if np.count_nonzero(good) >= 2:
            Fcont = np.interp(
                lam_um, lam_um[good], Fcont[good],
                left=Fcont[good][0], right=Fcont[good][-1]
            )
        else:
            Fcont[:] = _safe_median(flam, 0.0)

    mgood = np.isfinite(Fcont)
    if np.count_nonzero(mgood) >= max(deg + 2, 10):
        w_all = 1.0 / np.clip(sigma_flam[mgood], _safe_median(sigma_flam[mgood], 1.0) * 1e-3, np.inf)
        w_all /= np.nanmax(w_all)
        p_coefs = Polynomial.fit(lam_um[mgood], Fcont[mgood], deg, w=w_all).convert().coef
    else:
        p_coefs = np.array([_safe_median(Fcont[mgood], _safe_median(flam, 0.0))])

    return Fcont, p_coefs


def _prune_lines_without_data(
    lam_fit, resid_fit, sig_fit, which_lines, z, grating,
    min_pts_per_line: int = 3,
    window_sigma: float = 4.0,
    window_um: float | None = None,
    verbose: bool = False,
):
    """
    Keep only lines that have >= min_pts_per_line finite samples near the expected centre.
    """
    if not which_lines:
        return [], np.zeros(0, dtype=bool)

    expected_um  = np.array([REST_LINES_A[nm] * (1 + z) / 1e4 for nm in which_lines])
    sigma_gr_log = np.array([sigma_grating_logA(grating, mu_um) for mu_um in expected_um])
    sigmaA_inst  = expected_um * 1e4 * LN10 * sigma_gr_log
    sigma_um_inst = sigmaA_inst / 1e4
    

    finite = np.isfinite(resid_fit) & np.isfinite(sig_fit) & (sig_fit > 0)
    keep: List[bool] = []

    for mu_um, s_um, nm in zip(expected_um, sigma_um_inst, which_lines):
        half = float(window_um) if (window_um is not None) else (window_sigma * s_um)
        m = (lam_fit > mu_um - half) & (lam_fit < mu_um + half) & finite
        ok = (np.count_nonzero(m) >= int(min_pts_per_line))
        keep.append(ok)
        if verbose and not ok:
            print(f"[skip] {nm}: no usable pixels within ±{half:.5f} µm")

    keep = np.asarray(keep, bool)
    return [ln for ln, k in zip(which_lines, keep) if k], keep


def _find_local_peaks(lam_um, resid_flam, expected_um,
                      sigma_gr_um=None,
                      min_window_um=0.001,
                      per_line_halfwidth_um=None):
    """
    Find local peaks near expected wavelengths.
    """
    peaks = []
    expected_um = np.asarray(expected_um, float)
    if per_line_halfwidth_um is not None:
        per_line_halfwidth_um = np.asarray(per_line_halfwidth_um, float)

    for i, mu0 in enumerate(expected_um):
        if per_line_halfwidth_um is not None:
            w = float(max(per_line_halfwidth_um[i], min_window_um))
        else:
            if sigma_gr_um is None:
                w = float(min_window_um)
            else:
                w = float(max(5.0 * float(np.asarray(sigma_gr_um)[i]), min_window_um))

        m = (lam_um > mu0 - w) & (lam_um < mu0 + w)
        if np.count_nonzero(m) > 2:
            y = resid_flam[m]
            peaks.append(lam_um[m][np.nanargmax(y)] if np.any(np.isfinite(y)) else mu0)
        else:
            peaks.append(mu0)
    return np.array(peaks, float)

def _window_from_lines_um(rest_list, z, pad_A=250.0):
    """
    Return [lo_um, hi_um] spanning the given rest-Å lines at redshift z,
    with symmetric padding of pad_A (rest Å) on each side.
    """
    obs_um = _obs_um_from_rest_A(rest_list, z)
    lo = np.min(obs_um) - pad_A * (1 + z) / 1e4
    hi = np.max(obs_um) + pad_A * (1 + z) / 1e4
    return float(lo), float(hi)

def build_model_flam_linear(params, lam_um, z, grating, which_lines, mu_seed_um):
    """
    Sum of area-normalised Gaussians in F_lambda.
    """
    nL = len(which_lines)
    A = np.array(params[0:nL], float)
    sigmaA = np.array(params[nL:2*nL], float)
    muA = np.array(params[2*nL:3*nL], float)

    lam_A = lam_um * 1e4
    left_A, right_A = _pixel_edges_A(lam_A)

    model = np.zeros_like(lam_um, float)
    profiles: Dict[str, np.ndarray] = {}
    centers: Dict[str, Tuple[float, float]] = {}

    pix_A = np.median(np.diff(lam_A))
    min_sigma = max(0.35 * pix_A, 0.01)

    for j, name in enumerate(which_lines):
        sj = np.clip(sigmaA[j], min_sigma, 1e6)
        profA = _gauss_binavg_area_normalized_A(left_A, right_A, muA[j], sj)
        prof_flam = A[j] * profA
        model += prof_flam
        profiles[name] = prof_flam
        sigma_logA = sj / (muA[j] * LN10) if (np.isfinite(muA[j]) and muA[j] > 0) else np.nan
        centers[name] = (muA[j], sigma_logA)

    return model, profiles, centers


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


# --------------------------------------------------------------------
# Matched-filter + correlated-noise helpers (for SNR and BIC)
# --------------------------------------------------------------------

def _matched_filter_variance_A(lam_um, sig_flam, muA, sigmaA):
    """
    Matched-filter variance for an area-normalised Gaussian line with centroid muA,
    width sigmaA (Å), on wavelength grid lam_um with per-pixel σ_Fλ.
    Returns Var(Â) in the same flux units as line fluxes.
    """
    lamA = lam_um * 1e4
    left_A, right_A = _pixel_edges_A(lamA)
    t = _gauss_binavg_area_normalized_A(left_A, right_A, muA, sigmaA)
    w = 1.0 / np.clip(sig_flam, 1e-30, None)**2
    S = np.nansum(w * t**2)
    if not np.isfinite(S) or S <= 0:
        return np.inf
    return 1.0 / S


def _corr_noise_factor(resid, max_lag=3):
    """
    Simple correlated-noise inflation factor for the line flux error,
    based on the autocorrelation of whitened residuals.
    """
    r = resid[np.isfinite(resid)]
    if r.size < max_lag + 5:
        return 1.0
    r -= np.nanmean(r)
    rho = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, max_lag+1)]
    rho = np.nan_to_num(rho, nan=0.0)
    f_corr = np.sqrt(max(1.0, 1.0 + 2.0 * np.sum(np.clip(rho, -0.99, 0.99))))
    return f_corr

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


# --------------------------------------------------------------------
# Core worker: fit an arbitrary set of lines (incl. *_BROAD) + compute BIC
# --------------------------------------------------------------------

def _fit_emission_system(
    lam_fit,
    resid_fit,
    sig_fit,
    Fcont_fit,
    z,
    grating,
    which_lines,
    absorption_corrections=None,
    verbose=False,
):
    """
    Internal worker that mirrors the original excels_fit_poly()
    fitting method, but allows *_BROAD lines by widening their σ
    bounds. Returns a dict with BIC for model comparison.
    """
    if not which_lines:
        raise ValueError("No lines to fit in _fit_emission_system.")

    lam_fit   = np.asarray(lam_fit,   float)
    resid_fit = np.asarray(resid_fit, float)
    sig_fit   = np.asarray(sig_fit,   float)

    # --- Instrument σ and centroid seeds (exactly as in excels_fit_poly) ---
    expected_um  = np.array([REST_LINES_A[nm] * (1.0 + z) / 1e4 for nm in which_lines], float)
    sigma_gr_log = np.array([sigma_grating_logA(grating, mu_um) for mu_um in expected_um], float)
    muA_nom      = expected_um * 1e4                           # Å
    sigmaA_inst  = muA_nom * LN10 * sigma_gr_log               # Å
    sigma_um_inst = sigmaA_inst / 1e4                          # µm

    lamA_fit = lam_fit * 1e4
    pix_A_global = float(np.median(np.diff(lamA_fit))) if lamA_fit.size > 1 else 1.0

    def _local_pix_A_for(mu_um):
        mloc = np.abs(lam_fit - mu_um) < 0.02
        if np.count_nonzero(mloc) >= 3:
            return float(np.median(np.diff(lam_fit[mloc] * 1e4)))
        return pix_A_global

    pixA_local   = np.array([_local_pix_A_for(mu) for mu in expected_um], float)  # Å
    pix_um_local = pixA_local / 1e4                                               # µm

    g = str(grating).lower()
    if "prism" in g:
        # PRISM: tolerant, but never smaller than ±2 pixels
        peak_half_um = np.maximum(5.0 * sigma_um_inst, 2.0 * pix_um_local)
    else:
        # MED/HIGH: search only within ±2 pixels of expected centre
        peak_half_um = 3.0 * pix_um_local

    mu_seed_um = _find_local_peaks(
        lam_fit, resid_fit, expected_um,
        sigma_gr_um=sigma_um_inst,
        per_line_halfwidth_um=peak_half_um
    )
    muA_seed = mu_seed_um * 1e4  # Å

    nL = len(which_lines)

    # Pixel scale in Angstroms
    pix_A = float(np.median(np.diff(lamA_fit))) if lamA_fit.size > 1 else 1.0

    # Local S/N per line
    snr_loc = []
    for j, mu_um in enumerate(expected_um):
        w = np.abs(lam_fit - mu_um) < 5.0 * (sigmaA_inst[j] / 1e4)
        if np.any(w):
            peak  = np.nanmax(resid_fit[w])
            noise = np.nanmedian(sig_fit[w]) if np.any(np.isfinite(sig_fit[w])) else np.nan
            snr_loc.append(peak / (noise + 1e-30))
        else:
            snr_loc.append(0.0)
    snr_loc = np.asarray(snr_loc, float)
    good_snr = snr_loc >= 8.0

    # --- σ bounds as in excels_fit_poly ---
    if "prism" in g:
        sigmaA_lo = np.maximum(0.40 * pix_A, 0.45 * sigmaA_inst)
        sigmaA_hi = np.maximum(1.50 * pix_A, 1.70 * sigmaA_inst)
        seed_factor = 0.90
    elif any(k in g for k in ["m", "med"]):
        sigmaA_lo = np.maximum(0.12 * pix_A, 0.22 * sigmaA_inst)
        sigmaA_hi = np.maximum(0.75 * pix_A, 1.05 * sigmaA_inst)
        seed_factor = 0.60
    elif any(k in g for k in ["h", "high"]):
        sigmaA_lo = np.where(
            good_snr,
            np.maximum(0.10 * pix_A, 0.18 * sigmaA_inst),
            np.maximum(0.20 * pix_A, 0.40 * sigmaA_inst),
        )
        sigmaA_hi = np.where(
            good_snr,
            np.maximum(0.60 * pix_A, 0.95 * sigmaA_inst),
            np.maximum(0.75 * pix_A, 1.15 * sigmaA_inst),
        )
        seed_factor = np.where(good_snr, 0.50, 0.65)
    else:
        sigmaA_lo = np.maximum(0.30 * pix_A, 0.60 * sigmaA_inst)
        sigmaA_hi = np.maximum(0.70 * pix_A, 1.15 * sigmaA_inst)
        seed_factor = 0.85

    # --- BROAD components: σ ranges relative to the narrow Balmer width ---
    C_KMS = 299792.458

    # Factors (in σ_v space) relative to the *narrow* component.
    # i.e. σ_v(broad1) ≈ FACT_B1["seed"] × σ_v(narrow), etc.
    FACT_B1 = dict(lo=1.5, seed=3.0, hi=5.0)   # “intermediate” broad
    FACT_B2 = dict(lo=4.0, seed=7.0, hi=12.0)  # “very” broad

    # Start with instrument-based bounds for everyone
    sigmaA_lo   = np.array(sigmaA_lo,  float)
    sigmaA_hi   = np.array(sigmaA_hi,  float)
    sigmaA_seed = np.clip(seed_factor * sigmaA_inst, sigmaA_lo, sigmaA_hi)

    # ------------------------------------------------------------------
    # NEW LOGIC: Handle Blended Doublets
    # ------------------------------------------------------------------
    # These keys represent two physical lines fitted as one Gaussian.
    # We must relax the upper width bound and increase the seed width
    # so the fitter can encompass both peaks.
    BLENDED_KEYS = {"NV_doublet", "CIV_doublet", "OII_doublet"}

    for j, nm in enumerate(which_lines):
        if nm in BLENDED_KEYS:
            # Widen the seed: start at 1.5x the instrumental width
            sigmaA_seed[j] = max(sigmaA_seed[j], 1.5 * sigmaA_inst[j])
            
            # Widen the upper bound: allow up to 3.5x instrumental width
            # (Standard lines are usually capped around 1.15x - 1.5x)
            sigmaA_hi[j]   = max(sigmaA_hi[j],   3.5 * sigmaA_inst[j])
            
            if verbose:
                 print(f"  [Auto-Correction] Widening bounds for blended entry: {nm}")

    # Map each broad line to its narrow parent
    parent_narrow = {
        "H⍺_BROAD":     "H⍺",
        "H⍺_BROAD2":    "H⍺",
        "HBETA_BROAD":  "HBETA",
        "HBETA_BROAD2": "HBETA",
        "HDELTA_BROAD":  "HDELTA",
        "HDELTA_BROAD2": "HDELTA",
    }
    name_to_idx = {nm: j for j, nm in enumerate(which_lines)}

    # ------------------------------------------------------------------
    # 1. Set priors for each broad component individually (as before)
    # ------------------------------------------------------------------
    for j, nm in enumerate(which_lines):
        if "BROAD" not in nm:
            continue

        # -------- find the corresponding narrow component --------
        parent = parent_narrow.get(nm)
        if parent is not None and parent in name_to_idx:
            j_narrow = name_to_idx[parent]
            # "narrow peak width": you can swap this for a fitted σ if you pass it in
            sigma_narrow_A = sigmaA_inst[j_narrow]
            muA_narrow     = muA_nom[j_narrow]
        else:
            # Fallback: use local instrumental width as "narrow"
            sigma_narrow_A = sigmaA_inst[j]
            muA_narrow     = muA_nom[j]

        # Convert that narrow σ_λ → σ_v (km/s)
        sigma_narrow_v = (sigma_narrow_A / muA_narrow) * C_KMS

        # Choose factor set for BROAD vs BROAD2
        if "BROAD2" in nm:
            fcfg = FACT_B2
        else:
            fcfg = FACT_B1

        # Desired σ_v for this broad component
        sigma_lo_v   = fcfg["lo"]   * sigma_narrow_v
        sigma_hi_v   = fcfg["hi"]   * sigma_narrow_v
        sigma_seed_v = fcfg["seed"] * sigma_narrow_v

        # Convert back to σ_λ at this line’s wavelength
        mu_A = expected_um[j] * 1e4  # Å (broad component’s centre)
        sigma_lo_A   = (sigma_lo_v   / C_KMS) * mu_A
        sigma_hi_A   = (sigma_hi_v   / C_KMS) * mu_A
        sigma_seed_A = (sigma_seed_v / C_KMS) * mu_A

        # Enforce lower/upper bounds with instrumental and global limits
        sigmaA_lo[j] = max(sigma_lo_A, 1.2 * sigmaA_inst[j])
        sigmaA_hi[j] = min(sigma_hi_A, MAX_BROAD_SIGMA_A)

        if sigmaA_hi[j] <= sigmaA_lo[j]:
            # emergency widen if something pathological happens
            sigmaA_hi[j] = sigmaA_lo[j] * 1.5

        # Final σ seed for this broad component
        sigmaA_seed[j] = np.clip(sigma_seed_A, sigmaA_lo[j], sigmaA_hi[j])

    # ------------------------------------------------------------------
    # 2. Explicitly force BROAD2 > BROAD1 (for each Balmer line)
    # ------------------------------------------------------------------
    def _force_broad2_broader(name_b1, name_b2, factor=1.5):
        """
        Ensure σ(BROAD2) >= factor * σ(BROAD1) at the level of
        bounds AND seeds. Safe no-op if either line is absent.
        """
        i1 = name_to_idx.get(name_b1)
        i2 = name_to_idx.get(name_b2)
        if i1 is None or i2 is None:
            return

        # Lower bound: BROAD2 cannot be narrower than factor × BROAD1
        sigmaA_lo[i2]   = max(sigmaA_lo[i2],   factor * sigmaA_lo[i1])
        # Seed: push BROAD2 at least factor × BROAD1’s seed
        sigmaA_seed[i2] = max(sigmaA_seed[i2], factor * sigmaA_seed[i1])
        # Upper bound: allow BROAD2 to explore at least as wide a region
        sigmaA_hi[i2]   = max(sigmaA_hi[i2],   factor * sigmaA_hi[i1])

    _force_broad2_broader("H⍺_BROAD",   "H⍺_BROAD2",   factor=2)
    _force_broad2_broader("HBETA_BROAD","HBETA_BROAD2",factor=2)
    _force_broad2_broader("HDELTA_BROAD","HDELTA_BROAD2",factor=2)


    # Amplitude seeds for area-normalized Gaussians
    SQRT2PI = np.sqrt(2.0 * np.pi)
    A0, A_hi = [], []
    rms_flam = float(_safe_median(np.abs(resid_fit), 0.0))
    rms_flam = max(rms_flam, 1e-30)

    for j, mu_um in enumerate(expected_um):
        win = (lam_fit > mu_um - 0.05) & (lam_fit < mu_um + 0.05)
        peak_flam = np.nanmax(resid_fit[win]) if np.any(win) else 3.0 * rms_flam
        peak_flam = max(peak_flam, 3.0 * rms_flam)

        # Split peak flux between components to control relative *peak* heights.
        # We want:
        #   peak(BROAD1) ≈ 0.25 * peak(narrow)
        #   peak(BROAD2) ≈ 0.10 * peak(narrow)
        #
        # For area-normalised Gaussians, peak_j ∝ flux_fraction_j, so set:
        r1 = 0.25  # desired peak(BROAD1) / peak(narrow)
        r2 = 0.10  # desired peak(BROAD2) / peak(narrow)

        f_narrow = 1.0 / (1.0 + r1 + r2)   # ≈ 0.7407
        f_b1     = r1 * f_narrow          # ≈ 0.1852
        f_b2     = r2 * f_narrow          # ≈ 0.0741

        nm = which_lines[j]

        if "BROAD2" in nm:
            flux_fraction = f_b2
            # very broad wings — keep the integrated flux cap modest
            A_multiplier = 10.0  # tweak if you want broader allowed wings
        elif "BROAD" in nm:
            flux_fraction = f_b1
            # intermediate broad component
            A_multiplier = 40.0
        else:
            flux_fraction = f_narrow
            A_multiplier = 150.0

        A_seed = flux_fraction * peak_flam * SQRT2PI * max(
            sigmaA_seed[j], 0.9 * sigmaA_inst[j]
        )
        A0.append(A_seed)


        # Upper bound on integrated flux (area) for this Gaussian. BROAD2 is
        # intentionally capped more tightly to prevent the wings from absorbing
        # most of the flux during optimisation.
        A_upper = A_multiplier * max(peak_flam, 3.0 * rms_flam) * SQRT2PI * np.maximum(
            sigmaA_hi[j], sigmaA_seed[j]
        )
        A_hi.append(A_upper)

    A0 = np.asarray(A0, float)
    A_lo = np.zeros_like(A0)  # emission-only, as before
    A_hi = np.asarray(A_hi, float)

    # --- centroid bounds (with NII–Hα fences, but no constraints on *_BROAD) ---
    if "prism" in g:
        muA_lo = muA_seed - 12.0 * np.maximum(sigmaA_inst, 1.0)
        muA_hi = muA_seed + 12.0 * np.maximum(sigmaA_inst, 1.0)
    else:
        # Relaxed centroid bounds: allow ~25 A movement observed frame
        NPIX_CENT = 4.0
        CENTROID_HALF_WIDTH_A = 25.0

        pixA  = pixA_local
        halfA = np.maximum(NPIX_CENT * pixA, CENTROID_HALF_WIDTH_A)

        muA_lo = muA_seed - halfA
        muA_hi = muA_seed + halfA

        HaA       = REST_LINES_A["H⍺"] * (1.0 + z)
        mid_N2_Ha = 0.5 * (REST_LINES_A["NII_6549"] + REST_LINES_A["H⍺"]) * (1.0 + z)
        mid_Ha_N3 = 0.5 * (REST_LINES_A["H⍺"] + REST_LINES_A["NII_6585"]) * (1.0 + z)

        for j, nm in enumerate(which_lines):
            if nm == "NII_6549":
                muA_hi[j] = min(muA_hi[j], mid_N2_Ha)
            elif nm == "NII_6585":
                muA_lo[j] = max(muA_lo[j], mid_Ha_N3)
            # we let H⍺ and H⍺_BROAD float within halfA

    # --- pack params (full A, σ, μ as in original fitter) ---
    p0_full = np.r_[A0,          sigmaA_seed, muA_seed]
    lb_full = np.r_[A_lo,        sigmaA_lo,   muA_lo]
    ub_full = np.r_[A_hi,        sigmaA_hi,   muA_hi]
    p0_full, lb_full, ub_full = _finalize_seeds_and_bounds(p0_full, lb_full, ub_full)

    
    # ------------------------------------------------------------------
    # CONSTRAINT LOGIC: Tie NII Doublet & Weighting
    # ------------------------------------------------------------------
    idx_6549 = name_to_idx.get("NII_6549")
    idx_6585 = name_to_idx.get("NII_6585")
    do_tie_nii = (idx_6549 is not None) and (idx_6585 is not None)
    
    sig_fit_weighted = sig_fit.copy()
    
    # Increase weight on NII lines (6549 and 6585)
    # This forces the fitter to respect the narrow NII features and not let BROAD2 swallow them.
    for nm_nii in ["NII_6549", "NII_6585"]:
        idx_nii = name_to_idx.get(nm_nii)
        if idx_nii is not None:
             mu_um = p0_full[2*nL + idx_nii] / 1e4
             sig_um = p0_full[nL + idx_nii] / 1e4 * 1.0
             
             # Use +/- 3 sigma window
             mask_weight = (lam_fit >= mu_um - 3*sig_um) & (lam_fit <= mu_um + 3*sig_um)
             if np.any(mask_weight):
                 if verbose:
                     print(f"  [Weighting] Increasing weight for {nm_nii} (factor 9) on {np.count_nonzero(mask_weight)} pixels.")
                 # Divide sigma by 3.0 -> weight * 9
                 sig_fit_weighted[mask_weight] /= 3.0

    # Mask of parameters to OPTIMIZE
    # default: all True
    param_mask = np.ones(p0_full.size, dtype=bool)
    
    if do_tie_nii:
        if verbose:
            print("  [Constraints] Tying NII_6549 to NII_6585 (Flux ratio 2.96, tied kinematics).")
        # Remove 6549 parameters from optimization
        # Amplitude
        param_mask[idx_6549] = False
        # Sigma
        param_mask[nL + idx_6549] = False
        # Mu
        param_mask[2*nL + idx_6549] = False

    # ------------------------------------------------------------------
    # CONSTRAINT LOGIC: Tie Narrow Balmer lines to OIII_5007
    # ------------------------------------------------------------------
    idx_o3 = name_to_idx.get("OIII_5007")
    balmer_tie_list = ["HBETA", "H⍺", "HDELTA", "HGAMMA", "NII_6549", "NII_6585"]
    tied_indices = []
    
    if idx_o3 is not None:
        for nm in balmer_tie_list:
             if nm in name_to_idx:
                 tied_indices.append(name_to_idx[nm])
                 
    if idx_o3 is not None and len(tied_indices) > 0:
        if verbose:
             print(f"  [Constraints] Tying narrow Balmer width(s) to OIII_5007.")
        for idx in tied_indices:
             # Remove sigma from optimization (index + nL)
             param_mask[nL + idx] = False
        
    p0_opt = p0_full[param_mask]
    lb_opt = lb_full[param_mask]
    ub_opt = ub_full[param_mask]

    
    def _reconstruct_params(p_opt):
        # Start with full seeds (holds fixed values if any, though here we overwrite everything important)
        p_curr = p0_full.copy()
        
        # Fill in optimized values
        p_curr[param_mask] = p_opt
        
        if do_tie_nii:
            # 1. Amplitude: A_6549 = A_6585 / 2.96
            # Note: p_curr[idx_6585] is now updated from p_opt
            p_curr[idx_6549] = p_curr[idx_6585] / 2.96
            
            # 2. Kinematics
            # Need ratio of rest wavelengths
            lam_49 = REST_LINES_A["NII_6549"]
            lam_85 = REST_LINES_A["NII_6585"]
            ratio = lam_49 / lam_85
            
            # Tie Mu: z_49 = z_85
            # mu_85 = lam_85 * (1+z)
            # mu_49 = lam_49 * (1+z) = lam_49 * (mu_85 / lam_85) = mu_85 * ratio
            p_curr[2*nL + idx_6549] = p_curr[2*nL + idx_6585] * ratio
            
            # Tie Sigma (velocity): sigma_v_49 = sigma_v_85
            # sigma_lam_49 / lam_49 = sigma_lam_85 / lam_85
            # sigma_lam_49 = sigma_lam_85 * ratio
            p_curr[nL + idx_6549] = p_curr[nL + idx_6585] * ratio

        if idx_o3 is not None and len(tied_indices) > 0:
            lam_o3 = REST_LINES_A["OIII_5007"]
            # OIII sigma is at nL + idx_o3
            sigma_o3 = p_curr[nL + idx_o3] 
            
            for idx in tied_indices:
                nm = which_lines[idx]
                lam_target = REST_LINES_A[nm]
                ratio = lam_target / lam_o3
                p_curr[nL + idx] = sigma_o3 * ratio
            
        return p_curr

    # --- χ² residuals with gentle pixel-width weighting (same as before) ---
    dlA = np.gradient(lamA_fit)
    rat = dlA / np.nanmedian(dlA)
    rat = np.clip(rat, 0.6, 1.6)
    w_pix = (np.nanmedian(dlA) / np.clip(dlA, 1e-12, None)) ** 0.35
    w_pix = np.clip(w_pix, 0.8, 1.25)

    def fun(p):
        p_full = _reconstruct_params(p)
        model, _, _ = build_model_flam_linear(
            p_full, lam_fit, z, grating, which_lines, mu_seed_um
        )
        # Use weighted expected error
        return w_pix * (resid_fit - model) / np.clip(sig_fit_weighted, 1e-30, None)

    res = least_squares(
        fun, p0_opt, bounds=(lb_opt, ub_opt),
        max_nfev=80000, xtol=1e-8, ftol=1e-8
    )

    # --- best-fit model ---
    p_best_full = _reconstruct_params(res.x)
    model_flam_win, profiles_win, centers = build_model_flam_linear(
        p_best_full, lam_fit, z, grating, which_lines, mu_seed_um
    )

    # First nL params are fitted amplitudes
    A_fit = np.asarray(p_best_full[:nL], float)

    # --- fluxes & EWs using the standard helpers (same as excels_fit_poly) ---
    # NOTE: For overlapping components (e.g., broad Hα spilling into NII),
    # measure_fluxes_profile_weighted can double-count flux because it extracts
    # from the data independently for each component.
    # 
    # Since our Gaussians are area-normalized in build_model_flam_linear,
    # the fitted amplitudes A_fit[j] ARE the integrated line fluxes.
    # We use measure_fluxes_profile_weighted only for uncertainty estimation.
    
    fluxes_mf = measure_fluxes_profile_weighted(
        lam_fit, resid_fit, sig_fit, model_flam_win, profiles_win, centers
    )
    
    # Build corrected fluxes dict using fitted amplitudes as the true fluxes
    fluxes = {}
    for j, name in enumerate(which_lines):
        # Use fitted amplitude as the integrated flux (correct for overlapping components)
        F_line_correct = A_fit[j]
        
        # Use matched-filter uncertainty estimate (still valid for noise estimation)
        if name in fluxes_mf:
            sigma_line = fluxes_mf[name].get("sigma_line", np.nan)
            mask_idx = fluxes_mf[name].get("mask_idx", np.array([]))
        else:
            sigma_line = np.nan
            mask_idx = np.array([])
        
        fluxes[name] = dict(
            F_line=F_line_correct,
            sigma_line=sigma_line,
            mask_idx=mask_idx
        )
    
    ews = equivalent_widths_A(fluxes, lam_fit, Fcont_fit, z, centers)

    if absorption_corrections:
        fluxes = apply_balmer_absorption_correction(fluxes, absorption_corrections)

    resid_best = resid_fit - model_flam_win
    per_line = {}

    for j, name in enumerate(which_lines):
        F_line = fluxes.get(name, {}).get("F_line", np.nan)
        sig_line_nominal = fluxes.get(name, {}).get("sigma_line", np.nan)
        ew_obs = ews.get(name, {}).get("EW_obs_A", np.nan) if name in ews else np.nan
        ew0    = ews.get(name, {}).get("EW0_A", np.nan)     if name in ews else np.nan

        muA, sigma_logA = centers[name]
        sigma_A = sigma_logA * muA * LN10
        FWHM_A  = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_A

        # integrated SNR (matched filter + correlated noise, exactly as before)
        sig_mf = np.sqrt(_matched_filter_variance_A(lam_fit, sig_fit, muA, sigma_A))

        win_um = 3.0 * (sigma_A / 1e4)
        mask_loc = (lam_fit >= (muA/1e4 - win_um)) & (lam_fit <= (muA/1e4 + win_um))
        if np.any(mask_loc):
            r_loc = resid_best[mask_loc] / np.clip(sig_fit[mask_loc], 1e-30, None)
            f_corr = _corr_noise_factor(r_loc, max_lag=3)
        else:
            f_corr = 1.0

        sig_line_final = max(sig_line_nominal, sig_mf * f_corr)
        snr = (
            np.nan
            if (not np.isfinite(sig_line_final) or sig_line_final <= 0)
            else (F_line / sig_line_final)
        )

        # peak SNRs (data + model), same as original code
        win_um = 3.0 * (sigma_A / 1e4)
        mpeak = (lam_fit >= (muA/1e4 - win_um)) & (lam_fit <= (muA/1e4 + win_um))

        if np.any(mpeak):
            sigma_loc = np.nanmedian(sig_fit[mpeak])
            sigma_loc_eff = sigma_loc * f_corr
            peak_data = np.nanmax(resid_fit[mpeak])
            peak_snr_data = peak_data / np.clip(sigma_loc_eff, 1e-30, None)
        else:
            peak_snr_data = np.nan
            sigma_loc_eff = np.nan

        peak_model = F_line / (SQRT2PI * np.clip(sigma_A, 1e-30, None))
        peak_snr_model = peak_model / np.clip(
            sigma_loc_eff if np.any(mpeak) else np.nan,
            1e-30, None
        )

        per_line[name] = dict(
            F_line=F_line,
            sigma_line=sig_line_final,
            SNR=snr,
            SNR_peak_data=peak_snr_data,
            SNR_peak_model=peak_snr_model,
            EW_obs_A=ew_obs,
            EW0_A=ew0,
            lam_obs_A=muA,
            sigma_A=sigma_A,
            FWHM_A=FWHM_A,
            A_gauss=A_fit[j],
        )

    # --- robust BIC on whitened residuals ---
    resid_vec = fun(res.x)
    resid_vec = np.asarray(resid_vec, float)
    mfin = np.isfinite(resid_vec)
    N_data = int(np.count_nonzero(mfin))
    if N_data == 0:
        BIC = np.nan
        chi2 = np.nan
    else:
        chi2 = float(np.sum(resid_vec[mfin] ** 2))
        k_params = res.x.size
        BIC = chi2 + k_params * np.log(N_data)

    if verbose:
        print(f"Fit with lines={which_lines}: χ²={chi2:.2f}, k={res.x.size}, BIC={BIC:.2f}")

    return dict(
        res=res,
        model_flam=model_flam_win,
        profiles=profiles_win,
        centers=centers,
        per_line=per_line,
        which_lines=which_lines,
        BIC=BIC,
    )


#edit
# --------------------------------------------------------------------
# Main public fit: single_broad_fit
# --------------------------------------------------------------------

def fit_continuum_moving_average(
    lam_um, flam, z,
    windows=None,
    lyman_cut="lya",
    window_um=0.05,
    grating="PRISM",
    clip_sigma=3.0,
    percentile_bounds=(16, 84),
    min_points=5,
    exclude_regions=None,
    lyman_window_um=None,
    continuum_lya_mask_A=200.0,
):
    """
    Fit continuum using a moving average of flux between percentiles (default 16-84),
    with sigma clipping and line masking.

    Parameters
    ----------
    lam_um : array
        Wavelength in microns.
    flam : array
        Flux in F_lambda units.
    z : float
        Redshift.
    windows : list of tuples, optional
        (min, max) windows in microns to consider for the continuum.
        If None, uses the full range.
    lyman_cut : str or None
        If "lya", masks everything blue of Lyman-alpha.
    window_um : float
        Width of the moving window in microns.
    grating : str
        Used to determine line masking width based on resolution.
    clip_sigma : float
        Sigma threshold for clipping outliers within the moving window and overall.
    percentile_bounds : tuple
        (low, high) percentiles to compute the mean of (Truncated Mean).
    min_points : int
        Minimum number of valid points in a window to compute a value.
    exclude_regions : list of tuples, optional
        List of (min_um, max_um) regions to manually exclude from the fit.
    lyman_window_um : float, optional
        If provided, use this smaller window size when within `continuum_lya_mask_A`
        of the Lyman-alpha line. This preserves the sharp Lyman break.
    continuum_lya_mask_A : float
        Buffer in rest-frame Angstroms around LyA to consider "near" the break.

    Returns
    -------
    Fcont : array
        The estimated continuum flux array.
    coeffs : None
        Dummy return to match polynomial fitter signature.
    """
    lam_um = np.asarray(lam_um, float)
    flam = np.asarray(flam, float)
    n = len(lam_um)
    Fcont = np.full(n, np.nan)

    # 1. Global Masking (Lyman cut + Continuum Windows)
    mask = np.ones(n, dtype=bool)

    # Lyman cut
    if lyman_cut:
        ly_um = _lyman_cut_um(z, lyman_cut)
        mask &= (lam_um >= ly_um)

    # Continuum windows (if provided)
    if windows:
        in_windows = np.zeros(n, dtype=bool)
        for (lo, hi) in windows:
            in_windows |= (lam_um >= lo) & (lam_um <= hi)
        mask &= in_windows

    # 2. Line Masking (Mask known emission lines)
    # Similar logic to fit_continuum_polynomial
    lam_A = lam_um * 1e4
    for lam_rest in REST_LINES_A.values():
        muA = lam_rest * (1 + z)
        mu_um = muA / 1e4
        sig_gr_log = float(sigma_grating_logA(grating, mu_um))
        # Mask ±6 sigma around lines
        sigma_A = muA * LN10 * sig_gr_log * 1.5
        # slightly generous mask for moving average to avoid line wings
        core = (lam_A > muA - 6 * sigma_A) & (lam_A < muA + 6 * sigma_A)
        mask[core] = False

    # Manual exclude regions
    if exclude_regions:
        for (lo, hi) in exclude_regions:
            # Mask out region
            mask &= ~((lam_um >= lo) & (lam_um <= hi))

    # 3. Moving Average Calculation
    # We only compute Fcont where mask is True, but we use neighboring data
    # that is also valid (mask is True) and finite.

    valid_data_mask = mask & np.isfinite(flam)
    valid_indices = np.where(valid_data_mask)[0]

    if len(valid_indices) < min_points:
        # Not enough data for continuum
        return np.nan_to_num(Fcont, nan=np.nanmedian(flam) if len(flam) > 0 else 0.0), None

    # Pre-compute valid arrays to speed up
    lam_valid = lam_um[valid_indices]
    flam_valid = flam[valid_indices]

    # Pre-calculate Lyman-alpha parameters if adaptive window is used
    lam_lya_obs = 0.121567 * (1 + z)
    use_adaptive = (lyman_window_um is not None) and (lyman_window_um > 0)


    # Iterate over all points (or just valid ones? We want continuum everywhere ideally,
    # or at least interpolated everywhere).
    # Let's compute it at every wavelength point to be safe, using the valid data.

    half_win = window_um / 2.0

    # Optimization: iterate only finite points, then interpolate?
    # Or just iterate all points. For standard spectra (few k pixels), iterating is fast enough.
    
    # Let's calculate Fcont for each pixel i based on valid neighbors
    
    computed_vals = []
    computed_indices = []

    for i in range(n):
        # Center of window
        w_center = lam_um[i]
        
        # Determine window size
        current_window_um = window_um
        if use_adaptive:
            # check distance to LyA
            dist_um = abs(w_center - lam_lya_obs)
            dist_rest_A = dist_um * 1e4 / (1 + z)
            
            if dist_rest_A < continuum_lya_mask_A:
                current_window_um = lyman_window_um




        half_win = current_window_um / 2.0
        
        # Identify valid neighbors in window

        # We can use searchsorted or just boolean mask if array is small.
        # Assuming sorted lam_um.
        
        # Fast slice
        idx_start = np.searchsorted(lam_valid, w_center - half_win, side='left')
        idx_end = np.searchsorted(lam_valid, w_center + half_win, side='right')
        
        chunk = flam_valid[idx_start:idx_end]
        
        if len(chunk) < min_points:
            continue
            
        # Sigma Clipping in the window
        # Simple iterative clipping
        chunk_clipped = chunk.copy()
        
        # 1-pass or 2-pass clipping usually enough
        med = np.median(chunk_clipped)
        mad = np.median(np.abs(chunk_clipped - med))
        if mad == 0:
            std = np.std(chunk_clipped)
        else:
            std = 1.4826 * mad
            
        if std > 0:
            keep = np.abs(chunk_clipped - med) < clip_sigma * std
            chunk_clipped = chunk_clipped[keep]
        
        if len(chunk_clipped) < min_points:
            continue

        # Percentile Calculation (16th to 84th)
        p_lo, p_hi = np.percentile(chunk_clipped, list(percentile_bounds))
        
        # Mean of values within this range
        core_mask = (chunk_clipped >= p_lo) & (chunk_clipped <= p_hi)
        if np.sum(core_mask) > 0:
            val = np.mean(chunk_clipped[core_mask])
            computed_vals.append(val)
            computed_indices.append(i)

    # 4. Interpolate to fill gaps
    if len(computed_indices) > 1:
        Fcont[computed_indices] = computed_vals
        # Interpolate missing values (including those masked out)
        # We use the computed points as knots
        Fcont = np.interp(lam_um, lam_um[computed_indices], computed_vals, left=np.nan, right=np.nan)
        
        # Helper: Extrapolate ends if needed with nearest valid
        valid_cont = np.isfinite(Fcont)
        if np.any(valid_cont):
            Fcont[:np.argmax(valid_cont)] = Fcont[valid_cont][0] # fill left
            Fcont[len(Fcont) - np.argmax(valid_cont[::-1]):] = Fcont[valid_cont][-1] # fill right
            
    else:
        # Fallback if moving average failed completely
        Fcont[:] = np.nanmedian(flam_valid) if len(flam_valid) > 0 else 0.0

    return Fcont, None


def single_broad_fit(
    source,
    z,
    grating: str = "PRISM",
    lines_to_use=None,
    deg: int = 2,
    continuum_windows=None,
    lyman_cut="split",
    fit_window_um=None,
    plot: bool = True,
    verbose: bool = True,
    absorption_corrections=None,
    force_lines: Optional[List[str]] = None,
    bic_delta_prefer: float = 0.0,
    snr_broad_threshold: float = 5.0,
    broad_mode: str = "auto",
    continuum_lya_mask_A: float = 200.0,
    continuum_fit: str = "polyfit",
    continuum_movavg_window_um: float = 0.05,
    continuum_exclude_regions: Optional[List[Tuple[float, float]]] = None,
    continuum_lyman_window_um: Optional[float] = None,
    save_npz: bool = False,
    npz_name: Optional[str] = None,
):
    """
    Fit emission lines with optional broad Balmer components using BIC selection.

    Fits all narrow emission lines in the specified window, then uses local BIC
    comparisons to determine whether to add broad components to H-alpha, H-beta,
    and H-delta independently.

    Parameters
    ----------
    source : dict or HDUList
        Spectrum data with 'lam'/'wave', 'flux', and 'err' keys (dict) or
        FITS HDUList with SPEC1D extension.
    z : float
        Redshift of the source.
    grating : str, default='PRISM'
        Grating name for line list selection.
    lines_to_use : list of str, optional
        Subset of emission lines to fit. If None, uses all available lines.
    deg : int, default=2
        Polynomial degree for continuum fitting (used if continuum_fit='polyfit').
    continuum_windows : list of tuples or str, optional
        Wavelength windows for continuum fitting, or 'two_sided_lya' for automatic.
    lyman_cut : str, default='split'
        Lyman-alpha cutoff mode. 'split' fits both sides (default), 'lya' masks blue.
    fit_window_um : tuple of float or list of tuples, optional
        (low, high) or [(lo1, hi1), ...] wavelength range(s) in microns for fitting.
    plot : bool, default=True
        Whether to generate diagnostic plots.
    verbose : bool, default=True
        Whether to print diagnostic information.
    absorption_corrections : dict, optional
        Absorption corrections to apply.
    force_lines : list of str, optional
        If provided, skips BIC selection and fits this exact line list.
    bic_delta_prefer : float, default=0.0
        BIC improvement threshold for preferring broad components.
    snr_broad_threshold : float, default=5.0
        Minimum SNR required to consider adding broad components in auto mode.
    broad_mode : str, default='auto'
        Broad component selection mode: 'auto' (BIC-based), 'off' (narrow only),
        'broad1' (force BROAD), 'broad2' (force BROAD2), 'both' (force both).
    continuum_fit : str, default='polyfit'
        Continuum fitting method: 'polyfit' or 'moving_average'.
    continuum_movavg_window_um : float, default=0.05
        Window size for moving average continuum fit [um].
    continuum_exclude_regions : list of tuples, optional
        List of (min_um, max_um) regions to exclude from continuum fit.
    continuum_lyman_window_um : float, optional
        Smaller window size [um] for moving average near Lyman-alpha to preserve the break.
    continuum_fit : str, default='polyfit'
        Continuum fitting method: 'polyfit' (polynomial) or 'moving_average'.
    continuum_movavg_window_um : float, default=0.05
        Window size in microns for moving average continuum fit.

    Returns
    -------
    dict
        Fit results including continuum, model, line parameters, and BIC values.
    """

    valid_modes = {"auto", "off", "broad1", "broad2", "both"}
    if broad_mode not in valid_modes:
        raise ValueError(f"broad_mode must be one of {valid_modes}, got {broad_mode!r}")

    # Load spectrum
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

    # Handle "split" default: use two-sided LyA windows unless user overrides
    if str(lyman_cut).lower() == "split":
        if continuum_windows is None:
            continuum_windows = "two_sided_lya"
        # Disable hard cut so we can fit blue side if windows allow
        lyman_cut = None

    # Continuum windows around Lyman-alpha if requested
    auto_windows = None
    if isinstance(continuum_windows, str) and continuum_windows.lower() == "two_sided_lya":
        auto_windows = continuum_windows_two_sides_of_lya(
            lam_um, z, buffer_rest_A=continuum_lya_mask_A, min_pts=12, verbose=verbose
        )
    elif isinstance(continuum_windows, dict) and continuum_windows.get("mode", "").lower() == "two_sided_lya":
        brA = float(continuum_windows.get("buffer_rest_A", continuum_lya_mask_A))
        mpts = int(continuum_windows.get("min_pts", 12))
        auto_windows = continuum_windows_two_sides_of_lya(
            lam_um, z, buffer_rest_A=brA, min_pts=mpts, verbose=verbose
        )

    if auto_windows is not None:
        continuum_windows = auto_windows
        lyman_cut = None

    # Convert to F_lambda and fit continuum
    flam     = fnu_uJy_to_flam(flux_uJy, lam_um)
    sig_flam = fnu_uJy_to_flam(err_uJy,  lam_um)

    if continuum_fit == "moving_average":
        Fcont, _ = fit_continuum_moving_average(
            lam_um, flam, z,
            windows=continuum_windows,
            lyman_cut=lyman_cut,
            window_um=continuum_movavg_window_um,
            grating=grating,
            clip_sigma=3.0,
            percentile_bounds=(16, 84),
            exclude_regions=continuum_exclude_regions,
            lyman_window_um=continuum_lyman_window_um,
            continuum_lya_mask_A=continuum_lya_mask_A,
        )
    elif continuum_fit == "polyfit":
        Fcont, _ = fit_continuum_polynomial(
            lam_um, flam, z, deg=deg, windows=continuum_windows,
            lyman_cut=lyman_cut, sigma_flam=sig_flam, grating=grating
        )
    else:
        raise ValueError(f"Unknown continuum_fit mode: {continuum_fit}")
    resid_full   = flam - Fcont
    sig_flam_fit = rescale_uncertainties(resid_full, sig_flam)

    # Global fit window for all lines
    if fit_window_um is not None:
        if isinstance(fit_window_um, list):
             # List of tuples: [(lo1, hi1), (lo2, hi2), ...]
             w_all = np.zeros(lam_um.shape, dtype=bool)
             for (lo, hi) in fit_window_um:
                 w_all |= (lam_um >= lo) & (lam_um <= hi)
        else:
             # Single tuple: (lo, hi)
             lo_all, hi_all = fit_window_um
             w_all = (lam_um >= lo_all) & (lam_um <= hi_all)

        lam_all   = lam_um[w_all]
        resid_all = resid_full[w_all]
        sig_all   = sig_flam_fit[w_all]
        Fcont_all = Fcont[w_all]
    else:
        lam_all, resid_all, sig_all, Fcont_all = lam_um, resid_full, sig_flam_fit, Fcont
        lo_all, hi_all = lam_um[0], lam_um[-1]

    if lam_all.size == 0:
        raise ValueError("No data in the chosen fit_window_um / global window.")

    # ----------------------------------------------------------------
    # Helper: decide Hα model (none / one / two broad) in *local* window new!
    # ----------------------------------------------------------------
    def _select_ha_model(which_lines_all):
        """
        Use only a local Hα window for BIC comparison, but do not change
        the global continuum or residuals. Returns:

          broad_choice ∈ {"none","one","two"}
          BIC_narrow, BIC_1broad, BIC_2broad, BIC_broad
        """
        BIC_narrow = np.nan
        BIC_1broad = np.nan
        BIC_2broad = np.nan
        BIC_b2only = np.nan
        BIC_broad  = np.nan
        broad_choice = "none"

        # If Hα isn't even in the global narrow list, nothing to do
        if "H⍺" not in which_lines_all:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        ha_triplet = ["NII_6549", "H⍺", "NII_6585"]
        which_ha_narrow = [ln for ln in ha_triplet if ln in which_lines_all]
        if "H⍺" not in which_ha_narrow:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Local Hα window: NII_6549–NII_6585 ± padding
        rest_list = [REST_LINES_A["NII_6549"], REST_LINES_A["H⍺"], REST_LINES_A["NII_6585"]]
        lo_ha, hi_ha = _window_from_lines_um(rest_list, z, pad_A=150.0)
        m_ha = (lam_all >= lo_ha) & (lam_all <= hi_ha)
        if np.count_nonzero(m_ha) < 5:
            # basically no Hα coverage
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        lam_ha   = lam_all[m_ha]
        resid_ha = resid_all[m_ha]
        sig_ha   = sig_all[m_ha]
        Fcont_ha = Fcont_all[m_ha]

        # Base narrow-only fit (M0)
        fit_narrow = _fit_emission_system(
            lam_ha, resid_ha, sig_ha, Fcont_ha,
            z, grating, which_ha_narrow,
            absorption_corrections=absorption_corrections,
            verbose=verbose,
        )
        BIC_narrow = fit_narrow["BIC"]

        if broad_mode == "off":
            if verbose:
                print("broad_mode='off' → using narrow-only Hα+[N II] model.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Check Hα SNR for eligibility (auto mode)
        ha_info = fit_narrow["per_line"].get("H⍺", None)
        ha_snr  = ha_info.get("SNR", 0.0) if ha_info is not None else 0.0
        eligible = (ha_info is not None) and np.isfinite(ha_snr)

        if (broad_mode == "auto") and (not eligible or ha_snr < snr_broad_threshold):
            if verbose:
                print(f"No strong Hα line (SNR={ha_snr:.1f}) → staying narrow-only.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Try adding 1 and 2 broad Hα components on the local window
        def _safe_fit(which, label):
            try:
                fb = _fit_emission_system(
                    lam_ha, resid_ha, sig_ha, Fcont_ha,
                    z, grating, which,
                    absorption_corrections=absorption_corrections,
                    verbose=verbose,
                )
                return fb, fb["BIC"]
            except Exception as e:
                if verbose:
                    print(f"{label} model failed: {e}")
                return None, np.nan

        which_1 = list(which_ha_narrow)
        if "H⍺_BROAD" not in which_1:
            which_1.append("H⍺_BROAD")
        fit_1broad, BIC_1broad = _safe_fit(which_1, "1-broad Hα")
        if fit_1broad is not None:
            # Reject if any broad component is negative
            for ln, info in fit_1broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 1-broad Hα rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_1broad = np.nan
                    break

        which_2 = list(which_1)
        if "H⍺_BROAD2" not in which_2:
            which_2.append("H⍺_BROAD2")
        fit_2broad, BIC_2broad = _safe_fit(which_2, "2-broad Hα")
        if fit_2broad is not None:
            for ln, info in fit_2broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 2-broad Hα rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_2broad = np.nan
                    break

        # BROAD2-only (without BROAD) for complete 4-way comparison
        which_b2only = list(which_ha_narrow)
        if "H⍺_BROAD2" not in which_b2only:
            which_b2only.append("H⍺_BROAD2")
        fit_b2only, BIC_b2only = _safe_fit(which_b2only, "BROAD2-only Hα")
        if fit_b2only is not None:
            for ln, info in fit_b2only["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> BROAD2-only Hα rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_b2only = np.nan
                    break

        candidates = [(BIC_narrow, "none")]
        if fit_1broad is not None and np.isfinite(BIC_1broad):
            candidates.append((BIC_1broad, "one"))
        if fit_b2only is not None and np.isfinite(BIC_b2only):
            candidates.append((BIC_b2only, "broad2_only"))
        if fit_2broad is not None and np.isfinite(BIC_2broad):
            candidates.append((BIC_2broad, "two"))

        if len(candidates) == 1:
            broad_choice = "none"
            BIC_broad = np.nan
        else:
            if broad_mode == "broad1":
                # Force BROAD only
                if fit_1broad is not None and np.isfinite(BIC_1broad):
                    broad_choice = "one"
                else:
                    if verbose:
                        print("broad_mode='broad1' but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "broad2":
                # Force BROAD2 only (without BROAD)
                if fit_b2only is not None and np.isfinite(BIC_b2only):
                    broad_choice = "broad2_only"
                else:
                    if verbose:
                        print("broad_mode='broad2' but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "both":
                # Force both BROAD and BROAD2
                if fit_2broad is not None and np.isfinite(BIC_2broad):
                    broad_choice = "two"
                else:
                    if verbose:
                        print("broad_mode='both' but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            else:
                # auto → pick lowest BIC, but require ΔBIC improvement
                B_vals = [c[0] for c in candidates]
                i_best = int(np.argmin(B_vals))
                BIC_best, broad_choice = candidates[i_best]
                if broad_choice != "none":
                    if (not np.isfinite(BIC_narrow)) or (BIC_best + bic_delta_prefer >= BIC_narrow):
                        if verbose:
                            print("Broad Hα models do not improve BIC enough → reverting to narrow-only.")
                        broad_choice = "none"

            finite_bics = [b for b in (BIC_1broad, BIC_2broad) if np.isfinite(b)]
            BIC_broad = min(finite_bics) if finite_bics else np.nan

        if verbose:
            # Map internal choice to descriptive name
            model_desc = {
                "none": "narrow-only",
                "one": "narrow + BROAD",
                "broad2_only": "narrow + BROAD2",
                "two": "narrow + BROAD + BROAD2"
            }.get(broad_choice, broad_choice)
            
            print("Local Hα+[N II] BIC (Hα window only):")
            print(f"  narrow-only      : BIC = {BIC_narrow:.2f}")
            if np.isfinite(BIC_1broad):
                print(f"  +BROAD only      : BIC = {BIC_1broad:.2f}")
            else:
                print("  +BROAD only      : (fit failed)")
            if np.isfinite(BIC_b2only):
                print(f"  +BROAD2 only     : BIC = {BIC_b2only:.2f}")
            else:
                print("  +BROAD2 only     : (fit failed)")
            if np.isfinite(BIC_2broad):
                print(f"  +both BROAD      : BIC = {BIC_2broad:.2f}")
            else:
                print("  +both BROAD      : (fit failed)")
            print(f"  → Selected model: {model_desc}")

        return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

    # ----------------------------------------------------------------
    # Helper: decide Hδ model (none / one / two broad) in *local* window
    # ----------------------------------------------------------------
    def _select_hd_model(which_lines_all):
        """
        Use only a local Hδ+[O III]4363 window for BIC comparison, but do not
        change the global continuum or residuals. Returns:

          broad_choice ∈ {"none","one","two"}
          BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad
        """
        BIC_narrow = np.nan
        BIC_1broad = np.nan
        BIC_2broad = np.nan
        BIC_b2only = np.nan
        BIC_broad  = np.nan
        broad_choice = "none"

        # If Hδ isn't even in the global narrow list, nothing to do
        if "HDELTA" not in which_lines_all:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        hd_doublet = ["HDELTA", "OIII_4363"]
        which_hd_narrow = [ln for ln in hd_doublet if ln in which_lines_all]
        if "HDELTA" not in which_hd_narrow:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Local Hδ window: HDELTA–[O III]4363 ± padding
        rest_list = [
            REST_LINES_A["HDELTA"],
            REST_LINES_A["OIII_4363"],
        ]
        lo_hd, hi_hd = _window_from_lines_um(rest_list, z, pad_A=150.0)
        m_hd = (lam_all >= lo_hd) & (lam_all <= hi_hd)
        if np.count_nonzero(m_hd) < 5:
            # basically no Hδ coverage
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        lam_hd   = lam_all[m_hd]
        resid_hd = resid_all[m_hd]
        sig_hd   = sig_all[m_hd]
        Fcont_hd = Fcont_all[m_hd]

        # Base narrow-only fit (M0)
        fit_narrow = _fit_emission_system(
            lam_hd, resid_hd, sig_hd, Fcont_hd,
            z, grating, which_hd_narrow,
            absorption_corrections=absorption_corrections,
            verbose=verbose,
        )
        BIC_narrow = fit_narrow["BIC"]

        if broad_mode == "off":
            if verbose:
                print("broad_mode='off' → using narrow-only Hδ+[O III]4363 model.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Check Hδ SNR for eligibility (auto mode)
        hd_info = fit_narrow["per_line"].get("HDELTA", None)
        hd_snr  = hd_info.get("SNR", 0.0) if hd_info is not None else 0.0
        eligible = (hd_info is not None) and np.isfinite(hd_snr)

        if (broad_mode == "auto") and (not eligible or hd_snr < snr_broad_threshold):
            if verbose:
                print(f"No strong Hδ line (SNR={hd_snr:.1f}) → staying narrow-only.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Try adding 1 and 2 broad Hδ components on the local window
        def _safe_fit(which, label):
            try:
                fb = _fit_emission_system(
                    lam_hd, resid_hd, sig_hd, Fcont_hd,
                    z, grating, which,
                    absorption_corrections=absorption_corrections,
                    verbose=verbose,
                )
                return fb, fb["BIC"]
            except Exception as e:
                if verbose:
                    print(f"{label} model (Hδ) failed: {e}")
                return None, np.nan

        which_1 = list(which_hd_narrow)
        if "HDELTA_BROAD" not in which_1:
            which_1.append("HDELTA_BROAD")
        fit_1broad, BIC_1broad = _safe_fit(which_1, "1-broad Hδ")
        if fit_1broad is not None:
            for ln, info in fit_1broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 1-broad Hδ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_1broad = np.nan
                    break

        which_2 = list(which_1)
        if "HDELTA_BROAD2" not in which_2:
            which_2.append("HDELTA_BROAD2")
        fit_2broad, BIC_2broad = _safe_fit(which_2, "2-broad Hδ")
        if fit_2broad is not None:
            for ln, info in fit_2broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 2-broad Hδ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_2broad = np.nan
                    break

        # BROAD2-only (without BROAD) for complete 4-way comparison
        which_b2only = list(which_hd_narrow)
        if "HDELTA_BROAD2" not in which_b2only:
            which_b2only.append("HDELTA_BROAD2")
        fit_b2only, BIC_b2only = _safe_fit(which_b2only, "BROAD2-only Hδ")
        if fit_b2only is not None:
            for ln, info in fit_b2only["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> BROAD2-only Hδ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_b2only = np.nan
                    break

        candidates = [(BIC_narrow, "none")]
        if fit_1broad is not None and np.isfinite(BIC_1broad):
            candidates.append((BIC_1broad, "one"))
        if fit_b2only is not None and np.isfinite(BIC_b2only):
            candidates.append((BIC_b2only, "broad2_only"))
        if fit_2broad is not None and np.isfinite(BIC_2broad):
            candidates.append((BIC_2broad, "two"))

        if len(candidates) == 1:
            broad_choice = "none"
            BIC_broad = np.nan
        else:
            if broad_mode == "broad1":
                # Force BROAD only
                if fit_1broad is not None and np.isfinite(BIC_1broad):
                    broad_choice = "one"
                else:
                    if verbose:
                        print("broad_mode='broad1' (Hδ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "broad2":
                # Force BROAD2 only (without BROAD)
                if fit_b2only is not None and np.isfinite(BIC_b2only):
                    broad_choice = "broad2_only"
                else:
                    if verbose:
                        print("broad_mode='broad2' (Hδ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "both":
                # Force both BROAD and BROAD2
                if fit_2broad is not None and np.isfinite(BIC_2broad):
                    broad_choice = "two"
                else:
                    if verbose:
                        print("broad_mode='both' (Hδ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            else:
                # auto → pick lowest BIC, but require ΔBIC improvement
                B_vals = [c[0] for c in candidates]
                i_best = int(np.argmin(B_vals))
                BIC_best, broad_choice = candidates[i_best]
                if broad_choice != "none":
                    if (not np.isfinite(BIC_narrow)) or (BIC_best + bic_delta_prefer >= BIC_narrow):
                        if verbose:
                            print("Broad Hδ models do not improve BIC enough → reverting to narrow-only.")
                        broad_choice = "none"

            finite_bics = [b for b in (BIC_1broad, BIC_2broad) if np.isfinite(b)]
            BIC_broad = min(finite_bics) if finite_bics else np.nan

        if verbose:
            # Map internal choice to descriptive name
            model_desc = {
                "none": "narrow-only",
                "one": "narrow + BROAD",
                "broad2_only": "narrow + BROAD2",
                "two": "narrow + BROAD + BROAD2"
            }.get(broad_choice, broad_choice)
            
            print("Local Hδ+[O III]4363 BIC (Hδ window only):")
            print(f"  narrow-only      : BIC = {BIC_narrow:.2f}")
            if np.isfinite(BIC_1broad):
                print(f"  +BROAD only      : BIC = {BIC_1broad:.2f}")
            else:
                print("  +BROAD only      : (fit failed)")
            if np.isfinite(BIC_b2only):
                print(f"  +BROAD2 only     : BIC = {BIC_b2only:.2f}")
            else:
                print("  +BROAD2 only     : (fit failed)")
            if np.isfinite(BIC_2broad):
                print(f"  +both BROAD      : BIC = {BIC_2broad:.2f}")
            else:
                print("  +both BROAD      : (fit failed)")
            print(f"  → Selected model: {model_desc}")

        return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # ----------------------------------------------------------------
    # Helper: decide Hβ model (none / one / two broad) in *local* window
    # ----------------------------------------------------------------
    def _select_hb_model(which_lines_all):
        """
        Use only a local Hβ+[O III] window for BIC comparison, but do not
        change the global continuum or residuals. Returns:

          broad_choice ∈ {"none","one","two"}
          BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad
        """
        BIC_narrow = np.nan
        BIC_1broad = np.nan
        BIC_2broad = np.nan
        BIC_b2only = np.nan
        BIC_broad  = np.nan
        broad_choice = "none"

        # If Hβ isn't even in the global narrow list, nothing to do
        if "HBETA" not in which_lines_all:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        hb_triplet = ["HBETA", "OIII_4959", "OIII_5007"]
        which_hb_narrow = [ln for ln in hb_triplet if ln in which_lines_all]
        if "HBETA" not in which_hb_narrow:
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Local Hβ window: HBETA–[O III] ± padding
        rest_list = [
            REST_LINES_A["HBETA"],
            REST_LINES_A["OIII_4959"],
            REST_LINES_A["OIII_5007"],
        ]
        lo_hb, hi_hb = _window_from_lines_um(rest_list, z, pad_A=150.0)
        m_hb = (lam_all >= lo_hb) & (lam_all <= hi_hb)
        if np.count_nonzero(m_hb) < 5:
            # basically no Hβ coverage
            return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        lam_hb   = lam_all[m_hb]
        resid_hb = resid_all[m_hb]
        sig_hb   = sig_all[m_hb]
        Fcont_hb = Fcont_all[m_hb]

        # Base narrow-only fit (M0)
        fit_narrow = _fit_emission_system(
            lam_hb, resid_hb, sig_hb, Fcont_hb,
            z, grating, which_hb_narrow,
            absorption_corrections=absorption_corrections,
            verbose=verbose,
        )
        BIC_narrow = fit_narrow["BIC"]

        if broad_mode == "off":
            if verbose:
                print("broad_mode='off' → using narrow-only Hβ+[O III] model.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Check Hβ SNR for eligibility (auto mode)
        hb_info = fit_narrow["per_line"].get("HBETA", None)
        hb_snr  = hb_info.get("SNR", 0.0) if hb_info is not None else 0.0
        eligible = (hb_info is not None) and np.isfinite(hb_snr)

        if (broad_mode == "auto") and (not eligible or hb_snr < snr_broad_threshold):
            if verbose:
                print(f"No strong Hβ line (SNR={hb_snr:.1f}) → staying narrow-only.")
            return "none", BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad

        # Try adding 1 and 2 broad Hβ components on the local window
        def _safe_fit(which, label):
            try:
                fb = _fit_emission_system(
                    lam_hb, resid_hb, sig_hb, Fcont_hb,
                    z, grating, which,
                    absorption_corrections=absorption_corrections,
                    verbose=verbose,
                )
                return fb, fb["BIC"]
            except Exception as e:
                if verbose:
                    print(f"{label} model (Hβ) failed: {e}")
                return None, np.nan

        which_1 = list(which_hb_narrow)
        if "HBETA_BROAD" not in which_1:
            which_1.append("HBETA_BROAD")
        fit_1broad, BIC_1broad = _safe_fit(which_1, "1-broad Hβ")
        if fit_1broad is not None:
            for ln, info in fit_1broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 1-broad Hβ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_1broad = np.nan
                    break

        which_2 = list(which_1)
        if "HBETA_BROAD2" not in which_2:
            which_2.append("HBETA_BROAD2")
        fit_2broad, BIC_2broad = _safe_fit(which_2, "2-broad Hβ")
        if fit_2broad is not None:
            for ln, info in fit_2broad["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> 2-broad Hβ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_2broad = np.nan
                    break

        # BROAD2-only (without BROAD) for complete 4-way comparison
        which_b2only = list(which_hb_narrow)
        if "HBETA_BROAD2" not in which_b2only:
            which_b2only.append("HBETA_BROAD2")
        fit_b2only, BIC_b2only = _safe_fit(which_b2only, "BROAD2-only Hβ")
        if fit_b2only is not None:
            for ln, info in fit_b2only["per_line"].items():
                if "BROAD" in ln and info["A_gauss"] < 0:
                    if verbose:
                        print(f"  -> BROAD2-only Hβ rejected (negative {ln}: {info['A_gauss']:.2e})")
                    BIC_b2only = np.nan
                    break

        candidates = [(BIC_narrow, "none")]
        if fit_1broad is not None and np.isfinite(BIC_1broad):
            candidates.append((BIC_1broad, "one"))
        if fit_b2only is not None and np.isfinite(BIC_b2only):
            candidates.append((BIC_b2only, "broad2_only"))
        if fit_2broad is not None and np.isfinite(BIC_2broad):
            candidates.append((BIC_2broad, "two"))

        if len(candidates) == 1:
            broad_choice = "none"
            BIC_broad = np.nan
        else:
            if broad_mode == "broad1":
                # Force BROAD only
                if fit_1broad is not None and np.isfinite(BIC_1broad):
                    broad_choice = "one"
                else:
                    if verbose:
                        print("broad_mode='broad1' (Hβ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "broad2":
                # Force BROAD2 only (without BROAD)
                if fit_b2only is not None and np.isfinite(BIC_b2only):
                    broad_choice = "broad2_only"
                else:
                    if verbose:
                        print("broad_mode='broad2' (Hβ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            elif broad_mode == "both":
                # Force both BROAD and BROAD2
                if fit_2broad is not None and np.isfinite(BIC_2broad):
                    broad_choice = "two"
                else:
                    if verbose:
                        print("broad_mode='both' (Hβ) but fit failed → reverting to narrow-only.")
                    broad_choice = "none"
            else:
                # auto → pick lowest BIC, but require ΔBIC improvement
                B_vals = [c[0] for c in candidates]
                i_best = int(np.argmin(B_vals))
                BIC_best, broad_choice = candidates[i_best]
                if broad_choice != "none":
                    if (not np.isfinite(BIC_narrow)) or (BIC_best + bic_delta_prefer >= BIC_narrow):
                        if verbose:
                            print("Broad Hβ models do not improve BIC enough → reverting to narrow-only.")
                        broad_choice = "none"

            finite_bics = [b for b in (BIC_1broad, BIC_2broad) if np.isfinite(b)]
            BIC_broad = min(finite_bics) if finite_bics else np.nan

        if verbose:
            # Map internal choice to descriptive name
            model_desc = {
                "none": "narrow-only",
                "one": "narrow + BROAD",
                "broad2_only": "narrow + BROAD2",
                "two": "narrow + BROAD + BROAD2"
            }.get(broad_choice, broad_choice)
            
            print("Local Hβ+[O III] BIC (Hβ window only):")
            print(f"  narrow-only      : BIC = {BIC_narrow:.2f}")
            if np.isfinite(BIC_1broad):
                print(f"  +BROAD only      : BIC = {BIC_1broad:.2f}")
            else:
                print("  +BROAD only      : (fit failed)")
            if np.isfinite(BIC_b2only):
                print(f"  +BROAD2 only     : BIC = {BIC_b2only:.2f}")
            else:
                print("  +BROAD2 only     : (fit failed)")
            if np.isfinite(BIC_2broad):
                print(f"  +both BROAD      : BIC = {BIC_2broad:.2f}")
            else:
                print("  +both BROAD      : (fit failed)")
            print(f"  → Selected model: {model_desc}")

        return broad_choice, BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only, BIC_broad


    # ------------------------------------------------------------------
    # Branch A: force_lines → used by bootstrap to *freeze* the structure
    # ------------------------------------------------------------------
    if force_lines is not None:
        which_lines_global = list(force_lines)
        which_lines_global, _ = _prune_lines_without_data(
            lam_all, resid_all, sig_all, which_lines_global,
            z, grating, min_pts_per_line=3,
            window_sigma=4.0, window_um=None,
            verbose=bool(verbose),
        )
        if not which_lines_global:
            raise ValueError("force_lines provided, but no lines have local data.")

        fit_global = _fit_emission_system(
            lam_all, resid_all, sig_all, Fcont_all,
            z, grating, which_lines_global,
            absorption_corrections=absorption_corrections,
            verbose=verbose,
        )

        res             = fit_global["res"]
        model_flam_win  = fit_global["model_flam"]
        profiles_win    = fit_global["profiles"]
        centers         = fit_global["centers"]
        per_line        = fit_global["per_line"]
        which_lines_out = fit_global["which_lines"]
        BIC_best        = fit_global["BIC"]

        # No local BIC selection was done in force_lines path
        BIC_HA_narrow = BIC_HA_1broad = BIC_HA_2broad = BIC_HA_b2only = BIC_HA_broad = np.nan
        BIC_HB_narrow = BIC_HB_1broad = BIC_HB_2broad = BIC_HB_b2only = BIC_HB_broad = np.nan
        BIC_HD_narrow = BIC_HD_1broad = BIC_HD_2broad = BIC_HD_b2only = BIC_HD_broad = np.nan
        broad_choice_ha = "forced"
        broad_choice_hb = "forced"
        broad_choice_hd = "forced"


    else:
        # ------------------------------------------------------------------
        # Branch B: "normal" call → choose Hα & Hβ models, then global fit
        # ------------------------------------------------------------------
        # 1) all *narrow* lines in coverage in the global window
        use_lines = lines_to_use
        if use_lines is None:
            use_lines = get_default_line_list(grating)

        which_lines_all = _lines_in_range(z, lam_all, use_lines, margin_um=0.02)
        which_lines_all = [ln for ln in which_lines_all if "BROAD" not in ln]

        which_lines_all, _ = _prune_lines_without_data(
            lam_all, resid_all, sig_all, which_lines_all,
            z, grating, min_pts_per_line=3,
            window_sigma=4.0, window_um=None,
            verbose=bool(verbose),
        )
        if not which_lines_all:
            # Raise no error, just return continuum-only result
            if verbose:
                print("No emission lines with data in the global fit window (or empty list provided). Returning continuum only.")
            
            return {
                "success": True,
                "message": "Continuum only",
                "lam_fit": lam_all,
                "model_window_flam": np.zeros_like(lam_all),
                "model_flam": np.zeros_like(lam_all),  # for consistent return
                "continuum_flam": Fcont_all,
                "lines": {},
                "which_lines": [],
                "BIC": np.nan,
                # BICs
                "BIC_HA_narrow": np.nan, "BIC_HA_1broad": np.nan, "BIC_HA_2broad": np.nan, "BIC_HA_b2only": np.nan, "broad_choice_HA": "none",
                "BIC_HB_narrow": np.nan, "BIC_HB_1broad": np.nan, "BIC_HB_2broad": np.nan, "BIC_HB_b2only": np.nan, "broad_choice_HB": "none",
                "BIC_HD_narrow": np.nan, "BIC_HD_1broad": np.nan, "BIC_HD_2broad": np.nan, "BIC_HD_b2only": np.nan, "broad_choice_HD": "none",
            }

        # 2) decide Hα, Hβ, and Hδ models (none/one/two broad) using local windows
        broad_choice_ha, BIC_HA_narrow, BIC_HA_1broad, BIC_HA_2broad, BIC_HA_b2only, BIC_HA_broad = _select_ha_model(which_lines_all)
        broad_choice_hb, BIC_HB_narrow, BIC_HB_1broad, BIC_HB_2broad, BIC_HB_b2only, BIC_HB_broad = _select_hb_model(which_lines_all)
        broad_choice_hd, BIC_HD_narrow, BIC_HD_1broad, BIC_HD_2broad, BIC_HD_b2only, BIC_HD_broad = _select_hd_model(which_lines_all)

        # 3) build final global line list
        which_lines_global = list(which_lines_all)

        # Hα broad components
        if broad_choice_ha in {"one", "two"}:
            if "H⍺_BROAD" not in which_lines_global:
                which_lines_global.append("H⍺_BROAD")
        if broad_choice_ha == "two":
            if "H⍺_BROAD2" not in which_lines_global:
                which_lines_global.append("H⍺_BROAD2")
        if broad_choice_ha == "broad2_only":
            # Only BROAD2, no BROAD
            if "H⍺_BROAD2" not in which_lines_global:
                which_lines_global.append("H⍺_BROAD2")

        # Hβ broad components
        if broad_choice_hb in {"one", "two"}:
            if "HBETA_BROAD" not in which_lines_global:
                which_lines_global.append("HBETA_BROAD")
        if broad_choice_hb == "two":
            if "HBETA_BROAD2" not in which_lines_global:
                which_lines_global.append("HBETA_BROAD2")
        if broad_choice_hb == "broad2_only":
            # Only BROAD2, no BROAD
            if "HBETA_BROAD2" not in which_lines_global:
                which_lines_global.append("HBETA_BROAD2")

        # Hδ broad components
        if broad_choice_hd in {"one", "two"}:
            if "HDELTA_BROAD" not in which_lines_global:
                which_lines_global.append("HDELTA_BROAD")
        if broad_choice_hd == "two":
            if "HDELTA_BROAD2" not in which_lines_global:
                which_lines_global.append("HDELTA_BROAD2")
        if broad_choice_hd == "broad2_only":
            # Only BROAD2, no BROAD
            if "HDELTA_BROAD2" not in which_lines_global:
                which_lines_global.append("HDELTA_BROAD2")

        which_lines_global, _ = _prune_lines_without_data(
            lam_all, resid_all, sig_all, which_lines_global,
            z, grating, min_pts_per_line=3,
            window_sigma=4.0, window_um=None,
            verbose=bool(verbose),
        )
        if not which_lines_global:
            raise ValueError("After adding Balmer broad components, no lines remain to fit.")

        # 4) one *global* fit with the chosen Balmer parametrisations
        fit_global = _fit_emission_system(
            lam_all, resid_all, sig_all, Fcont_all,
            z, grating, which_lines_global,
            absorption_corrections=absorption_corrections,
            verbose=verbose,
        )

        res             = fit_global["res"]
        model_flam_win  = fit_global["model_flam"]
        profiles_win    = fit_global["profiles"]
        centers         = fit_global["centers"]
        per_line        = fit_global["per_line"]
        which_lines_out = fit_global["which_lines"]
        BIC_best        = fit_global["BIC"]

        # Backwards-compatible aliases: keep old "Hα-only" names pointing to Hα BICs
    BIC_narrow = BIC_HA_narrow
    BIC_1broad = BIC_HA_1broad
    BIC_2broad = BIC_HA_2broad
    BIC_broad  = BIC_HA_broad
    broad_choice = broad_choice_ha



    # For plotting we now interpret lam_fit as the *global* fit grid
    lam_fit   = lam_all
    Fcont_fit = Fcont_all

    # ------------------------------------------------------------------
    # Plotting
    if plot:
        # Embed model_window_flam (defined on lam_fit) back into the full lam_um grid
        if (lam_fit.size == lam_um.size) and np.allclose(lam_fit, lam_um):
            # Model already on the full grid
            model_full = model_flam_win
        else:
            # Model only defined on a subset (e.g. user-specified fit_window_um)
            model_full = np.zeros_like(lam_um)
            # lam_fit is always a subset slice of lam_um, so exact equality is fine
            mask = np.isin(lam_um, lam_fit)
            model_full[mask] = model_flam_win

        total_model_flam = Fcont + model_full

        cont_uJy  = flam_to_fnu_uJy(Fcont,            lam_um)
        model_uJy = flam_to_fnu_uJy(total_model_flam, lam_um)

        left_um, right_um = _pixel_edges_um(lam_um)
        edges_um = np.r_[left_um[0], right_um]

        fig, axes = plt.subplots(
            2, 1, figsize=(11, 7.5),
            gridspec_kw={"height_ratios": [1.4, 1.2]}
        )
        ax1, ax2 = axes


        # --- µJy panel ---
        ax1.stairs(flux_uJy, edges_um, label='Data (bins)', color='k', linewidth=0.9, alpha=0.85)
        ax1.errorbar(lam_um, flux_uJy, yerr=err_uJy, fmt='o', ms=2.5,
                     color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                     ecolor=(0,0,0,0.12), elinewidth=0.4, capsize=0, zorder=1,
                     label='Data ±1σ')
        ax1.stairs(cont_uJy, edges_um, label=f'Continuum (deg={deg})',
                   color='b', linestyle='--', linewidth=1.0)
        ax1.stairs(model_uJy, edges_um, label='Continuum + Lines',
           

 color='r', linewidth=1.2)

        # --- Hα complex: stacked components in µJy (narrow + broad + broad2) ---

        narrow_color = "tab:blue"
        broad_color  = "tab:orange"
        broad_color2 = "tab:pink"

        # Window around the Hα+[N II] complex
        rest_list = [REST_LINES_A["NII_6549"], REST_LINES_A["H⍺"], REST_LINES_A["NII_6585"]]
        lo_ha, hi_ha = _window_from_lines_um(rest_list, z, pad_A=150.0)
        m_ha = (lam_fit >= lo_ha) & (lam_fit <= hi_ha)

        if np.any(m_ha):
            lam_ha      = lam_fit[m_ha]
            cont_ha_flam = Fcont_fit[m_ha]

            # emission-only contribution of each component in Fλ
            em_narrow = np.zeros_like(lam_ha)
            for nm in ("NII_6549", "H⍺", "NII_6585"):
                if nm in profiles_win:
                    em_narrow += profiles_win[nm][m_ha]

            em_b1 = profiles_win.get("H⍺_BROAD",  np.zeros_like(lam_fit))[m_ha]
            em_b2 = profiles_win.get("H⍺_BROAD2", np.zeros_like(lam_fit))[m_ha]

            # successive baselines in Fλ
            base_flam      = cont_ha_flam
            top_narrow     = base_flam + em_narrow
            top_narrow_b1  = top_narrow + em_b1
            top_narrow_b12 = top_narrow_b1 + em_b2  # full Hα model

            # Create fine grid for smooth fills
            oversample = 6
            lam_ha_fine = np.linspace(lam_ha.min(), lam_ha.max(), 
                                      max(lam_ha.size * oversample, 200))
            
            # Interpolate each layer onto fine grid
            base_flam_fine = np.interp(lam_ha_fine, lam_ha, base_flam)
            top_narrow_fine = np.interp(lam_ha_fine, lam_ha, top_narrow)
            top_narrow_b1_fine = np.interp(lam_ha_fine, lam_ha, top_narrow_b1)
            top_narrow_b12_fine = np.interp(lam_ha_fine, lam_ha, top_narrow_b12)
            
            # Convert to µJy
            base_uJy_fine = flam_to_fnu_uJy(base_flam_fine, lam_ha_fine)
            top_narrow_uJy_fine = flam_to_fnu_uJy(top_narrow_fine, lam_ha_fine)
            top_b1_uJy_fine = flam_to_fnu_uJy(top_narrow_b1_fine, lam_ha_fine)
            top_b2_uJy_fine = flam_to_fnu_uJy(top_narrow_b12_fine, lam_ha_fine)

            # Smooth stacked areas
            ax1.fill_between(
                lam_ha_fine, base_uJy_fine, top_narrow_uJy_fine,
                color=narrow_color, alpha=0.35,
                label="Hα+[N II] narrow", linewidth=0,
            )

            if np.any(np.isfinite(em_b1)) and np.nanmax(np.abs(em_b1)) > 0:
                ax1.fill_between(
                    lam_ha_fine, top_narrow_uJy_fine, top_b1_uJy_fine,
                    color=broad_color, alpha=0.25,
                    label="Hα broad 1", linewidth=0,
                )

            if np.any(np.isfinite(em_b2)) and np.nanmax(np.abs(em_b2)) > 0:
                ax1.fill_between(
                    lam_ha_fine, top_b1_uJy_fine, top_b2_uJy_fine,
                    color=broad_color2, alpha=0.20,
                    label="Hα broad 2", linewidth=0,
                )



        ax1.set_ylabel('Flux density [µJy]')
        ax1.legend(ncol=3, fontsize=9, frameon=False)
        _annotate_lines(ax1, which_lines_out, z, per_line=per_line, min_dx_um=0.005)

        # --- F_lambda panel ---
        data_flam = flam
        data_flam_err = sig_flam
        ax2.stairs(data_flam, edges_um, label='Data (Fλ, bins)',
                   color='k', linewidth=0.9, alpha=0.85)
        ax2.errorbar(lam_um, data_flam, yerr=data_flam_err, fmt='o', ms=2.5,
                     color='0.35', mfc='none', mec=(0,0,0,0.25), mew=0.4,
                     ecolor=(0,0,0,0.12), elinewidth=0.4, capsize=0, zorder=1,
                     label='Data (Fλ) ±1σ')
        ax2.stairs(Fcont, edges_um, label='Continuum (Fλ)', color='b',
                   linestyle='--', linewidth=1.0)
        ax2.stairs(total_model_flam, edges_um, label='Model (Fλ)',
                   color='r', linewidth=1.2)

        # --- Hα complex: stacked components in Fλ ---

        cont_fit_flam = Fcont_fit  # continuum on lam_fit

        rest_list = [REST_LINES_A["NII_6549"], REST_LINES_A["H⍺"], REST_LINES_A["NII_6585"]]
        lo_ha, hi_ha = _window_from_lines_um(rest_list, z, pad_A=150.0)
        m_ha = (lam_fit >= lo_ha) & (lam_fit <= hi_ha)

        if np.any(m_ha):
            lam_ha      = lam_fit[m_ha]
            cont_ha_flam = cont_fit_flam[m_ha]

            em_narrow = np.zeros_like(lam_ha)
            for nm in ("NII_6549", "H⍺", "NII_6585"):
                if nm in profiles_win:
                    em_narrow += profiles_win[nm][m_ha]

            em_b1 = profiles_win.get("H⍺_BROAD",  np.zeros_like(lam_fit))[m_ha]
            em_b2 = profiles_win.get("H⍺_BROAD2", np.zeros_like(lam_fit))[m_ha]

            base_flam      = cont_ha_flam
            top_narrow     = base_flam + em_narrow
            top_narrow_b1  = top_narrow + em_b1
            top_narrow_b12 = top_narrow_b1 + em_b2

            # Create fine grid for smooth fills
            oversample = 6
            lam_ha_fine = np.linspace(lam_ha.min(), lam_ha.max(),
                                      max(lam_ha.size * oversample, 200))
            
            # Interpolate onto fine grid
            base_flam_fine = np.interp(lam_ha_fine, lam_ha, base_flam)
            top_narrow_fine = np.interp(lam_ha_fine, lam_ha, top_narrow)
            top_narrow_b1_fine = np.interp(lam_ha_fine, lam_ha, top_narrow_b1)
            top_narrow_b12_fine = np.interp(lam_ha_fine, lam_ha, top_narrow_b12)

            # Smooth stacked areas
            ax2.fill_between(
                lam_ha_fine, base_flam_fine, top_narrow_fine,
                color=narrow_color, alpha=0.35,
                label="Hα+[N II] narrow", linewidth=0,
            )

            if np.any(np.isfinite(em_b1)) and np.nanmax(np.abs(em_b1)) > 0:
                ax2.fill_between(
                    lam_ha_fine, top_narrow_fine, top_narrow_b1_fine,
                    color=broad_color, alpha=0.25,
                    label="Hα broad 1", linewidth=0,
                )

            if np.any(np.isfinite(em_b2)) and np.nanmax(np.abs(em_b2)) > 0:
                ax2.fill_between(
                    lam_ha_fine, top_narrow_b1_fine, top_narrow_b12_fine,
                    color=broad_color2, alpha=0.20,
                    label="Hα broad 2", linewidth=0,
                )


        ax2.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        ax2.set_xlabel('Observed wavelength [µm]')
        ax2.legend(ncol=3, fontsize=9, frameon=False)
        _annotate_lines(ax2, which_lines_out, z, per_line=per_line, min_dx_um=0.005)

        plt.tight_layout()
        plt.show()

    return dict(
        success=res.success,
        message=res.message,
        lam_fit=lam_fit,
        model_window_flam=model_flam_win,
        continuum_flam=Fcont,
        lines=per_line,
        which_lines=which_lines_out,
        BIC=BIC_best,   # global model BIC on the chosen window

        # Hα BIC summary + backward-compatible aliases
        BIC_HA_narrow=BIC_HA_narrow,
        BIC_HA_broad=BIC_HA_broad,
        BIC_HA_1broad=BIC_HA_1broad,
        BIC_HA_2broad=BIC_HA_2broad,
        BIC_HA_b2only=BIC_HA_b2only,
        broad_choice_HA=broad_choice_ha,

        BIC_narrow=BIC_narrow,
        BIC_broad=BIC_broad,
        BIC_1broad=BIC_1broad,
        BIC_2broad=BIC_2broad,
        broad_choice=broad_choice,

        # Hβ BIC summary
        BIC_HB_narrow=BIC_HB_narrow,
        BIC_HB_broad=BIC_HB_broad,
        BIC_HB_1broad=BIC_HB_1broad,
        BIC_HB_2broad=BIC_HB_2broad,
        BIC_HB_b2only=BIC_HB_b2only,
        broad_choice_HB=broad_choice_hb,

        # Hδ BIC summary
        BIC_HD_narrow=BIC_HD_narrow,
        BIC_HD_broad=BIC_HD_broad,
        BIC_HD_1broad=BIC_HD_1broad,
        BIC_HD_2broad=BIC_HD_2broad,
        BIC_HD_b2only=BIC_HD_b2only,
        broad_choice_HD=broad_choice_hd,

        profiles_window_flam=profiles_win,
    )





# --------------------------------------------------------------------
# Bootstrap variant that uses single_broad_fit
# --------------------------------------------------------------------

def _edges_median_spacing(lam_um: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam_um, float)
    if lam.size < 2:
        d = 1e-6
        return np.array([lam[0]-d, lam[0]+d], float)
    dlam = np.median(np.diff(lam))
    mids = 0.5 * (lam[1:] + lam[:-1])
    return np.concatenate(([lam[0] - dlam/2], mids, [lam[-1] + dlam/2]))


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


def broad_fit(
    source,
    z,
    grating: str = "PRISM",
    n_boot: int = 200,
    source_id: str | None = None,
    deg: int = 2,
    continuum_windows=None,
    lyman_cut="split",
    fit_window_um=None,
    absorption_corrections=None,
    random_state=None,
    verbose: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    save_path: str | None = None,
    save_dpi: int = 500,
    save_format: str = "png",
    save_transparent: bool = False,
    lines_to_use=None,
    broad_mode: str = "auto",
    plot_unit: str = "fnu",
    plot_continuum_subtracted: bool = False,
    continuum_lya_mask_A: float = 200.0,
    continuum_fit: str = "polyfit",
    continuum_movavg_window_um: float = 0.05,
    continuum_exclude_regions: Optional[List[Tuple[float, float]]] = None,
    continuum_lyman_window_um: Optional[float] = None,
    plot_ylim: Optional[Tuple[float, float]] = None,
    save_npz: bool = False,
    npz_name: Optional[str] = None,
) -> Dict:


    """
    Fit emission lines with bootstrap uncertainty estimation.

    Performs a base fit to determine the optimal line list (including broad
    components if warranted by BIC), then runs bootstrap iterations using the
    same line list to estimate uncertainties.

    Parameters
    ----------
    source : dict or HDUList
        Spectrum data with 'lam'/'wave', 'flux', and 'err' keys (dict) or
        FITS HDUList with SPEC1D extension.
    z : float
        Redshift of the source.
    grating : str, default='PRISM'
        Grating name for line list selection.
    n_boot : int, default=200
        Number of bootstrap iterations.
    source_id : str, optional
        Source identifier for plot titles.
    deg : int, default=2
        Polynomial degree for continuum fitting.
    continuum_windows : list of tuples or str, optional
        Wavelength windows for continuum fitting, or 'two_sided_lya' for automatic.
    lyman_cut : str, default='split'
        Lyman-alpha cutoff mode. 'split' fits both sides (default), 'lya' masks blue.
    fit_window_um : tuple of float, optional
        (low, high) wavelength range in microns for fitting.
    absorption_corrections : dict, optional
        Absorption corrections to apply.
    random_state : int or RandomState, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print diagnostic information.
    plot : bool, default=True
        Whether to generate diagnostic plots.
    show_progress : bool, default=True
        Whether to show progress bar during bootstrap.
    save_path : str, optional
        Path to save the output plot.
    save_dpi : int, default=500
        DPI for saved plot.
    save_format : str, default='png'
        Format for saved plot.
    save_transparent : bool, default=False
        Whether to use transparent background in saved plot.
    lines_to_use : list of str, optional
        Subset of emission lines to fit.
    broad_mode : str, default='auto'
        Broad component selection mode: 'auto', 'off', 'broad1', 'broad2', 'both'.
    plot_unit : str, default='fnu'
        Plotting unit: 'fnu' (µJy) or 'flam' (F_lambda).

    Returns
    -------
    dict
        Bootstrap results including mean line parameters, uncertainties, and plots.
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

    # -------- base run (chooses model structure) --------
    try:
        base = single_broad_fit(
            source=dict(lam=lam_um, flux=flux_uJy, err=err_uJy),
            z=z, grating=grating, deg=deg,
            continuum_windows=continuum_windows,
            lyman_cut=lyman_cut,
            fit_window_um=fit_window_um,
            plot=False, verbose=verbose,
            absorption_corrections=absorption_corrections,
            lines_to_use=lines_to_use,
            broad_mode=broad_mode,
            continuum_lya_mask_A=continuum_lya_mask_A,
            continuum_fit=continuum_fit,
            continuum_movavg_window_um=continuum_movavg_window_um,
            continuum_exclude_regions=continuum_exclude_regions,
            continuum_lyman_window_um=continuum_lyman_window_um,
        )
        which_lines = list(base.get("which_lines", []))
        if not which_lines:
            pass

    except ValueError as e:
        if verbose:
            print(f"Fit failed: {e}")
        
        # Return empty result structure to prevent crash
        return {
            "samples": {},
            "summary": {},
            "which_lines": [],
            "model_stack_flam": np.array([]),
            "keep_mask": np.array([]),
            "lam_um": lam_um,
            "data_flux_uJy": flux_uJy,
            # BIC scores
            "BIC_HA_narrow": np.nan,
            "BIC_HA_1broad": np.nan,
            "BIC_HA_2broad": np.nan,
            "BIC_HA_b2only": np.nan,
            "broad_choice_HA": "failed",
            "BIC_HB_narrow": np.nan,
            "BIC_HB_1broad": np.nan,
            "BIC_HB_2broad": np.nan,
            "BIC_HB_b2only": np.nan,
            "broad_choice_HB": "failed",
            "BIC_HD_narrow": np.nan,
            "BIC_HD_1broad": np.nan,
            "BIC_HD_2broad": np.nan,
            "BIC_HD_b2only": np.nan,
            "broad_choice_HD": "failed",
        }

    base_lines = base["lines"]


    base_sign = {}
    for ln in which_lines:
        d = base_lines.get(ln, {})
        F0 = d.get("F_line", np.nan)
        A0 = d.get("A_gauss", np.nan)

        if np.isfinite(F0) and F0 != 0.0:
            s = np.sign(F0)
        elif np.isfinite(A0) and A0 != 0.0:
            # fall back to the Gaussian amplitude sign
            s = np.sign(A0)
        else:
            s = np.nan

        base_sign[ln] = s



    flux_cap  = {ln: 10.0 * abs(base_lines[ln]["F_line"])  if ln in base_lines else np.inf
                 for ln in which_lines}
    width_cap = {ln: 10.0 * abs(base_lines[ln]["sigma_A"]) if ln in base_lines else np.inf
                 for ln in which_lines}

    if fit_window_um:
        if isinstance(fit_window_um, list):
             wfit = np.zeros_like(lam_um, dtype=bool)
             for (lo, hi) in fit_window_um:
                 wfit |= (lam_um >= lo) & (lam_um <= hi)
        else:
             lo, hi = fit_window_um
             wfit = (lam_um >= lo) & (lam_um <= hi)
    else:
        # Derive window from the base fit's lam_fit
        lam_fit_base = np.asarray(base.get("lam_fit", lam_um), float)

        # If the base fit used the full grid, keep the old behaviour
        if (lam_fit_base.size == lam_um.size) and np.allclose(lam_fit_base, lam_um, rtol=0, atol=0):
            wfit = slice(None)
        else:
            lo = float(np.min(lam_fit_base))
            hi = float(np.max(lam_fit_base))
            wfit = (lam_um >= lo) & (lam_um <= hi)


    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))

    samples = {
        ln: {
            "F_line": [], "sigma_A": [], "lam_obs_A": [],
            "EW0_A": [], "SNR": [], "SNR_peak_data": [], "SNR_peak_model": []
        }
        for ln in which_lines
    }
    model_stack_flam, keep_mask = [], []

    # -------- Bootstrap Loop (Robust) --------
    # We want exactly n_boot *valid* samples.
    # We will try up to max_attempts to get them.
    
    n_valid = 0
    n_attempts = 0
    max_attempts = n_boot * 30  # generous limit to prevent infinite loops
    
    if show_progress:
        pbar = tqdm(total=n_boot, desc="Bootstrap (valid draws)", unit="draw", dynamic_ncols=True)

    # Pre-compute pixel edges for model reconstruction
    lam_um_A = lam_um * 1e4
    leftA_boot, rightA_boot = _pixel_edges_A(lam_um_A)
    
    while n_valid < n_boot and n_attempts < max_attempts:
        n_attempts += 1
        
        flux_uJy_b = flux_uJy + rng.normal(0.0, err_uJy)

        try:
            fb = single_broad_fit(
                source=dict(lam=lam_um, flux=flux_uJy_b, err=err_uJy),
                z=z, grating=grating, deg=deg,
                continuum_windows=continuum_windows,
                lyman_cut=lyman_cut,
                fit_window_um=fit_window_um,
                plot=False, verbose=False,
                absorption_corrections=absorption_corrections,
                lines_to_use=lines_to_use,
                force_lines=which_lines,      # freeze line list
                broad_mode=broad_mode,
                continuum_lya_mask_A=continuum_lya_mask_A,
                continuum_fit=continuum_fit,
                continuum_movavg_window_um=continuum_movavg_window_um,
                continuum_exclude_regions=continuum_exclude_regions,
                continuum_lyman_window_um=continuum_lyman_window_um,
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
        
        # If rejected, just continue to next attempt (do not record nan)
        if not keep:
            continue

        # --- Check for crazy model fluxes ---
        cont = fb.get("continuum_flam", np.zeros_like(lam_um))
        
        # RECONSTRUCT the total model using the SIGN-CONSTRAINED fluxes.
        # This ensures the red "Mean model" line matches the blue dotted lines and table.
        reconstructed_model_flam = np.zeros_like(lam_um)
        
        for ln in which_lines:
            d = fb["lines"].get(ln, {})
            F_val = d.get("F_line", 0.0)
            muA   = d.get("lam_obs_A", np.nan)
            sA    = d.get("sigma_A", np.nan)
            
            # Apply sign constraint (same as stored samples)
            s0 = base_sign.get(ln, np.nan)
            if np.isfinite(s0) and s0 != 0 and np.isfinite(F_val):
                F_val = s0 * abs(F_val)
                
            if np.isfinite(F_val) and np.isfinite(muA) and np.isfinite(sA) and sA > 0:
                t = _gauss_binavg_area_normalized_A(leftA_boot, rightA_boot, muA, sA)
                reconstructed_model_flam += F_val * t

        total_flam = cont + reconstructed_model_flam

        if np.nanmax(np.abs(total_flam)) > 1e-11:
            # Reject crazy model
            continue

        # --- Valid Sample! Store it ---
        n_valid += 1
        if show_progress:
            pbar.update(1)

        model_stack_flam.append(total_flam)
        keep_mask.append(True) # All stored samples are valid now

        for ln in which_lines:
            d = fb["lines"][ln]

            # Start from the per-fit signed flux
            F_val = d["F_line"]
            s0 = base_sign.get(ln, np.nan)

            # If the base fit has a well-defined sign, enforce it on this draw
            if np.isfinite(s0) and s0 != 0 and np.isfinite(F_val):
                F_val = s0 * abs(F_val)

            samples[ln]["F_line"].append(F_val)
            samples[ln]["sigma_A"].append(d["sigma_A"])
            samples[ln]["lam_obs_A"].append(d["lam_obs_A"])
            samples[ln]["EW0_A"].append(d["EW0_A"])
            samples[ln]["SNR"].append(d["SNR"])
            samples[ln]["SNR_peak_data"].append(d["SNR_peak_data"])
            samples[ln]["SNR_peak_model"].append(d["SNR_peak_model"])

    if show_progress:
        pbar.close()
        
    if n_valid < n_boot:
        print(f"Warning: Only collected {n_valid}/{n_boot} valid bootstrap samples after {max_attempts} attempts.")

    # Convert lists to arrays
    model_stack_flam = np.asarray(model_stack_flam, float)
    keep_mask = np.ones(n_valid, dtype=bool) # All stored are valid
    for ln in which_lines:
        for k in samples[ln]:
            samples[ln][k] = np.asarray(samples[ln][k], float)

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
        vSNpkD, eSNpkD = _ve(samples[ln]["SNR_peak_data"])
        vSNpkM, eSNpkM = _ve(samples[ln]["SNR_peak_model"])
        summary[ln]["F_line"]    = {"value": vF,  "err": eF,  "text": f"{vF:.3e} ± {eF:.3e}"}
        summary[ln]["EW0_A"]     = {"value": vEW, "err": eEW, "text": f"{vEW:.2f} ± {eEW:.2f}"}
        summary[ln]["sigma_A"]   = {"value": vS,  "err": eS,  "text": f"{vS:.2f} ± {eS:.2f}"}
        summary[ln]["lam_obs_A"] = {"value": vMu, "err": eMu, "text": f"{vMu:.1f} ± {eMu:.1f}"}
        summary[ln]["SNR"]       = {"value": vSN, "err": eSN, "text": f"{vSN:.2f} ± {eSN:.2f}"}
        summary[ln]["SNR_peak_data"]  = {"value": vSNpkD, "err": eSNpkD, "text": f"{vSNpkD:.2f} ± {eSNpkD:.2f}"}
        summary[ln]["SNR_peak_model"] = {"value": vSNpkM, "err": eSNpkM, "text": f"{vSNpkM:.2f} ± {eSNpkM:.2f}"}

    # -------- DIAGNOSTIC: Check flux conservation in Hα region --------
    if verbose:
        # Define Hα region (NII_6549, H⍺, NII_6585, and broad components)
        ha_components = ["NII_6549", "H⍺", "NII_6585", "H⍺_BROAD", "H⍺_BROAD2"]
        ha_in_fit = [ln for ln in ha_components if ln in which_lines]
        
        if ha_in_fit:
            print("\n" + "="*70)
            print("FLUX DIAGNOSTIC: Hα Region")
            print("="*70)
            
            # Get Hα window
            rest_list = [REST_LINES_A[nm] for nm in ["NII_6549", "H⍺", "NII_6585"] if nm in REST_LINES_A]
            if rest_list:
                lo_ha, hi_ha = _window_from_lines_um(rest_list, z, pad_A=150.0)
                mask_ha = (lam_um >= lo_ha) & (lam_um <= hi_ha)
                
                if np.any(mask_ha):
                    # Integrate total model flux in Hα window
                    lam_ha_window = lam_um[mask_ha]
                    
                    # Get base fit model in this window
                    base_model_flam = base.get("model_window_flam", np.zeros_like(lam_um))
                    if isinstance(wfit, slice):
                        model_in_window = base_model_flam
                    else:
                        model_in_window = np.zeros_like(lam_um)
                        model_in_window[wfit] = base_model_flam
                    
                    model_ha = model_in_window[mask_ha]
                    dlam_A = np.gradient(lam_ha_window * 1e4)
                    
                    try:
                        total_model_flux = np.trapezoid(model_ha, lam_ha_window * 1e4)
                    except AttributeError:
                        total_model_flux = np.trapz(model_ha, lam_ha_window * 1e4)
                    
                    # Sum of individual component fluxes from base fit
                    sum_component_flux = 0.0
                    print(f"\nComponent fluxes in Hα region ({lo_ha:.4f} - {hi_ha:.4f} µm):")
                    for ln in ha_in_fit:
                        if ln in base_lines:
                            F_comp = base_lines[ln].get("F_line", 0.0)
                            sum_component_flux += F_comp
                            print(f"  {ln:15s}: {F_comp:.3e} erg/s/cm²")
                    
                    print(f"\n{'Total (sum of components)':30s}: {sum_component_flux:.3e} erg/s/cm²")
                    print(f"{'Total (integrated model)':30s}: {total_model_flux:.3e} erg/s/cm²")
                    
                    if abs(total_model_flux) > 1e-30:
                        ratio = sum_component_flux / total_model_flux
                        discrepancy = (ratio - 1.0) * 100
                        print(f"{'Ratio (components/model)':30s}: {ratio:.4f}")
                        print(f"{'Discrepancy':30s}: {discrepancy:+.2f}%")
                        
                        if abs(discrepancy) > 5:
                            print(f"\n⚠️  WARNING: Component flux sum differs from model by {abs(discrepancy):.1f}%!")
                            print("    This suggests flux measurement is double-counting overlapping components.")
                    
                    print("="*70 + "\n")
    
    
    # --- Extract continuum from base fit (available for return even if not plotting) ---
    cont_flam = np.asarray(base.get("continuum_flam", np.zeros_like(lam_um)))
    cont_uJy  = flam_to_fnu_uJy(cont_flam, lam_um)

    # -------- save results to .npz if requested --------
    # --- Calculate mean model stats (needed for saving OR plotting) ---
    mu_flam, sig_flam = _sigma_clip_mean_std(
        model_stack_flam[keep_mask], axis=0, sigma=3.0
    )

    # -------- save results to .npz if requested --------
    if save_npz:
        import os
        
        # Determine filename
        if npz_name:
             out_name = npz_name
             if not out_name.endswith(".npz"):
                 out_name += ".npz"
        elif save_path:
             base_name = os.path.splitext(os.path.basename(save_path))[0]
             out_name = base_name + ".npz"
             # If save_path has a directory, preserve it? 
             # save_path usually is just a prefix or full path. 
             # If save_path is "plots/target_fit", we want "plots/target_fit.npz"
             # But save_path usage in plotting often appends .pdf
             # Let's use the directory of save_path if it exists
             dirname = os.path.dirname(save_path)
             if dirname:
                 out_name = os.path.join(dirname, out_name)
        else:
             sid = source.get("id", "source") if isinstance(source, dict) else "source"
             out_name = f"broad_fit_{sid}.npz"
             
        if verbose:
             print(f"Saving fit results to {out_name}...")
             
        np.savez_compressed(
            out_name,
            lam_um=lam_um,
            flux_uJy=flux_uJy,
            err_uJy=err_uJy,
            z=z,
            grating=grating,
            which_lines=which_lines,
            cont_flam=cont_flam,
            # Use object array for dictionaries to allow pickling
            summary=np.array([summary], dtype=object),
            # Save mean model and std instead of full stack/samples
            mean_model_flam=mu_flam,
            std_model_flam=sig_flam,
        )

    # -------- plotting + optional saving --------
    if plot or save_path:
        # mu_flam, sig_flam are already computed above
        # cont_flam is now defined above

        # grid on which the base + all bootstrap fits are defined
        lam_fit_base = np.asarray(base.get("lam_fit", lam_um), float)
        lamA_base    = lam_fit_base * 1e4
        leftA_base, rightA_base = _pixel_edges_A(lamA_base)

        def _mean_profile_from_boot(line_name: str) -> np.ndarray:
            """
            Bootstrap-mean line *profile* (no continuum) in F_lambda
            on the base fit grid lam_fit_base.
            """
            if line_name not in samples:
                return np.zeros_like(lam_fit_base)

            F   = samples[line_name]["F_line"][keep_mask]
            muA = samples[line_name]["lam_obs_A"][keep_mask]
            sA  = samples[line_name]["sigma_A"][keep_mask]

            good = np.isfinite(F) & np.isfinite(muA) & np.isfinite(sA) & (sA > 0)
            if not np.any(good):
                return np.zeros_like(lam_fit_base)

            F, muA, sA = F[good], muA[good], sA[good]

            profs = []
            for Fi, mui, si in zip(F, muA, sA):
                # unit-area Gaussian in each pixel, then scale by flux
                t = _gauss_binavg_area_normalized_A(leftA_base, rightA_base, mui, si)
                profs.append(Fi * t)

            profs = np.asarray(profs, float)
            return np.nanmean(profs, axis=0)

        def _get_profile_stats_on_grid(line_name: str, lam_grid_um: np.ndarray):
            """
            Return stats for the line profile on an arbitrary wavelength grid (um).
            
            Returns:
                mean_prof: Mean of the bootstrap profiles at each pixel.
                           (Fixed: previously used profile of mean params, which was too sharp)
                std_prof:  Standard deviation of the bootstrap profiles at each pixel.
            """
            # Initialize
            mean_prof = np.zeros_like(lam_grid_um)
            std_prof  = np.zeros_like(lam_grid_um)
            
            if line_name not in samples:
                return mean_prof, std_prof

            # Get VALID samples
            F   = samples[line_name]["F_line"][keep_mask]
            muA = samples[line_name]["lam_obs_A"][keep_mask]
            sA  = samples[line_name]["sigma_A"][keep_mask]

            # Filter out any lingering NaNs
            good = np.isfinite(F) & np.isfinite(muA) & np.isfinite(sA) & (sA > 0)
            
            if not np.any(good):
                return mean_prof, std_prof

            F, muA, sA = F[good], muA[good], sA[good]

            # Prepare fine grid edges in Angstroms
            lam_grid_A = lam_grid_um * 1e4
            leftA, rightA = _pixel_edges_A(lam_grid_A)

            # Accumulate profiles
            profs = []
            for Fi, mui, si in zip(F, muA, sA):
                # Unit-area * Flux
                t = _gauss_binavg_area_normalized_A(leftA, rightA, mui, si)
                profs.append(Fi * t)
            
            profs = np.asarray(profs, float)
            
            # --- COMPUTE STATS ---
            # 1. Mean Profile: mean of the profiles (consistent with red total model line)
            mean_prof = np.nanmean(profs, axis=0)
            
            # 2. Std Profile: std of the profiles (uncertainty band)
            std_prof  = np.nanstd(profs, axis=0)

            return mean_prof, std_prof


        # also prepare µJy versions
        mu_uJy   = flam_to_fnu_uJy(mu_flam,  lam_um)
        sig_uJy  = flam_to_fnu_uJy(sig_flam, lam_um)
        cont_uJy = flam_to_fnu_uJy(cont_flam, lam_um)

        # shared display mask (Lyα, continuum windows, fit_window)
        disp_mask = np.ones_like(lam_um, dtype=bool)
        if (lyman_cut is not None) and (str(lyman_cut).lower() == "lya"):
            lya_edge = 0.1216 * (1.0 + z)
            disp_mask &= (lam_um >= lya_edge)

        if continuum_windows and isinstance(continuum_windows, (list, tuple)):
            cw_mask = np.zeros_like(lam_um, dtype=bool)
            for lo_cw, hi_cw in continuum_windows:
                cw_mask |= (lam_um >= lo_cw) & (lam_um <= hi_cw)
            disp_mask &= cw_mask

        # if fit_window_um:
        #    # User requested to see data outside fit window, so we do NOT mask the display.
        #    pass

        # -------------------------------
        # define line groups for zooms
        # -------------------------------
        o3hb_names   = ["HBETA", "OIII_4959", "OIII_5007"]
        aur_names    = ["HDELTA", "OIII_4363"]
        halpha_names = ["NII_6549", "H⍺", "NII_6585"]  # our naming in REST_LINES_A

        zoom_defs = [
            dict(title="Hδ + [O III]4363 (auroral)", names=aur_names),
            dict(title="Hβ + [O III]4959,5007",     names=o3hb_names),
            dict(title="Hα + [N II]6549,6585",      names=halpha_names),
        ]

        # Compute observed windows for each zoom region
        for zd in zoom_defs:
            restA = [REST_LINES_A[n] for n in zd["names"] if n in REST_LINES_A]
            if restA:
                zd["lo"], zd["hi"] = _window_from_lines_um(restA, z, pad_A=50.0)
            else:
                zd["lo"], zd["hi"] = np.nan, np.nan

        # Visibility flags (coverage + some fitted SNR)
        base_lines = base.get("lines", {})
        show_flags = []
        for zd in zoom_defs:
            cov   = _has_coverage_window(lam_um, zd["lo"], zd["hi"])
            fitok = _has_fit_in_window(base_lines, zd["names"], snr_min=1.0)
            show_flags.append(cov and fitok)

        # Any broad components in the chosen model? (includes HALPHA_BROAD2 etc.)
        has_broad_components_base = any(
            ("BROAD" in name) for name in base.get("which_lines", [])
        )

        # ----------------------------------------------------
        # helper: single plotting routine for a given unit
        # ----------------------------------------------------
        def _plot_unit(unit: str = "fnu", plot_continuum_subtracted: bool = False):
            unit = unit.lower()
            if unit == "flam":
                flux = fnu_uJy_to_flam(flux_uJy, lam_um)
                err  = fnu_uJy_to_flam(err_uJy,  lam_um)
                cont = cont_flam
                mu   = mu_flam
                ylabel = r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"
                tag = "flam"
            else:
                flux = flux_uJy
                err  = err_uJy
                cont = cont_uJy
                mu   = mu_uJy
                ylabel = r"$F_\nu$ [µJy]"
                tag = "fnu"

            # import matplotlib.gridspec as gridspec
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12.0, 8.2))
            gs = gridspec.GridSpec(
                2, 3, height_ratios=[1.6, 1.0],
                hspace=0.35, wspace=0.25
            )
            ax_full = fig.add_subplot(gs[0, :])
            ax_z    = [fig.add_subplot(gs[1, i]) for i in range(3)]

            # restrict to display mask
            lam_disp  = lam_um[disp_mask]
            flux_disp = flux[disp_mask]
            err_disp  = err[disp_mask]
            cont_disp = cont[disp_mask]
            mu_disp   = mu[disp_mask]

            # --- continuum on base-fit grid (always in Fλ here) ---
            cont_fit_base = np.interp(lam_fit_base, lam_um, cont_flam)

            # Colours for components
            narrow_color = "tab:blue"
            broad_color  = "tab:orange"
            broad_color2 = "tab:pink"

            # --- Apply continuum subtraction if requested ---
            if plot_continuum_subtracted: 
                flux = flux - cont
                # mu includes continuum so we subtract it
                mu = mu - cont
                # set continuum to zero for display
                cont = np.zeros_like(cont)
                ylabel += " (cont. sub.)"
                
                # Also subtract from base fit continuum for components
                # (components are added TO the continuum, so we just set base continuum to 0)
                cont_fit_base_for_plot = np.zeros_like(cont_fit_base)
            else:
                cont_fit_base_for_plot = cont_fit_base

            # --- build mean line profiles in Fλ on lam_fit_base ---
            # IMPORTANT: We must ensure that sum of component means = mean of total model.
            # To do this, we compute component profiles from the SAME bootstrap stack
            # that was used to compute mu_flam.
            
            sum_narrow_flam = np.zeros_like(lam_fit_base)
            sum_b1_flam     = np.zeros_like(lam_fit_base)
            sum_b2_flam     = np.zeros_like(lam_fit_base)
            has_b1 = False
            has_b2 = False

            for nm in which_lines:
                prof_flam_mean = _mean_profile_from_boot(nm)  # emission-only profile

                if "BROAD" in nm:
                    if "BROAD2" in nm:
                        sum_b2_flam += prof_flam_mean
                        has_b2 = True
                    else:
                        sum_b1_flam += prof_flam_mean
                        has_b1 = True
                else:
                    sum_narrow_flam += prof_flam_mean

            # --- convert continuum + grouped components to the chosen plotting unit ---
            # We use cont_fit_base_for_plot (which is 0 if subtracted) as the baseline for components
             
            if unit == "flam":
                cont_base_unit       = cont_fit_base_for_plot
                narrow_tot_unit      = cont_fit_base_for_plot + sum_narrow_flam
                narrow_plus_b1_unit  = narrow_tot_unit + sum_b1_flam
                full_unit            = narrow_plus_b1_unit + sum_b2_flam
            else:
                # If plotting Fnu but subtracting continuum, we must be careful.
                # The components (sum_narrow_flam etc) are in Flam.
                # If plot_continuum_subtracted is True, cont_fit_base_for_plot is 0 (in Flam).
                # To get Fnu of components, we just convert (0 + component_flam).
                
                cont_base_unit       = flam_to_fnu_uJy(cont_fit_base_for_plot, lam_fit_base)
                narrow_tot_unit      = flam_to_fnu_uJy(cont_fit_base_for_plot + sum_narrow_flam,
                                                      lam_fit_base)
                narrow_plus_b1_unit  = flam_to_fnu_uJy(cont_fit_base_for_plot + sum_narrow_flam + sum_b1_flam,
                                                      lam_fit_base)
                full_unit            = flam_to_fnu_uJy(cont_fit_base_for_plot + sum_narrow_flam +
                                                       sum_b1_flam + sum_b2_flam,
                                                      lam_fit_base)

            # --- FULL PANEL (data + stacked components) ---
            edges = _edges_median_spacing(lam_disp)

            # measurement error band
            ax_full.fill_between(
                lam_disp, flux_disp - err_disp, flux_disp + err_disp,
                step="mid", color="grey", alpha=0.25, linewidth=0,
            )

            ax_full.stairs(
                flux_disp, edges, color="black", lw=0.5, alpha=0.7,
                label="Data"
            )
            ax_full.stairs(
                cont_disp, edges, color="b", ls="--", lw=0.5,
                label="Continuum"
            )
            ax_full.stairs(
                mu_disp, edges, color="#ff0000a6", lw=0.5,
                label="Mean model"
            )

            # --- individual components on a fine grid (NOT stacked) ---
            # This shows peak heights more clearly than stacked fills
            oversample = 6
            lam_fine_full = np.linspace(
                lam_fit_base.min(), lam_fit_base.max(),
                max(lam_fit_base.size * oversample, 400)
            )
            cont_fine_full = np.interp(lam_fine_full, lam_fit_base, cont_base_unit)
            
            # Convert component sums to individual components edits...
            # OLD WAY: Interpolate from coarse grid (jagged)
            # narrow_only_fine = np.interp(lam_fine_full, lam_fit_base, sum_narrow_flam)
            # b1_only_fine = np.interp(lam_fine_full, lam_fit_base, sum_b1_flam)
            # b2_only_fine = np.interp(lam_fine_full, lam_fit_base, sum_b2_flam)

            # NEW WAY: Compute directly on fine grid (smooth, symmetric, with uncertainty)
            narrow_only_fine = np.zeros_like(lam_fine_full)
            narrow_std_fine  = np.zeros_like(lam_fine_full)
            
            b1_only_fine     = np.zeros_like(lam_fine_full)
            b1_std_fine      = np.zeros_like(lam_fine_full)
            
            b2_only_fine     = np.zeros_like(lam_fine_full)
            b2_std_fine      = np.zeros_like(lam_fine_full)

            for nm in which_lines:
                prof_mean, prof_std = _get_profile_stats_on_grid(nm, lam_fine_full)
                
                if "BROAD" in nm:
                    if "BROAD2" in nm:
                        b2_only_fine += prof_mean
                        b2_std_fine  = np.sqrt(b2_std_fine**2 + prof_std**2) # quadrature sum approx
                    else:
                        b1_only_fine += prof_mean
                        b1_std_fine  = np.sqrt(b1_std_fine**2 + prof_std**2)
                else:
                    narrow_only_fine += prof_mean
                    narrow_std_fine  = np.sqrt(narrow_std_fine**2 + prof_std**2)
            
            # Convert to plotting unit
            if unit == "flam":
                narrow_only_plot = narrow_only_fine
                narrow_std_plot  = narrow_std_fine
                b1_only_plot     = b1_only_fine
                b1_std_plot      = b1_std_fine
                b2_only_plot     = b2_only_fine
                b2_std_plot      = b2_std_fine
            else:
                # Note: STD conversion is approx linear for small errors
                narrow_only_plot = flam_to_fnu_uJy(narrow_only_fine, lam_fine_full)
                narrow_std_plot  = flam_to_fnu_uJy(narrow_std_fine, lam_fine_full)
                b1_only_plot     = flam_to_fnu_uJy(b1_only_fine, lam_fine_full)
                b1_std_plot      = flam_to_fnu_uJy(b1_std_fine, lam_fine_full)
                b2_only_plot     = flam_to_fnu_uJy(b2_only_fine, lam_fine_full)
                b2_std_plot      = flam_to_fnu_uJy(b2_std_fine, lam_fine_full)
                
            # Plot each component from continuum baseline
            # Using DOTTED lines for the mean-parameter profile
            # And SHADED band for uncertainty
            
            if np.nanmax(np.abs(narrow_only_plot)) > 0:
                # Uncertainty band
                ax_full.fill_between(
                    lam_fine_full,
                    cont_fine_full + narrow_only_plot - narrow_std_plot,
                    cont_fine_full + narrow_only_plot + narrow_std_plot,
                    color=narrow_color, alpha=0.5, linewidth=0,
                )
                # Main shape (dotted outline, light fill) edit
                
                ax_full.plot(
                    lam_fine_full, cont_fine_full + narrow_only_plot,
                    color=narrow_color, linestyle=":", linewidth=1,
                    label="Narrow components"
                )

            if has_b1 and np.nanmax(np.abs(b1_only_plot)) > 0:
                # Uncertainty band
                ax_full.fill_between(
                    lam_fine_full,
                    cont_fine_full + b1_only_plot - b1_std_plot,
                    cont_fine_full + b1_only_plot + b1_std_plot,
                    color=broad_color, alpha=0.5, linewidth=0,
                )
                # Main shape
    
                ax_full.plot(
                    lam_fine_full, cont_fine_full + b1_only_plot,
                    color=broad_color, linestyle=":", linewidth=1,
                    label="Broad component 1"
                )

            if has_b2 and np.nanmax(np.abs(b2_only_plot)) > 0:
                # Uncertainty band
                ax_full.fill_between(
                    lam_fine_full,
                    cont_fine_full + b2_only_plot - b2_std_plot,
                    cont_fine_full + b2_only_plot + b2_std_plot,
                    color=broad_color2, alpha=0.5, linewidth=0,
                )
                # Main shape
                
                ax_full.plot(
                    lam_fine_full, cont_fine_full + b2_only_plot,
                    color=broad_color2, linestyle=":", linewidth=1,
                    label="Broad component 2"
                )

            title_txt = f"{source_id}   (z = {z:.3f})" if source_id else f"z = {z:.3f}"
            ax_full.set_title(title_txt, fontsize=12, pad=8)
            ax_full.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
            ax_full.set_xlabel("Observed wavelength [µm]")
            ax_full.set_ylabel(ylabel)
            if plot_ylim is not None:
                ax_full.set_ylim(plot_ylim)
            ax_full.legend(ncol=3, fontsize=9, frameon=False)
            ax_full.grid(alpha=0.25, linestyle=":", linewidth=0.5)
            ax_full.tick_params(direction="in", top=True, right=True)

            _annotate_lines_above_model(
                ax=ax_full,
                lam=lam_disp,
                model=mu_disp,          # mean model in this unit
                line_names=which_lines, # all fitted lines
                z=z,
                per_line=base["lines"],
                label_offset_frac=0.06,
                x_margin_frac=0.01,
                fontsize=8,
            )

            # --- ZOOM PANELS ---
            for ax, zd, show in zip(ax_z, zoom_defs, show_flags):
                sel = (lam_um >= zd["lo"]) & (lam_um <= zd["hi"])
                if np.any(sel):
                    lam_z  = lam_um[sel]
                    flux_z = flux[sel]
                    err_z  = err[sel]
                    cont_z = cont[sel]
                    mu_z   = mu[sel]

                    edges_z = _edges_median_spacing(lam_z)

                    ax.fill_between(
                        lam_z, flux_z - err_z, flux_z + err_z,
                        step="mid", color="grey", alpha=0.25, linewidth=0
                    )
                    ax.stairs(
                        flux_z, edges_z, color="black",
                        lw=0.5, alpha=0.7, label="Data"
                    )
                    ax.stairs(
                        cont_z, edges_z, color="b",
                        ls="--", lw=0.5, label="Continuum"
                    )
                    ax.stairs(
                        mu_z, edges_z, color="#ff0000a6",
                        lw=0.5, label="Mean model"
                    )

                    # individual components (same logic, but restricted window)
                    mask_b = (lam_fit_base >= zd["lo"]) & (lam_fit_base <= zd["hi"])
                    if np.any(mask_b):
                        lam_sub = lam_fit_base[mask_b]
                        cont_sub_model = cont_base_unit[mask_b]
                        
                        # Get individual component profiles
                        narrow_sub = sum_narrow_flam[mask_b]
                        b1_sub = sum_b1_flam[mask_b]
                        b2_sub = sum_b2_flam[mask_b]

                        lam_fine = np.linspace(
                            lam_sub.min(), lam_sub.max(),
                            max(lam_sub.size * oversample, 200)
                        )
                        cont_fine = np.interp(lam_fine, lam_sub, cont_sub_model)
                        

                        narrow_fine = np.zeros_like(lam_fine)
                        narrow_std  = np.zeros_like(lam_fine)
                        
                        b1_fine     = np.zeros_like(lam_fine)
                        b1_std      = np.zeros_like(lam_fine)
                        
                        b2_fine     = np.zeros_like(lam_fine)
                        b2_std      = np.zeros_like(lam_fine)

                        for nm in which_lines: 
                             # Check if line is relevant for this window to avoid unnecessary computation
                             # (simple check: is the line center within window +/- margin?)
                             # If we don't check, it's fine, just slightly slower.
                             # But we must include BROAD lines which are not in zd["names"].
                             
                             prof_mean, prof_std = _get_profile_stats_on_grid(nm, lam_fine)
                             
                             # If the profile is all zeros (line far away), skip adding
                             if np.all(prof_mean == 0):
                                 continue

                             if "BROAD" in nm:
                                 if "BROAD2" in nm:
                                     b2_fine += prof_mean
                                     b2_std  = np.sqrt(b2_std**2 + prof_std**2)
                                 else:
                                     b1_fine += prof_mean
                                     b1_std  = np.sqrt(b1_std**2 + prof_std**2)
                             else:
                                 narrow_fine += prof_mean
                                 narrow_std  = np.sqrt(narrow_std**2 + prof_std**2)
                        
                        # Convert to plotting unit
                        if unit == "flam":
                            narrow_plot = narrow_fine
                            narrow_std_p = narrow_std
                            b1_plot = b1_fine
                            b1_std_p = b1_std
                            b2_plot = b2_fine
                            b2_std_p = b2_std
                        else:
                            narrow_plot = flam_to_fnu_uJy(narrow_fine, lam_fine)
                            narrow_std_p = flam_to_fnu_uJy(narrow_std, lam_fine)
                            b1_plot = flam_to_fnu_uJy(b1_fine, lam_fine)
                            b1_std_p = flam_to_fnu_uJy(b1_std, lam_fine)
                            b2_plot = flam_to_fnu_uJy(b2_fine, lam_fine)
                            b2_std_p = flam_to_fnu_uJy(b2_std, lam_fine)

                        # Plot each component from continuum baseline
                        if np.nanmax(np.abs(narrow_plot)) > 0:
                            ax.fill_between(
                                lam_fine, cont_fine + narrow_plot - narrow_std_p,
                                cont_fine + narrow_plot + narrow_std_p,
                                color=narrow_color, alpha=0.5, linewidth=0
                            )
                            # REMOVED FILL as per user request
                            ax.plot(
                                lam_fine, cont_fine + narrow_plot,
                                color=narrow_color, linestyle=":", linewidth=1
                            )

                        if has_b1 and np.nanmax(np.abs(b1_plot)) > 0:
                            ax.fill_between(
                                lam_fine, cont_fine + b1_plot - b1_std_p,
                                cont_fine + b1_plot + b1_std_p,
                                color=broad_color, alpha=0.5, linewidth=0
                            )
                            # REMOVED FILL as per user request
                            ax.plot(
                                lam_fine, cont_fine + b1_plot,
                                color=broad_color, linestyle=":", linewidth=1
                            )

                        if has_b2 and np.nanmax(np.abs(b2_plot)) > 0:
                            ax.fill_between(
                                lam_fine, cont_fine + b2_plot - b2_std_p,
                                cont_fine + b2_plot + b2_std_p,
                                color=broad_color2, alpha=0.5, linewidth=0
                            )
                            # REMOVED FILL as per user request
                            ax.plot(
                                lam_fine, cont_fine + b2_plot,
                                color=broad_color2, linestyle=":", linewidth=1
                            )

                    ax.set_xlim(zd["lo"], zd["hi"])
                    ax.set_title(zd["title"], fontsize=10)
                    ax.set_xlabel("Observed wavelength [µm]")
                    ax.set_ylabel(ylabel)
                    ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
                    ax.grid(alpha=0.25, linestyle=":", linewidth=0.5)
                    ax.tick_params(direction="in", top=True, right=True)

                    _annotate_lines_above_model(
                        ax=ax,
                        lam=lam_z,
                        model=mu_z,
                        line_names=zd["names"],
                        z=z,
                        per_line=base["lines"],
                        label_offset_frac=0.06,
                        x_margin_frac=0.02,
                        fontsize=8,
                    )
                else:
                    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                    ax.text(
                        0.5, 0.5, "No coverage",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=9, color="0.5",
                    )
                if not show:
                    ax.text(
                        0.5, 0.1, "No detection",
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=8, color="0.6", alpha=0.8,
                    )

            # --- save --- yes?
            if save_path:
                root, ext = os.path.splitext(save_path)
                fname = f"{root}_{tag}.{save_format}" if ext == "" else f"{root}_{tag}{ext}"
                fig.savefig(
                    fname, dpi=save_dpi,
                    bbox_inches="tight",
                    transparent=save_transparent,
                )

                if tag == "fnu" or (tag == "flam" and plot_unit.lower() == "flam"):
                    summary_txt = f"{root}_summary.txt"
                    print_bootstrap_line_table_broad(
                        dict(
                            which_lines=which_lines, 
                            summary=summary,
                            BIC_HA_narrow=base.get("BIC_HA_narrow", np.nan),
                            BIC_HA_1broad=base.get("BIC_HA_1broad", np.nan),
                            BIC_HA_2broad=base.get("BIC_HA_2broad", np.nan),
                            BIC_HA_b2only=base.get("BIC_HA_b2only", np.nan),
                            broad_choice_HA=base.get("broad_choice_HA", "unknown"),
                            BIC_HB_narrow=base.get("BIC_HB_narrow", np.nan),
                            BIC_HB_1broad=base.get("BIC_HB_1broad", np.nan),
                            BIC_HB_2broad=base.get("BIC_HB_2broad", np.nan),
                            BIC_HB_b2only=base.get("BIC_HB_b2only", np.nan),
                            broad_choice_HB=base.get("broad_choice_HB", "unknown"),
                            BIC_HD_narrow=base.get("BIC_HD_narrow", np.nan),
                            BIC_HD_1broad=base.get("BIC_HD_1broad", np.nan),
                            BIC_HD_2broad=base.get("BIC_HD_2broad", np.nan),
                            BIC_HD_b2only=base.get("BIC_HD_b2only", np.nan),
                            broad_choice_HD=base.get("broad_choice_HD", "unknown"),
                        ),
                        save_path=summary_txt,
                    )

            plt.tight_layout()
            if plot:
                plt.show()
            plt.close(fig)

        # --- make requested plot(s) ---
        if plot_unit.lower() in ("flam", "both"):
            _plot_unit("flam", plot_continuum_subtracted=plot_continuum_subtracted)
        if plot_unit.lower() in ("fnu", "both"):
            _plot_unit("fnu", plot_continuum_subtracted=plot_continuum_subtracted)



    return {
        "samples": samples,
        "summary": summary,
        "which_lines": which_lines,
        "model_stack_flam": model_stack_flam,
        "keep_mask": keep_mask,
        "lam_um": lam_um,
        "data_flux_uJy": flux_uJy,
        # BIC scores from base fit
        "BIC_HA_narrow": base.get("BIC_HA_narrow", np.nan),
        "BIC_HA_1broad": base.get("BIC_HA_1broad", np.nan),
        "BIC_HA_2broad": base.get("BIC_HA_2broad", np.nan),
        "BIC_HA_b2only": base.get("BIC_HA_b2only", np.nan),
        "broad_choice_HA": base.get("broad_choice_HA", "unknown"),
        "BIC_HB_narrow": base.get("BIC_HB_narrow", np.nan),
        "BIC_HB_1broad": base.get("BIC_HB_1broad", np.nan),
        "BIC_HB_2broad": base.get("BIC_HB_2broad", np.nan),
        "BIC_HB_b2only": base.get("BIC_HB_b2only", np.nan),
        "broad_choice_HB": base.get("broad_choice_HB", "unknown"),
        "BIC_HD_narrow": base.get("BIC_HD_narrow", np.nan),
        "BIC_HD_1broad": base.get("BIC_HD_1broad", np.nan),
        "BIC_HD_2broad": base.get("BIC_HD_2broad", np.nan),
        "BIC_HD_b2only": base.get("BIC_HD_b2only", np.nan),
        "broad_choice_HD": base.get("broad_choice_HD", "unknown"),
        
        # New return values: continuum and redshift
        "continuum_flam": cont_flam,
        "continuum_fnu": cont_uJy,
        "z": z,
    }
    
def print_bootstrap_line_table_broad(boot, save_path: str | None = None):
    """
    Print a formatted bootstrap summary to console, including peak SNRs and BIC scores.
    Optionally save the same output to a text file.
    """

    def _describe_broad_choice(choice: str, line_label: str) -> str:
        """
        Turn internal broad_choice flags into a clear human-readable description.
        BROAD  → medium broad
        BROAD2 → very broad
        """
        mapping = {
            "none":        f"{line_label}: narrow only",
            "one":         f"{line_label}: narrow + medium broad",
            "broad2_only": f"{line_label}: narrow + very broad",
            "two":         f"{line_label}: narrow + medium + very broad",
            "forced":      f"{line_label}: components forced (see which_lines)",
            "unknown":     f"{line_label}: model unknown",
        }
        return mapping.get(choice, f"{line_label}: model = {choice}")

    # BIC scores section
    bic_section = "\n=== MODEL SELECTION (BIC Scores) ===\n"
    
    # H-alpha BIC scores
    bic_ha_narrow = boot.get("BIC_HA_narrow", np.nan)
    bic_ha_1broad = boot.get("BIC_HA_1broad", np.nan)   # narrow + BROAD (medium)
    bic_ha_2broad = boot.get("BIC_HA_2broad", np.nan)   # narrow + BROAD + BROAD2
    bic_ha_b2only = boot.get("BIC_HA_b2only", np.nan)   # narrow + BROAD2 only
    choice_ha = boot.get("broad_choice_HA", "unknown")
    
    if np.isfinite(bic_ha_narrow):
        bic_section += f"\nH-alpha:\n"
        bic_section += f"  Narrow only                      : BIC = {bic_ha_narrow:.2f}\n"
        if np.isfinite(bic_ha_1broad):
            bic_section += f"  Narrow + medium broad            : BIC = {bic_ha_1broad:.2f}\n"
        if np.isfinite(bic_ha_2broad):
            bic_section += f"  Narrow + medium + very broad     : BIC = {bic_ha_2broad:.2f}\n"
        if np.isfinite(bic_ha_b2only):
            bic_section += f"  Narrow + very broad only         : BIC = {bic_ha_b2only:.2f}\n"
        bic_section += f"  → Selected: {_describe_broad_choice(choice_ha, 'Hα')}\n"
    
    # H-beta BIC scores
    bic_hb_narrow = boot.get("BIC_HB_narrow", np.nan)
    bic_hb_1broad = boot.get("BIC_HB_1broad", np.nan)
    bic_hb_2broad = boot.get("BIC_HB_2broad", np.nan)
    bic_hb_b2only = boot.get("BIC_HB_b2only", np.nan)
    choice_hb = boot.get("broad_choice_HB", "unknown")
    
    if np.isfinite(bic_hb_narrow):
        bic_section += f"\nH-beta:\n"
        bic_section += f"  Narrow only                      : BIC = {bic_hb_narrow:.2f}\n"
        if np.isfinite(bic_hb_1broad):
            bic_section += f"  Narrow + medium broad            : BIC = {bic_hb_1broad:.2f}\n"
        if np.isfinite(bic_hb_2broad):
            bic_section += f"  Narrow + medium + very broad     : BIC = {bic_hb_2broad:.2f}\n"
        if np.isfinite(bic_hb_b2only):
            bic_section += f"  Narrow + very broad only         : BIC = {bic_hb_b2only:.2f}\n"
        bic_section += f"  → Selected: {_describe_broad_choice(choice_hb, 'Hβ')}\n"
    
    # H-delta BIC scores (if available)
    bic_hd_narrow = boot.get("BIC_HD_narrow", np.nan)
    bic_hd_1broad = boot.get("BIC_HD_1broad", np.nan)
    bic_hd_2broad = boot.get("BIC_HD_2broad", np.nan)
    bic_hd_b2only = boot.get("BIC_HD_b2only", np.nan)
    choice_hd = boot.get("broad_choice_HD", "unknown")
    
    if np.isfinite(bic_hd_narrow):
        bic_section += f"\nH-delta:\n"
        bic_section += f"  Narrow only                      : BIC = {bic_hd_narrow:.2f}\n"
        if np.isfinite(bic_hd_1broad):
            bic_section += f"  Narrow + medium broad            : BIC = {bic_hd_1broad:.2f}\n"
        if np.isfinite(bic_hd_2broad):
            bic_section += f"  Narrow + medium + very broad     : BIC = {bic_hd_2broad:.2f}\n"
        if np.isfinite(bic_hd_b2only):
            bic_section += f"  Narrow + very broad only         : BIC = {bic_hd_b2only:.2f}\n"
        bic_section += f"  → Selected: {_describe_broad_choice(choice_hd, 'Hδ')}\n"
    
    bic_section += "\n" + "=" * 50 + "\n"
    
    header = (
        "\n=== BOOTSTRAP SUMMARY (value ± error) ===\n"
        f"{'Line':10s} {'F_line [erg/s/cm²]':>26s} "
        f"{'EW₀ [Å]':>16s} {'σ_A [Å]':>14s} {'μ_obs [Å]':>16s} "
        f"{'SNR_int':>10s} {'SNR_peak(data)':>16s} {'SNR_peak(model)':>18s}\n"
        + "-" * 125 + "\n"
    )

    lines = []
    for ln in boot["which_lines"]:
        s = boot["summary"][ln]
        snr_int   = s.get("SNR", {}).get("text", "—")
        snr_pdata = s.get("SNR_peak_data", {}).get("text", "—")
        snr_pmod  = s.get("SNR_peak_model", {}).get("text", "—")

        lines.append(
            f"{ln:10s} "
            f"{s['F_line']['text']:>26s} "
            f"{s['EW0_A']['text']:>16s} "
            f"{s['sigma_A']['text']:>14s} "
            f"{s['lam_obs_A']['text']:>16s} "
            f"{snr_int:>10s} "
            f"{snr_pdata:>16s} "
            f"{snr_pmod:>18s}"
        )

    table_text = bic_section + header + "\n".join(lines)
    print(table_text)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(table_text)
        print(f"\nSaved bootstrap summary → {save_path}")

#edit-more more
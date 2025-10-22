"""
PyRSR — line.py
================

Spectral line fitting, flux measurement, and equivalent width analysis for
JWST/NIRSpec and similar low–moderate resolution spectra.

Implements a self-contained EXCELS-style emission–line fitting workflow using
log-Gaussian line profiles, continuum estimation from percentile filtering,
and weighted flux integration.  

Main entry point:
-----------------
    excels_fit(source, z, grating="prism-clear", ...)

This function takes a 1D spectrum (FITS or dict), estimates the continuum,
fits emission lines via nonlinear least-squares, and returns line fluxes,
equivalent widths, and S/N estimates.

Author: Raunaq Rai (2025)
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares

# ============================================================
# -------------------- CONSTANTS & LINE LIST -----------------
# ============================================================

C_AA = 2.99792458e18  #: Speed of light [Å·Hz·s⁻¹]
LN10 = np.log(10.0)    #: Natural log of 10

#: Rest wavelengths of commonly fitted nebular and recombination lines [Å]
REST_LINES_A = {
    "HEII_1": 1640.420,  "OIII_05": 1663.4795, "CIII": 1908.734,
    "OII_UV_1": 3727.092, "OII_UV_2": 3729.875,
    "NEIII_UV_1": 3869.86, "NEIII_UV_2": 3968.59,
    "HDELTA": 4102.8922, "HGAMMA": 4341.6837, "OIII_1": 4364.436,
    "HEI_1": 4471.479, "HEII_2": 4685.710, "HBETA": 4862.6830,
    "OIII_2": 4960.295, "OIII_3": 5008.240, "HEI": 5877.252,
    "HALPHA": 6564.608, "SII_1": 6718.295, "SII_2": 6732.674,
}


# ============================================================
# ---------------------- UNIT CONVERSIONS --------------------
# ============================================================

def fnu_uJy_to_flam(flux_uJy: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    """
    Convert F_ν [µJy] at wavelength [µm] → F_λ [erg/s/cm²/Å].

    Parameters
    ----------
    flux_uJy : array_like
        Flux density per pixel in µJy.
    lam_um : array_like
        Wavelength array in µm.

    Returns
    -------
    Flam : ndarray
        Flux density per pixel in erg/s/cm²/Å.
    """
    lam_A = lam_um * 1e4
    Fnu_Jy = flux_uJy * 1e-6
    Flam = Fnu_Jy * (C_AA / lam_A**2) * 1e-23
    return Flam


def flam_to_fnu_uJy(flam: np.ndarray, lam_um: np.ndarray) -> np.ndarray:
    """
    Convert F_λ [erg/s/cm²/Å] → F_ν [µJy].

    Parameters
    ----------
    flam : array_like
        Flux density in erg/s/cm²/Å.
    lam_um : array_like
        Wavelength array in µm.

    Returns
    -------
    flux_uJy : ndarray
        Flux density in µJy.
    """
    lam_A = lam_um * 1e4
    Fnu_Jy = flam * (lam_A**2 / C_AA) * 1e23
    return Fnu_Jy * 1e6


# ============================================================
# ----------------- INSTRUMENTAL RESOLUTION ------------------
# ============================================================

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


# ============================================================
# -------------------- CONTINUUM ESTIMATION ------------------
# ============================================================

def continuum_running_percentile(lam_um, flam, z, rest_width_A=350.0):
    """
    Estimate continuum using a running 16–84th percentile window.

    For each wavelength, the continuum is defined as the mean of values within
    the 16th–84th percentile range inside a moving window of fixed rest-frame
    width.

    Parameters
    ----------
    lam_um : ndarray
        Observed wavelength [µm].
    flam : ndarray
        Spectrum in F_λ [erg/s/cm²/Å].
    z : float
        Redshift.
    rest_width_A : float, optional
        Width of the moving window in rest-frame Å (default: 350 Å).

    Returns
    -------
    Fcont : ndarray
        Continuum flux per pixel.
    STDc : ndarray
        Standard deviation within the window.
    SEc : ndarray
        Standard error of the mean.
    """
    lam_A = lam_um * 1e4
    obs_width_A = rest_width_A * (1.0 + z)
    half = 0.5 * obs_width_A

    Fcont = np.full_like(flam, np.nan)
    STDc  = np.full_like(flam, np.nan)
    SEc   = np.full_like(flam, np.nan)

    for i in range(len(lam_A)):
        mask = (lam_A > lam_A[i] - half) & (lam_A < lam_A[i] + half)
        vals = flam[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size < 5:
            continue
        F16, F84 = np.percentile(vals, [16, 84])
        sub = vals[(vals >= F16) & (vals <= F84)]
        if sub.size < 3:
            continue
        Fcont[i] = sub.mean()
        STDc[i]  = (F84 - F16) / 2.0
        SEc[i]   = STDc[i] / np.sqrt(sub.size)
    return Fcont, STDc, SEc


# ============================================================
# --------------------- UNCERTAINTY RESCALE ------------------
# ============================================================

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


# ============================================================
# --------------------- LINE PROFILE MODEL -------------------
# ============================================================

def line_centers_obs_A(z, rest_A):
    """Observed-frame line centres [Å] = λ_rest × (1 + z)."""
    return rest_A * (1.0 + z)


def sigma_tot_logA(sigma_intrinsic_logA, sigma_gr_logA):
    """Combine intrinsic and instrumental widths in quadrature (log space)."""
    return np.sqrt(sigma_intrinsic_logA**2 + sigma_gr_logA**2)


def gauss_log_profile(lam_A, mu_A, sigma_logA):
    """Return Gaussian profile in log₁₀λ space."""
    x = np.log10(lam_A)
    mu = np.log10(mu_A)
    return np.exp(-0.5 * ((x - mu) / sigma_logA)**2)


def build_model_flam(params, lam_um, z, grating, which_lines, tie_ratios=None, local_baseline=None):
    """
    Construct the multi-line emission model in F_λ space.

    Parameters
    ----------
    params : list
        Fit parameters = [σ_intrinsic_logA, A₀, A₁, ..., (b₀,b₁ if baseline)].
    lam_um : ndarray
        Observed wavelengths [µm].
    z : float
        Source redshift.
    grating : str
        NIRSpec grating name.
    which_lines : list
        Line identifiers to include in the fit.
    tie_ratios : dict, optional
        Amplitude ratio constraints, e.g. {'OIII_3':{'ref':'OIII_2','ratio':2.98}}.
    local_baseline : dict, optional
        Include a small local linear baseline {'use':bool,'lam0':float}.

    Returns
    -------
    model : ndarray
        Model F_λ across lam_um.
    profiles : dict
        Per-line individual profiles.
    centers : dict
        Per-line central λ and σ_logA.
    """
    lam_A = lam_um * 1e4
    sig_int = params[0]
    nL = len(which_lines)
    amps = np.array(params[1:1+nL], float)

    b0 = b1 = 0.0
    if local_baseline and local_baseline.get('use', False):
        b0, b1 = params[1+nL:3+nL]
        lam0 = local_baseline.get('lam0', np.nanmean(lam_um))
        baseline = b0 + b1*(lam_um - lam0)
    else:
        baseline = 0.0

    model = np.zeros_like(lam_um)
    profiles, centers = {}, {}
    name_to_index = {nm: i for i, nm in enumerate(which_lines)}

    for j, name in enumerate(which_lines):
        A = amps[j]
        if tie_ratios and name in tie_ratios:
            ref = tie_ratios[name]['ref']
            ratio = tie_ratios[name]['ratio']
            if ref in name_to_index:
                A = amps[name_to_index[ref]] * ratio

        muA = line_centers_obs_A(z, REST_LINES_A[name])
        mu_um = muA / 1e4
        sig_gr = float(sigma_grating_logA(grating, mu_um))
        sig_tot = sigma_tot_logA(sig_int, sig_gr)

        prof = gauss_log_profile(lam_A, muA, sig_tot)
        prof_flam = A * prof

        model += prof_flam
        profiles[name] = prof_flam
        centers[name] = (muA, sig_tot)

    return model + baseline, profiles, centers


# ============================================================
# ---------------------- FLUX MEASUREMENT --------------------
# ============================================================

def measure_fluxes_profile_weighted(lam_um, flam_sub, sigma_flam, model_flam, profiles, centers):
    """
    Compute line fluxes and uncertainties using profile-weighted integration.

    The method weights each pixel by the total model flux (handles blended lines)
    following the FastSpecFit philosophy.

    Parameters
    ----------
    lam_um : ndarray
        Observed wavelength grid [µm].
    flam_sub : ndarray
        Continuum-subtracted flux [erg/s/cm²/Å].
    sigma_flam : ndarray
        Flux uncertainty per pixel.
    model_flam : ndarray
        Total model profile.
    profiles : dict
        Individual line profiles from build_model_flam().
    centers : dict
        Central λ and σ_logA per line.

    Returns
    -------
    flux_dict : dict
        {line: {'F_line','sigma_line','mask_idx'}} for each fitted line.
    """
    lam_A = lam_um * 1e4
    dlam_A = np.gradient(lam_A)
    out = {}
    safe_sig2 = np.clip(sigma_flam, 1e-30, None)**2

    for name, prof in profiles.items():
        muA, sigma_logA = centers[name]
        sigma_ln = LN10 * sigma_logA
        sigma_A = muA * sigma_ln
        mask = (lam_A > muA - 3*sigma_A) & (lam_A < muA + 3*sigma_A)
        if np.count_nonzero(mask) < 3:
            continue

        G_i = prof[mask]
        M_i = model_flam[mask]
        Δλ_i = dlam_A[mask]
        σ_i2 = safe_sig2[mask]
        F_i  = flam_sub[mask]

        if np.sum(G_i) <= 0:
            continue

        P_i = M_i / np.sum(G_i)
        w_i = P_i / (Δλ_i * σ_i2)

        num = np.sum(w_i * Δλ_i * F_i)
        den = np.sum(w_i)
        F_line = num / den if den > 0 else np.nan
        sigma_line = np.sqrt(1.0 / np.sum(w_i)) if np.sum(w_i) > 0 else np.nan

        out[name] = dict(F_line=F_line, sigma_line=sigma_line, mask_idx=np.where(mask)[0])
    return out


# ============================================================
# -------------------- EQUIVALENT WIDTHS ---------------------
# ============================================================

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


# ============================================================
# ------------------- BALMER CORRECTIONS ---------------------
# ============================================================

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


# ============================================================
# -------------------- MAIN FITTING ROUTINE ------------------
# ============================================================

def excels_fit(source, z, grating="prism-clear", lines_to_use=None, tie_ratios=None,
               plot=True, rest_width_A=350.0, absorption_corrections=None,
               fit_window_um=None, use_local_baseline=False):
    """
    Perform emission line fitting on a single spectrum.

    Parameters
    ----------
    source : dict or FITS HDUList
        Spectrum container. Dict must have keys ['lam','flux','err'] in µm & µJy.
    z : float
        Source redshift.
    grating : str
        NIRSpec grating ('prism-clear', 'g235h', etc.).
    lines_to_use : list, optional
        Specific subset of REST_LINES_A keys to fit.
    tie_ratios : dict, optional
        Tied amplitude ratios for doublets/triplets.
    plot : bool
        Generate diagnostic plots.
    rest_width_A : float
        Window width (Å) for continuum percentile smoothing.
    absorption_corrections : dict, optional
        Balmer line multiplicative corrections.
    fit_window_um : tuple, optional
        Restrict fit to (λ_lo, λ_hi) µm.
    use_local_baseline : bool
        Add small local linear baseline.

    Returns
    -------
    results : dict
        {
          'lines': per-line flux/EW/SNR results,
          'continuum_flam': continuum model,
          'model_window_flam': best-fit line model,
          'sigma_flam': rescaled uncertainties,
          'params': {'sigma_intrinsic_logA', ...},
          ...
        }
    """
    # === Load data ===
    if isinstance(source, dict):
        lam_um = np.array(source.get("lam", source.get("wave")), float)
        flux_uJy = np.array(source["flux"], float)
        err_uJy  = np.array(source["err"], float)
    else:
        hdul = source if hasattr(source, "__iter__") else fits.open(source)
        d1 = hdul["SPEC1D"].data
        lam_um = np.array(d1["wave"], float)
        flux_uJy = np.array(d1["flux"], float)
        err_uJy  = np.array(d1["err"], float)

    good = np.isfinite(lam_um) & np.isfinite(flux_uJy) & np.isfinite(err_uJy) & (err_uJy > 0)
    lam_um, flux_uJy, err_uJy = lam_um[good], flux_uJy[good], err_uJy[good]

    # === Convert to F_lambda ===
    flam = fnu_uJy_to_flam(flux_uJy, lam_um)
    sig_flam_pipe = fnu_uJy_to_flam(err_uJy, lam_um)

    # === Continuum ===
    Fcont, _, _ = continuum_running_percentile(lam_um, flam, z, rest_width_A)
    if np.any(~np.isfinite(Fcont)):
        ix = np.isfinite(Fcont)
        Fcont = np.interp(lam_um, lam_um[ix], Fcont[ix])

    residual_full = flam - Fcont
    sig_flam_full = rescale_uncertainties(residual_full, sig_flam_pipe)

    # === Fit window ===
    if fit_window_um:
        lo, hi = fit_window_um
        win = (lam_um >= lo) & (lam_um <= hi)
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um[win], residual_full[win], sig_flam_full[win], Fcont[win]
    else:
        lam_fit, resid_fit, sig_fit, Fcont_fit = lam_um, residual_full, sig_flam_full, Fcont

    # === Select lines ===
    which_lines = list(lines_to_use or REST_LINES_A.keys())
    if fit_window_um:
        kept = []
        for nm in which_lines:
            muA0 = REST_LINES_A[nm]*(1.0 + z)
            mu_um0 = muA0 / 1e4
            if (lo-0.02) <= mu_um0 <= (hi+0.02):
                kept.append(nm)
        which_lines = kept
    if not which_lines:
        raise ValueError("No emission lines in selected window.")

    # === Initial parameter seeds ===
    rms = np.nanstd(resid_fit)
    p0 = [8e-4]  # intrinsic width
    A_seeds = []
    for nm in which_lines:
        mu_um0 = REST_LINES_A[nm]*(1+z)/1e4
        m = (lam_fit > mu_um0-0.03) & (lam_fit < mu_um0+0.03)
        A_guess = np.nanmax(resid_fit[m]) if np.count_nonzero(m) >= 3 else 0.0
        A_seeds.append(np.clip(A_guess, 0.2*rms, 20*rms))
    p0 += A_seeds

    # === Bounds ===
    sig_min, sig_max = 1e-5, 5e-3
    A_lo = np.full(len(which_lines), 0.0)
    A_hi = np.full(len(which_lines), max(50*rms, 1e-12))
    lb, ub = np.r_[sig_min, A_lo], np.r_[sig_max, A_hi]

    # === Least-squares fit ===
    def fun(p):
        model, _, _ = build_model_flam(p, lam_fit, z, grating, which_lines,
                                       tie_ratios=tie_ratios, local_baseline=None)
        return (resid_fit - model) / np.clip(sig_fit, 1e-30, None)

    res = least_squares(fun, p0, bounds=(lb, ub), max_nfev=200000, xtol=1e-12, ftol=1e-12)

    # === Model, fluxes, EWs ===
    model_flam_win, profiles_win, centers = build_model_flam(res.x, lam_fit, z, grating, which_lines)
    fluxes = measure_fluxes_profile_weighted(lam_fit, resid_fit, sig_fit, model_flam_win, profiles_win, centers)
    ews = equivalent_widths_A(fluxes, lam_fit, Fcont_fit, z, centers)
    if absorption_corrections:
        fluxes = apply_balmer_absorption_correction(fluxes, absorption_corrections)

    # === Pack results ===
    per_line = {}
    for name in fluxes:
        F_line, sigma_line = fluxes[name]["F_line"], fluxes[name]["sigma_line"]
        snr = np.nan if not np.isfinite(sigma_line) or sigma_line == 0 else F_line / sigma_line
        per_line[name] = dict(
            F_line=F_line, sigma_line=sigma_line, SNR=snr,
            EW_obs_A=ews[name]["EW_obs_A"], EW0_A=ews[name]["EW0_A"],
            lam_obs_A=centers[name][0]
        )

    # === Optional plots ===
    if plot:
        model_full = np.zeros_like(lam_um)
        if fit_window_um:
            model_full[win] = model_flam_win
        else:
            model_full = model_flam_win

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,7), sharex=True,
                                       gridspec_kw={'height_ratios':[1.0, 0.9]})
        ax1.plot(lam_um, flam_to_fnu_uJy(flam, lam_um), 'k-', lw=0.8, label='Data')
        ax1.plot(lam_um, flam_to_fnu_uJy(Fcont, lam_um), 'b--', lw=1.0, label='Continuum')
        ax1.plot(lam_um, flam_to_fnu_uJy(Fcont + model_full, lam_um), 'r-', lw=1.2, label='Model')
        ax1.set_ylabel('Flux density [µJy]')
        ax1.legend(ncol=3, fontsize=9)
        ax1.grid(alpha=0.2)

        ax2.plot(lam_um, flam_to_fnu_uJy(flam - Fcont, lam_um), '0.2', lw=0.8, label='Residual')
        ax2.plot(lam_um, flam_to_fnu_uJy(model_full, lam_um), 'r-', lw=1.2, label='Model')
        for name in per_line:
            mu_um = per_line[name]["lam_obs_A"]/1e4
            ax2.axvline(mu_um, color='gray', ls='--', lw=0.7, alpha=0.6)
            ax2.text(mu_um, ax2.get_ylim()[1]*0.85, name, rotation=90, va='top', ha='center', fontsize=8, color='gray')
        ax2.set_xlabel('Observed wavelength [µm]')
        ax2.set_ylabel('ΔFlux [µJy]')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    return dict(
        params=dict(sigma_intrinsic_logA=res.x[0]),
        continuum_flam=Fcont,
        sigma_flam=sig_flam_full,
        model_window_flam=model_flam_win,
        lines=per_line,
        which_lines=which_lines,
        lam_fit=lam_fit,
        Fcont_fit=Fcont_fit,
    )

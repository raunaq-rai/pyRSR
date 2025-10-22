"""
PyRSR — line_bootstrap.py
=========================

Bootstrap implementation of the EXCELS line fitting workflow.

This module calls :func:`PyRSR.line.excels_fit` multiple times on perturbed
versions of the observed spectrum to estimate robust uncertainties on
emission line fluxes and equivalent widths.

Each bootstrap realization adds Gaussian noise (σ = flux error) to the
spectrum, refits, and aggregates the resulting line parameters.

Example
-------
>>> from astropy.io import fits
>>> from PyRSR.line_bootstrap import excels_fit_bootstrap
>>> with fits.open("example.spec.fits") as hdul:
...     boot = excels_fit_bootstrap(hdul, z=8.22, n_boot=100, plot=True)
>>> for name, d in boot["line_summary"].items():
...     print(f"{name:10s}  ⟨F⟩={d['F_line_mean']:.3e} ± {d['F_line_std']:.3e}")

Author: Raunaq Rai (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

# Import base functions from line.py
from PyRSR.line import excels_fit, flam_to_fnu_uJy


def excels_fit_bootstrap(
    source,
    z,
    grating="prism-clear",
    lines_to_use=None,
    tie_ratios=None,
    n_boot=100,
    random_seed=42,
    rest_width_A=350.0,
    absorption_corrections=None,
    fit_window_um=None,
    use_local_baseline=False,
    plot=True,
):
    """
    Bootstrap version of :func:`PyRSR.line.excels_fit`.

    Parameters
    ----------
    source : dict or HDUList
        Spectrum containing wavelength, flux, and uncertainty arrays.
    z : float
        Redshift of the source.
    grating : str, default="prism-clear"
        JWST/NIRSpec grating configuration.
    lines_to_use : list, optional
        Subset of emission lines to fit.
    tie_ratios : dict, optional
        Dictionary of amplitude tie ratios between lines.
    n_boot : int, default=100
        Number of bootstrap realizations.
    random_seed : int, default=42
        Random seed for reproducibility.
    rest_width_A : float, default=350.0
        Rest-frame running percentile window for continuum estimation.
    absorption_corrections : dict, optional
        Multiplicative Balmer absorption corrections.
    fit_window_um : tuple, optional
        Wavelength window (µm) for fitting.
    use_local_baseline : bool, default=False
        Include a small local linear baseline in the model.
    plot : bool, default=True
        If True, plots the mean model ±1σ bootstrap envelope.

    Returns
    -------
    dict
        {
            "base_fit": dict,
            "bootstrap_results": list[dict],
            "line_summary": {
                line_name: {
                    "F_line_mean", "F_line_std",
                    "EW0_mean", "EW0_std",
                    "SNR_boot"
                }
            }
        }
    """

    rng = np.random.default_rng(random_seed)
    all_results = []

    # --- 1. Baseline fit
    base = excels_fit(
        source,
        z=z,
        grating=grating,
        lines_to_use=lines_to_use,
        tie_ratios=tie_ratios,
        plot=False,
        rest_width_A=rest_width_A,
        absorption_corrections=absorption_corrections,
        fit_window_um=fit_window_um,
        use_local_baseline=use_local_baseline,
    )

    if "lam_fit" not in base:
        raise KeyError("excels_fit() must return 'lam_fit' and 'Fcont_fit' for bootstrap plotting.")

    lam_fit = base["lam_fit"]
    Fcont_fit = base["Fcont_fit"]
    Fcont_full = base["continuum_flam"]
    which_lines = base["which_lines"]

    # --- 2. Load observed data
    if isinstance(source, dict):
        lam_data = np.array(source.get("lam", source.get("wave")), float)
        flux_data = np.array(source["flux"], float)
        err_data  = np.array(source["err"], float)
    else:
        from astropy.io import fits
        hdul = source if hasattr(source, "__iter__") else fits.open(source)
        d1 = hdul["SPEC1D"].data
        lam_data = np.array(d1["wave"], float)
        flux_data = np.array(d1["flux"], float)
        err_data  = np.array(d1["err"], float)

    good = np.isfinite(lam_data) & np.isfinite(flux_data) & np.isfinite(err_data) & (err_data > 0)
    lam_data, flux_data, err_data = lam_data[good], flux_data[good], err_data[good]

    # --- 3. Bootstrap resampling
    model_boot = []
    pbar = tqdm(range(n_boot), desc=f"Bootstrap fits (z={z:.3f})")

    for _ in pbar:
        flux_boot = flux_data + rng.normal(0, err_data)
        spec_dict = dict(lam=lam_data, flux=flux_boot, err=err_data)
        try:
            res = excels_fit(
                spec_dict,
                z=z,
                grating=grating,
                lines_to_use=which_lines,
                tie_ratios=tie_ratios,
                plot=False,
                rest_width_A=rest_width_A,
                absorption_corrections=absorption_corrections,
                fit_window_um=fit_window_um,
                use_local_baseline=use_local_baseline,
            )
            all_results.append(res)
            model_boot.append(res["model_window_flam"])
        except Exception as e:
            warnings.warn(f"Bootstrap iteration failed: {e}")
            continue

    if len(all_results) == 0:
        raise RuntimeError("All bootstrap iterations failed.")

    # --- 4. Combine models
    model_boot = np.array(model_boot)
    model_mean = np.nanmean(model_boot, axis=0)
    model_std  = np.nanstd(model_boot, axis=0)

    # --- 5. Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,7), sharex=True,
                                       gridspec_kw={'height_ratios':[1.0, 0.9]})
        ax1.plot(lam_data, flux_data, 'k-', lw=0.8, label='Data')
        ax1.plot(lam_data, flam_to_fnu_uJy(Fcont_full, lam_data),
                 'b--', lw=1.0, label='Continuum')
        ax1.plot(lam_fit, flam_to_fnu_uJy(Fcont_fit + model_mean, lam_fit),
                 'r-', lw=1.2, label='Mean model')
        ax1.fill_between(
            lam_fit,
            flam_to_fnu_uJy(Fcont_fit + model_mean - model_std, lam_fit),
            flam_to_fnu_uJy(Fcont_fit + model_mean + model_std, lam_fit),
            color='r', alpha=0.2, label='±1σ bootstrap'
        )
        ax1.set_ylabel("Flux density [µJy]")
        ax1.legend(ncol=3, fontsize=9)
        ax1.grid(alpha=0.2)

        resid = flux_data - flam_to_fnu_uJy(Fcont_full, lam_data)
        ax2.plot(lam_data, resid, '0.2', lw=0.8, label='Residuals')
        ax2.plot(lam_fit, flam_to_fnu_uJy(model_mean, lam_fit),
                 'r-', lw=1.2, label='Mean model')
        ax2.fill_between(
            lam_fit,
            flam_to_fnu_uJy(model_mean - model_std, lam_fit),
            flam_to_fnu_uJy(model_mean + model_std, lam_fit),
            color='r', alpha=0.2
        )

        for name in base["lines"].keys():
            try:
                mu_um = base["lines"][name]["lam_obs_A"] / 1e4
                ax2.axvline(mu_um, color='gray', ls='--', lw=0.7, alpha=0.6)
                ax2.text(mu_um, ax2.get_ylim()[1]*0.85, name,
                         rotation=90, va='top', ha='center', fontsize=8, color='gray')
            except KeyError:
                continue

        ax2.set_xlabel("Observed wavelength [µm]")
        ax2.set_ylabel("ΔFlux [µJy]")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    # --- 6. Per-line statistics
    line_summary = {}
    for name in which_lines:
        flux_vals = [r["lines"][name]["F_line"] for r in all_results if name in r["lines"]]
        ew_vals   = [r["lines"][name]["EW0_A"] for r in all_results if name in r["lines"]]
        if len(flux_vals) == 0:
            continue
        flux_vals = np.array(flux_vals)
        ew_vals   = np.array(ew_vals)
        F_mean, F_std = np.nanmean(flux_vals), np.nanstd(flux_vals)
        EW_mean, EW_std = np.nanmean(ew_vals), np.nanstd(ew_vals)
        line_summary[name] = dict(
            F_line_mean=F_mean,
            F_line_std=F_std,
            EW0_mean=EW_mean,
            EW0_std=EW_std,
            SNR_boot=F_mean / F_std if F_std > 0 else np.nan,
        )

    return dict(
        base_fit=base,
        bootstrap_results=all_results,
        line_summary=line_summary,
    )

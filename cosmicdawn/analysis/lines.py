"""
lines.py

Functions for visualizing emission-line fitting results.

Includes:
- plot_line_fitting_results: plot full stacked spectrum with fitted model and zoom-ins on emission-line regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from astropy.io import ascii, fits
from uncertainties import unumpy as unp
from cosmicdawn.general import utils
from cosmicdawn.general import conversions as conv


def plot_line_fitting_results(root, input_dir, output_dir, norm_range=(1500, 2000)):
    """
    Plot emission-line fitting results for a stacked spectrum.

    Parameters
    ----------
    root : str
        Base name of the stacked spectrum (e.g., "z6_stack").
    input_dir : str
        Path to directory containing input stacked spectra.
    output_dir : str
        Path to directory containing fitted models and emission line files.
    norm_range : tuple, default=(1500, 2000)
        Rest-frame wavelength range (Å) for continuum normalization.

    Returns
    -------
    None
        Displays a multi-panel matplotlib figure with:
        - Full spectrum (observed frame, normalized)
        - Zoomed panels on specific emission-line regions
    """
    # Define emission lines of interest
    lines = {
        "HEII_1": 1640.420,
        "OIII_05": 1663.480,
        "CIII": 1908.734,
        "OII_1": 3727.092,
        "OII_2": 3729.875,
        "NEIII_1": 3869.860,
        "NEIII_2": 3968.590,
        "HDELTA": 4102.892,
        "HGAMMA": 4341.684,
        "OIII_1": 4364.436,
        "HEI_1": 4471.479,
        "HEII_2": 4685.710,
        "HBETA": 4862.683,
        "OIII_2": 4960.295,
        "OIII_3": 5008.240,
        "HEI": 5877.252,
        "HALPHA": 6564.608,
        "SII_1": 6718.295,
        "SII_2": 6732.674,
    }

    def gauss(x, x0, A, sig, off=0):
        """Return Gaussian profile with uncertainties."""
        return A * unp.exp(-0.5 * ((x - x0 + off) ** 2) / (2 * sig**2))

    # -----------------
    # Full spectrum
    # -----------------
    infile = f"{input_dir}/{root}_full.txt"
    spec = ascii.read(infile)
    lam, flux, err = spec["lambda(AA)"], spec["flux_nu"], spec["eflux_nu_lines"]

    # crude z_spec from filename
    try:
        z_spec = float(root.split("z")[-1].split("_")[0].replace("p", "."))
    except Exception:
        raise ValueError("Could not infer redshift from root name.")

    lam_rest = lam / (1 + z_spec)
    lam_obs = lam / 10000.0

    # Normalize
    norm = np.median(flux[(lam_rest > norm_range[0]) & (lam_rest < norm_range[1])])
    flux /= norm
    err /= norm

    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(ncols=5, nrows=2)

    ax_full = fig.add_subplot(gs[0, :])
    ax_full.step(lam_obs, flux, lw=2.5, color="k", zorder=2)
    ax_full.fill_between(lam_obs, flux - err, flux + err, alpha=0.2, color="k")
    ax_full.axhline(0, ls="--", color="black", lw=1)
    ax_full.set_xlim(0.6, 5.3)
    ymax = np.nanmax(flux)
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ax_full.set_ylim(-0.5, ymax * 1.1)

    utils.style_axes(ax_full, "Observed Wavelength [μm]", "Flux Density [norm]")

    # Overlay continuum model
    model_file = f"{output_dir}/gsf_spec_{root}_full.fits"
    with fits.open(model_file) as fmod:
        lam_model = fmod[1].data["wave_model"] / 10000.0
        flux_model = fmod[1].data["f_model_noline_50"] * float(fmod[1].header["SCALE"])
        flux_model = conv.flam_to_fnu(lam_model * 1e4, flux_model) * 1e32
    flux_model /= norm
    ax_full.plot(lam_model, flux_model, color="darkred", lw=3)

    # -----------------
    # Zoomed emission-line regions
    # -----------------
    fit_file = f"{output_dir}/emission_fits/{root}_emission_lines.fits"
    fem = fits.open(fit_file)

    regions = ["CIII", "Balmer", "OIIIHbeta", "HeI", "Halpha"]
    for i, region in enumerate(regions):
        ax = fig.add_subplot(gs[1, i])
        infile = f"{input_dir}/{root}_{region}.txt"
        try:
            reg_spec = ascii.read(infile)
        except FileNotFoundError:
            continue
        lam_r, flux_r, err_r = (
            reg_spec["lambda(AA)"] / 10000.0,
            reg_spec["flux_nu"] / norm,
            reg_spec["eflux_nu_lines"] / norm,
        )
        ax.step(lam_r, flux_r, color="k", lw=2)
        ax.fill_between(lam_r, flux_r - err_r, flux_r + err_r, alpha=0.2)
        ax.axhline(0, ls="--", color="black", lw=1)

        # Continuum model for region
        model_file = f"{output_dir}/gsf_spec_{root}_{region}.fits"
        with fits.open(model_file) as fmod:
            lam_model = fmod[1].data["wave_model"] / 10000.0
            flux_model = fmod[1].data["f_model_noline_50"] * float(fmod[1].header["SCALE"])
            flux_model = conv.flam_to_fnu(lam_model * 1e4, flux_model) * 1e32
        ax.plot(lam_model, flux_model / norm, color="darkred", lw=2.5)

        # Add emission-line profiles if present
        lines_present = [k.split("FLUX_ERR_")[-1] for k in fem[1].header if "FLUX_ERR" in k]
        profile = np.copy(flux_model) / norm
        for line in lines_present:
            if line not in lines:
                continue
            lam_line = lines[line] * (1 + z_spec)
            A = fem[1].header.get(f"A_{line}", 0)
            sig = fem[1].header.get(f"SIGMA_{line}", 1)
            off = fem[1].header.get(f"OFFSET_{line}", 0)

            # Gaussian with uncertainties
            prof = gauss(lam_r * 1e4, lam_line, A, sig, off) * 1e-21
            prof = conv.flam_to_fnu(lam_r * 1e4, prof) * 1e32

            # Extract nominal values and errors
            prof_vals = unp.nominal_values(prof)
            prof_errs = unp.std_devs(prof)

            profile += prof_vals / norm
            ax.fill_between(lam_r, (profile - prof_errs) / norm,
                            (profile + prof_errs) / norm, color="orange", alpha=0.2)
            ax.axvline(lam_line / 1e4, ls="--", color="silver", lw=1.5)

        ax.plot(lam_r, profile, color="darkorange", lw=2.5)
        utils.style_axes(ax, "Obs. λ [μm]", "Flux [norm]", fontsize=18, labelsize=16)

    plt.tight_layout()
    plt.show()

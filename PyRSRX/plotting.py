import numpy as np
import matplotlib.pyplot as plt

# === REST wavelengths (Å) ===
REST_LINES_A = {
    "NIII_1": 989.790, "NIII_2": 991.514, "NIII_3": 991.579,
    "NV_1": 1238.821, "NV_2": 1242.804, "NIV": 1486.496,
    "HEII_1": 1640.420, "OIII_05": 1663.4795, "CIII": 1908.734,
    "OII_UV_1": 3727.092, "OII_UV_2": 3729.875,
    "NEIII_UV_1": 3869.86, "HEI_1":3889.749, "NEIII_UV_2": 3968.59, "HEPSILON":3971.1951,
    "HDELTA": 4102.8922, "HGAMMA": 4341.6837, "OIII_1": 4364.436,
    "HEI_2": 4471.479, "HEII_2": 4685.710, "HBETA": 4862.6830,
    "OIII_2": 4960.295, "OIII_3": 5008.240,
    "NII_1": 5756.19, "HEI_3": 5877.252, "NII_2": 6549.86,
    "HALPHA": 6564.608, "NII_3": 6585.27, "SII_1": 6718.295, "SII_2": 6732.674,
}

def _bin_edges_from_centers_um(lam_um):
    """Compute wavelength bin edges (µm) from central wavelengths."""
    lam_um = np.asarray(lam_um, float)
    if lam_um.size < 2:
        raise ValueError("Need ≥2 wavelength points to compute edges.")
    dlam = np.diff(lam_um)
    edges = np.concatenate((
        [lam_um[0] - dlam[0]/2],
        0.5 * (lam_um[1:] + lam_um[:-1]),
        [lam_um[-1] + dlam[-1]/2]
    ))
    return edges

from adjustText import adjust_text

def annotate_lines_no_overlap(ax, z, lam_min, lam_max, lam_data=None, flux_data=None, fontsize=7):
    """
    Annotate emission lines without overlaps, keeping labels inside the axes.
    Skips lines that fall in wavelength regions without valid data.
    """
    ax.autoscale(False)

    # --- Build observed wavelengths for all lines ---
    rest_waves_A = np.array(list(REST_LINES_A.values()))
    line_names   = np.array(list(REST_LINES_A.keys()))
    obs_um       = rest_waves_A * (1 + z) / 1e4

    # --- Only keep lines inside visible range ---
    m = (obs_um > lam_min) & (obs_um < lam_max)
    obs_um, names_in = obs_um[m], line_names[m]
    if len(obs_um) == 0:
        return

    # --- If lam_data provided, filter lines with no actual data coverage ---
    if lam_data is not None:
        lam_data = np.asarray(lam_data, float)
        valid_mask = np.isfinite(lam_data)
        if flux_data is not None:
            valid_mask &= np.isfinite(flux_data)
        lam_valid = lam_data[valid_mask]

        # define min/max of valid wavelength region(s)
        if len(lam_valid) > 0:
            lam_min_valid, lam_max_valid = np.nanmin(lam_valid), np.nanmax(lam_valid)
            in_coverage = (obs_um >= lam_min_valid) & (obs_um <= lam_max_valid)
            obs_um, names_in = obs_um[in_coverage], names_in[in_coverage]

    if len(obs_um) == 0:
        return

    # --- Sort lines by wavelength ---
    order = np.argsort(obs_um)
    obs_um, names_in = obs_um[order], names_in[order]

    # --- Label positioning ---
    ymin, ymax = ax.get_ylim()
    y_base = ymin + 0.90 * (ymax - ymin)

    # --- Draw vertical lines + text ---
    texts = []
    for x, label in zip(obs_um, names_in):
        ax.axvline(x, color="0.7", lw=0.4, alpha=0.5, zorder=1)
        texts.append(
            ax.text(
                x, y_base, label,
                rotation=90, ha="center", va="bottom",
                fontsize=fontsize, color="black", alpha=1,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.2),
                clip_on=False, zorder=10,
            )
        )

    # --- Tidy layout ---
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    adjust_text(
        texts,
        only_move={'points': 'none', 'text': 'xy'},
        expand_text=(1.1, 1.25),
        force_text=(0.05, 0.2),
        lim=(x0, x1, y0, y1),
        arrowprops=dict(arrowstyle='-', lw=0.4, color='0.4', alpha=0.7),
        autoalign='y',
        ax=ax,
    )



def plot_spectrum_basic(
    lam_um,
    flux_uJy,
    err_uJy=None,
    model_uJy=None,
    cont_uJy=None,
    title=None,
    ax=None,
    xlim=None,
    ylim=None,
    color="k",
    alpha=0.9,
    save_path=None,
    show=True,
    z=None,
    annotate_lines=True,
    fontsize=7,
):
    """
    Plot JWST/NIRSpec-style 1D spectrum with emission-line annotations.

    Parameters
    ----------
    lam_um, flux_uJy, err_uJy, model_uJy, cont_uJy : array-like
        Spectral data arrays.
    z : float, optional
        Redshift of the source (for line annotation).
    annotate_lines : bool, optional
        Whether to overlay line positions.
    """

    lam_um = np.asarray(lam_um, float)
    flux_uJy = np.asarray(flux_uJy, float)
    err_uJy = np.asarray(err_uJy, float) if err_uJy is not None else None

    edges_um = _bin_edges_from_centers_um(lam_um)
    centers_um = 0.5 * (edges_um[:-1] + edges_um[1:])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # --- shaded ±1σ envelope ---
    if err_uJy is not None and np.any(np.isfinite(err_uJy)):
        ax.fill_between(
            centers_um, flux_uJy - err_uJy, flux_uJy + err_uJy,
            step="mid", color="grey", alpha=0.25, linewidth=0, zorder=2,
            label="±1σ"
        )

    # --- stepped flux ---
    ax.stairs(flux_uJy, edges_um, color=color, lw=0.6, alpha=alpha,
              label="Data (bins)", zorder=3)

    # --- optional overlays ---
    if cont_uJy is not None:
        ax.stairs(cont_uJy, edges_um, color="b", linestyle="--", lw=0.7,
                  label="Continuum", zorder=2)
    if model_uJy is not None:
        ax.stairs(model_uJy, edges_um, color="r", lw=0.7,
                  label="Model", zorder=4)

    # --- cosmetics ---
    ax.set_xlabel("Observed wavelength [µm]")
    ax.set_ylabel("Flux density [µJy]")
    if title:
        ax.set_title(title, fontsize=11)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()

    # -----------------------------------------------------------------
    # === Line annotation section ===
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # === Line annotation section (non-overlapping labels) ===
    # -----------------------------------------------------------------
    if annotate_lines and z is not None:
        annotate_lines_no_overlap(ax, z, lam_um.min(), lam_um.max(), fontsize=fontsize)


    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=False)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

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


def plot_spectrum_with_2d(
    lam_um,
    flux_uJy,
    err_uJy=None,
    sci2d=None,
    model_uJy=None,
    cont_uJy=None,
    title=None,
    xlim=None,
    ylim=(-1, None),
    color="#6a0dad",
    alpha=0.9,
    save_path=None,
    show=True,
    z=None,
    annotate_lines=True,
    fontsize=7,
    cmap="plasma",
    vmin=None,
    vmax=None,
    # --- new options ---
    flux_space="fnu",   # "fnu" (µJy) or "flam" (erg/s/cm²/Å)
    frame="obs",        # "obs" or "rest" (requires z)
):
    lam_um   = np.asarray(lam_um, float)
    flux_uJy = np.asarray(flux_uJy, float)
    err_uJy  = np.asarray(err_uJy, float)  if err_uJy  is not None else None
    model_uJy = np.asarray(model_uJy,float) if model_uJy is not None else None
    cont_uJy  = np.asarray(cont_uJy,float)  if cont_uJy  is not None else None

    if frame not in {"obs", "rest"}:
        raise ValueError("frame must be 'obs' or 'rest'.")

    # ----------------------------------------------------
    # Wavelength frame (x-axis)
    # ----------------------------------------------------
    if frame == "rest":
        if z is None:
            raise ValueError("frame='rest' requested but z is None.")
        lam_plot = lam_um / (1.0 + z)
    else:
        lam_plot = lam_um.copy()

    # --- bin edges for 1D stairs (in chosen frame) ---
    dlam = np.median(np.diff(lam_plot))
    edges_plot = np.concatenate((
        [lam_plot[0] - dlam / 2],
        0.5 * (lam_plot[1:] + lam_plot[:-1]),
        [lam_plot[-1] + dlam / 2]
    ))

    # ----------------------------------------------------
    # Flux space: Fν(µJy) vs Fλ(cgs)
    # (inputs are always Fν in µJy)
    # ----------------------------------------------------
    flux_space = flux_space.lower()
    if flux_space == "fnu":
        flux_plot  = flux_uJy
        err_plot   = err_uJy
        model_plot = model_uJy
        cont_plot  = cont_uJy
        y_label = "Flux density [µJy]"
    elif flux_space == "flam":
        flux_plot  = fnu_uJy_to_flam(flux_uJy,  lam_um)
        err_plot   = fnu_uJy_to_flam(err_uJy,   lam_um) if err_uJy  is not None else None
        model_plot = fnu_uJy_to_flam(model_uJy, lam_um) if model_uJy is not None else None
        cont_plot  = fnu_uJy_to_flam(cont_uJy,  lam_um) if cont_uJy  is not None else None
        y_label = r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"
    else:
        raise ValueError("flux_space must be 'fnu' or 'flam'.")

    # ----------------------------------------------------
    # Create figure layout
    # ----------------------------------------------------
    if sci2d is not None:
        fig, (ax2d, ax1d) = plt.subplots(
            2, 1, figsize=(9, 5.5),
            gridspec_kw={"height_ratios": [0.35, 1.0], "hspace": 0.03},
            sharex=True,
        )
    else:
        fig, ax1d = plt.subplots(figsize=(8, 4))
        ax2d = None

    # ----------------------------------------------------
    # --- Top panel: 2D spectrum (auto-mask empty regions) ---
    # ----------------------------------------------------
    if sci2d is not None:
        ny, nx = sci2d.shape

        # wavelength grid for 2D (start in observed frame)
        lam2d = np.linspace(lam_um.min(), lam_um.max(), nx)
        lam2d_plot = lam2d / (1.0 + z) if (frame == "rest" and z is not None) else lam2d

        # detect wavelength columns with no data
        col_is_empty = np.all(~np.isfinite(sci2d) | (np.abs(sci2d) < 1e-8), axis=0)
        col_is_empty |= (np.nanstd(sci2d, axis=0) < 1e-8)

        # crop 2D array and wavelength grid to valid region only
        valid_cols = np.where(~col_is_empty)[0]
        if len(valid_cols) > 1:
            i0, i1 = valid_cols[0], valid_cols[-1] + 1
            sci2d_valid = sci2d[:, i0:i1]
            lam2d_valid = lam2d_plot[i0:i1]
            lam_valid_min, lam_valid_max = lam2d_valid[0], lam2d_valid[-1]
        else:
            sci2d_valid = sci2d
            lam2d_valid = lam2d_plot
            lam_valid_min, lam_valid_max = lam2d_valid.min(), lam2d_valid.max()

        # build edges for valid wavelength region
        dlam2d = np.median(np.diff(lam2d_valid))
        edges_2d = np.concatenate((
            [lam2d_valid[0] - dlam2d / 2],
            0.5 * (lam2d_valid[1:] + lam2d_valid[:-1]),
            [lam2d_valid[-1] + dlam2d / 2]
        ))

        # colour limits ignoring NaNs
        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(sci2d_valid, [5, 99.5])

        y_edges = np.arange(ny + 1)

        # plot cleaned 2D
        ax2d.pcolormesh(
            edges_2d, y_edges, sci2d_valid,
            shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )

        ax2d.set_ylabel("Spatial pixel", fontsize=9)
        ax2d.tick_params(labelbottom=False, direction="in")
        ax2d.set_ylim(ny * 0.25, ny * 0.75)

        # sync x-range to actual coverage
        ax2d.set_xlim(lam_valid_min, lam_valid_max)
        ax1d_xlim = (lam_valid_min, lam_valid_max)
    else:
        ax1d_xlim = (lam_plot.min(), lam_plot.max())

    # ----------------------------------------------------
    # --- Bottom panel: 1D spectrum ---
    # ----------------------------------------------------
    if err_plot is not None:
        ax1d.fill_between(
            lam_plot, flux_plot - err_plot, flux_plot + err_plot,
            step="mid", color="grey", alpha=0.25, linewidth=0
        )

    ax1d.stairs(flux_plot, edges_plot, color=color, lw=0.5, alpha=alpha, label="Data")

    if cont_plot is not None:
        ax1d.stairs(cont_plot, edges_plot, color="b", ls="--", lw=0.5, label="Continuum")
    if model_plot is not None:
        ax1d.stairs(model_plot, edges_plot, color="r", lw=0.5, label="Model")

    ax1d.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)

    if frame == "rest":
        ax1d.set_xlabel("Rest wavelength [µm]")
    else:
        ax1d.set_xlabel("Observed wavelength [µm]")
    ax1d.set_ylabel(y_label)

    # consistent x-limits
    if xlim is None:
        ax1d.set_xlim(*ax1d_xlim)
    else:
        ax1d.set_xlim(*xlim)

    # Y-limits:
    # - default (-1, None) is good for µJy
    # - but *terrible* for Fλ, so ignore that default and autoscale for Fλ
    ylim_use = None
    if ylim is not None:
        if isinstance(ylim, (tuple, list)):
            if not (flux_space == "flam" and tuple(ylim) == (-1, None)):
                ylim_use = ylim
        else:
            ylim_use = ylim

    if ylim_use is not None:
        ax1d.set_ylim(*ylim_use)

    ax1d.legend(fontsize=8, frameon=False, ncol=3)
    ax1d.grid(alpha=0.25, linestyle=":", linewidth=0.5)
    ax1d.tick_params(direction="in", top=True, right=True)

    # ----------------------------------------------------
    # --- Annotate emission lines ---
    # ----------------------------------------------------
    if annotate_lines and z is not None:
        if frame == "obs":
            # original behaviour: annotate at observed wavelengths
            annotate_lines_no_overlap(ax1d, z, *ax1d.get_xlim(), fontsize=fontsize)
            if sci2d is not None:
                rest_waves_A = np.array(list(REST_LINES_A.values()))
                obs_um = rest_waves_A * (1 + z) / 1e4
                mask = (obs_um > ax1d_xlim[0]) & (obs_um < ax1d_xlim[1])
                for x in obs_um[mask]:
                    ax2d.axvline(x, color="white", lw=0.4, ls=":", alpha=1, zorder=5)
        else:
            # rest-frame annotation at rest wavelengths
            x_min, x_max = ax1d.get_xlim()
            for name, lam_rest_A in REST_LINES_A.items():
                lam_rest_um = lam_rest_A / 1e4
                if not (x_min < lam_rest_um < x_max):
                    continue
                ax1d.axvline(lam_rest_um, color="k", lw=0.4, ls=":", alpha=0.5)
                ax1d.text(
                    lam_rest_um, 0.98, name, rotation=90,
                    va="top", ha="center",
                    transform=ax1d.get_xaxis_transform(),
                    fontsize=fontsize,
                )
                if sci2d is not None:
                    ax2d.axvline(lam_rest_um, color="white", lw=0.4, ls=":", alpha=1, zorder=5)

    # ----------------------------------------------------
    # --- Title and save ---
    # ----------------------------------------------------
    if title:
        fig.text(0.5, -0.02, title, ha="center", va="top", fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=False)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig

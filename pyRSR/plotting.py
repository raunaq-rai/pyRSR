import numpy as np
import matplotlib.pyplot as plt

# === REST wavelengths (Å) ===
REST_LINES_A = {
    "NIII_1": 989.790, "NIII_2": 991.514, "NIII_3": 991.579,
    "NV_1": 1238.821, "NV_2": 1242.804, "NIV_1": 1486.496,
    "HEII_1": 1640.420, "OIII_05": 1663.4795, "CIII": 1908.734,
    "OII_UV_1": 3727.092, "OII_UV_2": 3729.875,
    "NEIII_UV_1": 3869.86, "NEIII_UV_2": 3968.59,
    "HDELTA": 4102.8922, "HGAMMA": 4341.6837, "OIII_1": 4364.436,
    "HEI_1": 4471.479, "HEII_2": 4685.710, "HBETA": 4862.6830,
    "OIII_2": 4960.295, "OIII_3": 5008.240,
    "NII_1": 5756.19, "HEI": 5877.252, "NII_2": 6549.86,
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

def annotate_lines_no_overlap(ax, z, lam_min, lam_max, fontsize=7):
    """
    Annotate emission lines without overlaps, keeping labels inside the axes.
    """
    # Build observed wavelengths for lines in range
    rest_waves_A = np.array(list(REST_LINES_A.values()))
    line_names   = np.array(list(REST_LINES_A.keys()))
    obs_um       = rest_waves_A * (1 + z) / 1e4

    m = (obs_um > lam_min) & (obs_um < lam_max)
    if not np.any(m):
        return

    obs_um   = obs_um[m]
    names_in = line_names[m]

    # Sort by wavelength
    order = np.argsort(obs_um)
    obs_um, names_in = obs_um[order], names_in[order]

    # Base y near the top
    ymin, ymax = ax.get_ylim()
    yspan  = ymax - ymin
    y_base = ymin + 0.90 * yspan

    # Put a faint guide line for orientation (optional)
    for x in obs_um:
        ax.axvline(x, color="0.7", lw=0.4, alpha=0.5, zorder=1)

    # Create all Text artists first (same initial y); rotation vertical
    texts = []
    for x, label in zip(obs_um, names_in):
        t = ax.text(
            x, y_base, label,
            rotation=90, ha="center", va="bottom",
            fontsize=fontsize, color="0.2", alpha=0.9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.2),
            clip_on=False,  # adjustText keeps us in-bounds via lim=...
            zorder=10,
        )
        texts.append(t)

    # Tell adjustText to move labels but keep them in-bounds
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    adjust_text(
        texts,
        # allow both x & y motion for text; don’t move data points
        only_move={'points': 'none', 'text': 'xy'},
        # make collisions “bigger” so labels don’t touch
        expand_text=(1.1, 1.25),
        # slight repulsion strength
        force_text=(0.05, 0.2),
        # keep inside axes rectangle
        lim=(x0, x1, y0, y1),
        # draw short connectors to their x-position if moved sideways
        arrowprops=dict(arrowstyle='-', lw=0.4, color='0.4', alpha=0.7),
        autoalign='y',  # prefer vertical stacking
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

def plot_spectrum_with_2d(
    lam_um,
    flux_uJy,
    err_uJy=None,
    sci2d=None,
    model_uJy=None,
    cont_uJy=None,
    title=None,
    xlim=None,
    ylim=None,
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
):
    """
    Plot a JWST/NIRSpec 1D spectrum with optional 2D spectrum above.

    Parameters
    ----------
    lam_um : array
        Wavelength array [µm]
    flux_uJy : array
        Flux density [µJy]
    err_uJy : array, optional
        1σ uncertainty [µJy]
    sci2d : 2D array, optional
        Rectified 2D spectral image (for top panel)
    z : float, optional
        Redshift (for line annotation)
    cmap, vmin, vmax : optional
        Matplotlib colormap and limits for 2D display
    """

    lam_um = np.asarray(lam_um, float)
    flux_uJy = np.asarray(flux_uJy, float)
    err_uJy = np.asarray(err_uJy, float) if err_uJy is not None else None

    # --- bin edges for stairs/pcolormesh ---
    dlam = np.median(np.diff(lam_um))
    edges_um = np.concatenate(([lam_um[0] - dlam / 2],
                               0.5 * (lam_um[1:] + lam_um[:-1]),
                               [lam_um[-1] + dlam / 2]))

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
    # --- Top panel: 2D spectrum ---
    # ----------------------------------------------------
    if sci2d is not None:
        ny = sci2d.shape[0]
        y_edges = np.arange(ny + 1)
        if vmin is None or vmax is None:
            vmin, vmax = np.nanpercentile(sci2d, [5, 99.5])
        im = ax2d.pcolormesh(edges_um, y_edges, sci2d, shading="auto",
                             cmap=cmap, vmin=vmin, vmax=vmax)
        ax2d.set_ylabel("Spatial pixel", fontsize=9)
        ax2d.tick_params(labelbottom=False, direction="in")
        ax2d.set_ylim(ny * 0.25, ny * 0.75)

    # ----------------------------------------------------
    # --- Bottom panel: 1D spectrum ---
    # ----------------------------------------------------
    if err_uJy is not None:
        ax1d.fill_between(lam_um, flux_uJy - err_uJy, flux_uJy + err_uJy,
                          step="mid", color="grey", alpha=0.25, linewidth=0)

    ax1d.stairs(flux_uJy, edges_um, color=color, lw=0.5, alpha=alpha, label="Data")

    if cont_uJy is not None:
        ax1d.stairs(cont_uJy, edges_um, color="b", ls="--", lw=0.5, label="Continuum")

    if model_uJy is not None:
        ax1d.stairs(model_uJy, edges_um, color="r", lw=0.5, label="Model")

    ax1d.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
    ax1d.set_xlabel("Observed wavelength [µm]")
    ax1d.set_ylabel("Flux density [µJy]")
    if title:
        ax1d.set_title(title, fontsize=11)
    if xlim:
        ax1d.set_xlim(*xlim)
    if ylim:
        ax1d.set_ylim(*ylim)

    ax1d.legend(fontsize=8, frameon=False, ncol=3)
    ax1d.grid(alpha=0.25, linestyle=":", linewidth=0.5)
    ax1d.tick_params(direction="in", top=True, right=True)

    # ----------------------------------------------------
    # --- Annotate emission lines ---
    # ----------------------------------------------------
    if annotate_lines and z is not None:
        annotate_lines_no_overlap(ax1d, z, lam_um.min(), lam_um.max(), fontsize=fontsize)
        if sci2d is not None:
            # faint vertical lines on 2D panel for same features
            rest_waves_A = np.array(list(REST_LINES_A.values()))
            obs_um = rest_waves_A * (1 + z) / 1e4
            for x in obs_um[(obs_um > lam_um.min()) & (obs_um < lam_um.max())]:
                ax2d.axvline(x, color="white", lw=0.5, ls=":", alpha=0.5, zorder=5)


    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=False)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig

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
    if annotate_lines and z is not None:
        rest_waves_A = np.array(list(REST_LINES_A.values()))
        line_names = np.array(list(REST_LINES_A.keys()))

        obs_um = rest_waves_A * (1 + z) / 1e4  # Å → µm
        in_range = (obs_um > lam_um.min()) & (obs_um < lam_um.max())

        obs_um = obs_um[in_range]
        line_names = line_names[in_range]

        # avoid clutter: stagger labels vertically
        ymin, ymax = ax.get_ylim()
        label_height = ymax - ymin
        levels = np.linspace(0.9, 0.75, 6)  # staggered heights

        used_x = []
        for i, (lam_i, name) in enumerate(zip(obs_um, line_names)):
            if np.any(np.abs(np.array(used_x) - lam_i) < 0.015):  # avoid overlaps
                continue
            used_x.append(lam_i)
            y_frac = levels[i % len(levels)]
            ax.axvline(lam_i, color="gray", lw=0.4, alpha=0.5, zorder=1)
            ax.text(
                lam_i, ymin + y_frac * label_height,
                name, rotation=90,
                va="bottom", ha="center",
                fontsize=fontsize,
                color="0.2",
                alpha=0.85,
                clip_on=True,
            )

    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight", transparent=False)
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig

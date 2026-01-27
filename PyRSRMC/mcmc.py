
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Physics Imports from PyRSR
try:
    from PyRSR.fitting_helpers import (
        REST_LINES_A, 
        sigma_grating_logA, 
        fnu_uJy_to_flam,
        flam_to_fnu_uJy,
        LN10, 
        C_AA
    )
except ImportError:
    # Fallback if PyRSR structure is different (robustness)
    LN10 = np.log(10.0)
    REST_LINES_A = {} # Should not happen if installed correctly

try:
    from PyRSR.plotting import plot_spectrum_with_2d
except ImportError:
    plot_spectrum_with_2d = None

# MCMC Samplers
try:
    import emcee
except ImportError:
    emcee = None
    
try:
    import nautilus
except ImportError:
    nautilus = None

# -----------------------------------------------------------------------------
# Helpers for Plotting
# -----------------------------------------------------------------------------
def _annotate_lines_above_model(
    ax,
    lam,
    model,
    line_names,
    z,
    label_offset_frac=0.06,
    x_margin_frac=0.02,
    fontsize=8,
):
    """ Annotate emission lines cleanly above the model peak """
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
        # Simple rest frame logic
        if nm not in REST_LINES_A:
            continue
        lam_c = REST_LINES_A[nm] * (1.0 + z) / 1e4

        if not (x_lo <= lam_c <= x_hi):
            continue

        # Local model peak
        win = np.abs(lam - lam_c) <= 0.02
        if not np.any(win):
            continue

        y_loc = model[win]
        if not np.any(np.isfinite(y_loc)):
            continue

        y_peak = float(np.nanmax(y_loc))
        
        # Clean Label
        lab = nm.replace("BROAD", "br").replace("_", " ")
        
        xs.append(lam_c)
        y_peaks.append(y_peak)
        labels.append(lab)

    if not xs:
        return

    xs = np.asarray(xs, float)
    y_peaks = np.asarray(y_peaks, float)
    labels = np.asarray(labels, dtype=object)

    order = np.argsort(xs)
    xs, y_peaks, labels = xs[order], y_peaks[order], labels[order]

    dy = (y_max - y_min) * float(label_offset_frac)
    dx_thresh = (x_hi - x_lo) * float(x_margin_frac)

    placed_ys = []

    for i, (x, y0, lab) in enumerate(zip(xs, y_peaks, labels)):
        y = y0 + dy
        if i > 0 and abs(x - xs[i - 1]) < dx_thresh:
            y = max(y, placed_ys[-1] + 0.8 * dy)
        y = min(y, y_max - 0.3 * dy)
        placed_ys.append(y)

        ax.text(
            x, y, lab,
            ha="center", va="bottom",
            rotation=0, fontsize=fontsize, color="0.25", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )

# -----------------------------------------------------------------------------
# 1. Physics / Model Functions (Clean & Explicit)
# -----------------------------------------------------------------------------

def gaussian_pixel_integrated(lam_edges, mu, sigma):
    """
    Compute area-normalized Gaussian integrated over pixel bins.
    
    Args:
        lam_edges: 1D array of pixel edges (length N+1)
        mu: Centroid
        sigma: Standard deviation
        
    Returns:
        1D array of length N (values per pixel)
    """
    if sigma <= 0:
        return np.zeros(len(lam_edges)-1)
        
    # CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    inv_sq2sig = 1.0 / (sigma * np.sqrt(2.0))
    cdf = 0.5 * (1.0 + erf((lam_edges - mu) * inv_sq2sig))
    
    # Area in each bin is diff of CDF
    area = np.diff(cdf)
    return area

def build_model(params, lam, which_lines, fixed_z, fixed_sigma_inst):
    """
    Construct the model flux density (F_lambda).
    
    Args:
        params: Flat array of [Amplitudes..., Widths..., Offsets...]
             For now, we assume a simple parametrization:
             - One Amplitude (Total Flux) per line.
             - One Sigma broadening factor per line (or shared).
             - One Redshift (or velocity offset).
    
    Returns:
        model_flux: same shape as lam
    """
    # This is a placeholder for the more complex parameter mapping class
    pass

# -----------------------------------------------------------------------------
# 2. Parameter Handling Class
# -----------------------------------------------------------------------------

class LineModel:
    """
    Manages the translation between MCMC parameters and Physical Model.
    """
    def __init__(self, 
                 line_names: List[str], 
                 z_init: float, 
                 sigma_inst_A: List[float],
                 flux_seeds: List[float]):
        
        self.line_names = line_names
        self.n_lines = len(line_names)
        self.z_init = z_init
        self.sigma_inst_A = np.array(sigma_inst_A)
        self.flux_seeds = np.array(flux_seeds)
        
        # Parameter layout:
        # 0..N-1 : Fluxes (linear units)
        # N..2N-1: Broadening (Multiplicative factor over instrument sigma OR intrinsic velocity)
        #          Let's stick to Intrinsic Sigma (Angstroms) for broad lines?
        #          Actually, simpler: Total Sigma (Angstroms).
        # 2N     : Redshift offset (delta_z)
        
        self.n_dim = 2 * self.n_lines + 1
        
        # Bounds
        self.lb = np.zeros(self.n_dim)
        self.ub = np.zeros(self.n_dim)
        
        # Flux Bounds: 0 to 1000x seed
        self.lb[0:self.n_lines] = 0.0 # Strict positive
        self.ub[0:self.n_lines] = np.maximum(self.flux_seeds * 1000.0, 1e-10)
        
        # Sigma Bounds: Inst/2 to 500A (broad)
        # We need to distinguish narrow/broad logic here, but let's keep it generic first
        self.lb[self.n_lines : 2*self.n_lines] = self.sigma_inst_A * 0.5
        self.ub[self.n_lines : 2*self.n_lines] = np.maximum(self.sigma_inst_A * 10.0, 500.0)
        
        # Redshift Bounds: +/- 1000 km/s approx
        dz = 0.02
        self.lb[-1] = -dz
        self.ub[-1] = +dz

    def get_initial_guess(self):
        p0 = np.zeros(self.n_dim)
        p0[0:self.n_lines] = self.flux_seeds
        p0[self.n_lines : 2*self.n_lines] = self.sigma_inst_A * 1.1 # slightly broader start
        p0[-1] = 0.0
        return p0

    def compute_model(self, params, lam_um):
        fluxes = params[0 : self.n_lines]
        sigmas_A = params[self.n_lines : 2*self.n_lines]
        dz = params[-1]
        
        z_curr = self.z_init + dz
        
        lam_A = lam_um * 1e4
        # Pixel edges for accurate integration
        edges_A = np.zeros(len(lam_A)+1)
        edges_A[1:-1] = 0.5 * (lam_A[:-1] + lam_A[1:])
        edges_A[0] = lam_A[0] - (edges_A[1] - lam_A[0])
        edges_A[-1] = lam_A[-1] + (lam_A[-1] - edges_A[-2])
        
        model = np.zeros_like(lam_A)
        
        # Width of bins (approx for F_lambda scaling)
        widths_A = np.diff(edges_A)
        
        for i, name in enumerate(self.line_names):
            if name not in REST_LINES_A: continue
            
            lam0 = REST_LINES_A[name]
            mu_A = lam0 * (1.0 + z_curr)
            sig_A = sigmas_A[i]
            
            # Integrated area in each bin (sum=1)
            prof = gaussian_pixel_integrated(edges_A, mu_A, sig_A)
            
            # Scale by total Flux (F_lambda integrated)
            # F_lambda(pix) ~ (Flux / Width) * NormalizedArea
            # Actually, `prof` is the fraction of total flux in that bin.
            # So Flux_in_bin = Flux * prof
            # F_lambda = Flux_in_bin / d_lambda
            
            model += fluxes[i] * (prof / widths_A)
            
        return model

    def get_model_components(self, params, lam_um):
        """
        Return separate model components for detailed plotting.
        Returns:
            total_model: array
            dict_of_profiles: {line_name: flux_array}
        """
        fluxes = params[0 : self.n_lines]
        sigmas_A = params[self.n_lines : 2*self.n_lines]
        dz = params[-1]
        z_curr = self.z_init + dz
        
        lam_A = lam_um * 1e4
        edges_A = np.zeros(len(lam_A)+1)
        edges_A[1:-1] = 0.5 * (lam_A[:-1] + lam_A[1:])
        edges_A[0] = lam_A[0] - (edges_A[1] - lam_A[0])
        edges_A[-1] = lam_A[-1] + (lam_A[-1] - edges_A[-2])
        widths_A = np.diff(edges_A)
        
        profiles = {}
        total = np.zeros_like(lam_A)
        
        for i, name in enumerate(self.line_names):
            if name not in REST_LINES_A: continue
            
            lam0 = REST_LINES_A[name]
            mu_A = lam0 * (1.0 + z_curr)
            sig_A = sigmas_A[i]
            
            prof = gaussian_pixel_integrated(edges_A, mu_A, sig_A)
            comp_flux = fluxes[i] * (prof / widths_A)
            profiles[name] = comp_flux
            total += comp_flux
            
        return total, profiles

# -----------------------------------------------------------------------------
# 3. Main Fitter Class
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 3. Main Fitter Class
# -----------------------------------------------------------------------------

class MCMCFitter:
    def __init__(self, lam, flux, err, model_manager: LineModel):
        # 1. Filter Infinite/NaN Data immediately
        mask = np.isfinite(flux) & np.isfinite(err) & np.isfinite(lam) & (err > 0)
        self.lam = lam[mask]
        self.flux = flux[mask]
        self.err = err[mask]
        self.model = model_manager
        self.ndim = model_manager.n_dim
        
        if len(self.lam) == 0:
            raise ValueError("No valid data points (all NaN or masked) for MCMC fit.")

        # Pre-compute weights
        self.ivar = 1.0 / (self.err**2 + 1e-30)
        # Final cleanup just in case
        self.ivar[~np.isfinite(self.ivar)] = 0.0

    def log_likelihood(self, params):
        # 1. Model
        model_flux = self.model.compute_model(params, self.lam)
        
        # Check model validity
        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        # 2. Residual
        resid = self.flux - model_flux
        
        # 3. Chi2
        chi2 = np.sum(resid**2 * self.ivar)
        
        if not np.isfinite(chi2):
            return -np.inf
            
        return -0.5 * chi2

    def log_prior(self, params):
        if np.any(params < self.model.lb) or np.any(params > self.model.ub):
            return -np.inf
        return 0.0

    def log_probability(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp): return -np.inf
        
        ll = self.log_likelihood(params)
        if not np.isfinite(ll): return -np.inf
        
        return lp + ll

    def run_emcee(self, n_walkers=None, n_steps=1000, burn_in=200, verbose=True):
        if emcee is None:
            raise ImportError("emcee not installed.")
            
        if n_walkers is None:
            n_walkers = max(32, 2 * self.ndim + 2)
            
        if verbose:
            print(f"[MCMC] Running emcee with {n_walkers} walkers, {n_steps} steps, ndim={self.ndim}")

        # Initialize
        p0_center = self.model.get_initial_guess()
        p0 = np.zeros((n_walkers, self.ndim))
        
        # Robust ball initialization
        for i in range(self.ndim):
            val = p0_center[i]
            width = (self.model.ub[i] - self.model.lb[i])
            scale = width * 0.05
            p0[:, i] = val + scale * np.random.randn(n_walkers)
            p0[:, i] = np.clip(p0[:, i], self.model.lb[i] + 1e-6*width, self.model.ub[i] - 1e-6*width)
        
        # Final Check on Initial State
        # Ensure all walkers have finite log_prob before starting
        # If not, it crashes emcee
        valid_start = False
        retry_count = 0
        while not valid_start and retry_count < 5:
             # Check probs
             probs = np.array([self.log_probability(p) for p in p0])
             bad = ~np.isfinite(probs)
             if np.any(bad):
                 if verbose:
                     print(f"[MCMC] Found {np.sum(bad)} walkers with -inf/nan probability. Resampling...")
                 # Resample bad ones from good ones or random
                 good_indices = np.where(~bad)[0]
                 bad_indices = np.where(bad)[0]
                 
                 if len(good_indices) > 0:
                     for idx in bad_indices:
                         # Pick a random good walker and jitter it
                         clone_idx = np.random.choice(good_indices)
                         p0[idx] = p0[clone_idx] * (1 + 1e-4*np.random.randn(self.ndim))
                 else:
                     # Very bad initial guess, try re-centering tight on p0_center
                     p0 = p0_center + 1e-5 * p0_center * np.random.randn(n_walkers, self.ndim)
                 retry_count += 1
             else:
                 valid_start = True

        sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_probability)
        sampler.run_mcmc(p0, n_steps, progress=verbose, skip_initial_state_check=True)
        
        flat_chain = sampler.get_chain(discard=burn_in, flat=True, thin=1)
        return flat_chain

# -----------------------------------------------------------------------------
# 4. Usage Wrapper
# -----------------------------------------------------------------------------

def plot_mcmc_detailed(lam, flux, err, model_total, profiles, z, title, unit="fnu"):
    """
    Generate the 4-panel plot (Main + 3 Zooms) matching PyRSR style.
    """
    import matplotlib.gridspec as gridspec
    
    # 1. Setup Layout
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1.0], hspace=0.35, wspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_z1 = fig.add_subplot(gs[1, 0])
    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_z3 = fig.add_subplot(gs[1, 2])
    
    # Label
    ylabel = r"$F_{\nu}$ [$\mu$Jy]" if unit == "fnu" else r"$F_{\lambda}$"
    
    # 2. Main Panel
    # Data check
    if np.all(np.isnan(flux)):
         print("Warning: All flux is NaN, skipping plot.")
         return

    # Use steps for data
    # Create bin edges for steps
    dlam = np.median(np.diff(lam))
    edges = np.concatenate(([lam[0] - dlam/2], 0.5*(lam[1:]+lam[:-1]), [lam[-1] + dlam/2]))
    
    ax_main.stairs(flux, edges, color='k', lw=1, alpha=0.5, label='Data')
    if err is not None:
         ax_main.fill_between(lam, flux-err, flux+err, step="mid", color='gray', alpha=0.2, lw=0)
    
    ax_main.plot(lam, model_total, color='r', lw=1.5, alpha=0.9, label='Total Model')
    
    ax_main.set_xlim(lam.min(), lam.max())
    ax_main.set_xlabel(r"Observed Wavelength [$\mu$m]")
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.legend(loc='upper right', frameon=False)
    
    # 3. Decompose Profiles into Narrow / Broad for coloring
    # Simple heuristic: "BROAD" in name -> Broad
    narrow_sum = np.zeros_like(lam)
    broad_sum = np.zeros_like(lam)
    
    for name, prof in profiles.items():
        if "BROAD" in name.upper():
            broad_sum += prof
        else:
            narrow_sum += prof
            
    # 4. Zoom Panels Logic
    # We define 3 zones:
    # Z1: H-delta (4102) + OIII_4363
    # Z2: H-beta (4862) + OIII_4959/5007
    # Z3: H-alpha (6564) + NII
    
    def get_zoom_window(center_rest_A, z_val, width_rest_A=300):
        cen_um = center_rest_A * (1+z_val) / 1e4
        width_um = width_rest_A * (1+z_val) / 1e4
        return cen_um - width_um/2, cen_um + width_um/2
        
    zooms = [
        (ax_z1, 4102.0, "Auroral / H$\delta$"),
        (ax_z2, 4959.0, "H$\\beta$ + [OIII]"),
        (ax_z3, 6564.0, "H$\\alpha$ + [NII]")
    ]
    
    for ax, cen_rest, label in zooms:
        lo, hi = get_zoom_window(cen_rest, z)
        
        # Check coverage
        mask = (lam >= lo) & (lam <= hi)
        if np.sum(mask) < 2:
            ax.text(0.5, 0.5, "No Coverage", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
            
        # Plot Data
        lam_z = lam[mask]
        flux_z = flux[mask]
        edges_z = np.concatenate(([lam_z[0] - dlam/2], 0.5*(lam_z[1:]+lam_z[:-1]), [lam_z[-1] + dlam/2]))

        ax.stairs(flux_z, edges_z, color='k', lw=0.8, alpha=0.4)
        
        # Plot Components (Smooth?)
        # For now just plot the binned components to match resolution
        # Narrow = Blue, Broad = Orange
        
        n_z = narrow_sum[mask]
        b_z = broad_sum[mask]
        
        # Stacked Areas
        # Base is 0 (assuming continuum subtracted or user just wants line model)
        # Wait, MCMCFitter model includes continuum? No, my build_model is line-only.
        # So background is 0.
        
        ax.fill_between(lam_z, 0, n_z, color='tab:blue', alpha=0.4, label='Narrow')
        
        if np.max(b_z) > 1e-9:
             ax.fill_between(lam_z, n_z, n_z + b_z, color='tab:orange', alpha=0.4, label='Broad')
             ax.plot(lam_z, n_z + b_z, color='r', lw=1)
        else:
             ax.plot(lam_z, n_z, color='r', lw=1)
             
        ax.set_xlim(lo, hi)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        
    plt.tight_layout()
    plt.show()

def print_mcmc_summary(res, input_unit="fnu"):
    """
    Print a structured summary of the MCMC fit results matching PyRSR.broad_fit style.
    """
    if not res.get("success"):
        print("Fit Failed.")
        return
        
    lines = res.get("lines", {})
    z_fit = res.get("z_fit")
    
    # Constants
    C_AA = 2.99792458e18
    
    # Header matching user request
    # Line | F_line [erg/s/cm²] | EW₀ [Å] | σ_A [Å] | μ_obs [Å] | SNR_int | SNR_peak(data) | SNR_peak(model)
    
    print("\n=== BOOTSTRAP SUMMARY (value ± error) ===")
    header = (
        f"{'Line':<15} {'F_line [erg/s/cm²]':<25} {'EW₀ [Å]':<15} {'σ_A [Å]':<15} "
        f"{'μ_obs [Å]':<15} {'SNR_int':<12} {'SNR_peak(data)':<15} {'SNR_peak(model)':<16}"
    )
    print(header)
    print("-" * 135)
    
    for name, info in lines.items():
        # 1. Retrieve Params
        flux = info["flux"]
        flux_err = info["flux_err"]
        sigma = info["sigma_A"]
        sigma_err = info["sigma_A_err"]
        
        # 2. Get Rest Wavelength for Calc
        lam0 = REST_LINES_A.get(name, 0.0)
        lam_obs = lam0 * (1 + z_fit)
        
        # 3. Flux Conversion to erg/s/cm2
        # Param `flux` is integral in (InputUnit * Angstrom)
        f_cgs = np.nan
        f_cgs_err = np.nan
        
        if input_unit.lower() == "fnu":
            # Input: uJy. Param: uJy*A.
            # Conv: uJy*A -> erg/s/cm2
            # F_line = F_nu_int * (c / lam^2) * 1e-29
            conv = (C_AA / (lam_obs**2)) * 1e-29 if lam_obs > 0 else 0
            f_cgs = flux * conv
            f_cgs_err = flux_err * conv
        elif input_unit.lower() == "flam":
            # Input: erg/s/cm2/A. Param: erg/s/cm2.
            # No conversion needed (already integrated flam)
            f_cgs = flux
            f_cgs_err = flux_err
            
        # 4. SNR Integrated
        snr_int = flux / flux_err if flux_err > 0 else 0.0
        
        # 5. SNR Peak
        # We need the pixel data to calculate this properly, but 'res' dict only has params.
        # Ideally we'd calculate this during the fit and store it in 'lines'.
        # For now, I'll put placeholders or rudimentary estimates if available.
        # Current MCMCFitter doesn't store peak SNRs in res['lines'].
        # I will output "—" for now until I update the fitter.
        snr_peak_data = "—"
        snr_peak_model = "—"
        
        # 6. EW
        # Requires continuum. Using placeholder.
        ew = "—"
        
        # Format Strings
        s_flux = f"{f_cgs:.3e} ± {f_cgs_err:.3e}"
        s_ew   = f"{ew}"
        s_sig  = f"{sigma:.2f} ± {sigma_err:.2f}"
        s_mu   = f"{lam_obs:.1f}" # No error on z propagated yet?
        s_snri = f"{snr_int:.2f}"
        
        row = (
            f"{name:<15} {s_flux:<25} {s_ew:<15} {s_sig:<15} "
            f"{s_mu:<15} {s_snri:<12} {snr_peak_data:<15} {snr_peak_model:<16}"
        )
        print(row)
        
    print("-" * 135)
    print(f"all in erg/s/cm2\n")

def run_mcmc_fit(
    spec_data: Dict[str, np.ndarray],
    z_init: float,
    line_names: List[str],
    grating: str = "PRISM",
    n_walkers=None, 
    n_steps=1000,
    verbose=True,
    input_unit: str = "fnu", # "fnu" (uJy) or "flam" (erg/s/cm2/A)
    plot: bool = False,
    plot_unit: str = "fnu" # "fnu" or "flam"
):
    """
    Main entry point for MCMC fitting.
    
    Args:
        spec_data: dict with 'lam' (um), 'flux', 'err'
        z_init: initial redshift guess
        line_names: list of line names to fit
        grating: e.g. "PRISM", "G395M"
        input_unit: Unit of input flux. Default 'fnu' (uJy).
        plot: Whether to generate a plot.
        plot_unit: Unit for the plot ('fnu' or 'flam').
        
    Returns:
        dict with keys 'chains', 'best_fit', 'success'
    """
    lam = spec_data["lam"]
    flux = spec_data["flux"]
    err = spec_data["err"]
    
    # Estimate Inputs
    # 1. Sigma Inst
    lam_obs = np.array([REST_LINES_A.get(l, 0)*(1+z_init)/1e4 for l in line_names])
    if len(lam_obs) == 0:
         return {"success": False, "error": "No valid lines in REST_LINES_A"}
         
    sigma_log = sigma_grating_logA(grating, lam_obs)
    # Convert log-sigma to Angstroms: sigma_A = lam_A * ln10 * sigma_log
    sigma_inst_A = (lam_obs * 1e4) * LN10 * sigma_log
    
    # 2. Flux Seeds (basic estimator)
    flux_seeds = []
    # If input is Fnu, we can just find peak in Fnu and guess.
    
    for l_obs, sig_A in zip(lam_obs, sigma_inst_A):
        # find peak near l_obs
        mask = (lam > l_obs - 0.01) & (lam < l_obs + 0.01)
        if np.any(mask):
            peak = np.nanmax(flux[mask]) 
            peak = max(peak, 1e-20) # ensure positive
        else:
            peak = np.nanpercentile(flux, 95) if len(flux) > 0 else 1e-19
            peak = max(peak, 1e-20)
            
        # Initial guess area ~ Peak * Width
        # Factor 2.5 ~ sqrt(2pi)
        flux_est = peak * sig_A * 2.5 
        flux_seeds.append(flux_est)
        
    model_mgr = LineModel(line_names, z_init, sigma_inst_A, flux_seeds)
    fitter = MCMCFitter(lam, flux, err, model_mgr)
    
    try:
        chain = fitter.run_emcee(n_walkers=n_walkers, n_steps=n_steps, verbose=verbose)
        
        # Best fit (median)
        p_med = np.median(chain, axis=0)
        p_std = np.std(chain, axis=0)
        
        # Reconstruct result dict
        res_lines = {}
        for i, name in enumerate(line_names):
            res_lines[name] = {
                "flux": p_med[i],
                "flux_err": p_std[i],
                "sigma_A": p_med[len(line_names) + i],
                "sigma_A_err": p_std[len(line_names) + i]
            }
            
        # Plotting
        if plot:
            # 1. Compute Full Model + Components
            # We use the FULL wavelength grid from input spec_data, not just the finite one
            full_lam = spec_data["lam"] # Use original grid for plotting coverage
            
            # Note: params are fitted on filtered data, but we can evaluate model on full grid
            total_model, profiles = model_mgr.get_model_components(p_med, full_lam)
            
            # 2. Unit Conversion
            curr = input_unit.lower()
            targ = plot_unit.lower()
            
            p_flux = spec_data["flux"]
            p_err = spec_data["err"]
            p_model = total_model
            p_profiles = profiles
            
            if curr == "flam" and targ == "fnu":
                # input flam -> user wants fnu
                p_flux = flam_to_fnu_uJy(p_flux, full_lam)
                p_err = flam_to_fnu_uJy(p_err, full_lam) if p_err is not None else None
                p_model = flam_to_fnu_uJy(p_model, full_lam)
                for k in p_profiles:
                     p_profiles[k] = flam_to_fnu_uJy(p_profiles[k], full_lam)
                     
            elif curr == "fnu" and targ == "flam":
                # input fnu -> user wants flam
                p_flux = fnu_uJy_to_flam(p_flux, full_lam)
                p_err = fnu_uJy_to_flam(p_err, full_lam) if p_err is not None else None
                p_model = fnu_uJy_to_flam(p_model, full_lam)
                for k in p_profiles:
                     p_profiles[k] = fnu_uJy_to_flam(p_profiles[k], full_lam)
            
            # 3. Call Detailed Plotter
            plot_mcmc_detailed(
                lam=full_lam,
                flux=p_flux,
                err=p_err,
                model_total=p_model,
                profiles=p_profiles,
                z=z_init + p_med[-1],
                title=f"MCMC Fit ({grating})",
                unit=targ
            )

        res = {
            "success": True,
            "chains": chain,
            "best_params": p_med,
            "lines": res_lines,
            "z_fit": z_init + p_med[-1],
            "fit_stats": {"ndim": fitter.ndim, "n_steps": n_steps}
        }
        
        if verbose:
            print_mcmc_summary(res, input_unit=input_unit)
            
        return res
        
    except Exception as e:
        if verbose:
            print(f"MCMC Failed: {e}")
            import traceback
            traceback.print_exc()
        return {"success": False, "error": str(e)}

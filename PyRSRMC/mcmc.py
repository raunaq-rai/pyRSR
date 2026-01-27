
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

def gaussian_profile(lam, A, mu, sigma):
    """
    Compute a Gaussian profile for plotting.
    
    Args:
        lam: Wavelength array (Angstroms or microns, must match mu/sigma units)
        A: Integrated flux (area under curve)
        mu: Centroid (same units as lam)
        sigma: Standard deviation (same units as lam)
        
    Returns:
        F_lambda: Flux density at each wavelength (F_lambda = A * G(lam) / integral)
    """
    if sigma <= 0:
        return np.zeros_like(lam)
    
    SQRT2PI = np.sqrt(2 * np.pi)
    # Area-normalized Gaussian: integral = 1
    gauss = np.exp(-0.5 * ((lam - mu) / sigma)**2) / (sigma * SQRT2PI)
    return A * gauss

def reconstruct_line_profile(lam_um, line_params):
    """
    Reconstruct a line's Gaussian profile from fitted parameters.
    
    Args:
        lam_um: Wavelength array in microns
        line_params: dict with 'A_gauss', 'lam_obs_A', 'sigma_A'
        
    Returns:
        profile: F_lambda array
    """
    A = line_params["A_gauss"]
    mu_A = line_params["lam_obs_A"]
    sigma_A = line_params["sigma_A"]
    
    # Convert lam to Angstroms
    lam_A = lam_um * 1e4
    
    return gaussian_profile(lam_A, A, mu_A, sigma_A)

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
        
        # Flux Bounds: 0 to 10000x seed (very generous for real data)
        self.lb[0:self.n_lines] = 0.0 # Strict positive
        self.ub[0:self.n_lines] = np.maximum(self.flux_seeds * 10000.0, 1e-5)
        
        # Sigma Bounds: Handle narrow vs broad components
        # Base bounds: 0.2*inst to 15*inst for flexibility
        sig_lo = self.sigma_inst_A * 0.2
        sig_hi = np.maximum(self.sigma_inst_A * 15.0, 1000.0)
        
        # Widen bounds for BROAD components
        for j, nm in enumerate(self.line_names):
            if "BROAD2" in nm.upper():
                sig_lo[j] = self.sigma_inst_A[j] * 4.0
                sig_hi[j] = max(self.sigma_inst_A[j] * 25.0, 2000.0)
            elif "BROAD" in nm.upper():
                sig_lo[j] = self.sigma_inst_A[j] * 1.5
                sig_hi[j] = max(self.sigma_inst_A[j] * 12.0, 1500.0)
        
        self.lb[self.n_lines : 2*self.n_lines] = sig_lo
        self.ub[self.n_lines : 2*self.n_lines] = sig_hi
        
        # Redshift Bounds: +/- 0.01 (~3000 km/s)
        dz = 0.01
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
        return flat_chain, sampler

    def compute_bic(self, params):
        """
        Compute BIC = k*ln(n) - 2*ln(L)
        where k = ndim, n = number of data points, L = likelihood
        """
        ll = self.log_likelihood(params)
        n = len(self.lam)
        k = self.ndim
        bic = k * np.log(n) - 2.0 * ll
        return bic

def compute_bic_for_model(spec_data, z_init, line_names, grating, verbose=False):
    """
    Fast least-squares fit to compute BIC for a given line configuration.
    Uses scipy.optimize.least_squares, matching PyRSR's approach.
    """
    from scipy.optimize import least_squares
    
    lam = spec_data["lam"]
    flux = spec_data["flux"]
    err = spec_data["err"]
    
    # Filter valid lines
    valid_lines = [l for l in line_names if l in REST_LINES_A]
    if not valid_lines:
        return np.inf, None
    
    nL = len(valid_lines)
    
    # Compute observed wavelengths and instrument widths
    lam_obs_A = np.array([REST_LINES_A[l] * (1 + z_init) for l in valid_lines])
    lam_obs_um = lam_obs_A / 1e4
    
    sigma_log = sigma_grating_logA(grating, lam_obs_um)
    sigma_inst_A = lam_obs_A * LN10 * sigma_log
    
    # Convert wavelength to Angstroms for fitting
    lam_A = lam * 1e4
    
    # Pixel edges for area-normalized Gaussians
    dlam = np.median(np.diff(lam_A))
    edges_A = np.concatenate(([lam_A[0] - dlam/2], 0.5*(lam_A[1:]+lam_A[:-1]), [lam_A[-1] + dlam/2]))
    
    # Initial seeds for parameters: [A_0, A_1, ..., sigma_0, sigma_1, ..., mu_0, mu_1, ...]
    # A = flux amplitude (integrated flux)
    # sigma = line width in Angstroms
    # mu = centroid in Angstroms
    
    A0 = []
    for i, l_um in enumerate(lam_obs_um):
        win = (lam > l_um - 0.02) & (lam < l_um + 0.02)
        if np.any(win):
            peak = np.nanmax(flux[win])
            peak = max(peak, 1e-20)
        else:
            peak = np.nanpercentile(flux, 95) if len(flux) > 0 else 1e-19
            peak = max(peak, 1e-20)
        # Rough integrated flux estimate
        A0.append(peak * sigma_inst_A[i] * np.sqrt(2*np.pi))
    
    sigma0 = sigma_inst_A.copy()
    mu0 = lam_obs_A.copy()
    
    # Bounds - be generous to handle real data
    A_lo = np.zeros(nL)
    A_hi = np.array([1000 * max(a, 1e-10) for a in A0])  # More generous upper bound
    
    # Sigma bounds - widen for all gratings to handle broader lines
    g = str(grating).lower()
    if "prism" in g:
        sigma_lo = 0.4 * sigma_inst_A
        sigma_hi = 5.0 * sigma_inst_A  # Widened
    else:
        # G395M, G395H etc - be more generous
        sigma_lo = 0.2 * sigma_inst_A
        sigma_hi = 8.0 * sigma_inst_A  # Much wider to catch broad lines
    
    # Widen bounds for BROAD components
    for j, nm in enumerate(valid_lines):
        if "BROAD2" in nm:
            sigma_lo[j] = 4.0 * sigma_inst_A[j]
            sigma_hi[j] = 20.0 * sigma_inst_A[j]
            sigma0[j] = 7.0 * sigma_inst_A[j]
        elif "BROAD" in nm:
            sigma_lo[j] = 1.5 * sigma_inst_A[j]
            sigma_hi[j] = 10.0 * sigma_inst_A[j]
            sigma0[j] = 3.0 * sigma_inst_A[j]
    
    # Centroid bounds: +/- 100 Angstroms (wider for real data)
    mu_lo = lam_obs_A - 100.0
    mu_hi = lam_obs_A + 100.0
    
    # Pack parameters
    p0 = np.concatenate([A0, sigma0, mu0])
    lb = np.concatenate([A_lo, sigma_lo, mu_lo])
    ub = np.concatenate([A_hi, sigma_hi, mu_hi])
    
    # Ensure seeds are within bounds
    p0 = np.clip(p0, lb + 1e-10, ub - 1e-10)
    
    # Validate data before fitting
    valid_data = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if not np.any(valid_data):
        if verbose:
            print(f"  BIC fit failed: No valid data points")
        return np.inf, None
    
    # Filter data
    lam_valid = lam[valid_data]
    flux_valid = flux[valid_data]
    err_valid = err[valid_data]
    lam_A_valid = lam_valid * 1e4
    
    # Recompute edges for valid data
    dlam_v = np.median(np.diff(lam_A_valid))
    edges_valid = np.concatenate(([lam_A_valid[0] - dlam_v/2], 
                                   0.5*(lam_A_valid[1:]+lam_A_valid[:-1]), 
                                   [lam_A_valid[-1] + dlam_v/2]))
    
    def model_fn(params):
        A = params[:nL]
        sigma = params[nL:2*nL]
        mu = params[2*nL:3*nL]
        
        total = np.zeros_like(lam_A_valid)
        for j in range(nL):
            if sigma[j] <= 0:
                continue
            prof = gaussian_pixel_integrated(edges_valid, mu[j], sigma[j])
            total += A[j] * prof
        return total
    
    def residual_fn(params):
        model = model_fn(params)
        return (flux_valid - model) / np.clip(err_valid, 1e-30, None)
    
    try:
        res = least_squares(
            residual_fn, p0, 
            bounds=(lb, ub),
            max_nfev=20000,  # More iterations
            xtol=1e-7, ftol=1e-7
        )
        
        # Compute BIC = chi2 + k * ln(N)
        resid_vec = residual_fn(res.x)
        mfin = np.isfinite(resid_vec)
        N_data = int(np.count_nonzero(mfin))
        
        if N_data == 0:
            if verbose:
                print(f"  BIC fit failed: Zero valid residuals")
            return np.inf, None
            
        chi2 = float(np.sum(resid_vec[mfin]**2))
        k_params = res.x.size
        BIC = chi2 + k_params * np.log(N_data)
        
        if verbose:
            print(f"  Fit {valid_lines}: χ²={chi2:.1f}, k={k_params}, N={N_data}, BIC={BIC:.1f}")
        
        return BIC, {"params": res.x, "chi2": chi2, "lines": valid_lines}
        
    except Exception as e:
        if verbose:
            print(f"  BIC fit failed for {valid_lines}: {e}")
            import traceback
            traceback.print_exc()
        return np.inf, None

def select_broad_model(
    spec_data,
    z_init,
    base_lines,
    balmer_line="H⍺",
    grating="PRISM",
    broad_mode="auto",
    bic_delta=6.0,
    verbose=True
):
    """
    Select the best broad component configuration for a Balmer line using BIC.
    
    Args:
        spec_data: dict with lam, flux, err
        z_init: redshift
        base_lines: list of narrow lines to include
        balmer_line: which Balmer line to add broad components for (e.g. "H⍺", "HBETA")
        grating: instrument grating
        broad_mode: "auto", "off", "broad1", "broad2", "both"
        bic_delta: minimum BIC improvement to prefer broad model
        verbose: print diagnostics
        
    Returns:
        dict with:
            - chosen_lines: final line list
            - broad_choice: "none", "one", "broad2_only", "two"
            - BIC_narrow, BIC_1broad, BIC_2broad, BIC_b2only
    """
    broad1_name = f"{balmer_line}_BROAD"
    broad2_name = f"{balmer_line}_BROAD2"
    
    # Ensure broad names exist in REST_LINES_A
    if broad1_name not in REST_LINES_A and balmer_line in REST_LINES_A:
        REST_LINES_A[broad1_name] = REST_LINES_A[balmer_line]
    if broad2_name not in REST_LINES_A and balmer_line in REST_LINES_A:
        REST_LINES_A[broad2_name] = REST_LINES_A[balmer_line]
    
    results = {
        "broad_choice": "none",
        "BIC_narrow": np.nan,
        "BIC_1broad": np.nan,
        "BIC_2broad": np.nan,
        "BIC_b2only": np.nan,
        "chosen_lines": list(base_lines)
    }
    
    if balmer_line not in base_lines:
        if verbose:
            print(f"{balmer_line} not in base_lines, skipping broad selection.")
        return results
    
    if broad_mode == "off":
        if verbose:
            print(f"broad_mode='off' → using narrow-only {balmer_line} model.")
        return results
        
    # 1. Narrow-only fit
    if verbose:
        print(f"\n=== BIC Model Selection for {balmer_line} ===")
        print("Fitting narrow-only model...")
    
    bic_narrow, _ = compute_bic_for_model(spec_data, z_init, base_lines, grating, verbose=False)
    results["BIC_narrow"] = bic_narrow
    
    if verbose:
        print(f"  Narrow-only: BIC = {bic_narrow:.2f}")
    
    # 2. +BROAD fit
    lines_1broad = list(base_lines) + [broad1_name]
    if verbose:
        print("Fitting narrow + BROAD model...")
    bic_1broad, fit1 = compute_bic_for_model(spec_data, z_init, lines_1broad, grating, verbose=False)
    results["BIC_1broad"] = bic_1broad
    
    if verbose:
        print(f"  +BROAD: BIC = {bic_1broad:.2f}" if np.isfinite(bic_1broad) else "  +BROAD: (fit failed)")
    
    # 3. +BROAD2 only fit
    lines_b2only = list(base_lines) + [broad2_name]
    if verbose:
        print("Fitting narrow + BROAD2 model...")
    bic_b2only, fit_b2 = compute_bic_for_model(spec_data, z_init, lines_b2only, grating, verbose=False)
    results["BIC_b2only"] = bic_b2only
    
    if verbose:
        print(f"  +BROAD2 only: BIC = {bic_b2only:.2f}" if np.isfinite(bic_b2only) else "  +BROAD2 only: (fit failed)")
    
    # 4. +BROAD +BROAD2 fit
    lines_2broad = list(base_lines) + [broad1_name, broad2_name]
    if verbose:
        print("Fitting narrow + BROAD + BROAD2 model...")
    bic_2broad, fit2 = compute_bic_for_model(spec_data, z_init, lines_2broad, grating, verbose=False)
    results["BIC_2broad"] = bic_2broad
    
    if verbose:
        print(f"  +both BROAD: BIC = {bic_2broad:.2f}" if np.isfinite(bic_2broad) else "  +both BROAD: (fit failed)")
    
    # 5. Select best model
    candidates = [(bic_narrow, "none", base_lines)]
    if np.isfinite(bic_1broad):
        candidates.append((bic_1broad, "one", lines_1broad))
    if np.isfinite(bic_b2only):
        candidates.append((bic_b2only, "broad2_only", lines_b2only))
    if np.isfinite(bic_2broad):
        candidates.append((bic_2broad, "two", lines_2broad))
    
    if broad_mode == "broad1":
        if np.isfinite(bic_1broad):
            results["broad_choice"] = "one"
            results["chosen_lines"] = lines_1broad
        else:
            if verbose:
                print("broad_mode='broad1' but fit failed → reverting to narrow-only.")
    elif broad_mode == "broad2":
        if np.isfinite(bic_b2only):
            results["broad_choice"] = "broad2_only"
            results["chosen_lines"] = lines_b2only
        else:
            if verbose:
                print("broad_mode='broad2' but fit failed → reverting to narrow-only.")
    elif broad_mode == "both":
        if np.isfinite(bic_2broad):
            results["broad_choice"] = "two"
            results["chosen_lines"] = lines_2broad
        else:
            if verbose:
                print("broad_mode='both' but fit failed → reverting to narrow-only.")
    else:
        # auto mode: pick lowest BIC with threshold
        if len(candidates) > 1:
            bics = [c[0] for c in candidates]
            i_best = int(np.argmin(bics))
            best_bic, best_choice, best_lines = candidates[i_best]
            
            # Only accept if improvement exceeds threshold
            if best_choice != "none":
                if np.isfinite(bic_narrow) and (best_bic + bic_delta >= bic_narrow):
                    if verbose:
                        print(f"Broad models do not improve BIC by >{bic_delta:.1f} → reverting to narrow-only.")
                    best_choice = "none"
                    best_lines = base_lines
            
            results["broad_choice"] = best_choice
            results["chosen_lines"] = best_lines
    
    if verbose:
        model_desc = {
            "none": "narrow-only",
            "one": "narrow + BROAD",
            "broad2_only": "narrow + BROAD2",
            "two": "narrow + BROAD + BROAD2"
        }.get(results["broad_choice"], results["broad_choice"])
        print(f"→ Selected model: {model_desc}")
        print("=" * 50)
    
    return results

# -----------------------------------------------------------------------------
# 4. Usage Wrapper
# -----------------------------------------------------------------------------

def plot_mcmc_detailed(lam, flux, err, model_total, profiles, z, title, unit="fnu"):
    """
    Generate the 4-panel plot (Main + 3 Zooms) matching PyRSR style exactly.
    """
    import matplotlib.gridspec as gridspec
    
    # Colors
    narrow_color = "tab:blue"
    broad_color  = "tab:orange"
    broad_color2 = "tab:pink"
    
    # 1. Setup Layout
    fig = plt.figure(figsize=(12.0, 8.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_z1 = fig.add_subplot(gs[1, 0])
    ax_z2 = fig.add_subplot(gs[1, 1])
    ax_z3 = fig.add_subplot(gs[1, 2])
    
    ylabel = r"$F_{\nu}$ [$\mu$Jy]" if unit == "fnu" else r"$F_{\lambda}$"
    
    # 2. Main Panel
    if np.all(np.isnan(flux)):
         print("Warning: All flux is NaN, skipping plot.")
         return

    dlam = np.median(np.diff(lam))
    edges = np.concatenate(([lam[0] - dlam/2], 0.5*(lam[1:]+lam[:-1]), [lam[-1] + dlam/2]))
    
    # Measurement error band
    if err is not None:
         ax_main.fill_between(lam, flux-err, flux+err, step="mid", color='grey', alpha=0.25, lw=0)
    
    ax_main.stairs(flux, edges, color='k', lw=0.5, alpha=0.7, label='Data')
    ax_main.plot(lam, model_total, color='#ff0000a6', lw=0.5, label='Mean model')
    
    # Decompose model components for visualization
    # Note: `profiles` contains arrays of each line's contribution
    narrow_sum = np.zeros_like(lam)
    broad_sum = np.zeros_like(lam)
    broad2_sum = np.zeros_like(lam)
    
    has_b1, has_b2 = False, False
    
    for name, prof in profiles.items():
        if "BROAD" in name.upper():
            if "BROAD2" in name.upper():
                broad2_sum += prof
                has_b2 = True
            else:
                broad_sum += prof
                has_b1 = True
        else:
            narrow_sum += prof
            
    # Add Component Fills (Main Panel)
    # Stacked order: Continuum (0 here) -> Narrow -> B1 -> B2
    # But broad_fit uses fill_between logic:
    # narrow: 0 -> narrow
    # b1: narrow -> narrow+b1
    # b2: narrow+b1 -> narrow+b1+b2
    
    base = np.zeros_like(lam)
    top_narrow = base + narrow_sum
    top_b1 = top_narrow + broad_sum
    top_b2 = top_b1 + broad2_sum
    
    if np.any(narrow_sum > 0):
        ax_main.fill_between(lam, base, top_narrow, color=narrow_color, alpha=0.35, lw=0, label="Narrow")
        # Dotted outline
        ax_main.plot(lam, top_narrow, color=narrow_color, ls=":", lw=1)
        
    if has_b1 and np.any(broad_sum > 0):
        ax_main.fill_between(lam, top_narrow, top_b1, color=broad_color, alpha=0.25, lw=0, label="Broad 1")
        ax_main.plot(lam, top_b1, color=broad_color, ls=":", lw=1)
        
    if has_b2 and np.any(broad2_sum > 0):
        ax_main.fill_between(lam, top_b1, top_b2, color=broad_color2, alpha=0.20, lw=0, label="Broad 2")
        ax_main.plot(lam, top_b2, color=broad_color2, ls=":", lw=1)
    
    ax_main.set_xlim(lam.min(), lam.max())
    ax_main.set_xlabel(r"Observed Wavelength [$\mu$m]")
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title, fontsize=12, pad=8)
    ax_main.legend(ncol=3, fontsize=9, frameon=False)
    ax_main.grid(alpha=0.25, linestyle=":", linewidth=0.5)
    ax_main.tick_params(direction="in", top=True, right=True)
    
    # Annotate Lines
    _annotate_lines_above_model(
        ax_main, lam, model_total,
        profiles.keys(), z, fontsize=8
    )
    
    # 4. Zoom Panels
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
            ax.text(0.5, 0.5, "No Coverage", ha='center', va='center', transform=ax.transAxes, color='0.5')
            ax.set_title(label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.3)
            continue
            
        lam_z = lam[mask]
        flux_z = flux[mask]
        # model components in zoom
        n_z = narrow_sum[mask]
        b1_z = broad_sum[mask]
        b2_z = broad2_sum[mask]
        
        base_z = np.zeros_like(lam_z)
        tn_z = base_z + n_z
        tb1_z = tn_z + b1_z
        tb2_z = tb1_z + b2_z
        
        err_z = err[mask] if err is not None else np.zeros_like(lam_z)
        edges_z = np.concatenate(([lam_z[0] - dlam/2], 0.5*(lam_z[1:]+lam_z[:-1]), [lam_z[-1] + dlam/2]))

        # Data
        ax.fill_between(lam_z, flux_z - err_z, flux_z + err_z, step="mid", color="grey", alpha=0.25, lw=0)
        ax.stairs(flux_z, edges_z, color="black", lw=0.5, alpha=0.7, label="Data")
        
        # Model Components (Filled + Dotted)
        if np.any(n_z > 0):
             ax.fill_between(lam_z, base_z, tn_z, color=narrow_color, alpha=0.35, lw=0)
             ax.plot(lam_z, tn_z, color=narrow_color, ls=":", lw=1)
             
        if has_b1 and np.any(b1_z > 0):
             ax.fill_between(lam_z, tn_z, tb1_z, color=broad_color, alpha=0.25, lw=0)
             ax.plot(lam_z, tb1_z, color=broad_color, ls=":", lw=1)
        
        if has_b2 and np.any(b2_z > 0):
             ax.fill_between(lam_z, tb1_z, tb2_z, color=broad_color2, alpha=0.20, lw=0)
             ax.plot(lam_z, tb2_z, color=broad_color2, ls=":", lw=1)
        
        # Total Model
        total_z = tb2_z
        ax.stairs(total_z, edges_z, color="#ff0000a6", lw=0.5, label="Mean model")

        ax.set_xlim(lo, hi)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Observed wavelength [µm]")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
        ax.grid(alpha=0.25, linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.tick_params(labelsize=8)
        
        # Annotate
        _annotate_lines_above_model(
            ax, lam_z, total_z,
            profiles.keys(), z, fontsize=8
        )
        
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
    
    print("\n=== MCMC FIT SUMMARY ===")
    header = (
        f"{'Line':<15} {'A_gauss':<20} {'σ_A [Å]':<20} "
        f"{'μ_obs [Å]':<15} {'SNR':<10} {'FWHM [Å]':<12}"
    )
    print(header)
    print("-" * 100)
    
    for name, info in lines.items():
        # Extract Gaussian parameters
        A_gauss = info.get("A_gauss", np.nan)
        A_err = info.get("A_gauss_err", np.nan)
        sigma = info.get("sigma_A", np.nan)
        sigma_err = info.get("sigma_A_err", np.nan)
        lam_obs = info.get("lam_obs_A", np.nan)
        snr = info.get("SNR", np.nan)
        fwhm = info.get("FWHM_A", np.nan)
        
        # Format Strings
        s_A = f"{A_gauss:.3e} ± {A_err:.3e}"
        s_sig = f"{sigma:.2f} ± {sigma_err:.2f}"
        s_mu = f"{lam_obs:.1f}"
        s_snr = f"{snr:.1f}"
        s_fwhm = f"{fwhm:.1f}"
        
        row = f"{name:<15} {s_A:<20} {s_sig:<20} {s_mu:<15} {s_snr:<10} {s_fwhm:<12}"
        print(row)
        
    print("-" * 100)
    print(f"z_fit = {z_fit:.5f}\n")

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
    plot_unit: str = "fnu", # "fnu" or "flam"
    broad_mode: str = "auto", # "auto", "off", "broad1", "broad2", "both"
    bic_delta: float = 6.0, # BIC improvement threshold
):
    """
    Main entry point for MCMC fitting with automatic BIC-based broad component selection.
    
    Args:
        spec_data: dict with 'lam' (um), 'flux', 'err'
        z_init: initial redshift guess
        line_names: list of line names to fit (narrow components)
        grating: e.g. "PRISM", "G395M"
        input_unit: Unit of input flux. Default 'fnu' (uJy).
        plot: Whether to generate a plot.
        plot_unit: Unit for the plot ('fnu' or 'flam').
        broad_mode: Broad component selection mode:
            - "auto": Use BIC to decide (default)
            - "off": No broad components
            - "broad1": Force narrow + BROAD
            - "broad2": Force narrow + BROAD2 (very broad)
            - "both": Force narrow + BROAD + BROAD2
        bic_delta: Minimum BIC improvement to prefer broad model (default 6.0)
        
    Returns:
        dict with keys 'chains', 'best_fit', 'success', 'BIC_*' scores
    """
    lam = spec_data["lam"]
    flux = spec_data["flux"]
    err = spec_data["err"]
    
    # BIC Model Selection for Balmer lines
    # Check which Balmer lines are in the user's list and run selection for each
    final_lines = list(line_names)
    bic_results = {}
    
    balmer_lines = ["H⍺", "HBETA", "HDELTA"]
    for bl in balmer_lines:
        if bl in final_lines:
            sel = select_broad_model(
                spec_data, z_init, final_lines,
                balmer_line=bl,
                grating=grating,
                broad_mode=broad_mode,
                bic_delta=bic_delta,
                verbose=verbose
            )
            # Update line list with selected model
            final_lines = list(sel["chosen_lines"])
            bic_results[bl] = sel
    
    if verbose:
        print(f"\nFinal line list: {final_lines}")
    
    # Estimate Inputs
    # 1. Sigma Inst
    lam_obs = np.array([REST_LINES_A.get(l, 0)*(1+z_init)/1e4 for l in final_lines])
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
        
    model_mgr = LineModel(final_lines, z_init, sigma_inst_A, flux_seeds)
    fitter = MCMCFitter(lam, flux, err, model_mgr)
    
    try:
        chain, sampler = fitter.run_emcee(n_walkers=n_walkers, n_steps=n_steps, verbose=verbose)
        
        # Best fit (median)
        p_med = np.median(chain, axis=0)
        p_std = np.std(chain, axis=0)
        
        # Extract fitted parameters
        nL = len(final_lines)
        dz = p_med[-1]
        z_fit = z_init + dz
        
        # Reconstruct result dict with proper Gaussian parameters
        res_lines = {}
        for i, name in enumerate(final_lines):
            # Fitted values
            A_gauss = p_med[i]  # Integrated flux (area under Gaussian)
            A_gauss_err = p_std[i]
            sigma_A = p_med[nL + i]  # Width in Angstroms
            sigma_A_err = p_std[nL + i]
            
            # Compute centroid from rest wavelength + fitted z
            lam0_A = REST_LINES_A.get(name, 0.0)
            lam_obs_A = lam0_A * (1 + z_fit)
            
            # Compute derived quantities
            SQRT2PI = np.sqrt(2 * np.pi)
            peak_amplitude = A_gauss / (SQRT2PI * sigma_A) if sigma_A > 0 else 0
            FWHM_A = 2.355 * sigma_A  # FWHM = 2*sqrt(2*ln(2))*sigma
            
            # SNR estimate (integrated)
            SNR = A_gauss / A_gauss_err if A_gauss_err > 0 else 0
            
            res_lines[name] = {
                # Gaussian Parameters (for plotting)
                "A_gauss": A_gauss,           # Integrated flux (erg/s/cm² or uJy*A)
                "A_gauss_err": A_gauss_err,
                "sigma_A": sigma_A,           # Line width in Angstroms
                "sigma_A_err": sigma_A_err,
                "lam_obs_A": lam_obs_A,       # Observed centroid in Angstroms
                
                # Derived quantities
                "peak_amplitude": peak_amplitude,  # Peak F_lambda
                "FWHM_A": FWHM_A,
                "F_line": A_gauss,            # Same as A_gauss (integrated flux)
                "sigma_line": A_gauss_err,    # Error on flux
                "SNR": SNR,
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
            "which_lines": final_lines,
            "z_fit": z_init + p_med[-1],
            "fit_stats": {"ndim": fitter.ndim, "n_steps": n_steps},
            "bic_selection": bic_results,
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


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

def run_mcmc_fit(
    spec_data: Dict[str, np.ndarray],
    z_init: float,
    line_names: List[str],
    grating: str = "PRISM",
    n_walkers=None, 
    n_steps=1000,
    verbose=True,
    input_unit: str = "fnu", # "fnu" (uJy) or "flam" (erg/s/cm2/A)
    plot: Optional[str] = None # "fnu" or "flam"
):
    """
    Main entry point for MCMC fitting.
    
    Args:
        spec_data: dict with 'lam' (um), 'flux', 'err'
        z_init: initial redshift guess
        line_names: list of line names to fit
        grating: e.g. "PRISM", "G395M"
        input_unit: Unit of input flux. Default 'fnu' (uJy).
        plot: If "fnu" or "flam", generate a plot overlaid with best fit.
        
    Returns:
        dict with keys 'chains', 'best_fit', 'success'
    """
    lam = spec_data["lam"]
    flux = spec_data["flux"]
    err = spec_data["err"]
    
    # Estimate Inputs
    # 1. Sigma Inst
    lam_obs = np.array([REST_LINES_A.get(l, 0)*(1+z_init)/1e4 for l in line_names])
    sigma_log = sigma_grating_logA(grating, lam_obs)
    # Convert log-sigma to Angstroms: sigma_A = lam_A * ln10 * sigma_log
    sigma_inst_A = (lam_obs * 1e4) * LN10 * sigma_log
    
    # 2. Flux Seeds (basic estimator)
    flux_seeds = []
    # If input is Fnu, we can just find peak in Fnu and guess.
    # The MCMC parameters 'Flux' will thus be in integrated units of (InputUnit * Angstrom).
    # e.g. if uJy -> uJy*A. If flam -> erg/s/cm2.
    
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
        if plot is not None:
            # Reconstruct Model on FULL grid (not just non-NaN)
            model_flux = model_mgr.compute_model(p_med, lam)
            
            # Prepare plotting arrays
            x_plot = lam
            y_data = flux
            y_model = model_flux
            
            # Unit conversions
            target_unit = plot.lower()
            current_unit = input_unit.lower()
            
            ylabel = "Flux Density"
            
            if target_unit == "flam":
                ylabel = r"$F_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"
                if current_unit == "fnu":
                    # Convert uJy -> flam
                    y_data = fnu_uJy_to_flam(y_data, x_plot)
                    # Model (which is fitted to fnu) also needs conversion
                    y_model = fnu_uJy_to_flam(y_model, x_plot)
            elif target_unit == "fnu":
                ylabel = r"$F_{\nu}$ [$\mu$Jy]"
                if current_unit == "flam":
                    # Convert flam -> uJy
                    y_data = flam_to_fnu_uJy(y_data, x_plot)
                    y_model = flam_to_fnu_uJy(y_model, x_plot)
            
            plt.figure(figsize=(10, 5))
            plt.step(x_plot, y_data, where='mid', color='k', alpha=0.5, label='Data', lw=1)
            plt.plot(x_plot, y_model, color='r', lw=2, alpha=0.8, label='MCMC Model')
            
            # Add residuals or fill? User asked for "overlaid".
            plt.title(f"MCMC Fit ({grating}) z={z_init + p_med[-1]:.4f}")
            plt.xlabel(r"Observed Wavelength [$\mu$m]")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return {
            "success": True,
            "chains": chain,
            "best_params": p_med,
            "lines": res_lines,
            "z_fit": z_init + p_med[-1],
            "fit_stats": {"ndim": fitter.ndim, "n_steps": n_steps}
        }
        
    except Exception as e:
        if verbose:
            print(f"MCMC Failed: {e}")
            import traceback
            traceback.print_exc()
        return {"success": False, "error": str(e)}

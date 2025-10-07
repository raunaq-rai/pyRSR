### potentially add in automatic detection - at the moment i am defining the centres beforehand

"""
Gaussian line fitting utilities (least-squares + MCMC) for noisy spectra.

Outputs (per line): flux, flux_err, EW, EW_err, SNR, mu (center), mu_err, sigma, sigma_err.
Continuum is modeled locally as linear (c0 + c1*(x - x_ref)) inside a window.

Dependencies:
  - numpy
  - scipy (for least squares)
  - emcee (optional, for MCMC)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

# --- optional deps guarded ---
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

try:
    import emcee
except Exception:
    emcee = None


# ======= your lines dictionary =======
lines_dict = {
    'LYA':1215.670,
    'NV_1':1238.821,
    'NV_2':1242.804,
    'NIV_1':1486.496,
    'CIV_1':1548.187,
    'CIV_2':1550.772,
    'HEII_1':1640.420,
    'OIII_05':1663.4795,
    'NIII_1':1746.823,
    'NIII_2':1748.656,
    'CIII':1908.734,
    'FeIV_1':2829.360,
    'FeIV_2':2835.740,
    'NeV_1':3345.821,
    'NeV_2':3425.881,
    'OII_UV_1':3727.092,
    'OII_UV_2':3729.875,
    'NEIII_UV_1':3869.86,
    'NEIII_UV_2':3968.59,
    'HDELTA':4102.8922,
    'HGAMMA':4341.6837,
    'OIII_1':4364.436,
    'HEI_1':4471.479,
    'HEII_2':4685.710,
    'HBETA':4862.6830,
    'OIII_2':4960.295,
    'OIII_3':5008.240,
    'HEI':5877.252,
    'HALPHA':6564.608,
    'SII_1':6718.295,
    'SII_2':6732.674
}


# ======= data container =======
@dataclass
class LineFitResult:
    method: str
    line_name: str | None
    mu: float                 # center wavelength
    mu_err: float
    sigma: float              # Gaussian sigma
    sigma_err: float
    amp: float                # peak amplitude (above continuum)
    amp_err: float
    flux: float               # integrated flux
    flux_err: float
    ew: float                 # equivalent width (same units as wavelength)
    ew_err: float
    snr: float                # flux / flux_err
    continuum_mu: float       # continuum level evaluated at mu
    cov: np.ndarray | None    # covariance (for least squares) else None
    meta: dict

    def as_dict(self):
        d = asdict(self)
        if isinstance(self.cov, np.ndarray):
            d["cov"] = self.cov.tolist()
        return d


# ======= model / helpers =======
def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / np.clip(sigma, 1e-9, np.inf)) ** 2)

def _gaussian_w_continuum(x, amp, mu, sigma, c0, c1):
    return _gaussian(x, amp, mu, sigma) + c0 + c1 * (x - x.mean())

def _sigma_clip(y, mask, lo=3.0, hi=3.0, iters=3):
    m = mask.copy()
    for _ in range(iters):
        med = np.nanmedian(y[m])
        std = 1.4826 * np.nanmedian(np.abs(y[m] - med))  # robust sigma (MAD)
        m &= (y > med - lo * std) & (y < med + hi * std)
    return m

def _continuum_linear(x, y, yerr, mask):
    """Robust weighted linear fit for sidebands: c0 + c1*(x-xref)."""
    if not np.any(mask):
        c0 = float(np.nanmedian(y))
        c1 = 0.0
        cov = np.array([[np.nan, np.nan],[np.nan, np.nan]])
        return c0, c1, cov

    xref = x.mean()
    X = np.vstack([np.ones(mask.sum()), (x[mask] - xref)]).T
    w = np.ones_like(y[mask])
    if yerr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.clip(yerr[mask], 1e-12, np.inf)**2

    XT_W = X.T * w
    cov = np.linalg.pinv(XT_W @ X)
    beta = cov @ (XT_W @ y[mask])
    c0, c1 = beta
    return float(c0), float(c1), cov

def _prep_window(wave, flux, err, center, window=20.0, cont_inner=6.0):
    """Return local arrays (x,y,e), continuum fit (c0,c1,cov), and initial guesses (amp0,mu0,sig0)."""
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    err  = None if err is None else np.asarray(err)

    sel = (wave > center - window) & (wave < center + window)
    if sel.sum() < 8:
        raise ValueError("Not enough points in window. Increase `window` or check wavelength grid.")
    x = wave[sel]
    y = flux[sel]
    e = None if err is None else err[sel]

    # sidebands for continuum: exclude core
    sb = (np.abs(x - center) >= cont_inner) & (np.abs(x - center) <= window)
    sb = _sigma_clip(y, sb)
    c0, c1, cov_c = _continuum_linear(x, y, e, sb)

    # subtract continuum to init
    y_cont = y - (c0 + c1*(x - x.mean()))
    amp0 = float(np.nanmax(y_cont))
    try:
        mu0 = float(x[np.nanargmax(y_cont)])
    except Exception:
        mu0 = float(center)
    # second moment estimate (guarded)
    wpos = np.clip(y_cont - np.nanmin(y_cont), 0, np.inf)
    var = np.average((x - mu0)**2, weights=np.clip(wpos, 1e-12, np.inf))
    sig0 = float(np.sqrt(np.clip(var, 0.25, (0.25*window)**2)))  # min ~0.5 in wavelength units
    return x, y, e, (c0, c1, cov_c), (amp0, mu0, sig0)


# ======= least-squares =======
def fit_line_leastsq(
    wave, flux, err, center, window=20.0, cont_inner=6.0, name=None, bounds=None
) -> LineFitResult:
    """
    Weighted least-squares fit of Gaussian + linear continuum in a local window.
    `bounds` example: {'amp': (0,None), 'mu': (center-5, center+5), 'sigma': (0.05, 20)}
    """
    if curve_fit is None:
        raise ImportError("scipy is required: pip install scipy")

    x, y, e, (c0, c1, cov_c), (amp0, mu0, sig0) = _prep_window(wave, flux, err, center, window, cont_inner)
    p0 = [amp0, mu0, sig0, c0, c1]

    lb = [-np.inf, center - window, 0.05, -np.inf, -np.inf]
    ub = [ np.inf, center + window, window,  np.inf,  np.inf]
    if bounds:
        if "amp" in bounds:
            lb[0] = bounds["amp"][0] if bounds["amp"][0] is not None else lb[0]
            ub[0] = bounds["amp"][1] if bounds["amp"][1] is not None else ub[0]
        if "mu" in bounds:
            lb[1] = bounds["mu"][0] if bounds["mu"][0] is not None else lb[1]
            ub[1] = bounds["mu"][1] if bounds["mu"][1] is not None else ub[1]
        if "sigma" in bounds:
            lb[2] = bounds["sigma"][0] if bounds["sigma"][0] is not None else lb[2]
            ub[2] = bounds["sigma"][1] if bounds["sigma"][1] is not None else ub[2]

    sigma_w = None if e is None else e
    popt, pcov = curve_fit(
        _gaussian_w_continuum, x, y, p0=p0,
        sigma=sigma_w, absolute_sigma=True,
        bounds=(lb, ub), maxfev=20000
    )
    amp, mu, sig, c0_fit, c1_fit = popt
    perr = np.sqrt(np.diag(pcov))

    # flux, EW, SNR
    S = np.sqrt(2.0*np.pi)
    flux_line = float(S * amp * sig)

    # propagate flux error from (amp, sigma) covariance
    J = np.array([S*sig, S*amp])
    cov_as = pcov[np.ix_([0,2],[0,2])]
    flux_err = float(np.sqrt(J @ cov_as @ J))

    cont_at_mu = float(c0_fit + c1_fit*(mu - x.mean()))
    ew = flux_line / max(cont_at_mu, 1e-30)
    ew_err = abs(ew) * (flux_err / max(abs(flux_line), 1e-30))

    return LineFitResult(
        method="leastsq",
        line_name=name,
        mu=float(mu),         mu_err=float(perr[1]),
        sigma=float(sig),     sigma_err=float(perr[2]),
        amp=float(amp),       amp_err=float(perr[0]),
        flux=float(flux_line),flux_err=float(flux_err),
        ew=float(ew),         ew_err=float(ew_err),
        snr=float(flux_line / max(flux_err,1e-30)),
        continuum_mu=cont_at_mu,
        cov=pcov,
        meta={"window": window, "cont_inner": cont_inner, "init": [amp0, mu0, sig0]}
    )


# ======= MCMC (emcee) =======
def _log_prior(theta, center, window, amp_prior=(0, np.inf), sigma_prior=(0.05, None)):
    amp, mu, sig, c0, c1 = theta
    if not (amp_prior[0] <= amp <= (np.inf if amp_prior[1] is None else amp_prior[1])):
        return -np.inf
    if not (center - window <= mu <= center + window):
        return -np.inf
    upper_sig = window if sigma_prior[1] is None else sigma_prior[1]
    if not (sigma_prior[0] <= sig <= upper_sig):
        return -np.inf
    return 0.0

def _log_likelihood(theta, x, y, e):
    amp, mu, sig, c0, c1 = theta
    model = _gaussian_w_continuum(x, amp, mu, sig, c0, c1)
    if e is None:
        r = y - model
        return -0.5*np.sum(r**2)
    invvar = 1.0 / np.clip(e, 1e-12, np.inf)**2
    return -0.5*np.sum((y - model)**2 * invvar + np.log(2*np.pi) - np.log(invvar))

def _log_posterior(theta, x, y, e, center, window, amp_prior, sigma_prior):
    lp = _log_prior(theta, center, window, amp_prior, sigma_prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, x, y, e)

def fit_line_mcmc(
    wave, flux, err, center, window=20.0, cont_inner=6.0, name=None,
    nwalkers=32, nsteps=2000, nburn=500, amp_prior=(0.0, None), sigma_prior=(0.05, None),
    random_state=None
) -> LineFitResult:
    if emcee is None:
        raise ImportError("emcee is required: pip install emcee")

    x, y, e, (c0, c1, _), (amp0, mu0, sig0) = _prep_window(wave, flux, err, center, window, cont_inner)
    rng = np.random.default_rng(random_state)
    p0_center = np.array([amp0, mu0, sig0, c0, c1], dtype=float)
    p0_spread = np.array([
        max(1e-3, abs(amp0)*0.1),
        max(1e-3, window*0.02),
        max(0.02, sig0*0.2),
        max(1e-3, np.std(y)*0.1 + abs(c0)*0.1),
        max(1e-6, 0.1*abs(c1) + 1e-4)
    ], dtype=float)
    p0 = p0_center + rng.normal(scale=p0_spread, size=(nwalkers, 5))

    sampler = emcee.EnsembleSampler(
        nwalkers, 5, _log_posterior,
        args=(x, y, e, center, window, amp_prior, sigma_prior)
    )
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = sampler.get_chain(discard=nburn, flat=True)

    q16, q50, q84 = np.percentile(chain, [16,50,84], axis=0)
    med = q50
    perr = 0.5*(q84 - q16)
    amp, mu, sig, c0_fit, c1_fit = med

    # flux / EW from posterior
    flux_samples = np.sqrt(2.0*np.pi) * chain[:,0] * chain[:,2]
    flux_line = float(np.median(flux_samples))
    flux_err  = float(np.std(flux_samples))

    cont_mu_samples = c0_fit + c1_fit*(chain[:,1] - x.mean())
    ew_samples = flux_samples / np.maximum(cont_mu_samples, 1e-30)
    ew = float(np.median(ew_samples))
    ew_err = float(np.std(ew_samples))

    cont_at_mu = float(c0_fit + c1_fit*(mu - x.mean()))

    cont_at_mu = float(c0_fit + c1_fit * (mu - x.mean()))

    # Build result object
    result = LineFitResult(
        method="mcmc",
        line_name=name,
        mu=float(mu),         mu_err=float(perr[1]),
        sigma=float(sig),     sigma_err=float(perr[2]),
        amp=float(amp),       amp_err=float(perr[0]),
        flux=float(flux_line),flux_err=float(flux_err),
        ew=float(ew),         ew_err=float(ew_err),
        snr=float(flux_line / max(flux_err, 1e-30)),
        continuum_mu=cont_at_mu,
        cov=None,
        meta={
            "window": window, "cont_inner": cont_inner,
            "nwalkers": nwalkers, "nsteps": nsteps, "nburn": nburn,
            "amp_prior": amp_prior, "sigma_prior": sigma_prior
        }
    )

    # --- NEW additions ---
    result.samples = chain           # full posterior samples for corner plot
    result.continuum = (c0_fit, c1_fit)  # continuum slope/intercept

    return result



# ======= batch runner =======
def fit_lines_batch(wave, flux, err, line_centers: dict[str, float], method="leastsq", **kwargs):
    """Fit many lines from {name: center}. kwargs forwarded to fitter."""
    results = {}
    for name, center in line_centers.items():
        try:
            if method.lower() in ("lsq", "leastsq", "least_squares"):
                res = fit_line_leastsq(wave, flux, err, center=center, name=name, **kwargs)
            elif method.lower() == "mcmc":
                res = fit_line_mcmc(wave, flux, err, center=center, name=name, **kwargs)
            else:
                raise ValueError("method must be 'leastsq' or 'mcmc'")
            results[name] = res
        except Exception as e:
            results[name] = e  # store the exception so you can inspect failures
    return results


    # -----------------------------
    # 4. Bootstrap (parametric)
    # -----------------------------
def bootstrap_lines(wave, flux, err, line_centers, B=200, method="leastsq",
                    random_state=None, window=20.0, cont_inner=6.0, fitter_kwargs=None):
    """
    Parametric bootstrap: perturb flux with Gaussian noise ~N(0, err),
    refit all lines, and derive uncertainties on flux, EW, μ.
    """
    rng = np.random.default_rng(random_state)
    fitter_kwargs = {} if fitter_kwargs is None else dict(fitter_kwargs)

    results_all = {k: {"flux": [], "ew": [], "mu": []} for k in line_centers.keys()}

    for _ in range(B):
        flux_b = flux + rng.normal(0.0, np.asarray(err))
        batch = fit_lines_batch(
            wave, flux_b, err, line_centers,
            method=method, window=window, cont_inner=cont_inner, **fitter_kwargs
        )
        for name, res in batch.items():
            if isinstance(res, Exception):
                continue
            results_all[name]["flux"].append(res.flux)
            results_all[name]["ew"].append(res.ew)
            results_all[name]["mu"].append(res.mu)

    # Summarise results (median ± std)
    summary = {}
    for name, vals in results_all.items():
        summary[name] = {}
        for key, arr in vals.items():
            arr = np.asarray(arr)
            summary[name][key] = (np.median(arr), np.std(arr))
    return summary




import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:
    corner = None
    






# ======= MULTI-LINE DEMO =======
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(123)

    # -----------------------------
    # 1. Simulate synthetic spectrum
    # -----------------------------
    wave = np.linspace(4800, 5100, 3000)
    cont_c0, cont_c1 = 10.0, 0.002
    flux = cont_c0 + cont_c1 * (wave - wave.mean())

    # True Gaussian lines (μ, amp, σ)
    true_lines = {
        "[OIII]4959": (4960.295, 4.5, 1.4),
        "[OIII]5008": (5008.240, 5.0, 1.6),
        "Hβ": (4862.683, 3.5, 2.0)
    }

    for mu, amp, sig in true_lines.values():
        flux += amp * np.exp(-0.5 * ((wave - mu) / sig) ** 2)

    err = np.full_like(flux, 0.4)
    flux += rng.normal(0, err)

    # -----------------------------
    # 2. Least-squares fits
    # -----------------------------
    centers = {k: v[0] for k, v in true_lines.items()}
    results_lsq = fit_lines_batch(wave, flux, err, centers, method="leastsq", window=20.0, cont_inner=6.0)

    print("\n=== LEAST-SQUARES RESULTS ===")
    for name, res in results_lsq.items():
        if isinstance(res, Exception):
            print(f"{name}: FAILED ({res})")
        else:
            print(f"{name}: μ={res.mu:.2f}±{res.mu_err:.2f}, σ={res.sigma:.2f}±{res.sigma_err:.2f}, "
                  f"Flux={res.flux:.2f}±{res.flux_err:.2f}, SNR={res.snr:.1f},EW={res.ew:.2f}±{res.ew_err:.2f}")
            
        # Run bootstrap after LSQ fits
    print("\n=== BOOTSTRAP (parametric, B=200) ===")
    boot_summary = bootstrap_lines(
        wave, flux, err, centers,
        B=200, method="leastsq",
        random_state=42, window=20.0, cont_inner=6.0
    )

    for name, stats in boot_summary.items():
        f, f_std = stats["flux"]
        ew, ew_std = stats["ew"]
        mu, mu_std = stats["mu"]
        print(f"{name}: Flux={f:.3f}±{f_std:.3f}, EW={ew:.3f}±{ew_std:.3f}, μ={mu:.3f}±{mu_std:.3f}")

    # -----------------------------
    # 3. MCMC fits for all lines
    # -----------------------------
    if emcee is not None:
        results_mcmc = fit_lines_batch(
            wave, flux, err, centers,
            method="mcmc", window=20.0, cont_inner=6.0,
            nwalkers=12, nsteps=10000, nburn=5000, random_state=11
        )

        print("\n=== MCMC RESULTS ===")
        for name, res in results_mcmc.items():
            if isinstance(res, Exception):
                print(f"{name}: FAILED ({res})")
            else:
                print(f"{name}: μ={res.mu:.2f}±{res.mu_err:.2f}, σ={res.sigma:.2f}±{res.sigma_err:.2f}, "
                      f"Flux={res.flux:.2f}±{res.flux_err:.2f}, SNR={res.snr:.1f},EW={res.ew:.2f}±{res.ew_err:.2f}")

        # Optional: corner plot for one representative line
        if corner is not None:
            target_line = "[OIII]5008"
            res = results_mcmc[target_line]
            if hasattr(res, "samples"):
                labels = [r"$A$", r"$\mu$", r"$\sigma$", r"$c_0$", r"$c_1$"]
                fig = corner.corner(
                    res.samples,
                    labels=labels,
                    truths=[res.amp, res.mu, res.sigma, *res.continuum],
                    show_titles=True,
                    title_fmt=".3f",
                    quantiles=[0.16, 0.5, 0.84],
                    color="royalblue",
                    hist_kwargs={"density": True, "color": "lightblue"},
                    size=(8, 8)
                )
                fig.suptitle(f"MCMC Posterior Corner Plot: {target_line}", fontsize=12)
                plt.show()

        # -----------------------------
    # 4. Combined plot with LSQ, MCMC, and Bootstrap uncertainties
    # -----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(wave, flux, color="gray", lw=0.6, label="Synthetic data")

    # --- LSQ (red dashed + shaded ±μ_err) ---
    for name, res in results_lsq.items():
        if not isinstance(res, Exception):
            plt.axvline(res.mu, color="red", ls="--", alpha=0.7)
            plt.axvspan(
                res.mu - res.mu_err, res.mu + res.mu_err,
                color="red", alpha=0.15
            )
    lsq_proxy_line = plt.Line2D([0], [0], color="red", ls="--", lw=1.2, label="LSQ (±μ_err)")

    # --- MCMC (blue dotted + shaded ±μ_err) ---
    if emcee is not None and "results_mcmc" in locals():
        for name, res in results_mcmc.items():
            if not isinstance(res, Exception):
                plt.axvline(res.mu, color="blue", ls=":", alpha=0.7)
                plt.axvspan(
                    res.mu - res.mu_err, res.mu + res.mu_err,
                    color="blue", alpha=0.15
                )
        mcmc_proxy_line = plt.Line2D([0], [0], color="blue", ls=":", lw=1.2, label="MCMC (±μ_err)")
    else:
        mcmc_proxy_line = None

    # --- Bootstrap (gold band ±μ_std) ---
    if "boot_summary" in locals():
        for name, stats in boot_summary.items():
            mu, mu_std = stats["mu"]
            plt.axvspan(
                mu - mu_std, mu + mu_std,
                color="gold", alpha=0.25
            )
            plt.axvline(mu, color="goldenrod", lw=1.2, ls="-", alpha=0.6)
        boot_proxy_patch = plt.Rectangle((0, 0), 1, 1, color="gold", alpha=0.25, label="Bootstrap (±μ_std)")
    else:
        boot_proxy_patch = None

    # --- Plot formatting ---
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title("Synthetic Spectrum: LSQ, MCMC, and Bootstrap Fits with Uncertainties")

    # --- Legend (grouped by method) ---
    legend_elems = [plt.Line2D([0], [0], color="gray", lw=0.6, label="Synthetic data")]
    legend_elems.append(lsq_proxy_line)
    if mcmc_proxy_line:
        legend_elems.append(mcmc_proxy_line)
    if boot_proxy_patch:
        legend_elems.append(boot_proxy_patch)

    plt.legend(
        handles=legend_elems,
        fontsize=8, loc="best", frameon=False
    )

    plt.tight_layout()
    plt.show()

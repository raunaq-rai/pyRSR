"""
linefit.py — Gaussian line fitting (least-squares, MCMC, bootstrap, Monte Carlo)
================================================================================

Fits a Gaussian emission line plus linear continuum.

Model:
  F(λ) = A * exp[-(λ - μ)^2 / (2σ^2)] + (c₀ + c₁*(λ - λ̄))

Parameters:
  A       amplitude of the line
  μ       line centre (wavelength)
  σ       Gaussian width (Å)
  c₀, c₁  linear continuum coefficients

Outputs per line (LineFitResult):
  μ ± μ_err, σ ± σ_err, flux ± flux_err, EW ± EW_err, SNR, continuum(μ)
Dependencies: numpy, scipy, optional emcee
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict

# --- optional deps ---
try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None
try:
    import emcee
except ImportError:
    emcee = None


# ==================================================================
# --- Models & utilities
# ==================================================================
def _gauss(x, amp, mu, sig):
    """Pure Gaussian profile."""
    return amp * np.exp(-0.5 * ((x - mu) / np.clip(sig, 1e-9, np.inf)) ** 2)


def _gauss_lin(x, amp, mu, sig, c0, c1):
    """Gaussian + linear continuum."""
    return _gauss(x, amp, mu, sig) + (c0 + c1 * (x - x.mean()))


def _continuum(x, y, e, mask):
    """Weighted linear fit y = c0 + c1*(x - xmean) on sidebands."""
    if mask.sum() < 3:
        return np.nan, 0.0
    w = 1 / np.clip(e[mask], 1e-12, np.inf)**2
    X = np.vstack([np.ones(mask.sum()), (x[mask] - x.mean())]).T
    beta, *_ = np.linalg.lstsq(X * np.sqrt(w[:, None]), y[mask] * np.sqrt(w), rcond=None)
    return beta[0], beta[1]


def _sigma_clip(y, mask, lo=3.0, hi=3.0, iters=3):
    """Sigma-clip a boolean mask."""
    m = mask.copy()
    for _ in range(iters):
        if not np.any(m):
            break
        med = np.nanmedian(y[m])
        mad = np.nanmedian(np.abs(y[m] - med))
        std = 1.4826 * max(mad, 1e-20)
        m &= (y > med - lo*std) & (y < med + hi*std)
    return m


def _prep_window(w, f, e, center, win=20, inner=6):
    """Select fitting window and initial guesses."""
    sel = (w > center - win) & (w < center + win)
    if sel.sum() < 8:
        raise ValueError(f"Too few points near {center}")
    x, y, err = w[sel], f[sel], e[sel]
    cont_mask = _sigma_clip(y, np.abs(x - center) > inner)
    c0, c1 = _continuum(x, y, err, cont_mask)
    y_sub = y - (c0 + c1 * (x - x.mean()))
    amp0 = np.nanmax(y_sub)
    mu0 = x[np.nanargmax(y_sub)]
    sig0 = 1.5  # Å — reasonable starting guess
    return x, y, err, (c0, c1), (amp0, mu0, sig0)


def _ew_uncertainty(flux, flux_err, c0, c1, mu, xmean, cov_c=None):
    """Compute EW and its uncertainty (optionally propagating continuum covariance)."""
    dx = mu - xmean
    c_mu = np.clip(abs(c0 + c1 * dx), 1e-10, np.inf)
    var_c = 0.0
    if cov_c is not None and np.all(np.isfinite(cov_c)):
        var_c = cov_c[0,0] + dx**2*cov_c[1,1] + 2*dx*cov_c[0,1]
        var_c = max(var_c, 0.0)
    sigma_c = np.sqrt(var_c)
    ew = flux / c_mu
    ew_err = abs(ew) * np.sqrt((flux_err / flux)**2 + (sigma_c / c_mu)**2)
    return ew, ew_err


# ==================================================================
# --- Result container
# ==================================================================
@dataclass
class LineFitResult:
    method: str
    name: str
    mu: float
    mu_err: float
    sigma: float
    sigma_err: float
    flux: float
    flux_err: float
    ew: float
    ew_err: float
    snr: float
    cont_mu: float
    samples: Optional[np.ndarray] = None

    def as_dict(self):
        d = asdict(self)
        if isinstance(self.samples, np.ndarray):
            d["samples"] = None
        return d


# ==================================================================
# --- Fitting methods
# ==================================================================
def fit_line_lsq(w, f, e, center, name=None, window=20, inner=6):
    """Least-squares Gaussian+linear continuum fit."""
    if curve_fit is None:
        raise ImportError("scipy required for LSQ")

    x, y, err, (c0, c1), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)
    p0 = [A0, mu0, sig0, c0, c1]
    popt, pcov = curve_fit(_gauss_lin, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
    A, mu, sig, c0, c1 = popt
    perr = np.sqrt(np.diag(pcov))

    flux = np.sqrt(2 * np.pi) * A * sig
    flux_err = flux * np.sqrt((perr[0]/A)**2 + (perr[2]/sig)**2)
    cont_mu = c0 + c1 * (mu - x.mean())
    cov_c = pcov[3:5, 3:5]
    ew, ew_err = _ew_uncertainty(flux, flux_err, c0, c1, mu, x.mean(), cov_c)

    return LineFitResult("leastsq", name, mu, perr[1], sig, perr[2],
                         flux, flux_err, ew, ew_err, flux/flux_err, cont_mu)


def fit_line_bootstrap(w, f, e, center, name=None, B=200, **kw):
    """Bootstrap resampling using repeated LSQ fits."""
    rng = np.random.default_rng(42)
    fluxes, ews, mus, sigmas = [], [], [], []
    for _ in range(B):
        fb = f + rng.normal(0, e)
        try:
            r = fit_line_lsq(w, fb, e, center, name=name, **kw)
            fluxes.append(r.flux)
            ews.append(r.ew)
            mus.append(r.mu)
            sigmas.append(r.sigma)
        except Exception:
            continue
    if not fluxes:
        raise RuntimeError("Bootstrap failed")

    def robust_std(a):
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        return 1.4826 * mad

    f_med, f_std = np.median(fluxes), robust_std(fluxes)
    ew_med, ew_std = np.median(ews), robust_std(ews)
    mu_med, mu_std = np.median(mus), robust_std(mus)
    sig_med, sig_std = np.median(sigmas), robust_std(sigmas)
    snr = f_med / f_std if f_std > 0 else np.nan
    cont_mu0 = fit_line_lsq(w, f, e, center, name=name, **kw).cont_mu
    return LineFitResult("bootstrap", name, mu_med, mu_std, sig_med, sig_std,
                         f_med, f_std, ew_med, ew_std, snr, cont_mu0)


def fit_line_mcmc(w, f, e, center, name=None, window=20, inner=6,
                  nwalkers=32, nsteps=2000, nburn=500):
    """MCMC Gaussian+continuum fit (requires emcee)."""
    if emcee is None:
        raise ImportError("emcee not installed")

    x, y, err, (c0, c1), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)

    def log_prior(t):
        A, mu, sig, c0_, c1_ = t
        if A <= 0 or sig <= 0 or sig > window:
            return -np.inf
        return 0.0

    def log_like(t):
        model = _gauss_lin(x, *t)
        return -0.5 * np.sum(((y - model) / err) ** 2)

    def log_post(t):
        lp = log_prior(t)
        return lp + log_like(t) if np.isfinite(lp) else -np.inf

    # Initial ensemble with positive amplitudes and widths
    p0 = np.array([max(A0, 1e-6), mu0, abs(sig0), c0, c1])
    rng = np.random.default_rng(123)
    p0s = []
    for _ in range(nwalkers):
        A = abs(p0[0] * (1 + 0.1 * rng.standard_normal()))
        mu = p0[1] + 0.1 * rng.standard_normal()
        sig = abs(p0[2] * (1 + 0.1 * rng.standard_normal()))
        c0_ = p0[3] * (1 + 0.1 * rng.standard_normal())
        c1_ = p0[4] * (1 + 0.1 * rng.standard_normal())
        p0s.append([A, mu, sig, c0_, c1_])
    p0s = np.array(p0s)

    sampler = emcee.EnsembleSampler(nwalkers, 5, log_post)
    sampler.run_mcmc(p0s, nsteps, progress=False)

    chain = sampler.get_chain(discard=nburn, flat=True)
    med, std = np.median(chain, axis=0), np.std(chain, axis=0)
    A, mu, sig, c0, c1 = med
    flux = np.sqrt(2 * np.pi) * A * sig
    flux_err = flux * np.sqrt((std[0]/A)**2 + (std[2]/sig)**2)
    cont_mu = c0 + c1*(mu - x.mean())
    ew, ew_err = _ew_uncertainty(flux, flux_err, c0, c1, mu, x.mean())

    return LineFitResult("mcmc", name, mu, std[1], sig, std[2],
                         flux, flux_err, ew, ew_err, flux/flux_err, cont_mu, samples=chain)


def fit_line_mc(w, f, e, center, name=None, window=20, inner=6, nsamp=2000):
    """Parametric Monte Carlo around LSQ solution using LSQ covariance."""
    if curve_fit is None:
        raise ImportError("scipy required for MC")

    base = fit_line_lsq(w, f, e, center, name=name, window=window, inner=inner)
    x, y, err, (c0b, c1b), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)
    p0 = [A0, mu0, sig0, c0b, c1b]
    popt, pcov = curve_fit(_gauss_lin, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
    if not np.all(np.isfinite(pcov)):
        return base

    rng = np.random.default_rng(12345)
    draws = rng.multivariate_normal(popt, pcov, size=nsamp)
    fluxes, ews, mus, sigmas = [], [], [], []
    xbar = x.mean()
    for A, mu, sig, c0, c1 in draws:
        if A <= 0 or sig <= 0:
            continue
        flux = np.sqrt(2*np.pi)*A*sig
        c_mu = np.clip(abs(c0 + c1*(mu - xbar)), 1e-10, np.inf)
        ew = flux / c_mu
        fluxes.append(flux)
        ews.append(ew)
        mus.append(mu)
        sigmas.append(sig)

    def qstats(a):
        a = np.asarray(a)
        if a.size == 0:
            return np.nan, np.nan
        return np.median(a), 0.5*(np.percentile(a,84)-np.percentile(a,16))

    f_med, f_err = qstats(fluxes)
    ew_med, ew_err = qstats(ews)
    mu_med, mu_err = qstats(mus)
    sig_med, sig_err = qstats(sigmas)
    snr = f_med / f_err if np.isfinite(f_err) and f_err>0 else np.nan
    cont_mu = base.cont_mu

    return LineFitResult("montecarlo", name, mu_med, mu_err, sig_med, sig_err,
                         f_med, f_err, ew_med, ew_err, snr, cont_mu)


# ==================================================================
# --- Multi-line convenience
# ==================================================================
def fit_lines_all(w, f, e, lines: Dict[str, float], **kw):
    out = {"leastsq": {}, "bootstrap": {}, "mcmc": {}, "montecarlo": {}}
    for name, cen in lines.items():
        try:
            out["leastsq"][name] = fit_line_lsq(w, f, e, cen, name=name, **kw)
        except Exception as ex:
            out["leastsq"][name] = ex
        try:
            out["bootstrap"][name] = fit_line_bootstrap(w, f, e, cen, name=name, **kw)
        except Exception as ex:
            out["bootstrap"][name] = ex
        try:
            out["montecarlo"][name] = fit_line_mc(w, f, e, cen, name=name, **kw)
        except Exception as ex:
            out["montecarlo"][name] = ex
        if emcee:
            try:
                out["mcmc"][name] = fit_line_mcmc(w, f, e, cen, name=name, **kw)
            except Exception as ex:
                out["mcmc"][name] = ex
    return out



# ==================================================================
# --- Demo / main block
# ==================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        import corner
    except ImportError:
        corner = None

    # --------------------------------------------------------------
    # 1. Simulate synthetic spectrum
    # --------------------------------------------------------------
    rng = np.random.default_rng(123)
    wave = np.linspace(4800, 5100, 3000)
    continuum = 10 + 0.002 * (wave - wave.mean())
    flux = continuum.copy()

    # Add Gaussian emission lines
    true_lines = [
        ("Hβ", 4862.7, 3.5, 2.0),
        ("[OIII]4959", 4960.3, 4.5, 1.4),
        ("[OIII]5008", 5008.2, 5.0, 1.6),
    ]
    for _, mu, A, sig in true_lines:
        flux += A * np.exp(-0.5 * ((wave - mu) / sig) ** 2)

    # Add Gaussian noise
    err = np.full_like(flux, 0.4)
    flux += rng.normal(0, err)

    # --------------------------------------------------------------
    # 2. Define target lines
    # --------------------------------------------------------------
    lines = {n: mu for n, mu, _, _ in true_lines}

    # --------------------------------------------------------------
    # 3. Fit lines with all methods
    # --------------------------------------------------------------
    print("=== Fitting synthetic spectrum ===")
    res = fit_lines_all(wave, flux, err, lines, window=20)

    # --------------------------------------------------------------
    # 4. Print summary table
    # --------------------------------------------------------------
    for method, rd in res.items():
        print(f"\n=== {method.upper()} RESULTS ===")
        for name, r in rd.items():
            if isinstance(r, Exception):
                print(f"{name:10s} FAIL ({r})")
            else:
                print(
                    f"{name:10s} μ={r.mu:7.2f}±{r.mu_err:5.2f}  "
                    f"σ={r.sigma:4.2f}±{r.sigma_err:4.2f}  "
                    f"F={r.flux:7.2f}±{r.flux_err:5.2f}  "
                    f"EW={r.ew:7.2f}±{r.ew_err:5.2f}  "
                    f"SNR={r.snr:5.1f}"
                )

    # --------------------------------------------------------------
    # 5. Plot data and model fits
    # --------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.plot(wave, flux, color="gray", lw=0.8, label="Data")

    colors = {
        "leastsq": "crimson",
        "bootstrap": "goldenrod",
        "montecarlo": "teal",
        "mcmc": "royalblue",
    }
    linestyles = {"leastsq": "--", "bootstrap": "-.", "montecarlo": ":", "mcmc": "-"}

    for method, rd in res.items():
        for name, r in rd.items():
            if isinstance(r, Exception):
                continue
            A = r.flux / (np.sqrt(2 * np.pi) * r.sigma)
            model = r.cont_mu + A * np.exp(-0.5 * ((wave - r.mu) / r.sigma) ** 2)
            plt.plot(
                wave,
                model,
                color=colors[method],
                ls=linestyles[method],
                lw=1.2,
                label=f"{name} ({method})" if name == "[OIII]5008" else None,
            )

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux")
    plt.title("Gaussian Line Fits (synthetic test)")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # 6. Corner plot for one line (optional)
    # --------------------------------------------------------------
    if emcee and corner and "mcmc" in res and "[OIII]5008" in res["mcmc"]:
        r = res["mcmc"]["[OIII]5008"]
        if isinstance(r, LineFitResult) and r.samples is not None:
            fig = corner.corner(
                r.samples,
                labels=["A", "μ", "σ", "c0", "c1"],
                show_titles=True,
                title_fmt=".3e",
            )
            fig.suptitle("[OIII]5008 MCMC Posterior", fontsize=13)
            plt.show()

"""
line_fitting.py — Gaussian line fitting (least-squares, MCMC, bootstrap, Monte Carlo)
====================================================================================

Fits emission lines in astronomical spectra using a Gaussian model plus a
linear continuum:

    F(λ) = A * exp[-(λ - μ)² / (2σ²)] + (c₀ + c₁ * (λ - λ̄))

Continuum
---------
The continuum (c₀ + c₁*(λ - λ̄)) represents the baseline emission
(typically stellar or nebular background) approximated as a straight line.
A single global linear continuum is first fitted to the full spectrum and
then used for all line fits (so every line shares exactly the same continuum).
If a global continuum is not provided, each line fit will (for backward
compatibility) fit its own local continuum.

Outputs per line (LineFitResult)
--------------------------------
μ ± μ_err, σ ± σ_err, flux ± flux_err, EW ± EW_err, SNR, continuum(μ)

Fitting methods
---------------
1. Least-squares         — χ² minimisation
2. Bootstrap resampling  — noise resampling
3. Monte Carlo sampling  — draws from LSQ covariance
4. MCMC sampling         — full posterior exploration (Bayesian)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

# Optional dependencies
try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None
try:
    import emcee
except ImportError:
    emcee = None


# ==================================================================
# --- Models and helpers
# ==================================================================
def _gauss(x: np.ndarray, amp: float, mu: float, sig: float) -> np.ndarray:
    """Pure Gaussian profile."""
    sig = np.clip(sig, 1e-12, np.inf)
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def _gauss_lin(x: np.ndarray, amp: float, mu: float, sig: float, c0: float, c1: float) -> np.ndarray:
    """Gaussian emission line + linear continuum."""
    return _gauss(x, amp, mu, sig) + (c0 + c1 * (x - x.mean()))


def _fit_global_continuum(w: np.ndarray, f: np.ndarray, e: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Fit a global linear continuum f = c₀ + c₁*(λ - λ̄)."""
    if mask is None:
        mask = np.isfinite(f)
    w = np.asarray(w, float)
    f = np.asarray(f, float)
    e = np.asarray(e if e is not None else np.ones_like(f), float)

    wmean = np.nanmean(w)
    w_rel = w[mask] - wmean
    W = np.vstack([np.ones_like(w_rel), w_rel]).T
    weights = 1 / np.clip(e[mask], 1e-12, np.inf) ** 2
    beta, *_ = np.linalg.lstsq(W * np.sqrt(weights[:, None]),
                               f[mask] * np.sqrt(weights),
                               rcond=None)
    return float(beta[0]), float(beta[1])


def _prep_window(w, f, e, center, win=20, inner=6):
    """Extract window around line and get initial parameter guesses."""
    sel = (w > center - win) & (w < center + win)
    if sel.sum() < 8:
        raise ValueError(f"Too few points near {center}")
    x, y, err = w[sel], f[sel], (e[sel] if e is not None else np.ones(sel.sum()))
    ymed = np.median(y)
    amp0 = float(np.nanmax(y - ymed))
    mu0 = float(x[np.nanargmax(y)])
    sig0 = 1.5
    c0 = float(ymed)
    c1 = 0.0
    return x, y, err, (c0, c1), (amp0, mu0, sig0)


def _ew_uncertainty(flux, flux_err, c0, c1, mu, xmean, cov_c=None):
    """Compute EW and its uncertainty."""
    dx = mu - xmean
    c_mu = np.clip(abs(c0 + c1 * dx), 1e-10, np.inf)
    var_c = 0.0
    if cov_c is not None and np.all(np.isfinite(cov_c)):
        var_c = cov_c[0, 0] + dx**2 * cov_c[1, 1] + 2 * dx * cov_c[0, 1]
        var_c = max(var_c, 0.0)
    sigma_c = np.sqrt(var_c)
    ew = flux / c_mu
    ew_err = abs(ew) * np.sqrt((np.clip(flux_err, 0, np.inf) / np.clip(flux, 1e-30, np.inf))**2 +
                               (sigma_c / c_mu)**2)
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
    samples: Optional[np.ndarray] = None  # for MCMC chains, optional

    def as_dict(self):
        d = asdict(self)
        if isinstance(self.samples, np.ndarray):
            d["samples"] = None
        return d


# ==================================================================
# --- Fitting methods
# ==================================================================
def fit_line_lsq(w, f, e, center, name=None, window=20, inner=6, c0_global=None, c1_global=None):
    """Least-squares Gaussian fit."""
    if curve_fit is None:
        raise ImportError("scipy required for LSQ")

    x, y, err, (c0_guess, c1_guess), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)
    xmean = x.mean()

    if (c0_global is not None) and (c1_global is not None):
        def model_fixed(xx, A, mu, sig):
            return _gauss(xx, A, mu, sig) + (c0_global + c1_global * (xx - xmean))
        p0 = [max(A0, 1e-12), mu0, abs(sig0)]
        popt, pcov = curve_fit(model_fixed, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
        A, mu, sig = popt
        perr = np.sqrt(np.diag(pcov))
        flux = np.sqrt(2 * np.pi) * A * sig
        flux_err = abs(flux) * np.sqrt((perr[0]/np.clip(A, 1e-30, np.inf))**2 +
                                       (perr[2]/np.clip(sig, 1e-30, np.inf))**2)
        cont_mu = c0_global + c1_global * (mu - xmean)
        ew, ew_err = _ew_uncertainty(flux, flux_err, c0_global, c1_global, mu, xmean)
        print(f"[{name}] Continuum at μ = {cont_mu:.4e}")
        return LineFitResult("leastsq", name, mu, perr[1], sig, perr[2],
                             flux, flux_err, ew, ew_err, flux/flux_err, cont_mu)
    else:
        p0 = [A0, mu0, sig0, c0_guess, c1_guess]
        popt, pcov = curve_fit(_gauss_lin, x, y, p0=p0, sigma=err,
                               absolute_sigma=True, maxfev=20000)
        A, mu, sig, c0, c1 = popt
        perr = np.sqrt(np.diag(pcov))
        flux = np.sqrt(2 * np.pi) * A * sig
        flux_err = abs(flux) * np.sqrt((perr[0]/np.clip(A, 1e-30, np.inf))**2 +
                                       (perr[2]/np.clip(sig, 1e-30, np.inf))**2)
        cont_mu = c0 + c1*(mu - xmean)
        cov_c = pcov[3:5, 3:5]
        ew, ew_err = _ew_uncertainty(flux, flux_err, c0, c1, mu, xmean, cov_c)
        print(f"[{name}] Continuum at μ = {cont_mu:.4e}")
        return LineFitResult("leastsq", name, mu, perr[1], sig, perr[2],
                             flux, flux_err, ew, ew_err, flux/flux_err, cont_mu)


def fit_line_bootstrap(w, f, e, center, name=None, B=200, c0_global=None, c1_global=None, **kw):
    """Bootstrap resampling using repeated LSQ fits."""
    rng = np.random.default_rng(42)
    fluxes, ews, mus, sigmas = [], [], [], []

    for _ in range(B):
        fb = f + rng.normal(0, e if e is not None else 0.0, size=f.shape)
        try:
            r = fit_line_lsq(w, fb, e, center, name=name, c0_global=c0_global, c1_global=c1_global, **kw)
            fluxes.append(r.flux)
            ews.append(r.ew)
            mus.append(r.mu)
            sigmas.append(r.sigma)
        except Exception:
            continue

    if not fluxes:
        raise RuntimeError("Bootstrap failed")

    def robust_std(a):
        a = np.asarray(a)
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        return 1.4826 * mad

    f_med, f_std = np.median(fluxes), robust_std(fluxes)
    ew_med, ew_std = np.median(ews), robust_std(ews)
    mu_med, mu_std = np.median(mus), robust_std(mus)
    sig_med, sig_std = np.median(sigmas), robust_std(sigmas)
    snr = f_med / f_std if f_std > 0 else np.nan

    if (c0_global is not None) and (c1_global is not None):
        x, _, _, _, _ = _prep_window(w, f, e, center, kw.get("window", 20), kw.get("inner", 6))
        cont_mu0 = c0_global + c1_global * (mu_med - x.mean())
    else:
        cont_mu0 = fit_line_lsq(w, f, e, center, name=name, **kw).cont_mu

    print(f"[{name}] Continuum at μ = {cont_mu0:.4e}")
    return LineFitResult("bootstrap", name, mu_med, mu_std, sig_med, sig_std,
                         f_med, f_std, ew_med, ew_std, snr, cont_mu0)


def fit_line_mc(w, f, e, center, name=None, window=20, inner=6, nsamp=2000, c0_global=None, c1_global=None, **_):
    """Monte Carlo propagation based on LSQ covariance."""
    if curve_fit is None:
        raise ImportError("scipy required for MC")

    x, y, err, (c0_guess, c1_guess), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)
    xmean = x.mean()

    if (c0_global is not None) and (c1_global is not None):
        def model_fixed(xx, A, mu, sig):
            return _gauss(xx, A, mu, sig) + (c0_global + c1_global * (xx - xmean))
        p0 = [max(A0, 1e-12), mu0, abs(sig0)]
        popt, pcov = curve_fit(model_fixed, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
        if not np.all(np.isfinite(pcov)):
            base = fit_line_lsq(w, f, e, center, name=name, window=window, inner=inner,
                                c0_global=c0_global, c1_global=c1_global)
            return base

        rng = np.random.default_rng(12345)
        draws = rng.multivariate_normal(popt, pcov, size=nsamp)
        fluxes, ews, mus, sigmas = [], [], [], []
        for A, mu, sig in draws:
            if (A <= 0) or (sig <= 0):
                continue
            flux = np.sqrt(2*np.pi) * A * sig
            cont_mu = c0_global + c1_global * (mu - xmean)
            ew = flux / np.clip(abs(cont_mu), 1e-10, np.inf)
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
        snr = f_med / f_err if np.isfinite(f_err) and f_err > 0 else np.nan
        cont_mu_med = c0_global + c1_global * (mu_med - xmean)
        print(f"[{name}] Continuum at μ = {cont_mu_med:.4e}")
        return LineFitResult("montecarlo", name, mu_med, mu_err, sig_med, sig_err,
                             f_med, f_err, ew_med, ew_err, snr, cont_mu_med)

    else:
        base = fit_line_lsq(w, f, e, center, name=name, window=window, inner=inner)
        p0 = [base.flux/(np.sqrt(2*np.pi)*base.sigma), base.mu, base.sigma, base.cont_mu, 0.0]
        popt, pcov = curve_fit(_gauss_lin, x, y, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
        if not np.all(np.isfinite(pcov)):
            return base
        rng = np.random.default_rng(12345)
        draws = rng.multivariate_normal(popt, pcov, size=nsamp)
        fluxes, ews, mus, sigmas = [], [], [], []
        for A, mu, sig, c0, c1 in draws:
            if (A <= 0) or (sig <= 0):
                continue
            flux = np.sqrt(2*np.pi)*A*sig
            cont_mu = c0 + c1*(mu - xmean)
            ew = flux / np.clip(abs(cont_mu), 1e-10, np.inf)
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
        snr = f_med / f_err if np.isfinite(f_err) and f_err > 0 else np.nan
        cont_mu_med = np.median([c0 + c1*(m - xmean) for _, m, _, c0, c1 in draws])
        print(f"[{name}] Continuum at μ = {cont_mu_med:.4e}")
        return LineFitResult("montecarlo", name, mu_med, mu_err, sig_med, sig_err,
                             f_med, f_err, ew_med, ew_err, snr, cont_mu_med)


def fit_line_mcmc(w, f, e, center, name=None, window=20, inner=6,
                  nwalkers=32, nsteps=2000, nburn=500, c0_global=None, c1_global=None, **_):
    """MCMC Gaussian + continuum fit."""
    if emcee is None:
        raise ImportError("emcee not installed")

    x, y, err, (c0_guess, c1_guess), (A0, mu0, sig0) = _prep_window(w, f, e, center, window, inner)
    xmean = x.mean()

    if (c0_global is not None) and (c1_global is not None):
        def log_prior(t):
            A, mu, sig = t
            if A <= 0 or sig <= 0 or sig > window:
                return -np.inf
            return 0.0
        def log_like(t):
            A, mu, sig = t
            model = _gauss(x, A, mu, sig) + (c0_global + c1_global * (x - xmean))
            return -0.5 * np.sum(((y - model) / err) ** 2)
        def log_post(t):
            lp = log_prior(t)
            return lp + log_like(t) if np.isfinite(lp) else -np.inf
        p0 = np.array([max(A0, 1e-6), mu0, abs(sig0)], float)
        rng = np.random.default_rng(123)
        p0s = np.array([
            [abs(p0[0]*(1+0.1*rng.standard_normal())),
             p0[1] + 0.1*rng.standard_normal(),
             abs(p0[2]*(1+0.1*rng.standard_normal()))]
            for _ in range(nwalkers)
        ])
        sampler = emcee.EnsembleSampler(nwalkers, 3, log_post)
        sampler.run_mcmc(p0s, nsteps, progress=False)
        chain = sampler.get_chain(discard=nburn, flat=True)
        med, std = np.median(chain, axis=0), np.std(chain, axis=0)
        A, mu, sig = med
        flux = np.sqrt(2 * np.pi) * A * sig
        flux_err = abs(flux) * np.sqrt((std[0]/np.clip(A, 1e-30, np.inf))**2 +
                                       (std[2]/np.clip(sig, 1e-30, np.inf))**2)
        cont_mu = c0_global + c1_global*(mu - xmean)
        ew, ew_err = _ew_uncertainty(flux, flux_err, c0_global, c1_global, mu, xmean)
        print(f"[{name}] Continuum at μ = {cont_mu:.4e}")
        samples5 = np.column_stack([chain, np.full(chain.shape[0], c0_global),
                                    np.full(chain.shape[0], c1_global)])
        return LineFitResult("mcmc", name, mu, std[1], sig, std[2],
                             flux, flux_err, ew, ew_err, flux/flux_err, cont_mu,
                             samples=samples5)
    else:
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
        p0 = np.array([max(A0, 1e-6), mu0, abs(sig0), c0_guess, c1_guess], float)
        rng = np.random.default_rng(123)
        p0s = np.array([
            [abs(p0[0]*(1+0.1*rng.standard_normal())),
             p0[1] + 0.1*rng.standard_normal(),
             abs(p0[2]*(1+0.1*rng.standard_normal())),
             p0[3]*(1+0.1*rng.standard_normal()),
             p0[4]*(1+0.1*rng.standard_normal())]
            for _ in range(nwalkers)
        ])
        sampler = emcee.EnsembleSampler(nwalkers, 5, log_post)
        sampler.run_mcmc(p0s, nsteps, progress=False)
        chain = sampler.get_chain(discard=nburn, flat=True)
        med, std = np.median(chain, axis=0), np.std(chain, axis=0)
        A, mu, sig, c0, c1 = med
        flux = np.sqrt(2 * np.pi) * A * sig
        flux_err = abs(flux) * np.sqrt((std[0]/np.clip(A, 1e-30, np.inf))**2 +
                                       (std[2]/np.clip(sig, 1e-30, np.inf))**2)
        cont_mu = c0 + c1*(mu - x.mean())
        ew, ew_err = _ew_uncertainty(flux, flux_err, c0, c1, mu, x.mean())
        print(f"[{name}] Continuum at μ = {cont_mu:.4e}")
        return LineFitResult("mcmc", name, mu, std[1], sig, std[2],
                             flux, flux_err, ew, ew_err, flux/flux_err, cont_mu,
                             samples=chain)


# ==================================================================
# --- Multi-line wrapper
# ==================================================================
def fit_lines_all(w, f, e, lines: Dict[str, float], **kw):
    """Fit multiple emission lines using all four methods with shared continuum."""
    c0, c1 = _fit_global_continuum(w, f, e)
    print(f"[Global continuum initial guess] c0={c0:.3f}, c1={c1:.3e}")

    out = {"leastsq": {}, "bootstrap": {}, "montecarlo": {}, "mcmc": {}}
    for name, cen in lines.items():
        try:
            out["leastsq"][name] = fit_line_lsq(w, f, e, cen, name=name, c0_global=c0, c1_global=c1, **kw)
        except Exception as ex:
            out["leastsq"][name] = ex
        try:
            out["bootstrap"][name] = fit_line_bootstrap(w, f, e, cen, name=name, c0_global=c0, c1_global=c1, **kw)
        except Exception as ex:
            out["bootstrap"][name] = ex
        try:
            out["montecarlo"][name] = fit_line_mc(w, f, e, cen, name=name, c0_global=c0, c1_global=c1, **kw)
        except Exception as ex:
            out["montecarlo"][name] = ex
        if emcee:
            try:
                out["mcmc"][name] = fit_line_mcmc(w, f, e, cen, name=name, c0_global=c0, c1_global=c1, **kw)
            except Exception as ex:
                out["mcmc"][name] = ex
    return out


# ==================================================================
# --- Demo / test
# ==================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(123)
    wave = np.linspace(4800, 5100, 3000)
    continuum = 10 + 0.002*(wave - wave.mean())
    flux = continuum.copy()
    true_lines = [
        ("Hβ", 4862.7, 3.5, 2.0),
        ("[OIII]4959", 4960.3, 4.5, 1.4),
        ("[OIII]5008", 5008.2, 5.0, 1.6),
    ]
    for _, mu, A, sig in true_lines:
        flux += A * np.exp(-0.5 * ((wave - mu) / sig)**2)
    err = np.full_like(flux, 0.4)
    flux += rng.normal(0, err)
    lines = {n: mu for n, mu, _, _ in true_lines}
    res = fit_lines_all(wave, flux, err, lines, window=20)

    for method, rd in res.items():
        print(f"\n=== {method.upper()} RESULTS ===")
        for name, r in rd.items():
            if isinstance(r, Exception):
                print(f"{name:10s} FAIL ({r})")
            else:
                print(f"{name:10s} μ={r.mu:7.2f}±{r.mu_err:5.2f}  σ={r.sigma:4.2f}±{r.sigma_err:4.2f}  "
                      f"F={r.flux:7.2f}±{r.flux_err:5.2f}  EW={r.ew:7.2f}±{r.ew_err:5.2f}  "
                      f"Cont(μ)={r.cont_mu:7.3f}  SNR={r.snr:5.1f}")

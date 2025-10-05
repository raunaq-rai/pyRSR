#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for spectral_fitting.py
Run with:
    pytest -v test_spectral_fitting.py
"""

import numpy as np
import pytest
import cosmicdawn
from cosmicdawn.line_fitting import (
    _gaussian,
    _gaussian_w_continuum,
    _sigma_clip,
    _continuum_linear,
    _prep_window,
    fit_line_leastsq,
    fit_line_mcmc,
    fit_lines_batch,
    LineFitResult,
)

# Use a fixed random seed for reproducibility
rng = np.random.default_rng(42)


# ---------- BASIC GAUSSIAN UTILS ----------
def test_gaussian_shape_and_peak():
    x = np.linspace(-5, 5, 100)
    y = _gaussian(x, amp=1.0, mu=0.0, sigma=1.0)
    assert np.isclose(y.max(), 1.0, atol=1e-3)
    assert np.isclose(y[np.argmin(np.abs(x))], 1.0, atol=1e-3)
    assert np.all(np.isfinite(y))


def test_gaussian_w_continuum_linear_component():
    x = np.linspace(0, 10, 100)
    y = _gaussian_w_continuum(x, 2.0, 5.0, 0.5, 1.0, 0.1)
    assert np.isclose(y.mean(), (y.min() + y.max()) / 2, atol=1.0)
    assert np.all(np.isfinite(y))


# ---------- CONTINUUM + SIGMA CLIP ----------
def test_sigma_clip_removes_outliers():
    y = np.concatenate([np.random.normal(0, 1, 100), [10, -10]])
    mask = np.ones_like(y, dtype=bool)
    m2 = _sigma_clip(y, mask)
    assert m2.sum() < mask.sum()
    assert np.all(np.isfinite(y[m2]))


def test_continuum_linear_estimates_slope():
    x = np.linspace(0, 10, 20)
    y = 3.0 + 0.5 * (x - 5)
    c0, c1, cov = _continuum_linear(x, y, yerr=None, mask=np.ones_like(y, dtype=bool))
    assert np.isclose(c0, 3.0, atol=1e-3)
    assert np.isclose(c1, 0.5, atol=1e-3)
    assert cov.shape == (2, 2)


# ---------- PREP WINDOW ----------
def test_prep_window_estimates_parameters():
    wave = np.linspace(4900, 5100, 2000)
    flux = 10 + 3 * np.exp(-0.5 * ((wave - 5000) / 1.5) ** 2)
    err = np.full_like(flux, 0.1)
    x, y, e, cont, init = _prep_window(wave, flux, err, 5000, window=20, cont_inner=6)
    c0, c1, _ = cont
    amp0, mu0, sig0 = init
    assert np.isclose(c0, 10, atol=1e-1)
    assert np.isclose(mu0, 5000, atol=0.5)
    assert 0.5 < sig0 < 3.0


# ---------- LEAST SQUARES FITTING ----------
def test_fit_line_leastsq_recovers_parameters():
    wave = np.linspace(4800, 5100, 2000)
    amp_true, mu_true, sig_true = 5.0, 5008.24, 1.8
    cont_true = 10.0
    flux = _gaussian(wave, amp_true, mu_true, sig_true) + cont_true
    err = np.full_like(flux, 0.2)
    flux += rng.normal(0, err)

    res = fit_line_leastsq(wave, flux, err, center=5008.24, window=20.0, cont_inner=6.0, name="[OIII]")
    assert isinstance(res, LineFitResult)
    assert np.isclose(res.mu, mu_true, atol=0.5)
    assert np.isclose(res.sigma, sig_true, atol=0.3)
    assert res.snr > 5
    assert res.flux > 0
    assert res.method == "leastsq"


def test_fit_line_leastsq_handles_bad_bounds():
    wave = np.linspace(4800, 5100, 2000)
    flux = 10 + _gaussian(wave, 5, 5008.24, 1.5)
    err = np.full_like(flux, 0.2)
    res = fit_line_leastsq(wave, flux, err, center=5008.24, bounds={"amp": (0, 100)}, name="bound_test")
    assert res.mu > 0
    assert res.sigma > 0


# ---------- MCMC FITTING ----------
@pytest.mark.skipif("emcee" not in globals(), reason="emcee not installed")
def test_fit_line_mcmc_recovers_parameters():
    wave = np.linspace(4800, 5100, 2000)
    flux = 10 + _gaussian(wave, 5, 5008.24, 1.8)
    err = np.full_like(flux, 0.2)
    flux += rng.normal(0, err)

    res = fit_line_mcmc(
        wave, flux, err, center=5008.24, window=20.0,
        cont_inner=6.0, name="MCMC_OIII", nwalkers=24, nsteps=800, nburn=200
    )
    assert isinstance(res, LineFitResult)
    assert 5005 < res.mu < 5010
    assert res.snr > 5
    assert np.isfinite(res.flux)
    assert res.method == "mcmc"


# ---------- BATCH FITTING ----------
def test_fit_lines_batch_handles_multiple_lines():
    wave = np.linspace(4800, 6600, 5000)
    flux = 10 + _gaussian(wave, 5, 5008.24, 1.8) + _gaussian(wave, 4, 6564.6, 2.5)
    err = np.full_like(flux, 0.3)
    flux += rng.normal(0, err)

    line_centers = {"OIII": 5008.24, "HALPHA": 6564.6}
    results = fit_lines_batch(wave, flux, err, line_centers, method="leastsq", window=25)
    assert all(isinstance(v, LineFitResult) for v in results.values())
    assert len(results) == 2
    assert results["OIII"].snr > 5
    assert results["HALPHA"].flux > 0


# ---------- EDGE / FAILURE CASES ----------
def test_fit_lines_batch_handles_invalid_data():
    wave = np.linspace(4800, 5100, 2000)
    flux = np.zeros_like(wave)
    err = np.ones_like(wave) * 0.1
    lines = {"FAKE": 5000.0}
    res = fit_lines_batch(wave, flux, err, lines, method="leastsq")
    assert len(res) == 1
    assert "FAKE" in res
    assert isinstance(list(res.values())[0], LineFitResult) or isinstance(list(res.values())[0], Exception)


def test_prep_window_raises_for_small_region():
    wave = np.linspace(0, 10, 5)
    flux = np.ones_like(wave)
    with pytest.raises(ValueError):
        _prep_window(wave, flux, None, center=5, window=1)


def test_result_serialization_roundtrip():
    wave = np.linspace(4800, 5100, 2000)
    flux = 10 + _gaussian(wave, 5, 5008.24, 1.8)
    err = np.full_like(flux, 0.2)
    res = fit_line_leastsq(wave, flux, err, center=5008.24, name="test_line")
    d = res.as_dict()
    assert isinstance(d, dict)
    assert "flux" in d and "mu" in d
    assert np.isfinite(d["flux"])



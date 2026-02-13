import numpy as np
import pytest
from PyRSR.broad_line_fit import single_broad_fit, fit_continuum_moving_average
try:
    from PyRSRX.broad_line_fit import single_broad_fit as single_broad_fit_x
    HAS_X = True
except ImportError:
    HAS_X = False

def create_synthetic_spectrum():
    # Grid
    lam_um = np.linspace(0.6, 5.0, 3000)
    
    # Continuum: Sinusoidal
    flam_true = 10.0 + 5.0 * np.sin(lam_um * 2)
    
    # Add Emission Line (H-alpha at z=2)
    z = 2.0
    lam_ha = 0.6563 * (1 + z) # ~1.97 um
    sigma_um = 0.005
    line_flux = 50.0 # big line
    line_profile = line_flux * np.exp(-0.5 * ((lam_um - lam_ha)/sigma_um)**2)
    
    flux = flam_true + line_profile
    
    # Add noise
    noise = np.random.normal(0, 0.5, size=len(lam_um))
    flux += noise
    err = np.ones_like(flux) * 0.5
    
    # fake uJy units
    # for simplicity, let's just pretend flam is uJy for the fit helpers since we convert inside
    # but fit_continuum_moving_average takes flam locally.
    
    # broad_fit expects source dict with flux in uJy.
    # We constructed "flux" above as if it were flam.
    # To pass to broad_fit, let's just pass it as is and ignore unit conversion physics for this test
    # providing we use valid inputs.
    
    return {
        "lam": lam_um,
        "flux": flux,
        "err": err
    }, z, flam_true

def test_moving_average_logic():
    source, z, true_cont = create_synthetic_spectrum()
    lam_um = source["lam"]
    flam = source["flux"]
    
    # Direct help test
    fcont, _ = fit_continuum_moving_average(
        lam_um, flam, z,
        window_um=0.5, # Wide window to smooth
        clip_sigma=3.0,
        grating="PRISM"
    )
    
    # Check that continuum is finite
    assert np.all(np.isfinite(fcont))
    
    # Check that residuals near line are high (meaning we didn't fit the line as continuum)
    # The line is at ~1.97 um.
    idx_line = np.argmin(np.abs(lam_um - 0.6563*(1+z)))
    
    # The continuum fit should be close to true_cont, not flux (which includes line)
    print(f"At Line: Flux={flam[idx_line]:.2f}, TrueCont={true_cont[idx_line]:.2f}, FitCont={fcont[idx_line]:.2f}")
    assert abs(fcont[idx_line] - true_cont[idx_line]) < 2.0 # Tolerance
    assert abs(fcont[idx_line] - flam[idx_line]) > 10.0 # Should be far from line peak

def test_single_broad_fit_integration():
    source, z, _ = create_synthetic_spectrum()
    
    # Run with polyfit
    res_poly = single_broad_fit(source, z, plot=False, verbose=False, continuum_fit="polyfit")
    assert res_poly["success"]
    
    # Run with moving average
    res_mov = single_broad_fit(source, z, plot=False, verbose=False, continuum_fit="moving_average")
    assert res_mov["success"]
    assert "continuum_flam" in res_mov
    assert np.any(res_mov["continuum_flam"] != res_poly["continuum_flam"])

if HAS_X:
    def test_pxrsrx_integration():
        source, z, _ = create_synthetic_spectrum()
        res_mov = single_broad_fit_x(source, z, plot=False, verbose=False, continuum_fit="moving_average")
        assert res_mov["success"]

if __name__ == "__main__":
    test_moving_average_logic()
    test_single_broad_fit_integration()
    if HAS_X:
        test_pxrsrx_integration()
    print("All tests passed!")

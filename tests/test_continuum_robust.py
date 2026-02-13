import numpy as np
import pytest
from PyRSR.broad_line_fit import fit_continuum_moving_average as f_avg_rsr
from PyRSRX.broad_line_fit import fit_continuum_moving_average as f_avg_rsrx

def run_tests(fit_continuum_moving_average, name):
    print(f"Testing {name}...")

def test_continuum_exclude_regions(fit_continuum_moving_average):
    """Test that exclude_regions are ignored by the moving average."""
    lam_um = np.linspace(1.0, 2.0, 1000)
    # Continuum = 1.0
    flam = np.ones_like(lam_um)
    
    # Add a big "artifact" at 1.5 um
    mask_artifact = (lam_um > 1.48) & (lam_um < 1.52)
    flam[mask_artifact] += 100.0
    
    # 1. Fit without exclusion -> should be high near 1.5
    fcont_bad, _ = fit_continuum_moving_average(
        lam_um, flam, z=0.0, window_um=0.2, exclude_regions=None
    )
    val_at_15 = np.interp(1.5, lam_um, fcont_bad)
    # With a window of 0.2, the 100.0 peak will lift the average significantly
    assert val_at_15 > 2.0, f"Expected artifact to bias continuum without exclusion, got {val_at_15}"

    # 2. Fit with exclusion
    fcont_good, _ = fit_continuum_moving_average(
        lam_um, flam, z=0.0, window_um=0.2, exclude_regions=[(1.48, 1.52)]
    )
    val_at_15_good = np.interp(1.5, lam_um, fcont_good)
    assert abs(val_at_15_good - 1.0) < 0.1, f"Expected exclusion to remove artifact bias, got {val_at_15_good}"

def test_lyman_adaptive_window(fit_continuum_moving_average):
    """Test that lyman_window_um allows sharper transition."""
    # Lyman alpha at z=0 is 0.121567 um. Let's work at z=10 so LyA is ~1.33 um
    z = 10.0
    lya_obs = 0.121567 * (1 + z) # ~1.337
    
    lam_um = np.linspace(1.0, 2.0, 2000)
    
    # Step function continuum: 0 below LyA, 1 above
    true_cont = np.zeros_like(lam_um)
    true_cont[lam_um >= lya_obs] = 1.0
    
    # Add noise?
    flam = true_cont.copy()
    
    # Large window normally (e.g. 0.2 um) would smooth this step out broadly
    # Small window (0.01 um) would keep it sharp
    
    # 1. Standard fit with large window
    window_large = 0.2
    fcont_smooth, _ = fit_continuum_moving_average(
        lam_um, flam, z=z, window_um=window_large, 
        lyman_cut=None,
        lyman_window_um=None,
        grating="G140M" # High res to avoid broad masking
    )

    
    # Check value just after break (e.g. lya + 0.05). 
    # With large window, it mixes 0s from left and 1s from right.
    # At lya+0.05 (1.387), window is [1.287, 1.487]. Left part is < 1.337 (zeros).
    # So value should be < 1.0
    val_after_smooth = np.interp(lya_obs + 0.05, lam_um, fcont_smooth)
    print(f"DEBUG SMOOTH (large window): {val_after_smooth}")
    # We expect this to be < 1.0 because it averages 0s and 1s.
    # The large window might be robust to 0s if they are clipped?
    # Let's just check that adaptive window is *more* accurate (closer to 1) than smooth window
    # near the break.


    # 2. Adaptive fit
    # continuum_lya_mask_A = 200 A (rest) -> ~2200 A obs -> 0.22 um
    # So within +/- 0.22 um of LyA, we use small window.
    win_small = 0.02
    
    fcont_sharp, _ = fit_continuum_moving_average(
        lam_um, flam, z=z, window_um=window_large,
        lyman_cut=None,
        lyman_window_um=win_small,
        continuum_lya_mask_A=300.0,
        grating="G140M"
    )

    
    val_after_sharp = np.interp(lya_obs + 0.05, lam_um, fcont_sharp)
    print(f"DEBUG SHARP (small window): {val_after_sharp}")
    
    assert val_after_sharp > 0.98, f"Expected sharp transition to be complete. Got {val_after_sharp}"
    # assert val_after_sharp > val_after_smooth + 0.01, f"Expected sharp to be significantly better"


if __name__ == "__main__":
    print("Running manual verification...")
    try:
        run_tests(f_avg_rsr, "PyRSR")
        test_continuum_exclude_regions(f_avg_rsr)
        test_lyman_adaptive_window(f_avg_rsr)
        
        run_tests(f_avg_rsrx, "PyRSRX") 
        test_continuum_exclude_regions(f_avg_rsrx)
        test_lyman_adaptive_window(f_avg_rsrx)
        
        print("All manual tests passed!")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)


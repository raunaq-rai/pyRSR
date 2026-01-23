
import time
import numpy as np
from PyRSR.broad_line_fit import build_model_flam_linear as py_build
from PyRSRX.broad_line_fit import build_model_flam_linear as cy_build
from PyRSRX.broad_line_fit import HAS_CYTHON

def verify_and_benchmark():
    print(f"PyRSRX has Cython extension: {HAS_CYTHON}")
    if not HAS_CYTHON:
        print("WARNING: PyRSRX is falling back to Python! Build failed?")
        
    # Synthetic data setup
    n_pix = 10000
    lam_um = np.linspace(1.0, 5.0, n_pix)
    z = 6.0
    grating = "prism"
    
    lines = ["H⍺", "HBETA", "OIII_5007"]
    nL = len(lines)
    
    # Random params [A, sigma, mu]
    # A ~ 1e-18, sigma ~ 0.001 um, mu ~ 3.0 um
    A = np.array([1e-18, 5e-19, 2e-18])
    sigma = np.array([0.005, 0.003, 0.004])
    # Rest wavelengths approx to get centers
    mu = np.array([0.6563*(1+z), 0.4861*(1+z), 0.5007*(1+z)])
    
    params = np.concatenate([A, sigma, mu])
    
    # 1. Verification
    print("Verifying consistency...")
    model_py, profiles_py, centers_py = py_build(params, lam_um, z, grating, lines, None)
    model_cy, profiles_cy, centers_cy = cy_build(params, lam_um, z, grating, lines, None)
    
    # Check Model
    diff = np.max(np.abs(model_py - model_cy))
    print(f"Max difference in model: {diff:.5e}")
    if diff > 1e-20:  # Allow small float diffs
        # Scale by median flux?
        rel_diff = diff / np.max(np.abs(model_py))
        print(f"Relative difference: {rel_diff:.5e}")
        assert rel_diff < 1e-10, "Models mismatch!"
    else:
        print("Models match.")

    # Check Profiles
    for ln in lines:
        d = np.max(np.abs(profiles_py[ln] - profiles_cy[ln]))
        if d > 1e-20:
            rel = d / np.max(np.abs(profiles_py[ln]))
            assert rel < 1e-10, f"Profile {ln} mismatch"
            
    print("Consistency check PASSED.")
    
    # 2. Benchmark
    n_iter = 1000
    print(f"Benchmarking {n_iter} iterations...")
    
    t0 = time.time()
    for _ in range(n_iter):
        py_build(params, lam_um, z, grating, lines, None)
    t_py = time.time() - t0
    
    t0 = time.time()
    for _ in range(n_iter):
        cy_build(params, lam_um, z, grating, lines, None)
    t_cy = time.time() - t0
    
    print(f"Python time: {t_py:.4f} s ({t_py/n_iter*1e3:.4f} ms/call)")
    print(f"Cython time: {t_cy:.4f} s ({t_cy/n_iter*1e3:.4f} ms/call)")
    print(f"Speedup: {t_py/t_cy:.2f}x")

if __name__ == "__main__":
    verify_and_benchmark()

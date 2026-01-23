
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, erf, log, M_PI

# Initialize numpy C API
np.import_array()

cdef double SQRT2 = 1.4142135623730951

cdef inline double _erf_diff(double x1, double x2) nogil:
    """Computes erf(x2) - erf(x1)"""
    return erf(x2) - erf(x1)

def gauss_binavg_area_normalized_A_cython(
    double[:] lam_left_A,
    double[:] lam_right_A,
    double muA,
    double sigmaA
):
    """
    Mean value (per pixel) of a *unit-area* Gaussian over [left,right] bins (Å⁻¹).
    Cython optimized version.
    """
    cdef Py_ssize_t n = lam_left_A.shape[0]
    cdef double[:] mean = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double inv_sigma, inv_sqrt2_sigma
    cdef double cdf_l, cdf_r, area, width
    
    if sigmaA <= 0:
        return np.asarray(mean)

    inv_sqrt2_sigma = 1.0 / (SQRT2 * sigmaA)

    with nogil:
        for i in range(n):
            # cdf_r = 0.5 * (1.0 + erf((lam_right_A[i] - muA) * inv_sqrt2_sigma))
            # cdf_l = 0.5 * (1.0 + erf((lam_left_A[i]  - muA) * inv_sqrt2_sigma))
            # area = cdf_r - cdf_l
            # Optimization: 0.5 * (1 + erf2) - 0.5 * (1 + erf1) = 0.5 * (erf2 - erf1)
            area = 0.5 * (erf((lam_right_A[i] - muA) * inv_sqrt2_sigma) - 
                          erf((lam_left_A[i] - muA) * inv_sqrt2_sigma))
            
            width = lam_right_A[i] - lam_left_A[i]
            if width > 0:
                mean[i] = area / width
            else:
                mean[i] = 0.0

    return np.asarray(mean)


def build_model_flam_linear_cython(
    double[:] params,
    double[:] lam_um,
    int nL,
    list which_lines_names,
    dict lam_A_pixel_edges_cache=None
):
    """
    Sum of area-normalised Gaussians in F_lambda.
    
    params: [A0..An, sigma0..sigman, mu0..mun]
    lam_um: wavelength grid in microns
    nL: number of lines
    which_lines_names: list of line names (to populate profiles dict)
    """
    cdef Py_ssize_t n_pix = lam_um.shape[0]
    
    # Unpack params
    # Using memoryviews directly
    cdef double[:] A = params[0:nL]
    cdef double[:] sigmaA = params[nL:2*nL]
    cdef double[:] muA = params[2*nL:3*nL]

    # Pre-calculate edges in A if not cached, or pass them in?
    # Calculating edges might be fast enough, or we can do it in Python and pass.
    # For now, let's replicate the logic of _pixel_edges_A efficiently.
    # Note: _pixel_edges_A logic:
    # d = diff(lam); left = [lam[0]-0.5d[0], lam[:-1]+0.5d]; right = [lam[:-1]+0.5d, lam[-1]+0.5d[-1]]
    # This is a bit complex to do generic 'numpy-like' ops in cython fast without array creation overhead.
    # BUT, `lam_um` is constant for a fit. The caller should optimally pass `lam_left_A` and `lam_right_A`.
    
    # We will assume wrapping python code or the caller handles caching edges if performance is critical there,
    # but for now we'll do it inside here or accept them as args. 
    # To keep the signature similar to original, we might need adjustments, 
    # but the caller `broad_fit` builds the model inside the loop. 
    # Actually, `broad_fit` calls `build_model_flam_linear` many times inside `least_squares`.
    # So we SHOULD optimize the edge calculation or reuse it.
    
    # Let's assume lam_um is converted to A inside.
    cdef double[:] lam_A = np.empty(n_pix, dtype=np.float64)
    cdef Py_ssize_t i
    with nogil:
        for i in range(n_pix):
            lam_A[i] = lam_um[i] * 10000.0

    # Calculate edges (simplified logic matching original)
    cdef double[:] left_A = np.empty(n_pix, dtype=np.float64)
    cdef double[:] right_A = np.empty(n_pix, dtype=np.float64)
    cdef double d, d_prev, d_next

    if n_pix > 1:
        with nogil:
            # First pixel
            d = lam_A[1] - lam_A[0]
            left_A[0] = lam_A[0] - 0.5 * d
            right_A[0] = lam_A[0] + 0.5 * d
            
            # Middle pixels
            for i in range(1, n_pix - 1):
                d = lam_A[i+1] - lam_A[i]
                # In original: left[i] = lam[i-1] + 0.5 * (lam[i]-lam[i-1]) which is right[i-1]
                # right[i] = lam[i] + 0.5 * d
                left_A[i] = right_A[i-1]
                right_A[i] = lam_A[i] + 0.5 * d

            # Last pixel
            d = lam_A[n_pix-1] - lam_A[n_pix-2]
            left_A[n_pix-1] = right_A[n_pix-2]
            right_A[n_pix-1] = lam_A[n_pix-1] + 0.5 * d
    else:
        # Fallback for 1 pixel?
        left_A[0] = lam_A[0] - 0.5
        right_A[0] = lam_A[0] + 0.5

    # Main Loop
    cdef double[:] model = np.zeros(n_pix, dtype=np.float64)
    # profiles dict to hold numpy arrays per line
    profiles = {} # Dictionary of python objects
    centers = {}

    # Calculate median pixel width for MinSigma clamping
    # simple median estimate? or just take diff[0] if uniform?
    # Original: pix_A = np.median(np.diff(lam_A))
    # We can approximate or standard numpy call. Since this is setup, numpy call is fine.
    # But wait, we are inside the fit loop. 
    # Let's use a quick estimate or passed value. 
    # For robust median, we'd need sorting.
    # Let's assume the caller could pass min_sigma, but let's stick to simple logic:
    # Most spectra are somewhat uniform or slowly varying. Median ~ Mean or just sample a few.
    # Let's just do it via numpy, it's O(N).
    cdef double pix_A_val
    if n_pix > 1:
        # Avoid full sort if possible, but correctness matters.
        # Let's compute it once.
        diffs = np.diff(lam_A)
        pix_A_val = np.median(diffs)
    else:
        pix_A_val = 1.0

    cdef double min_sigma = 0.35 * pix_A_val
    if min_sigma < 0.01:
        min_sigma = 0.01
        
    cdef double[:] prof_flam
    cdef double sj, mu_val, inv_sqrt2_sj, width, area_j
    cdef Py_ssize_t j, k

    cdef double LN10 = 2.302585092994046

    # Iterate over lines
    for j in range(nL):
        sj = sigmaA[j]
        if sj < min_sigma:
            sj = min_sigma
        
        mu_val = muA[j]
        
        # Calculate profile
        prof_flam = np.zeros(n_pix, dtype=np.float64)
        inv_sqrt2_sj = 1.0 / (SQRT2 * sj)
        
        if sj > 0 and (mu_val > -1e9 and mu_val < 1e9): # Sanity check
            with nogil:
                for k in range(n_pix):
                    area_j = 0.5 * (erf((right_A[k] - mu_val) * inv_sqrt2_sj) - 
                                    erf((left_A[k] - mu_val) * inv_sqrt2_sj))
                    width = right_A[k] - left_A[k]
                    if width > 0:
                        # Multiply by Amplitude A[j] here directly?
                        # A[j] is the integrated flux. The unit-area profile * A[j] = flux density.
                        prof_flam[k] = A[j] * (area_j / width)
                        model[k] += prof_flam[k]
                    
        # Store for return
        # Note: We need to return copies or the array itself.
        # Since 'prof_flam' is overwritten next iter, we must save it.
        profiles[which_lines_names[j]] = np.asarray(prof_flam)
        
        # Centers info
        sigma_logA = sj / (mu_val * LN10) if (mu_val > 0) else float('nan')
        centers[which_lines_names[j]] = (mu_val, sigma_logA)

    return np.asarray(model), profiles, centers

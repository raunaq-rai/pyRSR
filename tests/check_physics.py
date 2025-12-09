
import numpy as np

# Speed of light in km/s
c_kms = 299792.458

def check_separation(name, lam1, lam2, R=1000):
    lam_mean = (lam1 + lam2) / 2
    dlam = abs(lam1 - lam2)
    
    # Velocity separation of the doublet
    dv_sep = c_kms * (dlam / lam_mean)
    
    # Resolution FWHM in km/s (approximate)
    dv_res = c_kms / R
    
    ratio = dv_sep / dv_res
    
    print(f"--- {name} ---")
    print(f"  Lambda: {lam1:.2f}, {lam2:.2f} A")
    print(f"  Separation: {dlam:.2f} A  ->  {dv_sep:.0f} km/s")
    print(f"  Resolution (R={R}): {dv_res:.0f} km/s")
    print(f"  Ratio (Sep/Res): {ratio:.2f}")
    if ratio < 1.0:
        print("  -> UNRESOLVED by instrument. Single Gaussian is VALID.")
    else:
        print("  -> RESOLVED by instrument (for narrow lines). Single Gaussian is APPROXIMATION.")
    print("")

print("Assuming R ~ 1000 (Medium Gratings)\n")

# NV
check_separation("NV Doublet", 1238.82, 1242.80)

# CIV
check_separation("CIV Doublet", 1548.19, 1550.77)

# OII
check_separation("OII Doublet", 3726.03, 3728.82)

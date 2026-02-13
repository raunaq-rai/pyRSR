import sys
import os
sys.path.insert(0, "/Users/raunaqrai/Documents/phd/y1/PyRSR")

import PyRSRX.broad_line_fit as blf
print(f"Imported from: {blf.__file__}")

try:
    # Try to call broad_fit with the new argument
    blf.broad_fit(
        source={"lam": [1, 2, 3], "flux": [1, 1, 1], "err": [0.1, 0.1, 0.1]},
        z=0.0,
        continuum_fit="moving_average"
    )
except TypeError as e:
    print(f"Caught expected TypeError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {e}")
else:
    print("Successfully called broad_fit with continuum_fit argument (it may fail later on logic, but argument was accepted)")

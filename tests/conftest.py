import numpy as np
import pytest

@pytest.fixture
def make_boxcar_filter(tmp_path):
    """
    Fixture: return a callable that generates a temporary boxcar filter file.
    """
    def _make(wmin=4000, wmax=8000, npts=2000):
        lam = np.linspace(3000, 9000, npts)
        resp = np.where((lam >= wmin) & (lam <= wmax), 1.0, 0.0)
        data = np.column_stack([lam, resp])

        # Use absolute string path to avoid NumPy DataSource + cwd issues
        filt_path = (tmp_path / "boxcar.txt").resolve()
        np.savetxt(str(filt_path), data)   # 👈 cast to str
        return filt_path

    return _make

import os
import pytest

@pytest.fixture(autouse=True)
def restore_cwd():
    """Ensure each test runs with a valid cwd."""
    old = os.getcwd()
    try:
        yield
    finally:
        if not os.path.exists(old):
            os.chdir(os.path.expanduser("~"))  # fallback
        else:
            os.chdir(old)

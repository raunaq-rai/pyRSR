import os
import numpy as np
import pytest
from cosmicdawn.general import igm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "inoue2014_table2.txt")

def test_get_Inoue14_trans_shape_and_bounds():
    rest_wavs = np.linspace(800, 1300, 100)
    trans = igm.get_Inoue14_trans(rest_wavs, z_obs=3.0, coef_file=DATA_PATH)

    assert trans.shape == rest_wavs.shape
    assert np.all(trans >= 0.0)
    assert np.all(trans <= 1.0)

def test_get_Inoue14_trans_highz_lower_transmission():
    rest_wavs = np.linspace(1100, 1300, 50)
    trans_lowz = igm.get_Inoue14_trans(rest_wavs, z_obs=2.0, coef_file=DATA_PATH)
    trans_highz = igm.get_Inoue14_trans(rest_wavs, z_obs=6.0, coef_file=DATA_PATH)

    # On average, transmission should be lower at higher redshift
    assert np.median(trans_highz) < np.median(trans_lowz)

def test_lyman_limit_break():
    rest_wavs = np.array([800, 910, 920, 1000])
    trans = igm.get_Inoue14_trans(rest_wavs, z_obs=4.0, coef_file=DATA_PATH)

    # Transmission should drop for λ < 912 Å
    assert trans[0] < trans[-1]
    assert trans[1] < trans[-1]

def test_get_IGM_absorption_interface():
    lam_obs, trans = igm.get_IGM_absorption(z_obs=5.0, coef_file=DATA_PATH)

    assert lam_obs.ndim == 1
    assert trans.ndim == 1
    assert lam_obs.shape == trans.shape
    assert np.all((0 <= trans) & (trans <= 1))

def test_missing_coef_file_raises(tmp_path):
    fake_file = tmp_path / "no_such_file.txt"
    rest_wavs = np.linspace(900, 1100, 10)
    with pytest.raises(FileNotFoundError):
        igm.get_Inoue14_trans(rest_wavs, z_obs=3.0, coef_file=str(fake_file))

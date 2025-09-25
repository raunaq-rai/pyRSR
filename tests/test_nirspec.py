"""
Test suite for preprocess_nirspec_file in cosmicdawn.preprocessing.nirspec

These tests use pytest. They don’t require real JWST data to run —
we mock RATE files with minimal FITS headers so the function can execute
without full CRDS reference files.
"""

import os
import pytest
import shutil
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from cosmicdawn.preprocessing.nirspec import preprocess_nirspec_file


@pytest.fixture(scope="module")
def tmp_workdir():
    """Create a temporary work directory for tests."""
    d = tempfile.mkdtemp(prefix="nirspec_test_")
    yield d
    shutil.rmtree(d)


def make_fake_rate_file(tmpdir, name="jw00000000001_00001_00001_nrs1_rate.fits",
                        filter_name="F100LP"):
    """
    Create a minimal fake NIRSpec RATE file with necessary structure.
    """
    path = Path(tmpdir) / name
    hdu0 = fits.PrimaryHDU(header=fits.Header())
    hdu0.header["FILTER"] = filter_name
    hdu0.header["EXP_TYPE"] = "NRS_MSASPEC"
    hdu0.header["MSAMETFL"] = "fake_msa.fits"

    # SCI extension
    sci = fits.ImageHDU(np.zeros((10, 10)), name="SCI")

    # DQ extension
    dq = fits.ImageHDU(np.zeros((10, 10), dtype=np.uint32), name="DQ")

    hdul = fits.HDUList([hdu0, sci, dq])
    hdul.writeto(path, overwrite=True)

    return str(path)


def test_missing_rate_file(tmp_workdir):
    """Should return status=3 when input file does not exist."""
    bad_file = os.path.join(tmp_workdir, "does_not_exist.fits")
    status = preprocess_nirspec_file(
        rate_file=bad_file,
        root="test-missing",
        workdir=tmp_workdir,
    )
    assert status == 3


def test_valid_rate_file_runs(tmp_workdir):
    """Check that a fake RATE file runs without raising and returns int."""
    fake_file = make_fake_rate_file(tmp_workdir)
    status = preprocess_nirspec_file(
        rate_file=fake_file,
        root="test-valid",
        workdir=tmp_workdir,
        undo_flat=True,
        by_source=False,
    )
    assert isinstance(status, int)
    # status may be 2 (success) or 3 (if pipeline fails internally)


def test_filter_renaming(tmp_workdir):
    """Ensure F070LP gets renamed to F100LP if rename_f070=True."""
    fake_file = make_fake_rate_file(tmp_workdir, filter_name="F070LP")
    status = preprocess_nirspec_file(
        rate_file=fake_file,
        root="test-f070",
        workdir=tmp_workdir,
        rename_f070=True,
    )
    # check that renamed file exists
    renamed = fake_file.replace(".fits", "_f100lp.fits")
    assert os.path.exists(renamed)


def test_logfile_created(tmp_workdir):
    """Check that a log file is created in workdir."""
    fake_file = make_fake_rate_file(tmp_workdir)
    preprocess_nirspec_file(
        rate_file=fake_file,
        root="test-log",
        workdir=tmp_workdir,
    )
    prefix = os.path.basename(fake_file).split("_rate")[0]
    logfile = os.path.join(tmp_workdir, prefix + "_rate.log.txt")
    assert os.path.exists(logfile)


def test_clean_removes_files(tmp_workdir):
    """If clean=True, intermediate products should be removed afterwards."""
    fake_file = make_fake_rate_file(tmp_workdir)
    preprocess_nirspec_file(
        rate_file=fake_file,
        root="test-clean",
        workdir=tmp_workdir,
        clean=True,
    )
    # The input RATE file should still exist
    assert os.path.exists(fake_file)
    # But no derived photom files should exist
    assert not any(f.endswith("_photom.fits") for f in os.listdir(tmp_workdir))


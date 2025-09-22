# tests/test_lines.py
import numpy as np
import pytest
from pathlib import Path
from astropy.io import ascii, fits
import matplotlib
matplotlib.use("Agg")  # avoid GUI issues
import matplotlib.pyplot as plt

from cosmicdawn.analysis import lines


@pytest.fixture
def fake_data(tmp_path):
    """Create a fake set of input/output files for line fitting plots."""
    root = "stack_z6.5"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output" / "emission_fits"
    input_dir.mkdir(parents=True)
    (tmp_path / "output").mkdir()
    output_dir.mkdir(parents=True)

    # ---- full spectrum (ASCII) ----
    lam = np.linspace(1000, 5000, 200)  # Å
    flux = np.ones_like(lam) * 1e-20 + 1e-21 * np.sin(lam / 200)
    err = np.full_like(flux, 1e-21)
    ascii.write(
        {"lambda(AA)": lam, "flux_nu": flux, "eflux_nu_lines": err},
        input_dir / f"{root}_full.txt",
        overwrite=True
    )

    # ---- region spectra ----
    regions = ["CIII", "Balmer", "OIIIHbeta", "HeI", "Halpha"]
    for region in regions:
        ascii.write(
            {
                "lambda(AA)": lam,
                "flux_nu": flux,
                "eflux_nu_lines": err,
            },
            input_dir / f"{root}_{region}.txt",
            overwrite=True
        )

    # ---- continuum model fits (FITS) ----
    hdu_cols = [
        fits.Column(name="wave_model", array=lam, format="D"),
        fits.Column(name="f_model_noline_50", array=np.ones_like(lam) * 1e-20, format="D"),
    ]
    hdu = fits.BinTableHDU.from_columns(hdu_cols)
    hdu.header["SCALE"] = 1.0
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        tmp_path / "output" / f"gsf_spec_{root}_full.fits", overwrite=True
    )
    for region in regions:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            tmp_path / "output" / f"gsf_spec_{root}_{region}.fits", overwrite=True
        )

    # ---- emission line fits ----
    fem = fits.HDUList([fits.PrimaryHDU(), hdu])
    fem[1].header["FLUX_ERR_HEII_1"] = 1e-21
    fem[1].header["A_HEII_1"] = 1e-20
    fem[1].header["SIGMA_HEII_1"] = 2.0
    fem.writeto(output_dir / f"{root}_emission_lines.fits", overwrite=True)

    return root, str(input_dir), str(tmp_path / "output")


def test_plot_line_fitting_results_runs(fake_data):
    """Ensure plotting function runs end-to-end with synthetic files."""
    root, input_dir, output_dir = fake_data

    # Call function (should pop up a matplotlib figure, but Agg backend avoids GUI)
    lines.plot_line_fitting_results(root, input_dir, output_dir)

    # Ensure a figure was created
    fig = plt.gcf()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0


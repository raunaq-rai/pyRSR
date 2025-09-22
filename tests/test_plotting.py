# tests/test_plotting.py

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import os
import tempfile

from cosmicdawn.general import plotting


def test_style_axes_labels():
    fig, ax = plt.subplots()
    ax = plotting.style_axes(ax, xlabel="λ [Å]", ylabel="Flux")
    assert ax.get_xlabel() == "λ [Å]"
    assert ax.get_ylabel() == "Flux"
    plt.close(fig)


def test_plot_spectrum_runs_and_labels():
    lam = np.linspace(1000, 2000, 100)
    flux = np.sin(lam / 200.) * 1e-18
    err = np.full_like(flux, 1e-19)

    ax = plotting.plot_spectrum(lam, flux, err=err, label="Test spectrum")
    # Check label added
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Test spectrum" in labels
    plt.close(ax.figure)


def test_plot_filters_runs(tmp_path):
    # Generate two fake filter files
    lam = np.linspace(4000, 8000, 50)
    resp1 = np.exp(-0.5*((lam-5000)/200.)**2)
    resp2 = np.exp(-0.5*((lam-7000)/300.)**2)

    f1 = tmp_path / "filt1.txt"
    f2 = tmp_path / "filt2.txt"
    np.savetxt(f1, np.column_stack([lam, resp1]))
    np.savetxt(f2, np.column_stack([lam, resp2]))

    ax = plotting.plot_filters([f1.name, f2.name], filter_dir=tmp_path)
    # Check that 2 lines were plotted
    assert len(ax.lines) == 2
    plt.close(ax.figure)


def test_make_cutout_runs(tmp_path):
    from astropy.io import fits

    # Create a fake FITS file with WCS keywords
    data = np.random.rand(100, 100)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CRPIX1"], header["CRPIX2"] = 50, 50
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CD1_1"], header["CD1_2"] = -0.000277, 0
    header["CD2_1"], header["CD2_2"] = 0, 0.000277
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    input_fits = tmp_path / "input.fits"
    output_fits = tmp_path / "cutout.fits"
    hdu.writeto(input_fits)

    out = plotting.make_cutout(
        input_fits,
        output_fits,
        ra=150.0,
        dec=2.0,
        size_arcsec=5.0,
        plot_cutout=False
    )

    assert os.path.exists(out)

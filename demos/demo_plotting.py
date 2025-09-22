# demos/demo_plotting.py

import numpy as np
import matplotlib.pyplot as plt
from cosmicdawn.general import plotting

def demo_plot_spectrum():
    lam = np.linspace(1000, 2000, 200)
    flux = np.exp(-0.5*((lam-1500)/50)**2) * 1e-18
    err = np.full_like(flux, 2e-19)

    ax = plotting.plot_spectrum(lam, flux, err=err, label="Gaussian line", color="blue")
    ax.set_title("Demo: Spectrum with Error Shading")
    plt.show()

def demo_plot_filters():
    lam = np.linspace(4000, 8000, 200)
    resp1 = np.exp(-0.5*((lam-5000)/200.)**2)
    resp2 = np.exp(-0.5*((lam-7000)/300.)**2)


    ax = plotting.plot_filters(filter_arrays=[(lam, resp1), (lam, resp2)])
    ax.set_title("Demo: Synthetic Filters (arrays, no files)")
    plt.show()

def demo_make_cutout():
    """
    Demo for making an in-memory cutout (no FITS files written).
    Uses a fake WCS header and random image data.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    # Fake FITS with WCS
    data = np.random.rand(100, 100)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header
    header["CRPIX1"], header["CRPIX2"] = 50, 50
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CD1_1"], header["CD1_2"] = -0.000277, 0
    header["CD2_1"], header["CD2_2"] = 0, 0.000277
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"

    # Make cutout directly in memory
    cutout_data, cutout_wcs = plotting.make_cutout_inmemory(
        data, header,
        ra=150.0, dec=2.0,
        size_arcsec=5.0,
        plot_cutout=True
    )

    return cutout_data, cutout_wcs


if __name__ == "__main__":
    demo_plot_spectrum()
    demo_plot_filters()
    demo_make_cutout()


# cosmicdawn/preprocessing/combine.py
import os, glob, warnings
import numpy as np
from astropy.io import fits
from jwst import datamodels

__all__ = ["make_level3_for_source"]

def _extract_1d_from_slit(slit):
    """
    Boxcar+IVAR 1D extraction from a jwst.datamodels slit (from *_s2d.fits).
    """
    sci = np.array(slit.data, dtype=float)

    # Fallbacks if err/dq are missing or wrong shape
    err = getattr(slit, "err", None)
    dq  = getattr(slit, "dq", None)

    if err is None or not hasattr(err, "shape") or err.shape != sci.shape:
        err = np.full_like(sci, np.nan, dtype=float)
    else:
        err = np.array(err, dtype=float)

    if dq is None or not hasattr(dq, "shape") or dq.shape != sci.shape:
        dq = np.zeros_like(sci, dtype=int)
    else:
        dq = np.array(dq, dtype=int)

    # Good pixels
    good = (dq == 0) & np.isfinite(sci) & np.isfinite(err) & (err > 0)

    ny, nx = sci.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    # Wavelength map
    out = slit.meta.wcs(xx, yy)
    if isinstance(out, tuple):
        lam = out[-1]  # take last output (usually λ)
    else:
        lam = out
    lam = np.array(lam, dtype=float)

    # Inverse-variance collapse
    ivar = np.zeros_like(sci)
    iv = np.where(good, 1.0 / np.square(err, dtype=float), 0.0)
    ivar[good] = iv[good]

    num = np.nansum(sci * ivar, axis=0)
    den = np.nansum(ivar,      axis=0)
    flux1d = np.where(den > 0, num / den, np.nan)
    err1d  = np.where(den > 0, 1.0 / np.sqrt(den), np.nan)

    # Wavelength = median across rows
    wave1d = np.nanmedian(lam, axis=0)

    det = getattr(slit.meta, "detector", "NRS?")
    m = np.isfinite(wave1d) & np.isfinite(flux1d) & np.isfinite(err1d)
    return wave1d[m], flux1d[m], err1d[m], str(det)


def _coadd_on_grid(waves, fluxes, errs, grid):
    """Linear-interp each spectrum to `grid` then IVAR coadd."""
    all_f = []
    all_iv = []
    for w, f, e in zip(waves, fluxes, errs):
        if w.size < 4:
            continue
        f_i = np.interp(grid, w, f, left=np.nan, right=np.nan)
        e_i = np.interp(grid, w, e, left=np.nan, right=np.nan)
        iv_i = np.where(np.isfinite(e_i) & (e_i > 0), 1.0/np.square(e_i), 0.0)
        all_f.append(f_i); all_iv.append(iv_i)

    if not all_f:
        return grid.copy(), np.full_like(grid, np.nan), np.full_like(grid, np.nan)

    F = np.vstack(all_f)
    IV= np.vstack(all_iv)
    num = np.nansum(F*IV, axis=0)
    den = np.nansum(IV,   axis=0)
    coadd = np.where(den>0, num/den, np.nan)
    err   = np.where(den>0, 1.0/np.sqrt(den), np.nan)
    return grid, coadd, err

def _write_c1d(path, wave, flux, err, meta_cards=None):
    """Write a simple PRISM 1D spectrum table."""
    cols = [
        fits.Column(name="WAVELENGTH", format="E", unit="um",  array=wave.astype(np.float32)),
        fits.Column(name="FLUX",       format="E", unit="",    array=flux.astype(np.float32)),
        fits.Column(name="ERR",        format="E", unit="",    array=err.astype(np.float32)),
    ]
    hdu1 = fits.BinTableHDU.from_columns(cols, name="SPEC1D")
    pri  = fits.PrimaryHDU()
    hdul = fits.HDUList([pri, hdu1])
    # minimal, helpful headers
    if meta_cards:
        for k, v in meta_cards.items():
            hdul[0].header[k] = v
    hdul.writeto(path, overwrite=True)

def make_level3_for_source(
    source_id,
    in_dir="rectifiedL2b",
    out_dir="level3",
    program=None,
    grid_dlam_factor=1.0,
    make_png=True,
):
    """
    Build L3 (combined 1D) for a given source_id from *_s2d.fits in `in_dir`.

    Outputs
    -------
    {out_dir}/{key}_NRS1_c1d.fits
    {out_dir}/{key}_NRS2_c1d.fits
    {out_dir}/{key}_merged_c1d.fits
    (+ optional PNG quicklooks)

    where key = f"jw{program}_{source_id}" if program provided, else f"{source_id}"
    """
    os.makedirs(out_dir, exist_ok=True)
    key = f"jw{program}_{source_id}" if program is not None else f"{source_id}"

    s2d_files = sorted(glob.glob(os.path.join(in_dir, "*_s2d.fits")))
    if len(s2d_files) == 0:
        raise FileNotFoundError(f"No *_s2d.fits in {in_dir}")

    det_w, det_f, det_e = {"NRS1": [], "NRS2": []}, {"NRS1": [], "NRS2": []}, {"NRS1": [], "NRS2": []}
    grabbed = 0

    for f in s2d_files:
        with datamodels.open(f) as model:
            if not hasattr(model, "slits"):
                continue
            for slit in model.slits:
                sid = getattr(slit, "source_id", None)
                if sid is None or int(sid) != int(source_id):
                    continue
                w, fl, er, det = _extract_1d_from_slit(slit)
                if det not in det_w:
                    det = "NRS1" if "1" in det else "NRS2"
                det_w[det].append(w); det_f[det].append(fl); det_e[det].append(er)
                grabbed += 1

    if grabbed == 0:
        raise FileNotFoundError(f"No slits with source_id={source_id} found in {in_dir}/*.s2d.fits")

    # Coadd per detector
    out_products = {}
    for det in ("NRS1", "NRS2"):
        if len(det_w[det]) == 0:
            continue
        # build a sensible grid
        wmin = max(np.min([w.min() for w in det_w[det] if w.size]), 0.0)
        wmax = np.max([w.max() for w in det_w[det] if w.size])
        dls  = [np.nanmedian(np.diff(w)) for w in det_w[det] if w.size > 5]
        dlam = np.nanmedian(dls) if dls else (wmax - wmin) / 2000.0
        grid = np.arange(wmin, wmax, grid_dlam_factor * dlam)

        wave_c, flux_c, err_c = _coadd_on_grid(det_w[det], det_f[det], det_e[det], grid)
        path = os.path.join(out_dir, f"{key}_{det}_c1d.fits")
        _write_c1d(path, wave_c, flux_c, err_c,
                   meta_cards={"DETECTOR": det, "SRCID": int(source_id), "PROGRAM": (program or 0)})
        out_products[det] = (wave_c, flux_c, err_c, path)

    # Merge NRS1+NRS2 (simple concatenate with small overlap trim)
    if "NRS1" in out_products and "NRS2" in out_products:
        w1, f1, e1, _ = out_products["NRS1"]
        w2, f2, e2, _ = out_products["NRS2"]
        cut = (w2 > (w1.max() + 1e-6))
        w_merged = np.concatenate([w1, w2[cut]])
        f_merged = np.concatenate([f1, f2[cut]])
        e_merged = np.concatenate([e1, e2[cut]])
        merged_path = os.path.join(out_dir, f"{key}_merged_c1d.fits")
        _write_c1d(merged_path, w_merged, f_merged, e_merged,
                   meta_cards={"DETECTOR": "NRS12", "SRCID": int(source_id), "PROGRAM": (program or 0)})
        out_products["MERGED"] = (w_merged, f_merged, e_merged, merged_path)

    # Optional quicklook
    if make_png:
        try:
            import matplotlib.pyplot as plt
            for det in ("NRS1", "NRS2", "MERGED"):
                if det not in out_products: 
                    continue
                w, f, e, p = out_products[det]
                png = p.replace(".fits", ".png")
                plt.figure(figsize=(8,3))
                plt.plot(w, f, lw=1)
                with np.errstate(divide="ignore"):
                    plt.fill_between(w, f-e, f+e, alpha=0.2, step="mid")
                plt.xlabel("Wavelength [μm]"); plt.ylabel("Flux")
                ttl = f"{key} {det} (N={len(det_w.get(det, []))})"
                plt.title(ttl); plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(png, dpi=140); plt.close()
        except Exception:
            pass

    return {k: v[3] for k, v in out_products.items()}

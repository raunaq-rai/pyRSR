# excels_pipeline/rectify.py

import os
import glob

from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep
from jwst.extract_2d import Extract2dStep
from jwst.resample import ResampleSpecStep


def rectify_photom_file(f, out_dir, context="jwst_1183.pmap"):
    """
    Rectify a single *_photom.fits file to Level-2b (_s2d.fits).
    """

    base = os.path.basename(f).replace("_photom.fits", "")
    wcs_file = os.path.join(out_dir, f"{base}_wcs.fits")
    x2d_file = os.path.join(out_dir, f"{base}_x2d.fits")
    s2d_file = os.path.join(out_dir, f"{base}_s2d.fits")

    os.makedirs(out_dir, exist_ok=True)

    # overwrite: just delete existing files if present
    for fn in [wcs_file, x2d_file, s2d_file]:
        if os.path.exists(fn):
            os.remove(fn)

    # 1. AssignWCS
    print("→ AssignWCS")
    wcs_model = AssignWcsStep.call(f)
    wcs_model.save(wcs_file, overwrite=True)

    # 2. Extract2D
    print("→ Extract2D")
    x2d_model = Extract2dStep.call(wcs_model)   # pass model, not path
    x2d_model.save(x2d_file, overwrite=True)

    # 3. ResampleSpec
    print("→ ResampleSpec")
    s2d_model = ResampleSpecStep.call(x2d_model)  # pass model again
    s2d_model.save(s2d_file, overwrite=True)

    # clean up
    wcs_model.close()
    x2d_model.close()
    s2d_model.close()

    print(f"✓ Rectified product written: {s2d_file}")
    return s2d_file


def batch_rectify(in_dir="calibratedL2", out_dir="rectifiedL2b", context="jwst_1183.pmap"):
    """
    Find and rectify all *_photom.fits in in_dir/*/, save results under out_dir.
    """
    photom_files = sorted(glob.glob(os.path.join(in_dir, "*", "*_photom.fits")))
    print(f"Found {len(photom_files)} *_photom.fits files in {in_dir}/*/")

    for f in photom_files:
        rectify_photom_file(f, out_dir, context=context)

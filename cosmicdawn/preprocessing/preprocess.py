# excels_pipeline/preprocess.py
import os, time, glob, yaml
import numpy as np
from astropy.io import fits as pyfits
import grizli
from grizli import utils, jwst_utils
import jwst
import msaexp
from msaexp import pipeline_extended, msa


def preprocess_nirspec_file(
    rate_file,
    root="excels-dataset",
    fixed_slit=None,
    rename_f070=False,
    context="jwst_1225.pmap",
    clean=False,
    extend_wavelengths=True,
    undo_flat=True,
    by_source=True,
):
    """
    Preprocess a single NIRSpec MSA or fixed-slit exposure with msaexp.

    Parameters
    ----------
    rate_file : str
        Path to *_rate.fits file
    root : str
        Output directory or tag for results
    fixed_slit : str or None
        If not None, e.g. "S200A1", force fixed slit mode
    by_source : bool
        If True, also save per-source rectified 2D spectra
    """

    # --- CRDS setup ---
    os.environ.setdefault("CRDS_PATH", os.path.expanduser("~/crds_cache"))
    os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
    os.environ["CRDS_CONTEXT"] = os.environ["CRDS_CTX"] = context
    jwst_utils.set_crds_context()

    file_prefix = os.path.basename(rate_file).replace("_rate.fits", "")
    WORKPATH = os.getcwd()

    # Logging
    _ORIG_LOGFILE = utils.LOGFILE
    _NEW_LOGFILE = os.path.join(WORKPATH, f"{file_prefix}_rate.log.txt")
    utils.LOGFILE = _NEW_LOGFILE

    # --- Log versions ---
    msg = f"""# {rate_file} {root}
  jwst version = {jwst.__version__}
grizli version = {grizli.__version__}
msaexp version = {msaexp.__version__}
"""
    utils.log_comment(utils.LOGFILE, msg, verbose=False)

    # --- Patch F070LP if needed ---
    use_file = rate_file
    if rename_f070:
        with pyfits.open(rate_file) as im:
            filt = im[0].header.get("FILTER", "")
            if filt == "F070LP":
                im[0].header["FILTER"] = "F100LP"
                new_file = rate_file.replace("_rate.fits", "_F100LP_rate.fits")
                im.writeto(new_file, overwrite=True)
                os.remove(rate_file)
                use_file = new_file
                utils.log_comment(utils.LOGFILE, f"Renamed {rate_file} → {new_file}", verbose=True)

    # --- Reset bad DQ=4 pixels ---
    with pyfits.open(use_file, mode="update") as im:
        im["DQ"].data -= im["DQ"].data & 4
        im.flush()

    # --- Run msaexp pipeline ---
    try:
        pipe = pipeline_extended.run_pipeline(
            use_file,
            slit_index=0,
            all_slits=True,
            write_output=True,   # writes per-slitlet files
            set_log=False,
            skip_existing_log=False,
            undo_flat=undo_flat,
            preprocess_kwargs={"do_nsclean": True, "n_sigma": 2},
        )
    except Exception as e:
        utils.log_comment(utils.LOGFILE, f"Pipeline failed: {str(e)}", verbose=False)
        utils.LOGFILE = _ORIG_LOGFILE
        return None

    # --- Save summary product ---
    if pipe is not None:
        if fixed_slit:
            photom_file = os.path.join(root, f"{file_prefix}_fs-photom.fits")
        else:
            photom_file = os.path.join(root, f"{file_prefix}_photom.fits")

        os.makedirs(os.path.dirname(photom_file), exist_ok=True)
        pipe.write(photom_file, overwrite=True)
        print(f"Write {photom_file}")

        # --- Extra: save per-source cutouts if requested ---
        from jwst.datamodels import SlitModel

# --- Extra: save per-source cutouts if requested ---
        if by_source and hasattr(pipe, "slits"):
            for slit in pipe.slits:
                sid = getattr(slit, "source_id", None)
                if sid is None:
                    continue

                out = os.path.join(root, f"{file_prefix}_photom.{sid}.fits")

                # Create a new SlitModel and fill it
                slit_model = SlitModel()
                if hasattr(slit, "data"):
                    slit_model.data = np.array(slit.data, copy=True)
                if hasattr(slit, "dq"):
                    slit_model.dq = np.array(slit.dq, copy=True)
                if hasattr(slit, "err"):
                    slit_model.err = np.array(slit.err, copy=True)

                # Copy over metadata fields if they exist
                for attr in ["name", "source_id", "shutter_id"]:
                    if hasattr(slit, attr):
                        setattr(slit_model, attr, getattr(slit, attr))

                # Save to disk
                slit_model.save(out, overwrite=True)
                slit_model.close()
                print(f"  ↳ wrote source_id={sid} → {out}")




    print(f"Processed {os.path.basename(rate_file)}")

    utils.LOGFILE = _ORIG_LOGFILE
    return pipe

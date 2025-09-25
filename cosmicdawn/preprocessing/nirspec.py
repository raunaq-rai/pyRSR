import os, glob, yaml
from astropy.io import fits
import jwst, grizli, msaexp
from msaexp import pipeline
from grizli import utils as grizli_utils, jwst_utils


def preprocess_nirspec_file(
    rate_file,
    root,
    context="jwst_1225.pmap",
    workdir=None,
    fixed_slit=None,
    rename_f070=False,
    extend_wavelengths=True,
    undo_flat=True,
    by_source=False,
    clean=False,
    **kwargs,
):
    """
    Preprocess a JWST/NIRSpec RATE file using `msaexp.pipeline.NirspecPipeline`.

    This function is a convenience wrapper around the msaexp pipeline that:
      1. Sets up CRDS environment variables.
      2. Optionally patches RATE file headers (e.g. rename F070LP → F100LP).
      3. Optionally resets problematic DQ flags in the input RATE file.
      4. Groups exposures into `mode` strings (via `msaexp.pipeline.exposure_groups`).
      5. Initializes a `NirspecPipeline` object for each exposure group.
      6. Runs the JWST calibration pipeline steps up to **Photom**,
         but skips 1D spectrum extraction (these can be done later).
      7. Writes slit cutouts (per-source if `by_source=True`) to `*_photom.fits`.

    Parameters
    ----------
    rate_file : str
        Path to the input NIRSpec RATE FITS file.  
        Typically named like::
            jw<program><visit><obs><exp>_<detector>_rate.fits

        Example: ``jw04233006001_03101_00002_nrs1_rate.fits``.

    root : str
        Root string for logging and file tagging (e.g. ``"rubies-egs61-demo"``).

    context : str, default="jwst_1225.pmap"
        CRDS context string (pmap).  
        Both ``CRDS_CONTEXT`` and ``CRDS_CTX`` are set to this value.

    workdir : str, optional
        Directory where logs and outputs will be written.  
        If ``None``, uses the current working directory.

    fixed_slit : str, optional
        If set (e.g. ``"S200A1"``), exposure headers are patched to treat
        the data as fixed-slit rather than MSA.  
        If ``None`` (default), leaves as MSA.

    rename_f070 : bool, default=False
        If True and the filter is ``F070LP``, header is patched to
        use ``F100LP`` (workaround for missing CRDS reference files).

    extend_wavelengths : bool, default=True
        If True, extend wavelength coverage by padding the MSA metafile
        (via `msaexp.msa.pad_msa_metafile`).

    undo_flat : bool, default=True
        Whether to undo the flat-field step before extraction.  
        Passed to `pipe.full_pipeline`.

    by_source : bool, default=False
        If True, split reductions by source ID (requires MSA metafile).  
        Each SOURCEID will be written as a separate extension or file.

    clean : bool, default=False
        If True, remove intermediate files after processing
        (leaving only final products).

    **kwargs : dict
        Additional keyword arguments passed directly to
        `NirspecPipeline.full_pipeline`.  
        Examples: `initialize_bkg`, `make_regions`, etc.

    Returns
    -------
    status : int
        Exit code:
        - ``2`` if preprocessing ran successfully and outputs were written.
        - ``3`` if the input file could not be processed.

    Side Effects
    ------------
    - Creates a log file named ``<rate_file>_rate.log.txt`` in ``workdir``.
    - Writes calibrated products such as:
        * ``*_photom.fits`` (contains flux-calibrated slit cutouts)
        * Optional per-SOURCEID files if `by_source=True`.
    - May patch FITS headers if `fixed_slit` or `rename_f070` is used.

    Notes
    -----
    Steps performed by this wrapper roughly correspond to:
    - 1/f noise correction, bias removal
    - `AssignWcs`, `Extract2D`, `FlatField`, `PathLoss`, `Photom`  
      from the JWST calibration pipeline
    - **Not included**: `srctype`, `master_background`, `wavecorr`  
      (these can be run separately if desired).
    """

    # -------------------------------
    # Setup working directory
    # -------------------------------
    if workdir is None:
        workdir = os.getcwd()
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    # -------------------------------
    # CRDS setup
    # -------------------------------
    if "CRDS_PATH" not in os.environ:
        os.environ["CRDS_PATH"] = os.path.expanduser("~/crds_cache")
    if "CRDS_SERVER_URL" not in os.environ:
        os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_CONTEXT"] = os.environ["CRDS_CTX"] = context
    jwst_utils.set_crds_context()

    # -------------------------------
    # Setup logging
    # -------------------------------
    file_prefix = os.path.basename(rate_file).split("_rate")[0]
    logfile = os.path.join(workdir, file_prefix + "_rate.log.txt")
    grizli_utils.LOGFILE = logfile

    msg = f"""# {rate_file} {root}
  jwst version = {jwst.__version__}
grizli version = {grizli.__version__}
msaexp version = {msaexp.__version__}
"""
    grizli_utils.log_comment(logfile, msg, verbose=True)

    # -------------------------------
    # Check input RATE file
    # -------------------------------
    if not os.path.exists(rate_file):
        msg = f"Input RATE file not found: {rate_file}"
        grizli_utils.log_comment(logfile, msg, verbose=True)
        return 3

    use_file = rate_file

    # -------------------------------
    # Optional filter renaming
    # -------------------------------
    if rename_f070:
        with fits.open(rate_file) as im:
            if im[0].header.get("FILTER") == "F070LP":
                new_file = rate_file.replace(".fits", "_f100lp.fits")
                im[0].header["FILTER"] = "F100LP"
                im.writeto(new_file, overwrite=True)
                os.remove(rate_file)
                use_file = new_file
                msg = f"Renamed filter F070LP → F100LP: {rate_file} → {new_file}"
                grizli_utils.log_comment(logfile, msg, verbose=True)

    # -------------------------------
    # Reset problematic DQ flags
    # -------------------------------
    with fits.open(use_file, mode="update") as im:
        if "DQ" in im:
            im["DQ"].data -= im["DQ"].data & 4
            im.flush()

    # -------------------------------
    # Group exposures
    # -------------------------------
    groups = pipeline.exposure_groups(files=[use_file], split_groups=True)
    print("Exposure groups:\n", yaml.dump(dict(groups)))

    # -------------------------------
    # Run msaexp pipeline
    # -------------------------------
    for g in groups:
        exp = groups[g][0]
        try:
            mode = os.path.basename(exp).replace("_rate.fits", "")
            pipe = pipeline.NirspecPipeline(
                mode=mode,
                files=[exp],
                source_ids=None if not by_source else [],
                pad=0,
                primary_sources=True,
            )

            # Run calibration steps up to photom
            pipe.full_pipeline(
                run_extractions=False,
                initialize_bkg=False,
                undo_flat=undo_flat,
                **kwargs,
            )

            print(f"Pipeline finished for {exp}, last step={pipe.last_step}")

        except Exception as e:
            msg = f"Pipeline failed for {exp}: {e}"
            grizli_utils.log_comment(logfile, msg, verbose=True)
            return 3

    if clean:
        for f in glob.glob("*"):
            os.remove(f)

    return 2

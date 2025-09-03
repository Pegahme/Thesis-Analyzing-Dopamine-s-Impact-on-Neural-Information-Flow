#!/usr/bin/env python
"""
Batch process a single *_preprocessed_SimonTask_epo.fif file passed on the command line,
and save source time courses in a 3D array (labels × timepoints × epochs).
"""

import sys
import os
import time
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.parallel import parallel_func
from mne.minimum_norm import (
    make_inverse_operator,
    prepare_inverse_operator,
    apply_inverse_epochs,
)

# =========================
#  INPUT & PATH SETTINGS
# =========================
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path_to_preprocessed_epoch_fif>")
    sys.exit(1)
fif_path = Path(sys.argv[1])

preprocessed_dir = Path("/home/majlepy2/myproject/PreProc")
save_dir_base = Path(os.environ.get(
    "SAVE_BASE", "/lustre/majlepy2/myproject/SourceLoc_3d_downsampled"
))
excel_file = preprocessed_dir / "preproc_ds003509.xlsx"

# Load metadata table
data = pd.read_excel(excel_file)

# Ensure input file exists and is in the right folder
if not fif_path.exists() or fif_path.parent != preprocessed_dir:
    raise ValueError(f"File {fif_path} not found in {preprocessed_dir}")

# =========================
#  MAIN PROCESSING
# =========================
fif = fif_path
try:
    start_all = time.time()
    parts = fif.stem.split("_")
    subject_id, session_id = parts[0], parts[1]

    print(f"\n=== Processing {fif.name} ({subject_id}, {session_id}) ===")

    # Check metadata entry exists
    if data[data["participant_id"] == subject_id].empty:
        raise ValueError(f"Metadata missing for {subject_id}")

    # Prepare output directory
    out_dir = save_dir_base / subject_id / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{subject_id}_{session_id}_label_tc3d.npy"
    if out_file.exists():
        print(f"Skipping {subject_id}/{session_id}, already exists → {out_file}")
        sys.exit(0)

    # -------------------------
    # Load & preprocess epochs
    # -------------------------
    epochs = mne.read_epochs(str(fif), preload=True)
    epochs.set_montage("standard_1020", match_case=False)
    epochs.resample(20, npad="auto")
    epochs.set_eeg_reference(ref_channels="average", projection=True)

    # -------------------------
    # Forward & noise model
    # -------------------------
    noise_cov = mne.compute_covariance(epochs, method="shrunk", rank=None)
    fs_dir = Path("/home/majlepy2/myproject/fsaverage")
    subjects_dir = fs_dir.parent
    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    fwd = mne.make_forward_solution(
        epochs.info, trans="fsaverage", src=src, bem=bem,
        eeg=True, meg=False, mindist=5.0
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True
    )

    # -------------------------
    # Inverse operator
    # -------------------------
    snr = 1.0
    lambda2 = 1.0 / snr**2
    inverse_operator = make_inverse_operator(
        epochs.info, fwd, noise_cov,
        fixed=True, depth=None, rank=None, use_cps=True
    )
    method = "eLORETA"
    print(f"  vertices: {sum(len(s['vertno']) for s in fwd['src'])}")

    # -------------------------
    # ROI labels
    # -------------------------
    labels = mne.read_labels_from_annot(
        "fsaverage", parc="aparc", subjects_dir=subjects_dir
    )
    labels = [lbl for lbl in labels if "unknown" not in lbl.name]

    # -------------------------
    # Warm-up inverse
    # -------------------------
    inv_op_prepared = prepare_inverse_operator(
        inverse_operator, nave=1, lambda2=lambda2, method=method, copy=True
    )
    _ = apply_inverse_epochs(
        epochs[:1], inv_op_prepared, lambda2, method,
        pick_ori=None, return_generator=False,
        prepared=True, use_cps=True
    )

    # -------------------------
    # Apply inverse in parallel
    # -------------------------
    parallel, run_inv, _ = parallel_func(
        apply_inverse_epochs, n_jobs=2
    )
    stc_lists = parallel(
        run_inv(
            epochs[[i]], inv_op_prepared, lambda2, method,
            pick_ori=None, return_generator=False,
            prepared=True, use_cps=True
        )
        for i in range(len(epochs))
    )
    stcs = [lst[0] for lst in stc_lists]

    # -------------------------
    # Extract label time courses
    # -------------------------
    all_tcs = [
        mne.extract_label_time_course(
            stc, labels, src=fwd["src"], mode="pca_flip"
        )
        for stc in stcs
    ]
    arr3d = np.stack(all_tcs, axis=2)
    np.save(out_file, arr3d)

    elapsed = (time.time() - start_all) / 60
    print(f"Done in {elapsed:.2f} min, saved 3D array → {out_file}")

except Exception:
    print(f"Error processing {fif.name}:", flush=True)
    traceback.print_exc()
    sys.exit(1)

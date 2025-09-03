#!/usr/bin/env python3
"""
Effective connectivity inference using IDTxl's Multivariate Transfer Entropy (Gaussian estimator).

Purpose
-------
Runs an MTE analysis for a SINGLE target node on a SINGLE .npy input file and
writes (a) a checkpoint for resuming partial runs and (b) a final pickled result.

Inputs
------
--input-file : path to a NumPy .npy array with shape (node, samples, trials)
               - 23 = number of nodes 
               - samples = time points per trial/epoch
               - trials = number of trials/epochs
--cmi-estimator : IDTxl CMI estimator string, default "JidtGaussianCMI"
--target : integer in [0..22], index of the target node

Outputs (per target)
--------------------
- Checkpoint files:   <output_dir>/<subj>_<ses>_gau_50_targetXX.ckp.*   (created by IDTxl)
- Final result (pkl): <output_dir>/<subj>_<ses>_gau_50_targetXX.pkl
- Log file:           <output_dir>/<subj>_<ses>_gau_50_targetXX.log

Conventions & Assumptions
-------------------------
- Subject/session inferred from the input filename stem via regex: (sub-\\d+)[_\\-](ses-\\d+)
- Data passed to IDTxl as 'psr' (process × sample × repetition) with normalise=True.
- Gaussian estimator via JIDT requires a working JAVA environment (JAVA_HOME set by the
  Slurm script). OMP threads pinned to 4 for reproducibility / cluster etiquette.
- Checkpoint resume is attempted first; on failure, a fresh analysis is run.

Reproducibility Notes
---------------------
- Permutation tests introduce randomness. This script does NOT set a global RNG seed.
  That means repeated runs may differ in the exact set of surrogate values and thus
  in borderline edge decisions. Results remain statistically reproducible but not
  bitwise identical. If strict determinism is required, consider pinning an RNG seed
  via IDTxl/JIDT (where supported) and/or containerizing the stack.
- The number of permutations per test (n_perm = 200) was chosen as a trade-off:
  enough to estimate significance at α = 0.05 reliably, while keeping runtime
  feasible given the large number of (file × target) analyses. Larger n_perm
  (e.g., 500–1000) increases sensitivity but comes with a substantial cost in
  CPU time and memory usage.
"""

import os
import sys
import argparse
import logging
import re
import pickle
from pathlib import Path
import numpy as np
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE

# Limit OpenMP threads (also exported in Slurm script).
os.environ['OMP_NUM_THREADS'] = '4'

def main():
    print("gaussian_mte.py started", flush=True)

    # ---------- CLI arguments ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True,
                        help="Path to .npy file (shape: 23 x samples x trials)")
    parser.add_argument('--cmi-estimator', type=str, default="JidtGaussianCMI",
                        help="IDTxl CMI estimator")
    parser.add_argument('--target', type=int, default=0,
                        help="Index of the target node (0-22)")
    args = parser.parse_args()

    print(f"args.input_file = {args.input_file}", flush=True)

    # ---------- Extract subject and session from filename ----------
    # Expecting something like ".../sub-01_ses-01_*.npy" or ".../sub-01-ses-01_*.npy"
    input_stem = Path(args.input_file).stem
    match = re.search(r'(sub-\d+)[_\-](ses-\d+)', input_stem)
    if match:
        subj, sess = match.groups()
    else:
        # Fail fast to avoid silently misfiling results
        raise ValueError(f"Could not extract subject/session from {args.input_file}")

    # ---------- Prepare output paths ----------
    # One folder per subject/session, one result per target
    output_dir = Path("/lustre/majlepy2/myproject/gaussian_mte_50") / subj / sess
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = output_dir / f"{subj}_{sess}_gau_50_target{args.target:02d}"
    ckp_file = filename_prefix.with_suffix(".ckp")   # presence indicates a prior attempt
    out_file = filename_prefix.with_suffix(".pkl")   # final pickle of the result dict
    log_file = filename_prefix.with_suffix(".log")   # run log

    # ---------- Logging to file + stdout ----------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    # ---------- Load data ----------
    logging.info(f"Loading data from: {args.input_file}")
    X = np.load(args.input_file).astype(np.float32)
    if X.shape[0] != 23:
        raise ValueError(f"Expected shape (23, samples, epochs), got: {X.shape}")
    logging.info(f"Input shape after load: {X.shape}")

    # IDTxl requires a Data object; 'psr' matches (process, sample, repetition)
    data = Data(data=X, dim_order='psr', normalise=True)
    mte = MultivariateTE()

    # ---------- IDTxl analysis settings ----------
    # Note: permutation counts and alphas are fixed here for Gaussian estimator runs.
    #       Checkpoint writing/resuming enabled via 'write_ckp' and 'filename_ckp'.
    settings = {
        'cmi_estimator': args.cmi_estimator,
        'target': args.target,
        'min_lag_sources': 1,
        'max_lag_sources': 5,
        'max_lag_target': 5,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 200,
        'alpha_max_stat': 0.05,
        'alpha_min_stat': 0.05,
        'alpha_omnibus': 0.05,
        'alpha_max_seq': 0.05,
        'permute_in_time': False,
        'verbose': True,
        'write_ckp': True,
        'filename_ckp': str(filename_prefix),
    }

    # ---------- Safe checkpoint resume ----------
    # If a checkpoint exists, try to complete from there; otherwise, start anew.
    try:
        if ckp_file.exists():
            logging.info(f"Checkpoint found. Trying to resume from checkpoint: {ckp_file}")
            result = mte.resume_checkpoint(str(filename_prefix))
            logging.info("Resumed and completed analysis from checkpoint.")
        else:
            raise FileNotFoundError
    except Exception as e:
        logging.warning(f"Could not resume from checkpoint. Starting new analysis. Reason: {e}")
        # Analyse a single target with all other nodes as candidate sources
        result = mte.analyse_single_target(settings, data, target=args.target, sources="all")
        logging.info("New analysis finished.")

    # ---------- Persist final result ----------
    logging.info(f"Saving result to: {out_file}")
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)
    logging.info("Finished and result saved.")

if __name__ == '__main__':
    main()

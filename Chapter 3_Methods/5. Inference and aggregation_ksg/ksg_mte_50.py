#!/usr/bin/env python3
"""
KSG (Kraskov) Multivariate Transfer Entropy with OpenCL acceleration via IDTxl.

Purpose
-------
Runs an MTE analysis for ONE target node on ONE .npy input file using a
KSG-style CMI estimator. Supports checkpoint resume and per-run logging.

Inputs
------
--input-file : path to a NumPy .npy array with shape (nodes, samples, trials)
               (axis order must be psr = process, sample, repetition)
--cmi-estimator : IDTxl CMI estimator string (default "OpenCLKraskovCMI")
                  Alternatives include "JidtKraskovCMI", "GaussianCMI", etc.
--target : integer in [0..22], index of the target node

Outputs
-------
For each (input file, target):
  - <subj>_<ses>_mte_50_targetTT.log  : run log
  - <subj>_<ses>_mte_50_targetTT.ckp* : IDTxl checkpoint shards (if any)
  - <subj>_<ses>_mte_50_targetTT.pkl  : pickled result dict

Assumptions
-----------
- Input filename stem contains both tokens: (sub-XX)[_-](ses-YY)
- data is (23, samples, trials) and will be normalised by IDTxl Data()
- OpenCL device selection uses the first entry of CUDA_VISIBLE_DEVICES
  purely as a convenience to mirror Slurm GPU allocation.

Reproducibility Notes
---------------------
- Permutation-based significance tests introduce randomness; no RNG seed
  is explicitly set here. Results are statistically reproducible but not
  guaranteed bitwise identical across runs/nodes.
- n_perm values are set to 200 (max/min/omnibus/max_seq) as a runtime–
  sensitivity trade-off at alpha=0.05 for large batch throughput.

GPU / Device Notes
------------------
- settings['gpuid'] is derived from CUDA_VISIBLE_DEVICES. On most clusters,
  Slurm sets CUDA_VISIBLE_DEVICES to the physical GPU(s) granted; we use
  the first listed to pick the OpenCL device index. If your OpenCL backend
  enumerates devices differently than CUDA, ensure the mapping is correct
  (or override gpuid explicitly).
"""

import os, sys, argparse, logging, re, pickle
from pathlib import Path
import numpy as np
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE

# --- CPU threading control ---
os.environ['OMP_NUM_THREADS'] = '4'  # Match --cpus-per-gpu in SLURM

def main():
    print("ksg_mte.py started", flush=True)

    # ---------- CLI ----------
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True,
                        help="Path to .npy file (shape: 23 x samples x trials or any axis order)")
    parser.add_argument('--cmi-estimator', type=str, default="OpenCLKraskovCMI",
                        help="IDTxl CMI estimator (e.g., OpenCLKraskovCMI, JidtKraskovCMI, GaussianCMI)")
    parser.add_argument('--target', type=int, default=0,
                        help="Index of the target node (0-22)")
    args = parser.parse_args()

    print(f"args.input_file = {args.input_file}", flush=True)

    # ---------- Subject/session parsing & output dir ----------
    # Expect input basename like: sub-XX_ses-YY_*.npy or sub-XX-ses-YY_*.npy
    input_stem = Path(args.input_file).stem
    match = re.search(r'(sub-\d+)[_\-](ses-\d+)', input_stem)
    if match:
        subj = match.group(1)
        sess = match.group(2)
    else:
        # Fail fast if filename doesn't contain both tokens
        raise ValueError(f"Could not extract subject/session from {args.input_file}")

    output_dir = Path("/lustre/majlepy2/myproject/ksg_mte_50") / subj / sess
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = output_dir / f"{subj}_{sess}_mte_50_target{args.target:02d}"
    ckp_file = filename_prefix.with_suffix(".ckp")
    out_file = filename_prefix.with_suffix(".pkl")
    log_file = filename_prefix.with_suffix(".log")

    # ---------- Logging ----------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    # ---------- Load & shape check ----------
    logging.info(f"Loading data from: {args.input_file}")
    X = np.load(args.input_file)
    X = X.astype(np.float32)
    if X.shape[0] != 23:
        # Enforce (process=23, sample, repetition)
        raise ValueError(f"Expected shape (23, samples, epochs), got: {X.shape}")
    logging.info(f"Input shape after fix_shape: {X.shape}")

    # IDTxl expects dim_order='psr' and can normalise internally
    data = Data(data=X, dim_order='psr', normalise=True)
    mte = MultivariateTE()

    # ---------- Analysis settings ----------
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
        'filename_ckp': str(filename_prefix),  # Correct: no double .ckp
    }

    # ---------- GPU device selection for OpenCL estimator ----------
    # Map the first visible CUDA device to IDTxl's gpuid (OpenCL device index).
    # If OpenCL enumeration differs from CUDA on your system, adjust externally.
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    gpuid = int(cuda_visible.split(',')[0])
    logging.info(f"Using GPU device ID: {gpuid}")
    settings['gpuid'] = gpuid

    # ---------- Run (resume if checkpoint exists) ----------
    try:
        if ckp_file.exists():
            logging.info(f"Checkpoint found. Resuming from checkpoint: {ckp_file}")
            result = mte.resume_checkpoint(str(filename_prefix))
            logging.info("Resumed and completed analysis from checkpoint.")
        else:
            logging.info("No checkpoint found. Starting new analysis.")
            result = mte.analyse_single_target(settings, data, target=args.target, sources="all")
            logging.info("Analysis finished.")
    except Exception as e:
        logging.error(f"Exception during analysis: {e}", exc_info=True)
        raise

    # ---------- Persist result ----------
    logging.info(f"Saving result to: {out_file}")
    with open(out_file, 'wb') as f:
        pickle.dump(result, f)
    logging.info("Finished and result saved.")

if __name__ == '__main__':
    main()

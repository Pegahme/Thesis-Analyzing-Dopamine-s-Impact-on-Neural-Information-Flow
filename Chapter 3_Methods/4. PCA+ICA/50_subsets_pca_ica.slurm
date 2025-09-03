#!/usr/bin/env bash
#
# SLURM job for group-level PCA + ICA step
# in the pipeline: preprocessing → source localization → PCA+ICA → ...

#SBATCH --job-name=subsets_pca_ica             # Job name
#SBATCH --account=kobelxwm_0000                # Account/project
#SBATCH --partition=cpu                        # Run on CPU partition
#SBATCH -N 1                                   # One node
#SBATCH --cpus-per-task=8                      # Use 8 CPU cores
#SBATCH --mem=32G                              # Memory per job
#SBATCH --time=12:00:00                        # Max runtime
#SBATCH --output=logs/subsets_pca_ica_%j.out   # Stdout log
#SBATCH --error=logs/subsets_pca_ica_%j.err    # Stderr log

# Ensure logs directory exists
mkdir -p logs

# Load Python and activate environment with sklearn, mne, etc.
module load python/3.11.7
source ~/idtxl_env_fixed/bin/activate

# Helpful diagnostics
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Current Directory: $(pwd)"
which python
python --version

# Ensure multithreading libs (numpy, sklearn, etc.) use allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the PCA+ICA script
python /home/majlepy2/myproject/PCA_ICA/50_subsets_pca_ica.py

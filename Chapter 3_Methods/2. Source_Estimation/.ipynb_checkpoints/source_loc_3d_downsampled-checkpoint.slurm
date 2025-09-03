#!/usr/bin/env bash
#
# SLURM job script for running source localization in parallel
# using an array job over multiple preprocessed .fif files.

#SBATCH --job-name=src3d_array              # Job name shown in SLURM queue
#SBATCH --output=logs/source_loc_3d_down_%A_%a.out   # STDOUT log (per array task)
#SBATCH --error=logs/source_loc_3d_down_%A_%a.err    # STDERR log (per array task)
#SBATCH --partition=cpu                     # Partition/queue to submit to
#SBATCH --account=kobelxwm_0000             # Project account for billing
#SBATCH -N 1                                # Number of nodes
#SBATCH --cpus-per-task=4                   # CPUs per task
#SBATCH --mem=128G                          # Memory per node
#SBATCH --time=24:00:00                     # Max runtime (24h)
#SBATCH --array=0-64                        # Array indices (65 tasks total)

# ----------------------------
# Job setup
# ----------------------------
set -e                                      # Exit immediately if a command fails
mkdir -p logs                               # Ensure log directory exists

# ----------------------------
# Environment setup
# ----------------------------
module purge                                # Clear any loaded modules
module load python/3.11.7                   # Load required Python version
source ~/myproject/idtxl_env_fixed/bin/activate   # Activate virtual environment

# Base directory for saving outputs
export SAVE_BASE=/lustre/majlepy2/myproject/SourceLoc_3d_downsampled

# ----------------------------
# Run the job
# ----------------------------
cd ~/myproject

# Select the N-th line from fif_list.txt according to SLURM_ARRAY_TASK_ID
# (SLURM_ARRAY_TASK_ID starts at 0, so +1 is needed for sed line indexing)
FIF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" fif_list.txt)

echo "[$(date)] Processing $FIF_FILE on task $SLURM_ARRAY_TASK_ID"

# Run the Python processing script on the selected file
python source_loc_3d_downsampled.py "$FIF_FILE"

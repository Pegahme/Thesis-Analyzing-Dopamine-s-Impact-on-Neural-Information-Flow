#!/bin/bash
#SBATCH --job-name=00_gauss_mte_batch_50          
#SBATCH --account=kobelxwm_0000
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                                 # One Python process
#SBATCH --cpus-per-task=4                          # Matches OMP_NUM_THREADS=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-68                               # 3 files × 23 targets = 69 tasks
#SBATCH --output=logs/50_gauss_mte_batch_00_%A_%a.out
#SBATCH --error=logs/50_gauss_mte_batch_00_%A_%a.err

# Fail on errors, unset vars, and pipeline errors
set -euo pipefail
mkdir -p logs

echo "Running on $(hostname) at $(date)"

# --- Environment / modules (cluster-specific) ---
module purge
module load openmpi/5.0.5-d3ii4pq
module load python/3.11.7-watv22a

# Activate project virtualenv containing IDTxl + deps
source ~/myproject/idtxl_env_fixed/bin/activate

# JIDT / Java requirement for Gaussian estimator
export JAVA_HOME=~/myproject/jdk/jdk-11.0.27+6
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"

# Keep Python output unbuffered; match threads to --cpus-per-task
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# --- Map array index -> (file, target) ---
NUM_TARGETS=23
FILE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_TARGETS))  # integer division
TARGET_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_TARGETS))# remainder in [0..22]

# Batch list: one absolute file path per line
BATCH_FILE="/lustre/majlepy2/myproject/50_epochs_idtxl/batches/batch_00.txt" 
files=($(cat "$BATCH_FILE"))

# Guard against accidental short lists
if [[ $FILE_INDEX -ge ${#files[@]} ]]; then
  echo "ERROR: FILE_INDEX=$FILE_INDEX out of bounds (only ${#files[@]} files)"
  exit 1
fi

# Select inputs for this array task
FILE="${files[$FILE_INDEX]}"
TARGET=$TARGET_INDEX

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "FILE_INDEX           = $FILE_INDEX"
echo "TARGET_INDEX         = $TARGET_INDEX"
echo "FILE                 = $FILE"
ls -lh "$FILE" || { echo "File not found: $FILE"; exit 1; }

# --- Run Gaussian MTE for the selected (file, target) ---
python /home/majlepy2/myproject/gauss/gaussian_mte_50.py \
  --input-file "$FILE" \
  --target "$TARGET"

echo "Finished FILE=$FILE, TARGET=$TARGET"

#!/bin/bash
#SBATCH --job-name=06_ksg_mte_batch_50
#SBATCH --account=kobelxwm_0000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-68  # 3 files × 23 targets = 69 jobs
#SBATCH --output=logs/50_ksg_mte_batch_06_%A_%a.out
#SBATCH --error=logs/50_ksg_mte_batch_06_%A_%a.err

set -euo pipefail
mkdir -p logs

echo "Running on $(hostname) at $(date)"

# --- Load environment ---
module purge
module load openmpi/5.0.5-d3ii4pq
module load python/3.11.7-watv22a
module load cuda/12.6.1-jr4ru3u

source ~/myproject/idtxl_env_fixed/bin/activate

export JAVA_HOME=~/myproject/jdk/jdk-11.0.27+6
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# --- Job/target logic ---
NUM_TARGETS=23
FILE_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_TARGETS))
TARGET_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_TARGETS))

BATCH_FILE="/lustre/majlepy2/myproject/50_epochs_idtxl/batches/batch_06.txt"
files=($(cat "$BATCH_FILE"))

if [[ $FILE_INDEX -ge ${#files[@]} ]]; then
  echo "ERROR: FILE_INDEX=$FILE_INDEX out of bounds (only ${#files[@]} files)"
  exit 1
fi

# Use path from batch file directly
FILE="${files[$FILE_INDEX]}"
TARGET=$TARGET_INDEX

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "FILE_INDEX           = $FILE_INDEX"
echo "TARGET_INDEX         = $TARGET_INDEX"
echo "FILE                 = $FILE"
ls -lh "$FILE" || { echo "File not found: $FILE"; exit 1; }

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- Run analysis ---
python /home/majlepy2/myproject/ksg/ksg_mte_50.py \
  --input-file "$FILE" \
  --cmi-estimator OpenCLKraskovCMI \
  --target "$TARGET"

echo "Finished FILE=$FILE, TARGET=$TARGET"

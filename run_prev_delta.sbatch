#!/bin/bash
#
# SLURM array job: runs all 6 prev_delta scripts in parallel on Euler.
#
# Submit with:
#     sbatch run_prev_delta.sbatch
#
# Each array task runs one of the 6 compare_*_pd.py scripts on its own
# 16-CPU allocation, with 32 GB RAM and up to 24 h wall time.
#
# Output, error, and per-script logs land in ./logs/.
# Results (.pt, .json) land in ./results/.
# Figures (.pdf) land in ./figures/.

#SBATCH --job-name=prev_delta
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Activate your Python environment (must have torch, numpy, matplotlib).
#    EDIT THIS LINE so it points at your environment's activate script.
# ---------------------------------------------------------------------------
source "$HOME/path/to/your/venv/bin/activate"   # <-- EDIT ME

# ---------------------------------------------------------------------------
# 2. Thread settings.
#    The Python script already calls torch.set_num_threads(SLURM_CPUS_PER_TASK),
#    but the BLAS libraries (OpenBLAS, MKL) and OMP inspect their own env vars
#    independently, so we set them here to avoid over-/under-subscription.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# 3. Pick the script for this array task index.
# ---------------------------------------------------------------------------
SCRIPTS=(
  compare_heston_asian_pd.py
  compare_heston_asian_reduced_pd.py
  compare_heston_barrier_pd.py
  compare_heston_barrier_reduced_pd.py
  compare_lookback_pd.py
  compare_lookback_reduced_pd.py
)
SCRIPT="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results figures

echo "================================================================"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Array task:    $SLURM_ARRAY_TASK_ID  ($SCRIPT)"
echo "  Node:          $(hostname)"
echo "  CPUs:          $SLURM_CPUS_PER_TASK"
echo "  Mem alloc:     ${SLURM_MEM_PER_CPU:-?} per CPU"
echo "  Started:       $(date)"
echo "  Python:        $(which python)"
echo "  Torch:         $(python -c 'import torch; print(torch.__version__)')"
echo "================================================================"

# -u so prints are unbuffered and appear in the log in real time
python -u prev_delta/"$SCRIPT"

echo "================================================================"
echo "  Finished:      $(date)"
echo "================================================================"

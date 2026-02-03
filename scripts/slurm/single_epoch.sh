#!/bin/bash
#SBATCH --job-name=pmseg_2d_gpu
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                     # RAM
#SBATCH --time=04:00:00               # Walltime
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

# activate venv
echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

# go to repo root
cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg

# make logs dir (so SLURM output path exists)
mkdir -p logs

# ensure imports work
export PYTHONPATH=$(pwd)

# optional: avoid oversubscribing threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "starting python... $(date)"
python -u scripts/train_2d_baseline.py \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0

echo "job finished: $(date)"

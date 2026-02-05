#!/bin/bash
#SBATCH --job-name=pmseg2d
#SBATCH --output=logs/pmseg2d_%j.out
#SBATCH --error=logs/pmseg2d_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

# go to project root (important so relative paths like runs/ work)
cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg || exit 1

# activate venv
echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

# optional: avoid CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "starting python... $(date)"
python -u scripts/run_train_2d.py \
  --epochs 50 \
  --num_workers 4 \
  --log_every 100

echo "job finished: $(date)"

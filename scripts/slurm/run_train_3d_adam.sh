#!/bin/bash
#SBATCH --job-name=pmseg3d_adam
#SBATCH --output=logs/pmseg3d_adam_%j.out
#SBATCH --error=logs/pmseg3d_adam_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

# go to project root (important so relative paths like runs_3d/ work)
cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg || exit 1

# activate venv
echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

# avoid CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# make sure logs dir exists (since SLURM writes there)
mkdir -p logs

python scripts/run_train_3d.py \
  --model_variant nnunet_fullres \
  --optimizer adam \
  --lr 1e-4 \
  --epochs 10 \
  --steps_per_epoch 250 \
  --batch_size 2 \
  --patch_z 14 --patch_y 256 --patch_x 448 \
  --oversample_fg 0.33 \
  --fg_thr 0.5 \
  --jitter_z 2 --jitter_y 32 --jitter_x 32 \
  --num_workers 4 \
  --log_every 25

echo "job finished: $(date)"

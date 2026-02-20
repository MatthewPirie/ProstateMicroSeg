#!/bin/bash
#SBATCH --job-name=pmseg3d_v2
#SBATCH --output=logs_v2/pmseg3d_v2_%j.out
#SBATCH --error=logs_v2/pmseg3d_v2_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

# go to project root (important so relative paths like runs_3d_v2/ work)
cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg || exit 1

# make sure logs dir exists (since SLURM writes there)
mkdir -p logs_v2

# activate venv
echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

# avoid CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_train_3d_v2.py \
  --runs_dir runs_3d_v2 \
  --run_name "v2_${SLURM_JOB_ID}" \
  --epochs 50 \
  --steps_per_epoch 250 \
  --batch_size 2 \
  --model_variant nnunet_fullres \
  --patch_z 14 --patch_y 256 --patch_x 448 \
  --oversample_fg 0.33 \
  --oversample_mode probabilistic \
  --optimizer adam \
  --lr 1e-4 \
  --weight_decay 3e-5 \
  --lr_scheduler polynomial \
  --poly_power 0.9 \
  --min_lr 1e-6 \
  --w_bce 1.0 \
  --w_dice 1.0 \
  --val_overlap 0.5 \
  --sw_batch_size 1 \
  --num_workers 4 

echo "job finished: $(date)"

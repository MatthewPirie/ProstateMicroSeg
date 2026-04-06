#!/bin/bash
#SBATCH --job-name=pmseg_convlstm
#SBATCH --output=a_logs/pmseg_convlstm_%j.out
#SBATCH --error=a_logs/pmseg_convlstm_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg || exit 1

mkdir -p scripts/slurm/logs_convlstm

echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_train_convlstm.py \
  --train_config configs/train_convlstm/fullinplane_zwindow.yaml \
  --runs_dir runs/c_runs_convlstm \
  --run_name "convlstm_${SLURM_JOB_ID}" \
  --resume_ckpt /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/runs/c_runs_convlstm/convlstm_2968121/checkpoint_last.pt \
  --epochs 11 \
  --steps_per_epoch 150 \
  --batch_size 4 \
  --model_variant base \
  --patch_z 45 --patch_y 256 --patch_x 256 \
  --oversample_fg 0 \
  --oversample_mode probabilistic \
  --optimizer adam \
  --lr 1e-4 \
  --weight_decay 3e-5 \
  --lr_scheduler polynomial \
  --poly_power 0.9 \
  --min_lr 1e-6 \
  --w_bce 1.0 \
  --w_dice 1.5 \
  --num_val_steps 50 \
  --fullval_every 50 \
  --val_overlap 0.5 \
  --sw_batch_size 1 \
  --val_carryover \
  --num_workers 8

echo "job finished: $(date)"
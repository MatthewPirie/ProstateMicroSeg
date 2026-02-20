#!/bin/bash
#SBATCH --job-name=pmseg2d
#SBATCH --output=logs/pmseg2d_%j.out
#SBATCH --error=logs/pmseg2d_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "job started: $(date)"
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi failed"
python -c "import torch; print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available());"
scontrol show job $SLURM_JOB_ID | egrep -i "command=|partition=|gres=|tres=|node"

# go to project root (important so relative paths like runs/ work)
cd /home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg || exit 1

# activate venv
echo "activating venv... $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated. $(date)"

# optional: avoid CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "starting python... $(date)"
python scripts/run_train_2d.py \
  --epochs 50 \
  --optimizer adam \
  --lr 3e-4 \
  --lr_scheduler cosine \
  --weight_decay 0.0 \
  --num_workers 4 \
  --w_bce 1.0 \
  --w_dice 1.5

echo "job finished: $(date)"

#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2
#SBATCH --error=slurm/logs/llama-3b-%j.err
#SBATCH --output=slurm/logs/llama-3b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
RECONS_LOSS=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9)
combinations=()
for rcl in "${RECONS_LOSS[@]}"; do
  combinations+=("${rcl}")
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r reconstruction_loss_factor <<< "$experiments"

# train debugging script for llama 3B model
WORK_DIR=/home/woody/iwbi/iwbi106h/suuraj
JOB_DIR=$WORK_DIR/slurm_scratch/$SLURM_JOB_ID
mkdir -p $JOB_DIR

rsync -av --exclude='__pycache__/' \
    --exclude='slurm/logs/' \
    --exclude='wandb/' \
    --exclude='checkpoints/' \
    $WORK_DIR/codes/predicting-hidden-states/predicting_hidden_states/ \
    $JOB_DIR

cd $JOB_DIR
printf "\nRunning job in $JOB_DIR\n"

python exp_script.py \
    metric_logger.mode=online \
    model.self_prediction_module.codebook_dim=128 \
    model.self_prediction_module.num_quantizers=3 \
    model.self_prediction_module.reconstruction_loss_factor=$reconstruction_loss_factor \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.2 \
    temperature_scheduler.global_steps=10000 \
    optimizer.lr=1e-2 \
    batch_size=8
#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
# #SBATCH --array=0-1%2
#SBATCH --error=slurm/logs/llama-3b-%j.err
#SBATCH --output=slurm/logs/llama-3b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
SELF_CRITIC_LOSS=(1e-1)
combinations=()
for lf in "${SELF_CRITIC_LOSS[@]}"; do
  combinations+=("${lf}")
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r self_critic_loss_factor <<< "$experiments"

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
    model.self_prediction_module.reconstruction_loss_factor=1e-6 \
    model.self_prediction_module.num_quantizers=2 \
    model.self_prediction_module.codebook_dim=3072 \
    model.self_critic_loss_factor="$self_critic_loss_factor" \
    model.phi_loss_factor=0.001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=10000 \
    batch_size=8 \
    max_total_steps=50000 \
    metric_logger.project="llama-3B-hyperparameter-search" \
    # debug=True \
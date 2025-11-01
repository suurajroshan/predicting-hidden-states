#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --error=slurm/logs/entropy-%j.err
#SBATCH --output=slurm/logs/entropy-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

WORK_DIR=/home/woody/iwbi/iwbi106h/suuraj
# JOB_DIR=$WORK_DIR/slurm_scratch/llama-3B/$SLURM_JOB_ID
JOB_DIR=$WORK_DIR/slurm_scratch/$SLURM_JOB_ID
mkdir -p $JOB_DIR

rsync -av --exclude='predicting_hidden_states/__pycache__/' \
    --exclude='predicting_hidden_states/slurm/logs/' \
    --exclude='predicting_hidden_states/wandb/' \
    --exclude='predicting_hidden_states/checkpoints/' \
    --exclude='data/' \
    --exclude='assets/' \
    $WORK_DIR/codes/predicting-hidden-states \
    $JOB_DIR

cd $JOB_DIR/predicting_hidden_states/
printf "\nRunning job in $JOB_DIR\n"

git checkout entropy

python exp_script.py \
    metric_logger.mode=online \
    model.self_prediction_information_bottleneck=continuous \
    model.self_prediction_module.codebook_dim=30720 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    batch_size=8 \
    model.self_prediction_module.reconstruction_loss_factor=0.000001 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.00001

# self_critic_loss_factor = 0.1
# next_hidden_loss_factor = 0.001 \ 0.005 \ 0.0001

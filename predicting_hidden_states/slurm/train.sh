#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --error=slurm/logs/steering-phi-loss-%j.err
#SBATCH --output=slurm/logs/steering-phi-loss-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

WORK_DIR=/home/woody/iwbi/iwbi106h/suuraj
JOB_DIR=$WORK_DIR/slurm_scratch/$SLURM_JOB_ID
mkdir -p $JOB_DIR

git checkout steering-phi-loss

rsync -av --exclude='__pycache__/' --exclude='slurm/logs/' $WORK_DIR/codes/predicting-hidden-states/predicting_hidden_states/ $JOB_DIR
cd $JOB_DIR
printf "\nRunning job in $JOB_DIR\n"

python exp_script.py \
    metric_logger.mode=offline \
    model.self_prediction_module.reconstruction_loss_factor=0.001 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.0001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    batch_size=16 \
    model.alpha=-0.25

# self_critic_loss_factor = 0.1
# next_hidden_loss_factor = 0.001 \ 0.005 \ 0.0001
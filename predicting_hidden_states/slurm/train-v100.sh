#!/bin/bash -l
#
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:05:00
#SBATCH --error=slurm/logs/steering-phi-loss-v100-%j.err
#SBATCH --output=slurm/logs/steering-phi-loss-v100-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp


JOB_DIR=/home/woody/iwi5/iwi5368h/predicting_hidden_states_scratch/slurm_scratch/$SLURM_JOB_ID
mkdir -p $JOB_DIR

git checkout steering-phi-loss

rsync -av --exclude='__pycache__/' --exclude='slurm/logs/' ~/predicting-hidden-states/predicting_hidden_states/ $JOB_DIR
cd $JOB_DIR
printf "Running job in $JOB_DIR\n"

python exp_script.py \
    metric_logger.mode=offline \
    model.self_prediction_module.reconstruction_loss_factor=0.001 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.0001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    batch_size=8 \
    model.alpha=-0.25

# self_critic_loss_factor = 0.1
# next_hidden_loss_factor = 0.001 \ 0.005 \ 0.0001
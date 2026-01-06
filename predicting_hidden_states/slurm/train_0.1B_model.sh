#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --error=slurm/logs/llama0.1b-%j.err
#SBATCH --output=slurm/logs/llama0.1b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

JOB_DIR=/home/woody/iwi5/iwi5368h/slurm_scratch/llama-0.1B/$SLURM_JOB_ID
mkdir -p $JOB_DIR

rsync -av --exclude='__pycache__/' \
    --exclude='slurm/logs/' \
    --exclude='wandb/' \
    --exclude='checkpoints/' \
    --exclude='notebooks/' \
    --exclude='data/' \
    --exclude='.ipynb_checkpoints/' \
    $HOME/predicting-hidden-states/predicting_hidden_states/ \
    $JOB_DIR
cd $JOB_DIR
printf "\nRunning job in $JOB_DIR\n"

config_file="configs/llama_0.1B_PHi.yaml" 

python exp_script.py \
    metric_logger.mode=online \
    config_file=$config_file \
    model.self_critic_loss_factor=0.1 \
    model.latent_loss_factor=1e-4 \
    model.detach_targets=True \
    model.phi_loss_factor=0.001 \
    batch_size=8
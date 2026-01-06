#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --error=slurm/logs/llama0.1b-%j.err
#SBATCH --output=slurm/logs/llama0.1b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

CODEBOOK_DIM=768

# train debugging script for llama 3B model
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

config_file="configs/llama_0.1B_PHi_vector-quantizer.yaml"

python exp_script.py \
    metric_logger.mode=online \
    config_file=$config_file \
    model.self_prediction_module.codebook_dim=$CODEBOOK_DIM \
    model.self_prediction_module.codeword_dim=768 \
    metric_logger.project="llama-0.1B-vq-scaling-codebook" \
    metric_logger.name="codebook-dim-$CODEBOOK_DIM" \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    batch_size=16 \
    beta_scheduler.saturation_steps=5000 \
    beta_scheduler.beta_max=0.5 
    
# 128^3 for 3B / lr ramp up 1e-2 / batch-size 8 / reconstruction loss 0 / self critic loss 0.1 / 
# 64^2 for 0.1B

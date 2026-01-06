#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4%8
#SBATCH --error=slurm/logs/llama-0.1b-%j.err
#SBATCH --output=slurm/logs/llama-0.1b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
SEED=(40 42 44 46 48)
LAYERS=(10)
for s in "${SEED[@]}"; do
  for l in "${LAYERS[@]}"; do
    combinations+=("${s},${l}")
  done
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r s l <<< "$experiments"

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
  seed=$s \
  metric_logger.project="llama-0-1B-all-layer-10" \
  metric_logger.name="seed-$s-layer-$l-llf-1e-4" \
  model.self_critic_loss_factor=0.1 \
  model.phi_loss_factor=0.001 \
  model.latent_loss_factor=1e-4 \
  model.self_prediction_layer=$l \
  batch_size=8 \
  model.detach_targets=false \
  max_total_steps=30000 \
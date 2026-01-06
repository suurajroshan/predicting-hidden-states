#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%4
#SBATCH --error=slurm/logs/llama-3b-%j.err
#SBATCH --output=slurm/logs/llama-3b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
SEED=(46 48)
LAYERS=(19)
for s in "${SEED[@]}"; do
  for l in "${LAYERS[@]}"; do
    combinations+=("${s},${l}")
  done
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r s l <<< "$experiments"

JOB_DIR=/home/woody/iwi5/iwi5368h/slurm_scratch/llama-3B/$SLURM_JOB_ID
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

config_file="configs/llama_3B_PHi.yaml" 

python exp_script.py \
  metric_logger.mode=online \
  config_file=$config_file \
  seed=$s \
  metric_logger.project="llama-3B-all-layer-19" \
  metric_logger.name="seed-$s-layer-$l-llf-0.0" \
  model.self_critic_loss_factor=0.1 \
  model.phi_loss_factor=0.001 \
  model.self_prediction_layer=$l \
  batch_size=8 \
  model.detach_targets=true \
  max_total_steps=10000 \

    # metric_logger.project="llama-3B-vae-layers" \
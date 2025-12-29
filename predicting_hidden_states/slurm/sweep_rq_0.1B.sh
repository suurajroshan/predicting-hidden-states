#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%4
#SBATCH --error=slurm/logs/llama-0.1b-%A-%a.err
#SBATCH --output=slurm/logs/llama-0.1b-%A-%a.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

define sweep values
SEED=(40 42 44)
LAYERS=(9)
for s in "${SEED[@]}"; do
  for l in "${LAYERS[@]}"; do
    combinations+=("${s},${l}")
  done
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r s l <<< "$experiments"

WORK_DIR=/home/woody/iwbi/iwbi106h/suuraj
JOB_DIR=$WORK_DIR/slurm_scratch/llama-0.1B/$SLURM_JOB_ID
mkdir -p $JOB_DIR

rsync -av --exclude='__pycache__/' \
    --exclude='slurm/logs/' \
    --exclude='wandb/' \
    --exclude='checkpoints/' \
    --exclude='notebooks/' \
    --exclude='data/' \
    --exclude='.ipynb_checkpoints/' \
    $WORK_DIR/codes/predicting-hidden-states/predicting_hidden_states/ \
    $JOB_DIR
cd $JOB_DIR
printf "\nRunning job in $JOB_DIR\n"

config_file='configs/llama_0.1B_PHi_residual-gumbel-quantizer.yaml'

python exp_script.py \
    metric_logger.mode=online \
    config_file=$config_file \
    metric_logger.name="seed-$s-layer-$l-llf-1e-4" \
    metric_logger.project="llama-0.1B-rq-layers" \
    model.self_prediction_module.reconstruction_loss_factor=1e-5 \
    model.self_prediction_module.num_quantizers=2 \
    model.self_prediction_module.codebook_dim=64 \
    model.self_critic_loss_factor=0.1 \
    model.phi_loss_factor=0.001 \
    model.self_prediction_module.latent_loss_factor=1e-4 \
    seed=$s \
    model.self_prediction_layer=$l \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.2 \
    temperature_scheduler.global_steps=10000 \
    batch_size=8 \
    model.detach_targets=true \
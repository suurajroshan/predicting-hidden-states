#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --error=slurm/logs/llama-0.1b-%j-%A.err
#SBATCH --output=slurm/logs/llama-0.1b-%j-%A.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
LATENT_LOSS=(1e-1 1e-2 1e-3)
combinations=()
for lf in "${LATENT_LOSS[@]}"; do
  combinations+=("${lf}")
done

experiments="${combinations[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read -r llf <<< "$experiments"

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

config_file="configs/llama_0.1B_PHi.yaml"



python exp_script.py \
    metric_logger.mode=online \
    config_file=$config_file \
    seed=$seed \
    metric_logger.name="$seed" \
    model.latent_loss_factor=0.0 \
    metric_logger.project="final-runs-vanilla-100M" \
#!/bin/bash -l
#
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
## #SBATCH --array=0-2%4
#SBATCH --error=slurm/logs/llama-0.1b-%j.err
#SBATCH --output=slurm/logs/llama-0.1b-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

#define sweep values
LATENT_LOSS=(0.0)
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
    metric_logger.project="llama-0.1B-vae-sweep" \
    metric_logger.name="llf-$llf-target-detach" \
    model.latent_loss_factor=$llf \
    batch_size=8
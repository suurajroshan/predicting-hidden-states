#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --error=slurm/logs/%A_%a.err
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --array=0-9%4   # total jobs, max 3 run at once
#SBATCH --job-name=emb_sweep

unset SLURM_EXPORT_ENV
module load python
conda activate hsp

# Define parameter grids
temp_end=(0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)

params=()
for te in "${temp_end[@]}"; do
  # for cw in "${codeword_dim[@]}"; do
  #     params+=("$cb,$cw")
  # done
  params+=("$te")
done

# # Pick the current combination based on array index
# IFS=',' read w f <<< "${params[$SLURM_ARRAY_TASK_ID]}"
IFS=',' read w <<< "${params[$SLURM_ARRAY_TASK_ID]}"

python /home/woody/iwbi/iwbi106h/suuraj/codes/hidden-state-predictions/hidden-state-prediction-master/hidden_state_prediction/exp_script.py \
    metric_logger.mode=offline \
    batch_size=16 \
    recon_loss_weight=0.001 \
    model.self_critic_loss_factor=0.1 \
    model.next_hidden_loss_factor=0.005 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    model.next_hidden_loss_factor=$w 

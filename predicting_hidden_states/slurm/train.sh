#!/bin/bash -l
#
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --error=slurm/logs/phi-moreinf-%j.err
#SBATCH --output=slurm/logs/phi-moreinf-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

/home/woody/iwbi/iwbi106h/software/private/conda/envs/hsp/bin/python /home/woody/iwbi/iwbi106h/suuraj/codes/hidden-state-predictions/hidden-state-prediction-master/hidden_state_prediction/exp_script.py \
    metric_logger.mode=offline \
    recon_loss_weight=0.001 \
    model.self_critic_loss_factor=0.1 \
    model.next_hidden_loss_factor=0.0001 \
    temperature_scheduler.temp_start=1 \
    temperature_scheduler.temp_end=0.1 \
    temperature_scheduler.global_steps=30000 \
    # batch_size=16 

# self_critic_loss_factor = 0.1
# next_hidden_loss_factor = 0.001 \ 0.005 \ 0.0001
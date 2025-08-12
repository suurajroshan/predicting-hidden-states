#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --error=slurm/logs/%j.err
#SBATCH --output=slurm/logs/%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp 

python /home/woody/iwbi/iwbi106h/suuraj/codes/hidden-state-predictions/predicting-hidden-states/predicting_hidden_states/training_script.py
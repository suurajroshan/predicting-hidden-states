# #!/bin/bash -l
# #
# #SBATCH --partition=a100
# #SBATCH --gres=gpu:a100:1
# #SBATCH --time=06:00:00
# #SBATCH --error=slurm/logs/llama0.1b-%j.err
# #SBATCH --output=slurm/logs/llama0.1b-%j.out

# unset SLURM_EXPORT_ENV

# module load python
# conda activate hsp

wandb agent \
    hidden-state-predictions/self_prediction/ne6sishu \
    --count 1

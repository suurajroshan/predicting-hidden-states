#!/bin/bash -l
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --error=slurm/logs/logit-divergence-%j.err
#SBATCH --output=slurm/logs/logit-divergence-%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate hsp

WORK_DIR=/home/woody/iwbi/iwbi106h/suuraj
JOB_DIR=$WORK_DIR/slurm_scratch/$SLURM_JOB_ID
mkdir -p $JOB_DIR

git checkout logit-divergence

rsync -av --exclude='__pycache__/' --exclude='slurm/logs/' $WORK_DIR/codes/predicting-hidden-states/predicting_hidden_states/ $JOB_DIR
cd $JOB_DIR
printf "\nRunning job in $JOB_DIR\n"


python exp_script.py \
    metric_logger.mode=offline \
    model.depth_efficiency=True \
    model.fork_layer_position=10 \
    train_shortcut_loss=True
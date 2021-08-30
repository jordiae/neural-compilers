#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=slurm_logs/gen_eval_big_chkpt5_%j.log

DATA_PATH="data/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f"
CHECKPOINTS_PATH="runs/big-AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/checkpoints/checkpoint5.pt"

source ~/.bashrc
source venv/bin/activate

fairseq-generate $DATA_PATH \
    -s c -t s --path $CHECKPOINTS_PATH \
    --batch-size 16 --beam 5 --remove-bpe=' ##'
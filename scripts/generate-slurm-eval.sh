#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=logs/gen_eval_%j.log

DATA_PATH="data/AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045"
CHECKPOINTS_PATH="runs/baseline-AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045/checkpoints/checkpoint_best.pt"

source ~/.bashrc
source venv/bin/activate

fairseq-generate $DATA_PATH \
    -s c -t s --path $CHECKPOINTS_PATH \
    --batch-size 128 --beam 5 --remove-bpe=' ##'
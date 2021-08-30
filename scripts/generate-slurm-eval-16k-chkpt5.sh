#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=slurm_logs/gen_eval_16k_chkpt5_%j.log

DATA_PATH="experiments/AnghaBench-supervised-2021-07-25-1628-5a69ccfe0ea3eaf9fd9a09bff9555d9a-17e5-b3ad"
CHECKPOINTS_PATH="experiments/AnghaBench-supervised-2021-07-25-1628-5a69ccfe0ea3eaf9fd9a09bff9555d9a-17e5-b3ad/model_16k/checkpoints/checkpoint5.pt"

source ~/.bashrc
source venv/bin/activate

fairseq-generate $DATA_PATH \
    -s c -t s --path $CHECKPOINTS_PATH \
    --batch-size 16 --beam 5 --remove-bpe=' ##'
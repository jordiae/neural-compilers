#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=slurm_logs/gen_eval_250k_chkpt5_%j.log

DATA_PATH="output/AnghaBench-supervised-2021-07-24-0506-4df2227e3af912140f3c3cdb37f8dcb6-a767-3294"
CHECKPOINTS_PATH="output/AnghaBench-supervised-2021-07-24-0506-4df2227e3af912140f3c3cdb37f8dcb6-a767-3294/model250k/checkpoints/checkpoint5.pt"


source ~/.bashrc
source venv/bin/activate

fairseq-generate $DATA_PATH \
    -s c -t s --path $CHECKPOINTS_PATH \
    --batch-size 16 --beam 5 --remove-bpe=' ##'
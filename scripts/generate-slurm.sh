#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=logs/gen_%j.log

source venv/bin/activate

cat data/AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045/test.tok.c | fairseq-interactive data/AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045 -s c -t s --path runs/baseline-AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045/checkpoints/checkpoint_best.pt
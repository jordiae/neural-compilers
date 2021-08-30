#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=baseline_%j.log

source venv/bin/activate

DATA_PATH="data/AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045"
RUN="runs/baseline-AnghaBench-supervised-2021-04-12-2011-1ad3533bfc9df19cc9bff3ba72487765-1f9a-5045"

mkdir -p $RUN

fairseq-train \
    $DATA_PATH \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --tensorboard-logdir $RUN/tb \
    --save-dir $RUN/checkpoints

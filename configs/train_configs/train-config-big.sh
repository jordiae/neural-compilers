#!/usr/bin/env bash

DATA_PATH="data/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f"
RUN="runs/big-AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f"

mkdir -p $RUN

fairseq-train \
    $DATA_PATH \
    --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07'\
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1024 \
    --tensorboard-logdir $RUN/tb \
    --save-dir $RUN/checkpoints -s c -t s \
    --update-freq 16 --encoder-layers 8 --decoder-layers 8

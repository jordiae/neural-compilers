#!/usr/bin/env bash

DATA_PATH="output/AnghaBench-supervised-2021-07-24-0506-4df2227e3af912140f3c3cdb37f8dcb6-a767-3294"
RUN="output/AnghaBench-supervised-2021-07-24-0506-4df2227e3af912140f3c3cdb37f8dcb6-a767-3294/model250k"

mkdir -p $RUN

fairseq-train \
    $DATA_PATH \
    --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07'\
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --tensorboard-logdir $RUN/tb \
    --save-dir $RUN/checkpoints -s c -t s \
    --update-freq 4

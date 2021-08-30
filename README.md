# Neural Compilers

Code for building a neural compiler (a Transformer that learns to translate C to x86), from "Learning C to x86 Translation: An Experiment in Neural Compilation" (https://arxiv.org/abs/2108.07639).

**Warning: The code is not properly documented and organized since this is an ongoing effort.**

## Steps to run

1 - Create anbd activate virtual environment, install dependencies and get data:


      bash setup.sh


2 - Prepare training data:
      
      python prepare-train-data.py --config-file configs/data_configs/data-config1.json

3 - Train model (with Fairseq)
      
      DATA_PATH="data/YOUR_DATA" # Path from data generated in step 2
      RUN="$DATA_PATH/YOUR_MODEL"

      mkdir -p $RUN

      fairseq-train \
          $DATA_PATH \
          --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
          --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
          --lr 0.001 --lr-scheduler inverse_sqrt --min-lr '1e-09' --warmup-updates 4000 --warmup-init-lr '1e-07'\
          --weight-decay 0.0001 \
          --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
          --max-tokens 4096 \
          --eval-bleu \
          --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
          --eval-bleu-detok moses \
          --eval-bleu-remove-bpe \
          --eval-bleu-print-samples \
          --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --tensorboard-logdir $RUN/tb \
          --save-dir $RUN/checkpoints -s c -t s \
          --update-freq 4

4 - Run inference on I/O benchmark

      python infer.py --config-file YOUR_CONFIG.json  # See configs/infer_configs/infer-config-4k-ch5.json

5 - Evaluate on I/O examples:

      
      python evaluate-io-legacy.py --synthesis-eval-path synthesis-eval \
                                   --predictions-path PATH_TO_PREDICTIONS # Predictions generated in step 4


## How to cite
If you use any of these resources (datasets or models) in your work, please cite our latest paper:
```bibtex
@misc{armengolestape2021learning,
      title={Learning C to x86 Translation: An Experiment in Neural Compilation}, 
      author={Jordi Armengol-Estap\'e and Michael F. P. O'Boyle},
      year={2021},
      eprint={2108.07639},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


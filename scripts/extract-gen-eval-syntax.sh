#!/usr/bin/env bash

# Usage: bash scripts/extract-gen-eval-syntax.sh PATH_TO_FAIRSEQ_GENERATE_LOG

NAME=$1

grep "^H" $NAME | awk 'BEGIN {FS="\t"}; {print $3}' | sed -r 's/(##)//g' > $NAME.detok

venv8/bin/python evaluate-syntax.py --system-output-path $NAME.detok

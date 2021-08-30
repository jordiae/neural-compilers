#!/bin/bash

NAME="experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-3f2d/eval-io-legacy-2021-08-05-1430-949b-6b75/eval-io-legacy-2021-08-05-1430-949b-6b75.log"
grep $NAME -e "OK |" | sed 's/INFO:root://g'

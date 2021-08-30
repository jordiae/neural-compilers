#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BEST_MODEL="experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-3f2d/eval-io-legacy-2021-08-03-0138-5a1d-b5c0/eval-io-legacy-2021-08-03-0138-5a1d-b5c0.log"

ALL="experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-6b96/eval-io-legacy-2021-08-03-0121-5a1d-f263/eval-io-legacy-2021-08-03-0121-5a1d-f263.log
       experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-6463/eval-io-legacy-2021-08-03-0117-5a1d-675e/eval-io-legacy-2021-08-03-0117-5a1d-675e.log
       data/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-07ce/eval-io-legacy-2021-08-03-0111-5a1d-1a20/eval-io-legacy-2021-08-03-0111-5a1d-1a20.log
       experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-3a2d/eval-io-legacy-2021-08-03-0127-5a1d-1000/eval-io-legacy-2021-08-03-0127-5a1d-1000.log
       experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-3f2d/eval-io-legacy-2021-08-03-0138-5a1d-b5c0/eval-io-legacy-2021-08-03-0138-5a1d-b5c0.log
       experiments/AnghaBench-supervised-2021-07-25-1758-5a69ccfe0ea3eaf9fd9a09bff9555d9a-7b23-e039/infer-beam5-topn5-2021-07-28-1408-a148-80dc/eval-io-legacy-2021-08-03-0142-5a1d-941e/eval-io-legacy-2021-08-03-0142-5a1d-941e.log
       output/AnghaBench-supervised-2021-07-24-0506-4df2227e3af912140f3c3cdb37f8dcb6-a767-3294/infer-beam5-topn5-2021-07-26-0621-c473-2a5d/eval-io-legacy-2021-08-03-0148-5a1d-a23b/eval-io-legacy-2021-08-03-0148-5a1d-a23b.log
       experiments/AnghaBench-supervised-2021-07-25-1628-5a69ccfe0ea3eaf9fd9a09bff9555d9a-17e5-b3ad/infer-beam5-topn5-2021-07-28-1406-84ea-84b7/eval-io-legacy-2021-08-03-0152-5a1d-670a/eval-io-legacy-2021-08-03-0152-5a1d-670a.log
       data/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-08-05-1320-56d3-bb77/eval-io-legacy-2021-08-05-1234-e80d-64f0/eval-io-legacy-2021-08-05-1234-e80d-64f0.log"

grep $BEST_MODEL -e "IO = OK" | awk -F  ":" '{print $3}' > $SCRIPT_DIR/best_ok.txt

for path in $ALL
  do
  echo $path
  grep $path -e "IO = OK" | awk -F  ":" '{print $3}' | comm -12 - $SCRIPT_DIR/best_ok.txt | wc -l
  done
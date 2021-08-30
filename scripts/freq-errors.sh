#!/bin/bash

grep -A1  experiments/AnghaBench-supervised-2021-05-09-1331-1ad3533bfc9df19cc9bff3ba72487765-3840-f78f/infer-beam5-topn5-2021-07-26-0621-c473-3f2d/eval-io-legacy-2021-08-03-0138-5a1d-b5c0/eval-io-legacy-2021-08-03-0138-5a1d-b5c0.log -e "IO: N/A " | awk -F"Error: " '{print $2}' | sed '/experiments/d' | sort| uniq -c  | sort -k1 -n -r

#!/usr/bin/env bash

NAME=gen_3958

grep "^H" $NAME.log | awk 'BEGIN {FS="\t"}; {print $3}' | sed -r 's/(##)//g' > $NAME.detok.s

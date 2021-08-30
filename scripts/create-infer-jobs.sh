#!/usr/bin/env bash

for CONFIG in infer-config-16k-ch5.json  infer-config-250k-ch5.json  infer-config-4k-ch5.json  infer-config-base-ch10.json  infer-config-base-ch5.json  infer-config-big-ch5.json  infer-config-small-ch5.json  infer-config-wd-ch5.json
do
	python job-gen.py --system calcula --config-path configs/infer_configs/$CONFIG --job-type infer
done
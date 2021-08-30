#!/usr/bin/env bash

# Get data
bash scripts/get-anghabench-data.sh

# Get benchmark

bash scripts/get-synthesis-benchmark.sh

# Install dependencies
python3.7 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
apt install gcc

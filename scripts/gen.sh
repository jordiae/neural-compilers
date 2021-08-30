#!/bin/bash

cwd=$(pwd)
for example in synthesis-eval/examples/*
do
	cd $example && python3 gen.py
	cd $cwd
done

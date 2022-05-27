#!/usr/bin/env bash

GPU=1
TASK=scrfd_2.5g_bnkps
SOURCE=tests/data

CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd.py --model-file onnx/"$TASK".onnx --source "$SOURCE"
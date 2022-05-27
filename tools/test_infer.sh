#!/usr/bin/env bash

GPU=1
TASK=scrfd_2.5g_bnkps

CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd.py onnx/"$TASK".onnx --source test/data
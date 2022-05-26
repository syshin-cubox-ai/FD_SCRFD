#!/usr/bin/env bash

GPU=1
GROUP=scrfd
TASK=scrfd_2.5g

CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/model.pth
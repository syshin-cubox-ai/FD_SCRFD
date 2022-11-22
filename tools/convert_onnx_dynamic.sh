#!/usr/bin/env bash

TASK=scrfd_2.5g_bnkps

python tools/scrfd2onnx.py ./configs/scrfd/"$TASK".py ./work_dirs/"$TASK"/model.pth --dynamic

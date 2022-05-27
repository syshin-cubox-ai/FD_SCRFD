#!/usr/bin/env bash

GPU=1
GROUP=scrfd
TASK=scrfd_2.5g_bnkps

CUDA_VISIBLE_DEVICES="$GPU" python tools/get_flops.py ./configs/"$GROUP"/"$TASK".py
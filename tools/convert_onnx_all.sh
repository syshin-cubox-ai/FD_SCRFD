#!/usr/bin/env bash

GPU=1
GROUP=scrfd

CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_1g.py ./work_dirs/scrfd_1g/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_2.5g.py ./work_dirs/scrfd_2.5g/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_2.5g_bnkps.py ./work_dirs/scrfd_2.5g_bnkps/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_10g.py ./work_dirs/scrfd_10g/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_10g_bnkps.py ./work_dirs/scrfd_10g_bnkps/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_34g.py ./work_dirs/scrfd_34g/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_500m.py ./work_dirs/scrfd_500m/model.pth --simplify
CUDA_VISIBLE_DEVICES="$GPU" python tools/scrfd2onnx.py ./configs/"$GROUP"/scrfd_500m_bnkps.py ./work_dirs/scrfd_500m_bnkps/model.pth --simplify

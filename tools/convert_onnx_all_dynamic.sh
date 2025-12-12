#!/usr/bin/env bash

python tools/scrfd2onnx.py ./configs/scrfd/scrfd_500m.py ./work_dirs/scrfd_500m/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_1g.py ./work_dirs/scrfd_1g/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_2.5g.py ./work_dirs/scrfd_2.5g/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_10g.py ./work_dirs/scrfd_10g/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_34g.py ./work_dirs/scrfd_34g/model.pth --dynamic

python tools/scrfd2onnx.py ./configs/scrfd/scrfd_500m_bnkps.py ./work_dirs/scrfd_500m_bnkps/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_2.5g_bnkps.py ./work_dirs/scrfd_2.5g_bnkps/model.pth --dynamic
python tools/scrfd2onnx.py ./configs/scrfd/scrfd_10g_bnkps.py ./work_dirs/scrfd_10g_bnkps/model.pth --dynamic
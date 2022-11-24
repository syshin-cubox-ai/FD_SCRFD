import argparse
import math
import os
from typing import Tuple

import cv2
import numpy as np
import onnx.shape_inference
import onnxruntime.tools.symbolic_shape_infer
import onnxsim
import torch

import mmdet.core


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: img_size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def resize_preserving_aspect_ratio(img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
    # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
    h, w = img.shape[:2]
    scale = img_size // scale_ratio / max(h, w)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img, scale


def transform_image(img: np.ndarray, img_size: int) -> np.ndarray:
    img, _ = resize_preserving_aspect_ratio(img, img_size)

    pad = (0, img_size - img.shape[0], 0, img_size - img.shape[1])
    img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = cv2.dnn.blobFromImage(img, 1 / 128, img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--img_size', type=int, default=640, help='image size')
    parser.add_argument('--dynamic', action='store_true', help='use dynamic axes')
    parser.add_argument('--skip_simplify', action='store_true', help='skip onnx-simplifier')
    args = parser.parse_args()
    print(args)

    # Create torch model
    model = mmdet.core.build_model_from_cfg(args.config, args.checkpoint)

    # Create input data
    img = cv2.imread('tests/data/2.jpg')
    img = transform_image(img, check_img_size(args.img_size, 32))
    img = torch.from_numpy(img)

    # Define output file path
    output_dir = 'onnx_files'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.config).replace('.py', '.onnx'))

    # Define input and output names
    input_names = ['img']
    output_names = ['pred']

    # Define dynamic_axes
    if args.dynamic:
        dynamic_axes = {input_names[0]: {2: 'H', 3: 'W'},
                        output_names[0]: {0: 'Candidates', 1: 'dyn_15'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    try:
        onnx_model = onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
    except Exception as e:
        print(f'ERROR: {e}, skip symbolic shape inference.')
    onnx.shape_inference.infer_shapes_path(output_path, output_path, check_type=True, strict_mode=True, data_prop=True)

    # Compare output with torch model and ONNX model
    torch_out = model(img, force_onnx_export=True).detach().numpy()
    session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        print(e)
        stdin = input('Do you want to ignore the error and proceed with the export ([y]/n)? ')
        if stdin == 'n':
            os.remove(output_path)
            exit(1)

    # Simplify ONNX model
    if not args.skip_simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

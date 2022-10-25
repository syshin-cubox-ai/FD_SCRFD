import argparse
import os

import cv2
import numpy as np
import onnx
import onnxruntime
import onnxsim
import torch

import mmdet.core


def transform_image(img: np.ndarray, img_size: int) -> np.ndarray:
    """
    Resizes the input image to fit img_size while maintaining aspect ratio.
    This performs BGR to RGB, HWC to CHW, normalization, and adding batch dimension.
    (mean=(127.5, 127.5, 127.5), std=(128.0, 128.0, 128.0))
    """
    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    if scale != 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    pad = (0, img_size - img.shape[0], 0, img_size - img.shape[1])
    img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = cv2.dnn.blobFromImage(img, 1 / 128, img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic', action='store_true', help='use dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='use onnx-simplifier')
    args = parser.parse_args()
    print(args)

    # Create model and input data
    model = mmdet.core.build_model_from_cfg(args.config, args.checkpoint)
    img = cv2.imread('tests/data/2.jpg')
    img = transform_image(img, 640)
    img = torch.from_numpy(img)

    # Define output file path
    output_dir = 'onnx'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(args.config).replace('.py', '.onnx'))

    # Define input and output names
    input_names = ['img']
    output_names = ['pred']

    # Define dynamic_axes
    if args.dynamic:
        dynamic_axes = {input_names[0]: {0: 'N', 2: 'H', 3: 'W'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # only work with version 11
        dynamic_axes=dynamic_axes,
    )

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    onnx.shape_inference.infer_shapes_path(output_path, output_path)

    # Compare output with torch model and ONNX model
    try:
        torch_out = model(img, force_onnx_export=True).detach().numpy()
        session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        print(e)
        stdin = input('Do you want to ignore the error and proceed with the export ([y]/n)? ')
        if stdin == 'n':
            os.remove(output_path)
            exit(1)

    # Simplify ONNX model
    if args.simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

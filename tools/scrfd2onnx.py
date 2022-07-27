import argparse
import os

import cv2
import numpy as np
import onnx
import onnxruntime
import onnxsim
import torch

import mmdet.core


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def transform_image(img: np.ndarray, img_size) -> np.ndarray:
    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    if scale != 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    padded_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded_img[:img.shape[0], :img.shape[1], :] = img
    img = cv2.dnn.blobFromImage(padded_img, 1 / 128, padded_img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic', type=bool, default=True, help='use dynamic axes')
    parser.add_argument('--simplify', type=bool, default=True, help='use onnx-simplifier')
    args = parser.parse_args()
    print(args)

    img_path = 'tests/data/2.jpg'
    img_size = 640
    mean = (127.5, 127.5, 127.5)
    std = (128.0, 128.0, 128.0)

    # Create model and input data
    model = mmdet.core.build_model_from_cfg(args.config, args.checkpoint)
    img = cv2.imread(img_path)
    img = transform_image(img, img_size)
    img = torch.from_numpy(img)
    input_data = (img, torch.tensor(640, dtype=torch.int32), torch.tensor(0.3), torch.tensor(0.5))

    # Define output file path
    output_dir = 'onnx'
    os.makedirs(output_dir, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.dynamic:
        output_path = os.path.join(output_dir, f'{cfg_name}.onnx')
    else:
        output_path = os.path.join(output_dir, f'{cfg_name}_static_axis.onnx')

    # Define input and output names
    input_names = ['img', 'img_size', 'conf_thres', 'iou_thres']
    output_names = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32']

    # If model graph contains keypoints strides add keypoints to outputs
    if 'stride_kps' in str(model):
        output_names += ['kps_8', 'kps_16', 'kps_32']

    # Define dynamic axes for export
    if args.dynamic:
        dynamic_axes = {name: {0: 'N', 1: 'C'} for name in output_names}
        dynamic_axes[input_names[0]] = {0: 'N', 2: 'H', 3: 'W'}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    torch.onnx.export(
        model,
        input_data,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # only work with version 11
        dynamic_axes=dynamic_axes,
    )

    # Compare the exported onnx model with the torch model
    session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    onnx_inputs = {name: to_numpy(data) for name, data in zip(input_names, input_data)}
    onnx_ouputs = session.run(None, onnx_inputs)

    torch_outputs = model(*input_data, force_onnx_export=True)
    torch_outputs = [to_numpy(out) for out in torch_outputs]

    for onnx_ouput, torch_output in zip(onnx_ouputs, torch_outputs):
        np.testing.assert_allclose(onnx_ouput, torch_output, rtol=1e-03, atol=1e-05)

    # Simplify ONNX model
    if args.simplify:
        model = onnx.load(output_path)
        if args.dynamic:
            input_shapes = {model.graph.input[0].name: img.shape}
            model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        else:
            model, check = onnxsim.simplify(model)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

import argparse
import os

import mmcv
import numpy as np
import onnx
import onnxruntime
import onnxsim
import torch

import mmdet.core


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic', type=bool, default=True, help='use dynamic axes')
    parser.add_argument('--simplify', type=bool, default=True, help='use onnx-simplifier')
    args = parser.parse_args()
    print(args)

    input_shape = [1, 3, 640, 640]
    input_path = 'tests/data/2.jpg'
    mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    std = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    # Create model and input data
    model = mmdet.core.build_model_from_cfg(args.config, args.checkpoint)
    img = mmcv.imread(input_path)
    img = mmcv.imresize(img, input_shape[2:][::-1])
    img = mmcv.imnormalize(img, mean, std)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
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
            input_shape = {model.graph.input[0].name: input_shape}
            model, check = onnxsim.simplify(model, test_input_shapes=input_shape)
        else:
            model, check = onnxsim.simplify(model)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')

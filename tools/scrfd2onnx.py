import argparse
import os

import onnx
import torch

from mmdet.core import generate_inputs_and_wrap_model


def pytorch2onnx(
        config_path,
        checkpoint_path,
        input_img,
        input_shape,
        opset_version=11,
        output_file='tmp.onnx',
        simplify=True,
        dynamic=True,
        normalize_cfg=None
):
    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    tmp_ckpt_file = None
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
        tmp_ckpt_file = checkpoint_path + '_slim.pth'
        torch.save(checkpoint, tmp_ckpt_file)
        print('remove optimizer params and save to', tmp_ckpt_file)
        checkpoint_path = tmp_ckpt_file

    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config)

    if tmp_ckpt_file is not None:
        os.remove(tmp_ckpt_file)

    if simplify or dynamic:
        ori_output_file = output_file.split('.')[0] + '_ori.onnx'
    else:
        ori_output_file = output_file

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ['input.1']
    output_names = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32']

    # If model graph contains keypoints strides add keypoints to outputs
    if 'stride_kps' in str(model):
        output_names += ['kps_8', 'kps_16', 'kps_32']

    # Define dynamic axes for export
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {out: {0: 'N', 1: 'C'} for out in output_names}
        dynamic_axes[input_names[0]] = {0: 'N', 2: 'H', 3: 'W'}

    torch.onnx.export(
        model,
        tensor_data,
        ori_output_file,
        keep_initializers_as_inputs=False,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version)

    if simplify or dynamic:
        model = onnx.load(ori_output_file)
        if simplify:
            from onnxsim import simplify
            if dynamic:
                input_shapes = {model.graph.input[0].name: list(input_shape)}
                model, check = simplify(model, input_shapes=input_shapes, dynamic_input_shape=True)
            else:
                model, check = simplify(model)
            assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_file)
        os.remove(ori_output_file)
    print(f'Successfully exported ONNX model: {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--dynamic', type=bool, default=True, help='use dynamic axes')
    parser.add_argument('--simplify', type=bool, default=True, help='use onnx-simplifier')
    args = parser.parse_args()
    print(args)

    # Create model and tensor data
    input_config = {
        'input_shape': (1, 3, 640, 640),
        'input_path': 'tests/data/t1.jpg',
        'normalize_cfg': {'mean': [127.5, 127.5, 127.5], 'std': [128.0, 128.0, 128.0]}
    }
    model, input_data = generate_inputs_and_wrap_model(args.config, args.checkpoint, input_config)

    # Define output file path
    output_dir = 'onnx'
    os.makedirs(output_dir, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.dynamic:
        output_file = os.path.join(output_dir, f'{cfg_name}.onnx')
    else:
        output_file = os.path.join(output_dir, f'{cfg_name}_static_axes.onnx')

    if args.simplify:
        ori_output_file = output_file.replace('.onnx', '_ori.onnx')
    else:
        ori_output_file = output_file

    # Define input and output names
    input_names = ['input.1']
    output_names = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32']

    # If model graph contains keypoints strides add keypoints to outputs
    if 'stride_kps' in str(model):
        output_names += ['kps_8', 'kps_16', 'kps_32']

    # Define dynamic axes for export
    if args.dynamic:
        dynamic_axes = {out: {0: 'N', 1: 'C'} for out in output_names}
        dynamic_axes[input_names[0]] = {0: 'N', 2: 'H', 3: 'W'}
    else:
        dynamic_axes = None

    torch.onnx.export(
        model,
        input_data,
        ori_output_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # only work with version 11
        dynamic_axes=dynamic_axes,
    )

    if args.simplify:
        from onnxsim import simplify
        model = onnx.load(ori_output_file)
        if args.dynamic:
            input_shapes = {model.graph.input[0].name: list(input_config['input_shape'])}
            model, check = simplify(model, input_shapes=input_shapes, dynamic_input_shape=True)
        else:
            model, check = simplify(model)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_file)
        os.remove(ori_output_file)
    print(f'Successfully exported ONNX model: {output_file}')

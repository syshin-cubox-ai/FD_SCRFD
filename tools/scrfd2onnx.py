import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import onnxruntime
import onnxslim
import torch
import torch.nn as nn

import mmdet.core


def convert_onnx(
    model: nn.Module,
    img_path: Union[str, Path],
    output_path: Union[str, Path],
    dynamic=False,
):
    model.eval()
    output_path = Path(output_path)

    img = cv2.imread(str(img_path))
    assert img is not None
    img = cv2.dnn.blobFromImage(
        img, 1 / 128, img.shape[:2][::-1], (127.5, 127.5, 127.5), swapRB=True
    )
    img = torch.from_numpy(img)

    # Define input and output names
    input_names = ["image"]
    output_names = ["pred"]

    # Define dynamic_axes
    if dynamic:
        dynamic_axes = {input_names[0]: {0: "N"}, output_names[0]: {0: "N"}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (img,),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )

    # Simplify ONNX model
    onnxslim.slim(str(output_path), output_model=str(output_path))

    # Compare output with torch model and ONNX model
    torch_out = model(img, force_onnx_export=True).detach().numpy()
    session = onnxruntime.InferenceSession(
        output_path, providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(None, {input_names[0]: img.numpy()})[0]
    try:
        np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-4)
    except AssertionError as e:
        print(e)
    print(f"Successfully export: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MMDetection models to ONNX")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("--dynamic", action="store_true", help="use dynamic axes")
    args = parser.parse_args()
    print(args)

    # Create torch model
    model = mmdet.core.build_model_from_cfg(args.config, args.checkpoint)

    convert_onnx(
        model,
        "tests/data/debug.jpg",
        Path("onnx_files") / Path(args.config).with_suffix(".onnx").name,
        args.dynamic,
    )

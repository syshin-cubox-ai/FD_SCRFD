import onnxruntime

sess_options = onnxruntime.SessionOptions()
sess_options.log_severity_level = 0
session = onnxruntime.InferenceSession('onnx/scrfd_2.5g_bnkps.onnx',
                                       sess_options,
                                       ['TensorrtExecutionProvider'])

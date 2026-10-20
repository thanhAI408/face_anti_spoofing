"""Small CPU runtime for the exported, unquantized liveness model."""
import os
import numpy as np


class _SilenceStderr:
    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_RDWR)
            self.save_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None

    def __exit__(self, *_):
        if getattr(self, 'null_fd', None) is not None:
            try:
                os.dup2(self.save_fd, 2)
                os.close(self.null_fd)
                os.close(self.save_fd)
            except Exception:
                pass


class LiteModel:
    input_shape = (None, 32, 32, 3)
    output_shape = (None, 2)

    def __init__(self, path):
        with _SilenceStderr():
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=str(path), num_threads=2)
            self.interpreter.allocate_tensors()
            self.input = self.interpreter.get_input_details()[0]['index']
            self.output = self.interpreter.get_output_details()[0]['index']
            # Fail startup if the exported graph is invalid.
            self(np.zeros((1, 32, 32, 3), dtype=np.float32))

    def __call__(self,batch,training=False):
        result=[]
        for face in batch:
            self.interpreter.set_tensor(self.input,face[None].astype(np.float32))
            self.interpreter.invoke()
            result.append(self.interpreter.get_tensor(self.output)[0].copy())
        return np.stack(result)

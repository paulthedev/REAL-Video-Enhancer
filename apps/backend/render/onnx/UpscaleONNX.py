import onnx
import os
import numpy as np
import time
import cv2

from .session import create_session, prepare_model, resolve_precision as _resolve_precision


def getONNXScale(modelPath: str = "") -> int:
    paramName = os.path.basename(modelPath).lower()
    for i in range(100):
        if f"{i}x" in paramName or f"x{i}" in paramName:
            return i


class UpscaleONNX:
    @classmethod
    def getModelScale(cls, modelPath: str) -> int:
        model_name = os.path.basename(modelPath).lower()
        for i in range(100):
            if f"{i}x" in model_name or f"x{i}" in model_name:
                return i
        raise ValueError("Scale not found in model name!")

    def handlePrecision(self, precision: str) -> np.dtype:
        return _resolve_precision(precision)

    def __init__(
        self,
        modelPath: str,
        device="default",
        tile_pad: int = 10,
        precision: str = "auto",
        width: int = 1920,
        height: int = 1080,
        scale: int = 2,
        tilesize: int = 0,
        gpu_id: int = 0,
        hdr_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.modelPath = modelPath
        self.device = device
        self.i0 = None
        self.precision = self.handlePrecision(precision)

        self.input_buffer = np.empty((1, 3, self.height, self.width), dtype=self.precision)
        self.output_buffer = np.empty((1, self.height * self.scale, self.width * self.scale, 3), dtype=self.precision)

        # Pre-compute normalization factor
        self.norm_factor = np.array(1.0/255.0, dtype=self.precision)


        # load model + build session through the shared helper
        self.model = prepare_model(self.modelPath, self.precision)
        self.inference_session = create_session(self.model, 0)

    

    def bytesToFrame(self, image: bytes) -> tuple:
        temp_view = np.frombuffer(image, dtype=np.uint8)
        temp_view = temp_view.reshape(self.height, self.width, 3)

        # Use direct assignment to pre-allocated buffer
        self.input_buffer[0] = np.transpose(temp_view, (2, 0, 1))
        
        # In-place normalization
        self.input_buffer *= self.norm_factor
            
        return self.input_buffer
        image = np.frombuffer(image, dtype=np.uint8).reshape(1080, 1920, 3)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(self.precision)
        image = image.__mul__(1.0 / 255.0)
        return np.ascontiguousarray(image)

    def renderTensor(self, image_as_np_array: np.ndarray) -> np.ndarray:
        onnx_input = {self.inference_session.get_inputs()[0].name: image_as_np_array}
        onnx_output = self.inference_session.run(None, onnx_input)[0]
        return onnx_output

    def frameToBytes(self, image: np.ndarray) -> bytes:
        
        
        self.output_buffer[0] = image.clip(0, 1).squeeze().transpose(1, 2, 0)
        self.output_buffer[0] /= self.norm_factor
        image = self.output_buffer[0].astype(np.uint8).reshape(self.height* self.scale, self.width*self.scale, 3)
        return np.ascontiguousarray(image).tobytes()
    
    def hotUnload(self):
        self.paused = True

    def hotReload(self):
        self.paused = False
    
    def __call__(self, image: bytes) -> bytes:
        while self.paused:
            time.sleep(1)
        image_as_np_array = self.bytesToFrame(image)
        output = self.renderTensor(image_as_np_array)
        return self.frameToBytes(output)

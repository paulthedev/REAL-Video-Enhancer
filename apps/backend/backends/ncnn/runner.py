"""
NCNN backend runner for backend-agnostic models.

This module handles running inference with NCNN backend.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class NCNNBackendRunner:
    """
    NCNN backend runner.
    
    Runs inference using NCNN backend.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ncnn"
    
    def run(
        self,
        model_info: Dict[str, Any],
        input_data: np.ndarray,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Run inference with NCNN model.
        
        Args:
            model_info: Model info dictionary from loader
            input_data: Input numpy array
            input_names: Input tensor names
            output_names: Output tensor names
        
        Returns:
            Output numpy array
        """
        import ncnn

        net = model_info["net"]
        input_names = input_names or model_info.get("input_names", ["input"])
        output_names = output_names or model_info.get("output_names", ["output"])

        # Accept NCHW float [0,1] or NHWC uint8; ncnn wants NHWC uint8 pixels
        data = input_data
        if data.ndim == 4:
            data = data[0]  # batch of 1
            if data.shape[0] == 3 and data.shape[-1] != 3:
                data = np.transpose(data, (1, 2, 0))
        if data.dtype != np.uint8:
            data = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)

        h, w = data.shape[:2]
        mat = ncnn.Mat.from_pixels(data, ncnn.Mat.PixelType.PIXEL_RGB, w, h)

        extractor = net.create_extractor()
        extractor.input(input_names[0], mat)
        ret, out_mat = extractor.extract(output_names[0])
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed with code {ret}")

        out = out_mat.numpy()  # CHW float
        if out.ndim == 3 and out.shape[0] == 3:
            out = np.clip(out.transpose(1, 2, 0), 0.0, 1.0)
        return out[None]  # restore batch dim
    
    def run_batch(
        self,
        model_info: Dict[str, Any],
        input_data: List[np.ndarray],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Run batch inference with NCNN model.
        
        Args:
            model_info: Model info dictionary from loader
            input_data: List of input numpy arrays
            input_names: Input tensor names
            output_names: Output tensor names
        
        Returns:
            List of output numpy arrays
        """
        results = []
        for data in input_data:
            result = self.run(model_info, data, input_names, output_names)
            results.append(result)
        return results
    
    def get_input_shape(self, model_info: Dict[str, Any]) -> Tuple[int, ...]:
        """
        Get expected input shape.
        
        Args:
            model_info: Model info dictionary
        
        Returns:
            Input shape tuple
        """
        # NCNN typically uses fixed input sizes
        return (1, 3, 512, 512)  # Default
    
    def get_output_shape(self, model_info: Dict[str, Any], input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get expected output shape.
        
        Args:
            model_info: Model info dictionary
            input_shape: Input shape
        
        Returns:
            Output shape tuple
        """
        # For most models, output shape matches input shape (1:1)
        return input_shape

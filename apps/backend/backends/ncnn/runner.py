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
        net = model_info["net"]
        input_names = input_names or model_info.get("input_names", ["input"])
        output_names = output_names or model_info.get("output_names", ["output"])
        
        # Create NCNN Mat from numpy array
        # NCNN expects NHWC format, convert from NCHW if needed
        if len(input_data.shape) == 4 and input_data.shape[1] == 3:
            # NCHW to NHWC
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        
        # Convert to NCNN Mat
        mat = model_info["net"].create_input(input_names[0])
        mat.from_pixels(input_data[0])  # Assume batch size 1
        
        # Run inference
        predictor = net.create_extractor()
        predictor.set_input(mat, input_names[0])
        predictor.extract(output_names[0])
        
        # Get output
        output_mat = predictor.extract(output_names[0])
        output_data = output_mat.to_pixels()
        
        # Convert back to NCHW if needed
        if len(output_data.shape) == 3:
            output_data = np.transpose(output_data, (2, 0, 1))
        
        return output_data.reshape(1, *output_data.shape)
    
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

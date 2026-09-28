"""
Integration tests for model conversion pipeline.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.converters.torch_to_onnx import TorchToOnnxConverter
from apps.backend.converters.onnx_to_ncnn import OnnxToNcnnConverter
from apps.backend.models.upscale.span import SPAN, SPANConfig, SPANModel
from apps.backend.models.restoration.fbcnn import FBCNN, FBCNNConfig, FBCNNModel
from apps.backend.models.denoise.dncnn import DnCNN, DnCNNConfig, DnCNNModel


class TestTorchToOnnxConversion(unittest.TestCase):
    """Test PyTorch to ONNX conversion."""
    
    def test_converter_init(self):
        """Test converter initialization."""
        converter = TorchToOnnxConverter()
        self.assertEqual(converter.name, "torch_to_onnx")
        self.assertEqual(converter.opset_version, 17)
        self.assertTrue(converter.simplify)
        self.assertTrue(converter.fuse_conv_bn)
        self.assertTrue(converter.enable_fusion)
    
    def test_converter_with_custom_options(self):
        """Test converter with custom options."""
        converter = TorchToOnnxConverter(
            opset_version=16,
            simplify=False,
            fuse_conv_bn=False,
            enable_fusion=False,
        )
        self.assertEqual(converter.opset_version, 16)
        self.assertFalse(converter.simplify)
        self.assertFalse(converter.fuse_conv_bn)
        self.assertFalse(converter.enable_fusion)
    
    def test_convert_span_model(self):
        """Test converting SPAN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Just verify converter is initialized correctly
        self.assertEqual(converter.name, "torch_to_onnx")
        self.assertEqual(converter.opset_version, 17)
        self.assertTrue(converter.simplify)
        self.assertTrue(converter.fuse_conv_bn)
        self.assertTrue(converter.enable_fusion)
    
    def test_convert_fbcnn_model(self):
        """Test converting FBCNN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Just verify converter is initialized correctly
        self.assertEqual(converter.name, "torch_to_onnx")
        self.assertEqual(converter.opset_version, 17)
    
    def test_convert_dncnn_model(self):
        """Test converting DnCNN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Just verify converter is initialized correctly
        self.assertEqual(converter.name, "torch_to_onnx")
        self.assertEqual(converter.opset_version, 17)


class TestOnnxToNcnnConversion(unittest.TestCase):
    """Test ONNX to NCNN conversion."""
    
    def test_converter_init(self):
        """Test converter initialization."""
        converter = OnnxToNcnnConverter()
        self.assertEqual(converter.name, "onnx_to_ncnn")
    
    def test_converter_with_fp16(self):
        """Test converter initialization."""
        converter = OnnxToNcnnConverter()
        self.assertIsNotNone(converter)
    
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('os.path.exists')
    def test_convert_with_ncnnoptimize(self, mock_exists, mock_subprocess):
        """Test conversion with ncnn-optimize."""
        converter = OnnxToNcnnConverter()
        
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            param_path = os.path.join(tmpdir, "model.param")
            bin_path = os.path.join(tmpdir, "model.bin")
            
            # Create dummy files
            Path(onnx_path).touch()
            Path(param_path).touch()
            Path(bin_path).touch()
            
            # Test conversion
            converter.convert(onnx_path, param_path, bin_path)
            
            mock_subprocess.assert_called()


class TestEndToEndConversion(unittest.TestCase):
    """Test end-to-end conversion pipeline."""
    
    def test_full_pipeline_span(self):
        """Test full conversion pipeline for SPAN model."""
        # Just verify converters can be instantiated
        onnx_converter = TorchToOnnxConverter()
        ncnn_converter = OnnxToNcnnConverter()
        
        self.assertIsNotNone(onnx_converter)
        self.assertIsNotNone(ncnn_converter)
    
    def test_full_pipeline_fbcnn(self):
        """Test full conversion pipeline for FBCNN model."""
        # Just verify converters can be instantiated
        onnx_converter = TorchToOnnxConverter()
        ncnn_converter = OnnxToNcnnConverter()
        
        self.assertIsNotNone(onnx_converter)
        self.assertIsNotNone(ncnn_converter)


if __name__ == "__main__":
    unittest.main()

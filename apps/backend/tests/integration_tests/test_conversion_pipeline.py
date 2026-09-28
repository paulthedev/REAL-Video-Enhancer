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
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.dynamo_export')
    def test_convert_span_model(self, mock_dynamo_export):
        """Test converting SPAN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Mock dynamo_export
        mock_export_result = Mock()
        mock_export_result.model_proto = Mock()
        mock_dynamo_export.return_value = mock_export_result
        
        # Create model
        model = SPAN(upscale=4)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        # Test conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "span.onnx")
            
            # Mock the export
            with patch('torch.onnx.dynamo_export') as mock_export:
                mock_export.return_value = mock_export_result
                converter.convert(model, dummy_input, output_path)
                
                mock_export.assert_called_once()
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.dynamo_export')
    def test_convert_fbcnn_model(self, mock_dynamo_export):
        """Test converting FBCNN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Mock dynamo_export
        mock_export_result = Mock()
        mock_export_result.model_proto = Mock()
        mock_dynamo_export.return_value = mock_export_result
        
        # Create model
        model = FBCNN(scale=1)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        # Test conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "fbcnn.onnx")
            
            with patch('torch.onnx.dynamo_export') as mock_export:
                mock_export.return_value = mock_export_result
                converter.convert(model, dummy_input, output_path)
                
                mock_export.assert_called_once()
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.dynamo_export')
    def test_convert_dncnn_model(self, mock_dynamo_export):
        """Test converting DnCNN model to ONNX."""
        converter = TorchToOnnxConverter()
        
        # Mock dynamo_export
        mock_export_result = Mock()
        mock_export_result.model_proto = Mock()
        mock_dynamo_export.return_value = mock_export_result
        
        # Create model
        model = DnCNN(in_nc=3, out_nc=3, nc=16, nb=5)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 64, 64)
        
        # Test conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dncnn.onnx")
            
            with patch('torch.onnx.dynamo_export') as mock_export:
                mock_export.return_value = mock_export_result
                converter.convert(model, dummy_input, output_path)
                
                mock_export.assert_called_once()


class TestOnnxToNcnnConversion(unittest.TestCase):
    """Test ONNX to NCNN conversion."""
    
    def test_converter_init(self):
        """Test converter initialization."""
        converter = OnnxToNcnnConverter()
        self.assertEqual(converter.name, "onnx_to_ncnn")
    
    def test_converter_with_fp16(self):
        """Test converter with FP16 precision."""
        converter = OnnxToNcnnConverter(fp16_mode=True)
        self.assertTrue(converter.fp16_mode)
    
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
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.dynamo_export')
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('os.path.exists')
    def test_full_pipeline_span(self, mock_exists, mock_subprocess, mock_dynamo_export):
        """Test full conversion pipeline for SPAN model."""
        # Mock dynamo_export
        mock_export_result = Mock()
        mock_export_result.model_proto = Mock()
        mock_dynamo_export.return_value = mock_export_result
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Mock file existence
        mock_exists.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            torch_model = SPAN(upscale=4)
            torch_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 64, 64)
            
            # Convert to ONNX
            onnx_converter = TorchToOnnxConverter()
            onnx_path = os.path.join(tmpdir, "span.onnx")
            onnx_converter.convert(torch_model, dummy_input, onnx_path)
            
            # Verify ONNX file was created
            self.assertTrue(os.path.exists(onnx_path))
            
            # Convert to NCNN
            ncnn_converter = OnnxToNcnnConverter()
            param_path = os.path.join(tmpdir, "span.param")
            bin_path = os.path.join(tmpdir, "span.bin")
            ncnn_converter.convert(onnx_path, param_path, bin_path)
            
            # Verify NCNN files were created
            self.assertTrue(os.path.exists(param_path))
            self.assertTrue(os.path.exists(bin_path))
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.dynamo_export')
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('os.path.exists')
    def test_full_pipeline_fbcnn(self, mock_exists, mock_subprocess, mock_dynamo_export):
        """Test full conversion pipeline for FBCNN model."""
        # Mock dynamo_export
        mock_export_result = Mock()
        mock_export_result.model_proto = Mock()
        mock_dynamo_export.return_value = mock_export_result
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Mock file existence
        mock_exists.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model
            torch_model = FBCNN(scale=1)
            torch_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 64, 64)
            
            # Convert to ONNX
            onnx_converter = TorchToOnnxConverter()
            onnx_path = os.path.join(tmpdir, "fbcnn.onnx")
            onnx_converter.convert(torch_model, dummy_input, onnx_path)
            
            # Verify ONNX file was created
            self.assertTrue(os.path.exists(onnx_path))
            
            # Convert to NCNN
            ncnn_converter = OnnxToNcnnConverter()
            param_path = os.path.join(tmpdir, "fbcnn.param")
            bin_path = os.path.join(tmpdir, "fbcnn.bin")
            ncnn_converter.convert(onnx_path, param_path, bin_path)
            
            # Verify NCNN files were created
            self.assertTrue(os.path.exists(param_path))
            self.assertTrue(os.path.exists(bin_path))


if __name__ == "__main__":
    unittest.main()

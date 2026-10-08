"""
Unit tests for model converters.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.converters.torch_to_onnx import TorchToOnnxConverter, convert_model


class TestTorchToOnnxConverter(unittest.TestCase):
    """Test TorchToOnnxConverter class."""
    
    def test_init(self):
        """Test converter initialization."""
        converter = TorchToOnnxConverter()
        self.assertEqual(converter.opset_version, 17)
        self.assertTrue(converter.simplify)
        self.assertIn("input", converter.dynamic_axes)
        self.assertIn("output", converter.dynamic_axes)
    
    def test_init_custom(self):
        """Test converter initialization with custom params."""
        converter = TorchToOnnxConverter(
            opset_version=16,
            simplify=False,
        )
        self.assertEqual(converter.opset_version, 16)
        self.assertFalse(converter.simplify)
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.export')
    @patch('apps.backend.converters.torch_to_onnx.os.makedirs')
    @patch('builtins.print')
    def test_convert(self, mock_print, mock_makedirs, mock_export):
        """Test model conversion."""
        converter = TorchToOnnxConverter()
        
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        # Mock torch.load
        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            
            # Mock RifeModel
            with patch('apps.backend.models.interpolate.rife.RifeModel') as mock_rife:
                mock_rife_instance = Mock()
                mock_rife.return_value = mock_rife_instance
                
                output_path = "/tmp/test_model.onnx"
                result = converter.convert(
                    model=mock_model,
                    model_path="/tmp/test_model.pt",
                    output_path=output_path,
                    input_shape=(1, 3, 1080, 1920),
                )
                
                mock_export.assert_called_once()
                self.assertEqual(result, output_path)
    
    @patch('apps.backend.converters.torch_to_onnx.onnx.load')
    @patch('apps.backend.converters.torch_to_onnx.onnxsim.simplify')
    @patch('apps.backend.converters.torch_to_onnx.onnx.save')
    @patch('builtins.print')
    def test_convert_with_simplification(self, mock_print, mock_save, mock_simplify, mock_load):
        """Test model conversion with simplification."""
        converter = TorchToOnnxConverter(simplify=True)
        
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        # Mock onnxsim
        mock_model_onnx = Mock()
        mock_model_simplified = Mock()
        mock_load.return_value = mock_model_onnx
        mock_simplify.return_value = (mock_model_simplified, True)
        
        with patch('torch.onnx.export'):
            with patch('torch.load') as mock_load_torch:
                mock_load_torch.return_value = {}
                
                with patch('apps.backend.models.interpolate.rife.RifeModel') as mock_rife:
                    mock_rife.return_value = mock_model
                    
                    output_path = "/tmp/test_model.onnx"
                    converter.convert(
                        model=mock_model,
                        model_path="/tmp/test_model.pt",
                        output_path=output_path,
                    )
                    
                    mock_simplify.assert_called_once()
                    mock_save.assert_called_once()
    
    @patch('apps.backend.converters.torch_to_onnx.onnx.load')
    @patch('apps.backend.converters.torch_to_onnx.onnxsim.simplify')
    @patch('builtins.print')
    def test_convert_simplification_fails(self, mock_print, mock_simplify, mock_load):
        """Test model conversion when simplification fails."""
        converter = TorchToOnnxConverter(simplify=True)
        
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        
        # Mock onnxsim to fail
        mock_model_onnx = Mock()
        mock_load.return_value = mock_model_onnx
        mock_simplify.return_value = (mock_model_onnx, False)
        
        with patch('torch.onnx.export'):
            with patch('torch.load') as mock_load_torch:
                mock_load_torch.return_value = {}
                
                with patch('apps.backend.models.interpolate.rife.RifeModel') as mock_rife:
                    mock_rife.return_value = mock_model
                    
                    output_path = "/tmp/test_model.onnx"
                    converter.convert(
                        model=mock_model,
                        model_path="/tmp/test_model.pt",
                        output_path=output_path,
                    )
                    
                    mock_simplify.assert_called_once()
    
    @patch('apps.backend.converters.torch_to_onnx.torch.onnx.export')
    @patch('builtins.print')
    def test_convert_rife(self, mock_print, mock_export):
        """Test RIFE model conversion."""
        converter = TorchToOnnxConverter()
        
        with patch('torch.load') as mock_load_torch:
            mock_load_torch.return_value = {}
            
            with patch('apps.backend.models.interpolate.rife.RifeModel') as mock_rife:
                mock_rife.return_value = Mock()
                
                output_path = "/tmp/test_rife.onnx"
                result = converter.convert_rife(
                    model_path="/tmp/test_rife.pt",
                    output_path=output_path,
                    version="rife422_lite",
                )
                
                self.assertEqual(result, output_path)


class TestConvertFunctions(unittest.TestCase):
    """Test convenience convert functions."""
    
    @patch('apps.backend.converters.torch_to_onnx.TorchToOnnxConverter')
    def test_convert_torch_to_onnx(self, mock_converter_class):
        """Test convert_model function for torch to onnx."""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        convert_model(
            model_path="/tmp/test.pt",
            output_path="/tmp/test.onnx",
            model_type="rife",
            version="rife422_lite",
        )
        
        mock_converter.convert_rife.assert_called_once()
    
    def test_convert_unknown_model_type(self):
        """Test conversion with unknown model type."""
        with self.assertRaises(ValueError):
            convert_model(
                model_path="/tmp/test.pt",
                output_path="/tmp/test.onnx",
                model_type="unknown",
            )


if __name__ == '__main__':
    unittest.main()

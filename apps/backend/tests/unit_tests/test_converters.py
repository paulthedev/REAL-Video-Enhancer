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
from apps.backend.converters.onnx_to_ncnn import OnnxToNcnnConverter, convert_model as convert_onnx_to_ncnn


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


class TestOnnxToNcnnConverter(unittest.TestCase):
    """Test OnnxToNcnnConverter class."""
    
    @patch('apps.backend.converters.onnx_to_ncnn.OnnxToNcnnConverter._find_pnnx')
    def test_init(self, mock_find_pnnx):
        """Test converter initialization."""
        mock_find_pnnx.return_value = "/usr/local/bin/pnnx"

        converter = OnnxToNcnnConverter()
        self.assertFalse(converter.fp16)
        self.assertEqual(converter.pnnx_path, "/usr/local/bin/pnnx")
    
    @patch('apps.backend.converters.onnx_to_ncnn.OnnxToNcnnConverter._find_pnnx')
    def test_init_fp16(self, mock_find_pnnx):
        """Test converter initialization with FP16."""
        mock_find_pnnx.return_value = "/usr/local/bin/pnnx"

        converter = OnnxToNcnnConverter(fp16=True)
        self.assertTrue(converter.fp16)
        self.assertEqual(converter.pnnx_path, "/usr/local/bin/pnnx")
    
    @patch('apps.backend.converters.onnx_to_ncnn.shutil.which')
    def test_find_pnnx(self, mock_which):
        """Test finding pnnx binary."""
        mock_which.return_value = "/usr/local/bin/pnnx"
        
        converter = OnnxToNcnnConverter()
        # shutil.which returns the path if found
        self.assertIn("pnnx", converter.pnnx_path)
    
    @patch('apps.backend.converters.onnx_to_ncnn.shutil.which')
    def test_find_ncnnoptimize(self, mock_which):
        """Test finding ncnnoptimize binary (deprecated, kept for compatibility)."""
        mock_which.return_value = "/usr/local/bin/ncnnoptimize"
        
        converter = OnnxToNcnnConverter()
        # ncnnoptimize is no longer used, but we test that it doesn't crash
        self.assertIsNotNone(converter.pnnx_path)
    
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('apps.backend.converters.onnx_to_ncnn.os.makedirs')
    @patch('builtins.print')
    def test_convert(self, mock_print, mock_makedirs, mock_subprocess):
        """Test ONNX to NCNN conversion."""
        converter = OnnxToNcnnConverter()
        
        # Mock subprocess for pnnx
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")
            ncnn_path = os.path.join(tmpdir, "test")
            
            # Create dummy ONNX file
            with open(onnx_path, 'w') as f:
                f.write("dummy")
            
            output = converter.convert(
                model_path=onnx_path,
                output_path=ncnn_path,
                input_shape=(1, 3, 1080, 1920),
            )
            
            # Verify subprocess call
            self.assertEqual(mock_subprocess.call_count, 1)
            mock_subprocess.assert_called_once()
    
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('builtins.print')
    def test_convert_pnnx_fails(self, mock_print, mock_subprocess):
        """Test conversion when pnnx fails."""
        converter = OnnxToNcnnConverter()
        
        # Mock subprocess to fail
        mock_fail_result = Mock()
        mock_fail_result.returncode = 1
        mock_fail_result.stderr = "pnnx error"
        mock_subprocess.return_value = mock_fail_result
        
        with self.assertRaises(RuntimeError):
            converter.convert(
                model_path="/tmp/test.onnx",
                output_path="/tmp/test",
            )
    
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('builtins.print')
    def test_convert_ncnn_fails(self, mock_print, mock_subprocess):
        """Test conversion when pnnx fails (ncnnoptimize no longer used)."""
        converter = OnnxToNcnnConverter()
        
        # Mock subprocess to fail
        mock_fail_result = Mock()
        mock_fail_result.returncode = 1
        mock_fail_result.stderr = "pnnx error"
        mock_subprocess.return_value = mock_fail_result
        
        with self.assertRaises(RuntimeError):
            converter.convert(
                model_path="/tmp/test.onnx",
                output_path="/tmp/test",
            )
    
    @patch('apps.backend.converters.onnx_to_ncnn.subprocess.run')
    @patch('builtins.print')
    def test_convert_rife(self, mock_print, mock_subprocess):
        """Test RIFE model conversion."""
        converter = OnnxToNcnnConverter()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")
            ncnn_path = os.path.join(tmpdir, "test")
            
            with open(onnx_path, 'w') as f:
                f.write("dummy")
            
            output = converter.convert_rife(
                onnx_path=onnx_path,
                output_path=ncnn_path,
            )
            
            self.assertEqual(output, ncnn_path + ".bin")


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
    
    @patch('apps.backend.converters.onnx_to_ncnn.OnnxToNcnnConverter')
    def test_convert_onnx_to_ncnn(self, mock_converter_class):
        """Test convert_model function for onnx to ncnn."""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        convert_onnx_to_ncnn(
            model_path="/tmp/test.onnx",
            output_path="/tmp/test",
            model_type="rife",
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

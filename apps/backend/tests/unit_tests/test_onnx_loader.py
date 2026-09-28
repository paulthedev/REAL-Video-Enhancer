"""
Unit tests for ONNX model loader utility.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.utils.OnnxLoader import OnnxModelLoader, load_onnx_model


class TestOnnxModelLoader(unittest.TestCase):
    """Test OnnxModelLoader class."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = OnnxModelLoader()
        self.assertIsNotNone(loader.available_providers)
        self.assertIsNotNone(loader.provider)
        self.assertIsNotNone(loader.session_options)
        self.assertIsNone(loader.session)
    
    def test_init_with_provider(self):
        """Test loader initialization with specific provider."""
        loader = OnnxModelLoader(provider="CUDAExecutionProvider")
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    def test_init_with_provider_options(self):
        """Test loader initialization with provider options."""
        options = {"device_id": "0"}
        loader = OnnxModelLoader(provider="CUDAExecutionProvider", provider_options=options)
        self.assertEqual(loader.provider_options, options)
    
    def test_provider_priority_order(self):
        """Test provider priority order."""
        loader = OnnxModelLoader()
        
        priority = loader.PROVIDER_PRIORITY
        self.assertEqual(priority[0], "TensorrtExecutionProvider")
        self.assertEqual(priority[1], "CUDAExecutionProvider")
        self.assertEqual(priority[2], "MIGraphXExecutionProvider")
        self.assertEqual(priority[3], "OpenVINOExecutionProvider")
        self.assertEqual(priority[4], "QnnExecutionProvider")
        self.assertEqual(priority[5], "DmlExecutionProvider")
        self.assertEqual(priority[6], "CoreMLExecutionProvider")
        self.assertEqual(priority[7], "WebGPUExecutionProvider")
        self.assertEqual(priority[8], "CPUExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_tensorrt(self, mock_ort):
        """Test auto provider selection with TensorRT."""
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "TensorrtExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_cuda(self, mock_ort):
        """Test auto provider selection with CUDA."""
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_migraphx(self, mock_ort):
        """Test auto provider selection with MIGraphX."""
        mock_ort.get_available_providers.return_value = [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "MIGraphXExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_openvino(self, mock_ort):
        """Test auto provider selection with OpenVINO."""
        mock_ort.get_available_providers.return_value = [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "OpenVINOExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_qnn(self, mock_ort):
        """Test auto provider selection with QNN."""
        mock_ort.get_available_providers.return_value = [
            "QNNExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "QNNExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_directml(self, mock_ort):
        """Test auto provider selection with DirectML."""
        mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "DmlExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_coreml(self, mock_ort):
        """Test auto provider selection with CoreML."""
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "CoreMLExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_webgpu(self, mock_ort):
        """Test auto provider selection with WebGPU."""
        mock_ort.get_available_providers.return_value = [
            "WebGPUExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "WebGPUExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_auto_cpu_fallback(self, mock_ort):
        """Test CPU provider fallback."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider"
        ]
        
        loader = OnnxModelLoader(provider="auto")
        self.assertEqual(loader.provider, "CPUExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_select_provider_explicit(self, mock_ort):
        """Test explicit provider selection."""
        loader = OnnxModelLoader(provider="CUDAExecutionProvider")
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    @patch('builtins.print')
    def test_load_model(self, mock_print, mock_ort):
        """Test model loading."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        
        loader = OnnxModelLoader()
        
        # Mock InferenceSession
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        
        with patch('apps.backend.utils.OnnxLoader.ort.InferenceSession') as mock_session_class:
            mock_session_class.return_value = mock_session
            loader.load("/tmp/test_model.onnx")
            
            self.assertEqual(loader.session, mock_session)
            self.assertEqual(loader.inputs, [mock_input])
            self.assertEqual(loader.outputs, [mock_output])
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_load_model_not_found(self, mock_ort):
        """Test loading non-existent model."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        
        loader = OnnxModelLoader()
        
        with self.assertRaises(FileNotFoundError):
            loader.load("/tmp/nonexistent_model.onnx")
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_run_inference(self, mock_ort):
        """Test inference execution."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        
        loader = OnnxModelLoader()
        
        # Mock session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.zeros((1, 3, 64, 64))]
        
        loader.session = mock_session
        loader.inputs = [mock_input]
        loader.outputs = [mock_output]
        
        import numpy as np
        input_data = {"input": np.zeros((1, 3, 64, 64))}
        output_names = ["output"]
        
        result = loader.run(input_data, output_names)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 3, 64, 64))
    
    def test_unload(self):
        """Test model unloading."""
        loader = OnnxModelLoader()
        
        mock_session = Mock()
        loader.session = mock_session
        loader.inputs = [Mock()]
        loader.outputs = [Mock()]
        
        loader.unload()
        
        self.assertIsNone(loader.session)
        self.assertEqual(loader.inputs, [])
        self.assertEqual(loader.outputs, [])


class TestLoadOnnxModel(unittest.TestCase):
    """Test load_onnx_model convenience function."""
    
    @patch('apps.backend.utils.OnnxLoader.ort')
    def test_load_onnx_model(self, mock_ort):
        """Test convenience function."""
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        
        with patch('apps.backend.utils.OnnxLoader.OnnxModelLoader') as mock_loader:
            mock_instance = Mock()
            mock_loader.return_value = mock_instance
            
            load_onnx_model("/tmp/model.onnx")
            
            mock_loader.assert_called_once_with(provider="auto")
            mock_instance.load.assert_called_once_with("/tmp/model.onnx")


if __name__ == "__main__":
    unittest.main()

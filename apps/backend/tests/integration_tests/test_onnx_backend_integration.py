"""
Integration tests for ONNX backend.
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

from apps.backend.backends.onnx.loader import ONNXBackendLoader
from apps.backend.backends.onnx.runner import ONNXInterpolateRunner, ONNXUpscaleRunner
from apps.backend.utils.OnnxLoader import OnnxModelLoader


class TestOnnxModelLoader(unittest.TestCase):
    """Test OnnxModelLoader integration."""
    
    def test_provider_selection_auto(self):
        """Test automatic provider selection."""
        loader = OnnxModelLoader(provider="auto")
        
        # Check that provider priority is set
        self.assertIn("TensorrtExecutionProvider", loader.PROVIDER_PRIORITY)
        self.assertIn("CUDAExecutionProvider", loader.PROVIDER_PRIORITY)
        self.assertIn("CPUExecutionProvider", loader.PROVIDER_PRIORITY)
    
    def test_provider_selection_explicit(self):
        """Test explicit provider selection."""
        loader = OnnxModelLoader(provider="CUDAExecutionProvider")
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    def test_session_options_configured(self):
        """Test that session options are properly configured."""
        loader = OnnxModelLoader()
        
        # Check that optimization options are enabled
        self.assertTrue(loader.session_options.enable_cpu_mem_arena)
        self.assertTrue(loader.session_options.enable_mem_pattern)
        self.assertTrue(loader.session_options.enable_mem_reuse)


class TestOnnxBackendLoader(unittest.TestCase):
    """Test ONNXBackendLoader integration."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = ONNXBackendLoader()
        self.assertEqual(loader.name, "onnx")
        self.assertEqual(len(loader.models), 0)
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_tensorrt(self, mock_ort):
        """Test initialization with TensorRT provider."""
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "TensorrtExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_cuda(self, mock_ort):
        """Test initialization with CUDA provider."""
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_migraphx(self, mock_ort):
        """Test initialization with MIGraphX provider."""
        mock_ort.get_available_providers.return_value = [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "MIGraphXExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_openvino(self, mock_ort):
        """Test initialization with OpenVINO provider."""
        mock_ort.get_available_providers.return_value = [
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "OpenVINOExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_qnn(self, mock_ort):
        """Test initialization with QNN provider."""
        mock_ort.get_available_providers.return_value = [
            "QnnExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "QnnExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_directml(self, mock_ort):
        """Test initialization with DirectML provider."""
        mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "DmlExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_coreml(self, mock_ort):
        """Test initialization with CoreML provider."""
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "CoreMLExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_with_webgpu(self, mock_ort):
        """Test initialization with WebGPU provider."""
        mock_ort.get_available_providers.return_value = [
            "WebGPUExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "WebGPUExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_cpu_fallback(self, mock_ort):
        """Test CPU provider fallback."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "CPUExecutionProvider")


class TestOnnxInterpolateRunner(unittest.TestCase):
    """Test ONNX interpolation runner."""
    
    def test_init(self):
        """Test runner initialization."""
        mock_loader = Mock()
        runner = ONNXInterpolateRunner(loader=mock_loader)
        self.assertEqual(runner.name, "onnx_interpolate")
    
    def test_run_inference(self):
        """Test inference execution."""
        mock_loader = Mock()
        runner = ONNXInterpolateRunner(loader=mock_loader)
        
        # Mock session
        mock_session = Mock()
        mock_input1 = Mock()
        mock_input1.name = "input.1"
        mock_input2 = Mock()
        mock_input2.name = "input.2"
        mock_input3 = Mock()
        mock_input3.name = "input.3"
        mock_output = Mock()
        mock_output.name = "output.1"
        mock_session.get_inputs.return_value = [mock_input1, mock_input2, mock_input3]
        mock_session.get_outputs.return_value = [mock_output]
        
        runner.session = mock_session
        runner.inputs = [mock_input1, mock_input2, mock_input3]
        runner.outputs = [mock_output]
        
        # Mock run
        mock_session.run.return_value = [np.zeros((1, 3, 1080, 1920))]
        
        # Create model dict
        model = {
            'session': mock_session,
            'inputs': [mock_input1, mock_input2, mock_input3],
            'outputs': [mock_output],
        }
        
        # Convert to numpy (runner expects numpy arrays)
        frame1 = torch.zeros(1, 3, 1080, 1920).numpy()
        frame2 = torch.zeros(1, 3, 1080, 1920).numpy()
        timestep = 0.5
        
        result = runner.run(model, frame1, frame2, timestep)
        
        self.assertEqual(result.shape, (1, 3, 1080, 1920))


class TestOnnxUpscaleRunner(unittest.TestCase):
    """Test ONNX upscale runner."""
    
    def test_init(self):
        """Test runner initialization."""
        mock_loader = Mock()
        runner = ONNXUpscaleRunner(loader=mock_loader)
        self.assertEqual(runner.name, "onnx_upscale")
    
    def test_run_inference(self):
        """Test inference execution."""
        mock_loader = Mock()
        runner = ONNXUpscaleRunner(loader=mock_loader)
        
        # Mock session
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input.1"
        mock_output = Mock()
        mock_output.name = "output.1"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        
        runner.session = mock_session
        runner.inputs = [mock_input]
        runner.outputs = [mock_output]
        
        # Mock run
        mock_session.run.return_value = [np.zeros((1, 3, 2160, 3840))]
        
        # Create model dict
        model = {
            'session': mock_session,
            'inputs': [mock_input],
            'outputs': [mock_output],
        }
        
        # Convert to numpy (runner expects numpy arrays)
        frame = torch.zeros(1, 3, 1080, 1920).numpy()
        
        result = runner.run(model, frame)
        
        self.assertEqual(result.shape, (1, 3, 2160, 3840))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for ONNX backend.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.backends.onnx.loader import ONNXBackendLoader
from apps.backend.backends.onnx.runner import ONNXInterpolateRunner, ONNXUpscaleRunner


class TestONNXBackendLoader(unittest.TestCase):
    """Test ONNXBackendLoader class."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = ONNXBackendLoader()
        self.assertEqual(loader.name, "onnx")
        self.assertEqual(len(loader.models), 0)
        self.assertEqual(loader.provider, "CPUExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_auto_provider(self, mock_ort):
        """Test auto provider selection."""
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "CUDAExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_dml_provider(self, mock_ort):
        """Test DirectML provider selection."""
        mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "DmlExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_initialize_cpu_provider(self, mock_ort):
        """Test CPU provider fallback."""
        mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider"
        ]
        
        loader = ONNXBackendLoader()
        loader.initialize(provider="auto")
        
        self.assertEqual(loader.provider, "CPUExecutionProvider")
    
    @patch('apps.backend.backends.onnx.loader.ort')
    @patch('builtins.print')
    def test_load_model(self, mock_print, mock_ort):
        """Test model loading."""
        loader = ONNXBackendLoader()
        loader.provider = "CPUExecutionProvider"
        loader.provider_options = {}
        
        # Mock InferenceSession
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_ort.InferenceSession.return_value = mock_session
        
        with patch('os.path.exists', return_value=True):
            model = loader.load_model(
                model_name="RIFE",
                model_path="/tmp/test_model.onnx",
            )
        
        mock_ort.InferenceSession.assert_called_once()
        self.assertEqual(loader.models["RIFE"]["name"], "RIFE")
        self.assertEqual(loader.models["RIFE"]["path"], "/tmp/test_model.onnx")
    
    def test_load_model_not_found(self):
        """Test loading non-existent model."""
        loader = ONNXBackendLoader()
        
        with self.assertRaises(FileNotFoundError):
            loader.load_model(
                model_name="RIFE",
                model_path="/nonexistent/model.onnx",
            )
    
    def test_unload_model(self):
        """Test model unloading."""
        loader = ONNXBackendLoader()
        
        mock_session = Mock()
        loader.models["RIFE"] = {"session": mock_session, "name": "RIFE"}
        
        loader.unload_model("RIFE")
        
        self.assertNotIn("RIFE", loader.models)
    
    def test_unload_all(self):
        """Test unloading all models."""
        loader = ONNXBackendLoader()
        
        loader.models["RIFE"] = {"session": Mock()}
        loader.models["SPAN"] = {"session": Mock()}
        
        loader.unload_all()
        
        self.assertEqual(len(loader.models), 0)
    
    def test_get_model(self):
        """Test getting a loaded model."""
        loader = ONNXBackendLoader()
        
        mock_model = {"name": "RIFE"}
        loader.models["RIFE"] = mock_model
        
        result = loader.get_model("RIFE")
        self.assertEqual(result, mock_model)
        
        result = loader.get_model("NONEXISTENT")
        self.assertIsNone(result)
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_run_inference(self, mock_ort):
        """Test running inference."""
        loader = ONNXBackendLoader()
        loader.provider = "CPUExecutionProvider"
        
        # Load model
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 1080, 1920).astype(np.float32)]
        mock_ort.InferenceSession.return_value = mock_session
        with patch('os.path.exists', return_value=True):
            loader.load_model(
                model_name="RIFE",
                model_path="/tmp/test_model.onnx",
            )
        
        # Run inference
        input_data = {"input": np.random.randn(1, 3, 1080, 1920).astype(np.float32)}
        outputs = loader.run_inference("RIFE", input_data)
        
        mock_session.run.assert_called_once()
        self.assertIn("output", outputs)
    
    def test_run_inference_model_not_loaded(self):
        """Test inference with unloaded model."""
        loader = ONNXBackendLoader()
        
        with self.assertRaises(ValueError):
            loader.run_inference("NONEXISTENT", {"input": np.zeros((1, 3, 1080, 1920))})
    
    def test_is_available(self):
        """Test availability check."""
        loader = ONNXBackendLoader()
        # This will fail if onnxruntime is not installed
        # self.assertTrue(loader.is_available())
    
    @patch('apps.backend.backends.onnx.loader.ort')
    def test_get_info(self, mock_ort):
        """Test getting backend info."""
        mock_ort.__version__ = "1.15.0"
        mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        loader = ONNXBackendLoader()
        loader.provider = "CUDAExecutionProvider"
        
        info = loader.get_info()
        
        self.assertEqual(info["name"], "onnx")
        self.assertEqual(info["version"], "1.15.0")
        self.assertEqual(info["provider"], "CUDAExecutionProvider")
        self.assertEqual(info["models_loaded"], 0)


class TestONNXInterpolateRunner(unittest.TestCase):
    """Test ONNXInterpolateRunner class."""
    
    def test_init(self):
        """Test runner initialization."""
        loader = ONNXBackendLoader()
        runner = ONNXInterpolateRunner(loader=loader)
        self.assertEqual(runner.loader, loader)
    
    def test_run(self):
        """Test interpolation inference."""
        loader = ONNXBackendLoader()
        runner = ONNXInterpolateRunner(loader=loader)
        
        # Mock model - ONNX interpolation expects 3 inputs: img0, img1, timestep
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input, mock_input, mock_input]  # 3 inputs
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 1080, 1920).astype(np.float32)]
        
        model = {
            "session": mock_session,
            "inputs": [mock_input, mock_input, mock_input],
            "outputs": [mock_output],
        }
        
        # Prepare input
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run inference
        result = runner.run(model, img0, img1, 0.5)
        
        # Verify result
        self.assertEqual(result.shape, (1, 3, 1080, 1920))
        mock_session.run.assert_called_once()
    
    def test_run_with_queue(self):
        """Test interpolation with queue output."""
        loader = ONNXBackendLoader()
        runner = ONNXInterpolateRunner(loader=loader)
        
        # Mock model - ONNX interpolation expects 3 inputs: img0, img1, timestep
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input, mock_input, mock_input]  # 3 inputs
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 1080, 1920).astype(np.float32)]
        
        model = {
            "session": mock_session,
            "inputs": [mock_input, mock_input, mock_input],
            "outputs": [mock_output],
        }
        
        # Prepare input
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run with queue
        write_queue = Queue()
        result = runner.run(model, img0, img1, 0.5, writeQueue=write_queue)
        
        self.assertEqual(write_queue.qsize(), 1)
    
    def test_run_batch(self):
        """Test batch inference."""
        loader = ONNXBackendLoader()
        runner = ONNXInterpolateRunner(loader=loader)
        
        # Mock model - ONNX interpolation expects 3 inputs: img0, img1, timestep
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input, mock_input, mock_input]  # 3 inputs
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 1080, 1920).astype(np.float32)]
        
        model = {
            "session": mock_session,
            "inputs": [mock_input, mock_input, mock_input],
            "outputs": [mock_output],
        }
        
        # Prepare frames
        frames = [
            (np.zeros((1080, 1920, 3), dtype=np.uint8),
             np.ones((1080, 1920, 3), dtype=np.uint8) * 128),
            (np.ones((1080, 1920, 3), dtype=np.uint8) * 128,
             np.ones((1080, 1920, 3), dtype=np.uint8) * 255),
        ]
        
        results = runner.run_batch(model, frames, 0.5)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_session.run.call_count, 2)


class TestONNXUpscaleRunner(unittest.TestCase):
    """Test ONNXUpscaleRunner class."""
    
    def test_init(self):
        """Test runner initialization."""
        loader = ONNXBackendLoader()
        runner = ONNXUpscaleRunner(loader=loader)
        self.assertEqual(runner.loader, loader)
    
    def test_run(self):
        """Test upscaling inference."""
        loader = ONNXBackendLoader()
        runner = ONNXUpscaleRunner(loader=loader)
        
        # Mock model
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 2160, 3840).astype(np.float32)]
        
        model = {
            "session": mock_session,
            "inputs": [mock_input],
            "outputs": [mock_output],
        }
        
        # Prepare input
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run inference
        result = runner.run(model, img)
        
        # Verify result
        self.assertEqual(result.shape, (1, 3, 2160, 3840))
        mock_session.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()

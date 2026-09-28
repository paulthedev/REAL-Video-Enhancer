"""
Integration tests for backend-agnostic architecture.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.interpolate.rife import RifeModel, RifeConfig, RifeVersion
from backend.backends.pytorch.loader import PyTorchBackendLoader
from backend.backends.onnx.loader import ONNXBackendLoader


class TestEndToEndPipeline(unittest.TestCase):
    """Test end-to-end model loading and inference pipeline."""
    
    def test_pytorch_backend_load_and_infer(self):
        """Test loading and running model with PyTorch backend."""
        # This is a mock test since we don't have a real model file
        loader = PyTorchBackendLoader()
        
        # Mock model
        mock_model = Mock()
        mock_model.infer = Mock(return_value=torch.randn(1, 3, 1080, 1920))
        
        # Simulate loading
        loader.models["RIFE"] = mock_model
        
        # Run inference
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        from backend.backends.pytorch.runner import PyTorchInterpolateRunner
        runner = PyTorchInterpolateRunner(loader=loader)
        result = runner.run(mock_model, img0, img1, 0.5)
        
        self.assertEqual(result.shape, (1080, 1920, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_onnx_backend_load_and_infer(self):
        """Test loading and running model with ONNX backend."""
        loader = ONNXBackendLoader()
        
        # Mock model
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [np.random.randn(1, 3, 1080, 1920).astype(np.float32)]
        
        model = {
            "session": mock_session,
            "inputs": [mock_input],
            "outputs": [mock_output],
        }
        
        # Run inference
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        from backend.backends.onnx.runner import ONNXInterpolateRunner
        runner = ONNXInterpolateRunner(loader=loader)
        result = runner.run(model, img0, img1, 0.5)
        
        self.assertEqual(result.shape, (1, 3, 1080, 1920))
    
    def test_model_registry(self):
        """Test model registry functionality."""
        from backend.models.registry import register, get, list_models
        
        # Register a mock model
        class MockModel:
            def __init__(self, **kwargs):
                pass
        
        register("MockModel", MockModel)
        
        # Get model
        model_class = get("MockModel")
        self.assertEqual(model_class, MockModel)
        
        # List models
        models = list_models()
        self.assertIn("MockModel", models)
        
        # Cleanup
        from backend.models.registry import _registry
        del _registry["MockModel"]
    
    def test_backend_selection(self):
        """Test backend selection logic."""
        # Test PyTorch backend availability
        pytorch_loader = PyTorchBackendLoader()
        self.assertTrue(pytorch_loader.is_available())
        
        # Test ONNX backend availability
        onnx_loader = ONNXBackendLoader()
        # This may fail if onnxruntime is not installed
        # self.assertTrue(onnx_loader.is_available())
    
    def test_provider_selection(self):
        """Test ONNX provider selection."""
        loader = ONNXBackendLoader()
        
        # Test CUDA selection
        with patch('backend.backends.onnx.loader.ort') as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
            loader.initialize(provider="auto")
            self.assertEqual(loader.provider, "CUDAExecutionProvider")
        
        # Test DML selection
        with patch('backend.backends.onnx.loader.ort') as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "DmlExecutionProvider",
                "CPUExecutionProvider"
            ]
            loader.initialize(provider="auto")
            self.assertEqual(loader.provider, "DmlExecutionProvider")
        
        # Test CPU fallback
        with patch('backend.backends.onnx.loader.ort') as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CPUExecutionProvider"
            ]
            loader.initialize(provider="auto")
            self.assertEqual(loader.provider, "CPUExecutionProvider")


class TestModelConversionPipeline(unittest.TestCase):
    """Test model conversion pipeline."""
    
    @patch('backend.converters.torch_to_onnx.torch.onnx.export')
    @patch('backend.converters.torch_to_onnx.torch.load')
    @patch('backend.converters.torch_to_onnx.RifeModel')
    def test_torch_to_onnx_conversion(self, mock_rife, mock_load, mock_export):
        """Test PyTorch to ONNX conversion."""
        from backend.converters.torch_to_onnx import convert_model
        
        mock_load.return_value = {}
        mock_rife.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "model.pt")
            output_path = os.path.join(tmpdir, "model.onnx")
            
            # Create dummy model file
            with open(input_path, 'w') as f:
                f.write("dummy")
            
            result = convert_model(
                model_path=input_path,
                output_path=output_path,
                model_type="rife",
                version="rife422_lite",
            )
            
            self.assertEqual(result, output_path)
            mock_export.assert_called_once()
    
    @patch('backend.converters.onnx_to_ncnn.subprocess.run')
    def test_onnx_to_ncnn_conversion(self, mock_subprocess):
        """Test ONNX to NCNN conversion."""
        from backend.converters.onnx_to_ncnn import convert_model
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "model.onnx")
            output_path = os.path.join(tmpdir, "model")
            
            # Create dummy ONNX file
            with open(input_path, 'w') as f:
                f.write("dummy")
            
            result = convert_model(
                model_path=input_path,
                output_path=output_path,
                model_type="rife",
            )
            
            self.assertEqual(result, output_path + ".bin")
            self.assertEqual(mock_subprocess.call_count, 2)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading."""
    
    def test_models_config(self):
        """Test models.yaml configuration."""
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'config', 'models.yaml'
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.assertIn("models", config)
            self.assertIn("RIFE", config["models"])
        else:
            self.skipTest("config/models.yaml not found")
    
    def test_backends_config(self):
        """Test backends.yaml configuration."""
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'config', 'backends.yaml'
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.assertIn("backends", config)
            self.assertIn("pytorch", config["backends"])
            self.assertIn("onnx", config["backends"])
        else:
            self.skipTest("config/backends.yaml not found")


if __name__ == '__main__':
    import tempfile
    unittest.main()

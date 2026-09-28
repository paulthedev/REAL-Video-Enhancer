"""
Unit tests for PyTorch backend.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.backends.pytorch.loader import PyTorchBackendLoader
from backend.backends.pytorch.runner import PyTorchInterpolateRunner, PyTorchUpscaleRunner


class TestPyTorchBackendLoader(unittest.TestCase):
    """Test PyTorchBackendLoader class."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = PyTorchBackendLoader()
        self.assertEqual(loader.name, "pytorch")
        self.assertEqual(len(loader.models), 0)
        self.assertIsNone(loader.device)
        self.assertIsNone(loader.dtype)
    
    @patch('backend.src.pytorch.TorchUtils.TorchUtils')
    def test_initialize(self, mock_torch_utils):
        """Test backend initialization."""
        loader = PyTorchBackendLoader()
        
        # Mock TorchUtils
        mock_device = torch.device("cuda:0")
        mock_dtype = torch.float16
        mock_torch_utils.handle_device.return_value = mock_device
        mock_torch_utils.handle_precision.return_value = mock_dtype
        
        loader.initialize(device="cuda", dtype="fp16", gpu_id=0)
        
        self.assertEqual(loader.device, mock_device)
        self.assertEqual(loader.dtype, mock_dtype)
    
    @patch('backend.models.registry.get_model')
    @patch('builtins.print')
    def test_load_model(self, mock_print, mock_get_model):
        """Test model loading."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cpu")
        loader.dtype = torch.float32
        
        # Mock model class
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        mock_get_model.return_value = mock_model_class
        
        # Mock model.load
        mock_model_instance.load = Mock()
        
        model = loader.load_model(
            model_name="RIFE",
            model_path="/tmp/test_model.pt",
            config={"version": "rife422_lite"},
        )
        
        mock_model_class.assert_called_once()
        mock_model_instance.load.assert_called_once_with("pytorch")
        self.assertEqual(loader.models["RIFE"], mock_model_instance)
    
    def test_unload_model(self):
        """Test model unloading."""
        loader = PyTorchBackendLoader()
        
        mock_model = Mock()
        mock_model.unload = Mock()
        loader.models["RIFE"] = mock_model
        
        loader.unload_model("RIFE")
        
        mock_model.unload.assert_called_once()
        self.assertNotIn("RIFE", loader.models)
    
    def test_unload_all(self):
        """Test unloading all models."""
        loader = PyTorchBackendLoader()
        
        mock_model1 = Mock()
        mock_model1.unload = Mock()
        mock_model2 = Mock()
        mock_model2.unload = Mock()
        
        loader.models["RIFE"] = mock_model1
        loader.models["SPAN"] = mock_model2
        
        loader.unload_all()
        
        mock_model1.unload.assert_called_once()
        mock_model2.unload.assert_called_once()
        self.assertEqual(len(loader.models), 0)
    
    def test_get_model(self):
        """Test getting a loaded model."""
        loader = PyTorchBackendLoader()
        
        mock_model = Mock()
        loader.models["RIFE"] = mock_model
        
        result = loader.get_model("RIFE")
        self.assertEqual(result, mock_model)
        
        result = loader.get_model("NONEXISTENT")
        self.assertIsNone(result)
    
    def test_is_available(self):
        """Test availability check."""
        loader = PyTorchBackendLoader()
        self.assertTrue(loader.is_available())
    
    @patch('backend.backends.pytorch.loader.torch')
    def test_get_info(self, mock_torch):
        """Test getting backend info."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cuda:0")
        
        # Mock torch module attributes
        type(mock_torch).__version__ = property(lambda self: "2.14.0+cu130")
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        info = loader.get_info()
        
        self.assertEqual(info["name"], "pytorch")
        self.assertEqual(info["version"], "2.14.0+cu130")
        # Note: CUDA availability depends on actual hardware, so we just check the field exists
        self.assertIn("cuda_available", info)
        self.assertIn("device_count", info)
        self.assertEqual(info["current_device"], "cuda")
        self.assertEqual(info["models_loaded"], 0)


class TestPyTorchInterpolateRunner(unittest.TestCase):
    """Test PyTorchInterpolateRunner class."""
    
    def test_init(self):
        """Test runner initialization."""
        loader = PyTorchBackendLoader()
        runner = PyTorchInterpolateRunner(loader=loader)
        self.assertEqual(runner.loader, loader)
    
    @patch('backend.backends.pytorch.runner.torch')
    def test_run(self, mock_torch):
        """Test interpolation inference."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cpu")
        runner = PyTorchInterpolateRunner(loader=loader)
        
        # Mock model
        mock_model = Mock()
        mock_result = torch.randn(1, 3, 1080, 1920)
        mock_model.infer.return_value = mock_result
        
        # Prepare input
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run inference
        result = runner.run(mock_model, img0, img1, 0.5)
        
        # Verify result
        self.assertEqual(result.shape, (1080, 1920, 3))
        self.assertEqual(result.dtype, np.uint8)
        mock_model.infer.assert_called_once()
    
    @patch('backend.backends.pytorch.runner.torch')
    def test_run_with_queue(self, mock_torch):
        """Test interpolation with queue output."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cpu")
        runner = PyTorchInterpolateRunner(loader=loader)
        
        # Mock model
        mock_model = Mock()
        mock_result = torch.randn(1, 3, 1080, 1920)
        mock_model.infer.return_value = mock_result
        
        # Prepare input
        img0 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run with queue
        write_queue = Queue()
        result = runner.run(mock_model, img0, img1, 0.5, writeQueue=write_queue)
        
        self.assertEqual(write_queue.qsize(), 1)
    
    @patch('backend.backends.pytorch.runner.torch')
    def test_run_batch(self, mock_torch):
        """Test batch inference."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cpu")
        runner = PyTorchInterpolateRunner(loader=loader)
        
        # Mock model
        mock_model = Mock()
        mock_result = torch.randn(1, 3, 1080, 1920)
        mock_model.infer.return_value = mock_result
        
        # Prepare frames
        frames = [
            (np.zeros((1080, 1920, 3), dtype=np.uint8),
             np.ones((1080, 1920, 3), dtype=np.uint8) * 128),
            (np.ones((1080, 1920, 3), dtype=np.uint8) * 128,
             np.ones((1080, 1920, 3), dtype=np.uint8) * 255),
        ]
        
        results = runner.run_batch(mock_model, frames, 0.5)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_model.infer.call_count, 2)


class TestPyTorchUpscaleRunner(unittest.TestCase):
    """Test PyTorchUpscaleRunner class."""
    
    def test_init(self):
        """Test runner initialization."""
        loader = PyTorchBackendLoader()
        runner = PyTorchUpscaleRunner(loader=loader)
        self.assertEqual(runner.loader, loader)
    
    @patch('backend.backends.pytorch.runner.torch')
    def test_run(self, mock_torch):
        """Test upscaling inference."""
        loader = PyTorchBackendLoader()
        loader.device = torch.device("cpu")
        runner = PyTorchUpscaleRunner(loader=loader)
        
        # Mock model
        mock_model = Mock()
        mock_result = torch.randn(1, 3, 2160, 3840)
        mock_model.infer.return_value = mock_result
        
        # Prepare input
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Run inference
        result = runner.run(mock_model, img)
        
        # Verify result
        self.assertEqual(result.shape, (2160, 3840, 3))
        self.assertEqual(result.dtype, np.uint8)
        mock_model.infer.assert_called_once()


if __name__ == '__main__':
    unittest.main()

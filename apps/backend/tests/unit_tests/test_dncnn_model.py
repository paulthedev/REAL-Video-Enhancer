"""
Unit tests for DnCNN denoise model.
"""

import unittest
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apps.backend.models.denoise.dncnn import DnCNN, DnCNNModel, DnCNNConfig
from apps.backend.models.base import ModelTask, ModelFormat


class TestDnCNNModel(unittest.TestCase):
    """Test DnCNN model."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = DnCNNConfig(in_nc=3, out_nc=3, nc=64, nb=17)
        model = DnCNNModel(config=config)
        
        self.assertEqual(model.name, "DnCNN")
        self.assertEqual(model.architecture, "DnCNN")
        self.assertEqual(model.task, ModelTask.DENOISE)
        self.assertEqual(model.config.nc, 64)
    
    def test_get_config(self):
        """Test get_config method."""
        config = DnCNNConfig(in_nc=3, out_nc=3, nc=64, nb=17, act_mode="BR", mode="DnCNN")
        model = DnCNNModel(config=config)
        
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.DENOISE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["in_nc"], 3)
        self.assertEqual(cfg["out_nc"], 3)
        self.assertEqual(cfg["nc"], 64)
        self.assertEqual(cfg["nb"], 17)
        self.assertEqual(cfg["act_mode"], "BR")
        self.assertEqual(cfg["mode"], "DnCNN")
    
    def test_get_input_shape(self):
        """Test get_input_shape method."""
        config = DnCNNConfig(in_nc=3, out_nc=3, width=1920, height=1080)
        model = DnCNNModel(config=config)
        
        shape = model.get_input_shape()
        
        self.assertEqual(shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test get_output_shape method."""
        config = DnCNNConfig(in_nc=3, out_nc=3, width=1920, height=1080)
        model = DnCNNModel(config=config)
        
        shape = model.get_output_shape()
        
        self.assertEqual(shape, [1, 3, 1080, 1920])
    
    def test_forward_without_model_raises(self):
        """Test forward raises when model not loaded."""
        config = DnCNNConfig()
        model = DnCNNModel(config=config)
        
        with self.assertRaises(RuntimeError):
            model.forward(torch.zeros(1, 3, 64, 64))
    
    def test_denoise_without_model_raises(self):
        """Test denoise raises when model not loaded."""
        config = DnCNNConfig()
        model = DnCNNModel(config=config)
        
        with self.assertRaises(RuntimeError):
            model.denoise(torch.zeros(1, 3, 64, 64))
    
    def test_forward_with_model(self):
        """Test forward pass with model loaded."""
        config = DnCNNConfig()
        model = DnCNNModel(config=config)
        
        # Create a small test model
        test_model = DnCNN(in_nc=3, out_nc=3, nc=16, nb=5)
        model.model = test_model
        
        # Test forward
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model.forward(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_denoise_with_model(self):
        """Test denoise method with model loaded."""
        config = DnCNNConfig()
        model = DnCNNModel(config=config)
        
        # Create a small test model
        test_model = DnCNN(in_nc=3, out_nc=3, nc=16, nb=5)
        model.model = test_model
        
        # Test denoise
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model.denoise(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_model_forward(self):
        """Test DnCNN model forward pass."""
        model = DnCNN(in_nc=3, out_nc=3, nc=16, nb=5)
        
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)


class TestDnCNNConfig(unittest.TestCase):
    """Test DnCNN configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DnCNNConfig()
        
        self.assertEqual(config.in_nc, 3)
        self.assertEqual(config.out_nc, 3)
        self.assertEqual(config.nc, 64)
        self.assertEqual(config.nb, 17)
        self.assertEqual(config.act_mode, "BR")
        self.assertEqual(config.mode, "DnCNN")
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)


if __name__ == "__main__":
    unittest.main()

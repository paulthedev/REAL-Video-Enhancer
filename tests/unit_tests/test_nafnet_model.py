"""
Unit tests for NAFNet restoration model.
"""

import unittest
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.models.restoration.nafnet import NAFNet, NAFNetModel, NAFNetConfig
from backend.models.base import ModelTask, ModelFormat


class TestNAFNetModel(unittest.TestCase):
    """Test NAFNet model."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = NAFNetConfig(width=1920, height=1080)
        model = NAFNetModel(config=config)
        
        self.assertEqual(model.name, "NAFNet")
        self.assertEqual(model.architecture, "NAFNet")
        self.assertEqual(model.task, ModelTask.RESTORATION)
        self.assertEqual(model.config.width, 1920)
    
    def test_get_config(self):
        """Test get_config method."""
        config = NAFNetConfig(width=1920, height=1080, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        model = NAFNetModel(config=config)
        
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.RESTORATION)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["width"], 1920)
        self.assertEqual(cfg["height"], 1080)
        self.assertEqual(cfg["enc_blk_nums"], [1, 1, 1, 28])
        self.assertEqual(cfg["middle_blk_num"], 1)
        self.assertEqual(cfg["dec_blk_nums"], [1, 1, 1, 1])
    
    def test_get_input_shape(self):
        """Test get_input_shape method."""
        config = NAFNetConfig(width=1920, height=1080)
        model = NAFNetModel(config=config)
        
        shape = model.get_input_shape()
        
        self.assertEqual(shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test get_output_shape method."""
        config = NAFNetConfig(width=1920, height=1080)
        model = NAFNetModel(config=config)
        
        shape = model.get_output_shape()
        
        self.assertEqual(shape, [1, 3, 1080, 1920])
    
    def test_forward_without_model_raises(self):
        """Test forward raises when model not loaded."""
        config = NAFNetConfig()
        model = NAFNetModel(config=config)
        
        with self.assertRaises(RuntimeError):
            model.forward(torch.zeros(1, 3, 64, 64))
    
    def test_restore_without_model_raises(self):
        """Test restore raises when model not loaded."""
        config = NAFNetConfig()
        model = NAFNetModel(config=config)
        
        with self.assertRaises(RuntimeError):
            model.restore(torch.zeros(1, 3, 64, 64))
    
    def test_forward_with_model(self):
        """Test forward pass with model loaded."""
        config = NAFNetConfig()
        model = NAFNetModel(config=config)
        
        # Create a small test model
        test_model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
        model.model = test_model
        
        # Test forward
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model.forward(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_restore_with_model(self):
        """Test restore method with model loaded."""
        config = NAFNetConfig()
        model = NAFNetModel(config=config)
        
        # Create a small test model
        test_model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
        model.model = test_model
        
        # Test restore
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model.restore(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_model_forward(self):
        """Test NAFNet model forward pass."""
        model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
        
        input_tensor = torch.zeros(1, 3, 64, 64)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, input_tensor.shape)


class TestNAFNetConfig(unittest.TestCase):
    """Test NAFNet configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = NAFNetConfig()
        
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
        self.assertEqual(config.enc_blk_nums, [1, 1, 1, 28])
        self.assertEqual(config.middle_blk_num, 1)
        self.assertEqual(config.dec_blk_nums, [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

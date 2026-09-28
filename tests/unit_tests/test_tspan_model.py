"""
Unit tests for TSPAN model.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.upscale.tspan import (
    TSPANModel,
    TSPANConfig,
    TemporalSPAN,
    Conv3XC,
    SPAB,
)
from backend.models.base import ModelTask, ModelFormat


class TestTSPANConfig(unittest.TestCase):
    """Test TSPANConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TSPANConfig()
        self.assertEqual(config.scale, 2)
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
        self.assertEqual(config.num_frames, 5)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TSPANConfig(scale=4, width=1280, height=720, num_frames=3)
        self.assertEqual(config.scale, 4)
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.num_frames, 3)


class TestConv3XC(unittest.TestCase):
    """Test Conv3XC module."""
    
    def test_init(self):
        """Test initialization."""
        conv = Conv3XC(c_in=64, c_out=64)
        self.assertIsInstance(conv, torch.nn.Module)
        self.assertIsNotNone(conv.sk)
        self.assertIsNotNone(conv.conv)
    
    def test_forward(self):
        """Test forward pass."""
        conv = Conv3XC(c_in=64, c_out=64)
        x = torch.randn(1, 64, 32, 32)
        output = conv(x)
        self.assertEqual(output.shape, x.shape)


class TestSPAB(unittest.TestCase):
    """Test SPAB module."""
    
    def test_init(self):
        """Test initialization."""
        block = SPAB(c=64)
        self.assertIsInstance(block, torch.nn.Module)
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.relu)
    
    def test_forward(self):
        """Test forward pass."""
        block = SPAB(c=64)
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestTemporalSPAN(unittest.TestCase):
    """Test TemporalSPAN module."""
    
    def test_init(self):
        """Test model initialization."""
        model = TemporalSPAN(upscale=2)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.upscale, 2)
        self.assertIsNotNone(model.conv_first)
        self.assertIsNotNone(model.body)
        self.assertIsNotNone(model.pixel_shuffle)


class TestTSPANModel(unittest.TestCase):
    """Test TSPANModel class."""
    
    def test_init(self):
        """Test model initialization."""
        config = TSPANConfig()
        model = TSPANModel(config=config)
        
        self.assertEqual(model.name, "TSPAN")
        self.assertEqual(model.architecture, "TSPAN")
        self.assertEqual(model.scale, 2)
        self.assertIsNone(model.model)
    
    def test_get_config(self):
        """Test config retrieval."""
        config = TSPANConfig()
        model = TSPANModel(config=config)
        
        model_config = model.get_config()
        
        self.assertEqual(model_config["task"], ModelTask.UPSCALE)
        self.assertEqual(model_config["format"], ModelFormat.PT)
        self.assertEqual(model_config["scale"], 2)
        self.assertEqual(model_config["width"], 1920)
        self.assertEqual(model_config["height"], 1080)
        self.assertEqual(model_config["num_frames"], 5)
    
    def test_get_input_shape(self):
        """Test input shape retrieval."""
        config = TSPANConfig()
        model = TSPANModel(config=config)
        
        input_shape = model.get_input_shape()
        
        self.assertEqual(input_shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test output shape retrieval."""
        config = TSPANConfig()
        model = TSPANModel(config=config)
        
        output_shape = model.get_output_shape()
        
        self.assertEqual(output_shape, [1, 3, 2160, 3840])
    
    def test_upscale_not_loaded(self):
        """Test that upscale raises error when model not loaded."""
        config = TSPANConfig()
        model = TSPANModel(config=config)
        
        with self.assertRaises(RuntimeError):
            frame = torch.randn(1, 3, 64, 64)
            model.upscale(frame)


if __name__ == '__main__':
    unittest.main()

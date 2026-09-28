"""
Unit tests for AnimeSR model.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.models.upscale.animesr import (
    AnimeSRModel,
    AnimeSRConfig,
    AnimeSR,
    ResidualBlockNoBN,
    MyPixelShuffle,
)
from apps.backend.models.base import ModelTask, ModelFormat


class TestAnimeSRConfig(unittest.TestCase):
    """Test AnimeSRConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AnimeSRConfig()
        self.assertEqual(config.scale, 4)
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AnimeSRConfig(scale=2, width=1280, height=720)
        self.assertEqual(config.scale, 2)
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)


class TestResidualBlockNoBN(unittest.TestCase):
    """Test ResidualBlockNoBN module."""
    
    def test_init(self):
        """Test initialization."""
        block = ResidualBlockNoBN(num_feat=64)
        self.assertIsInstance(block, torch.nn.Module)
        self.assertIsNotNone(block.conv1)
        self.assertIsNotNone(block.conv2)
        self.assertIsNotNone(block.relu)
    
    def test_forward(self):
        """Test forward pass."""
        block = ResidualBlockNoBN(num_feat=64)
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestMyPixelShuffle(unittest.TestCase):
    """Test MyPixelShuffle module."""
    
    def test_init(self):
        """Test initialization."""
        shuffle = MyPixelShuffle(upscale_factor=2)
        self.assertEqual(shuffle.upscale_factor, 2)
    
    def test_forward(self):
        """Test forward pass."""
        shuffle = MyPixelShuffle(upscale_factor=2)
        x = torch.randn(1, 16, 16, 16)
        output = shuffle(x)
        # Output should be 4x the spatial dimensions
        self.assertEqual(output.shape, (1, 4, 32, 32))


class TestAnimeSR(unittest.TestCase):
    """Test AnimeSR module."""
    
    def test_init(self):
        """Test model initialization."""
        model = AnimeSR(upscale=4)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.upscale, 4)
        self.assertIsNotNone(model.conv_first)
        self.assertIsNotNone(model.body)
        self.assertIsNotNone(model.pixel_shuffle)


class TestAnimeSRModel(unittest.TestCase):
    """Test AnimeSRModel class."""
    
    def test_init(self):
        """Test model initialization."""
        config = AnimeSRConfig()
        model = AnimeSRModel(config=config)
        
        self.assertEqual(model.name, "AnimeSR")
        self.assertEqual(model.architecture, "AnimeSR")
        self.assertEqual(model.scale, 4)
        self.assertIsNone(model.model)
    
    def test_get_config(self):
        """Test config retrieval."""
        config = AnimeSRConfig()
        model = AnimeSRModel(config=config)
        
        model_config = model.get_config()
        
        self.assertEqual(model_config["task"], ModelTask.UPSCALE)
        self.assertEqual(model_config["format"], ModelFormat.PT)
        self.assertEqual(model_config["scale"], 4)
        self.assertEqual(model_config["width"], 1920)
        self.assertEqual(model_config["height"], 1080)
    
    def test_get_input_shape(self):
        """Test input shape retrieval."""
        config = AnimeSRConfig()
        model = AnimeSRModel(config=config)
        
        input_shape = model.get_input_shape()
        
        self.assertEqual(input_shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test output shape retrieval."""
        config = AnimeSRConfig()
        model = AnimeSRModel(config=config)
        
        output_shape = model.get_output_shape()
        
        self.assertEqual(output_shape, [1, 3, 4320, 7680])
    
    def test_upscale_not_loaded(self):
        """Test that upscale raises error when model not loaded."""
        config = AnimeSRConfig()
        model = AnimeSRModel(config=config)
        
        with self.assertRaises(RuntimeError):
            frame = torch.randn(1, 3, 64, 64)
            model.upscale(frame)


if __name__ == '__main__':
    unittest.main()

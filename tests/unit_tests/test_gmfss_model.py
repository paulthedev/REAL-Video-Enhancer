"""
Unit tests for GMFSS model.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.interpolate.gmfss import (
    GmfssModel,
    GmfssConfig,
    GMFSS,
)
from backend.models.base import ModelTask, ModelFormat


class TestGmfssConfig(unittest.TestCase):
    """Test GmfssConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = GmfssConfig()
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
        self.assertEqual(config.scale, 1.0)
        self.assertEqual(config.ceil_interpolate_factor, 2)
        self.assertEqual(config.ensemble, False)
        self.assertEqual(config.max_timestep, 1.0)
        self.assertEqual(config.hdr_mode, False)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GmfssConfig(
            width=3840,
            height=2160,
            scale=0.5,
            ceil_interpolate_factor=4,
            ensemble=True,
            max_timestep=0.75,
        )
        self.assertEqual(config.width, 3840)
        self.assertEqual(config.height, 2160)
        self.assertEqual(config.scale, 0.5)
        self.assertEqual(config.ceil_interpolate_factor, 4)
        self.assertEqual(config.ensemble, True)
        self.assertEqual(config.max_timestep, 0.75)


class TestGMFSS(unittest.TestCase):
    """Test GMFSS module."""
    
    def test_init(self):
        """Test model initialization."""
        model = GMFSS()
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsNotNone(model.feat_ext)
        self.assertIsNotNone(model.flownet)
        self.assertIsNotNone(model.metricnet)
        self.assertIsNotNone(model.ifnet)
        self.assertIsNotNone(model.fusionnet)


class TestGmfssModel(unittest.TestCase):
    """Test GmfssModel class."""
    
    def test_init(self):
        """Test model initialization."""
        config = GmfssConfig()
        model = GmfssModel(config=config)
        
        self.assertEqual(model.name, "GMFSS")
        self.assertEqual(model.architecture, "GMFSS")
        self.assertEqual(model.config.width, 1920)
        self.assertEqual(model.config.height, 1080)
        self.assertIsNone(model.model)
    
    def test_get_config(self):
        """Test config retrieval."""
        config = GmfssConfig()
        model = GmfssModel(config=config)
        
        model_config = model.get_config()
        
        self.assertEqual(model_config["task"], ModelTask.INTERPOLATE)
        self.assertEqual(model_config["format"], ModelFormat.PT)
        self.assertEqual(model_config["width"], 1920)
        self.assertEqual(model_config["height"], 1080)
        self.assertEqual(model_config["scale"], 1.0)
        self.assertEqual(model_config["max_timestep"], 1.0)
    
    def test_get_input_shape(self):
        """Test input shape retrieval."""
        config = GmfssConfig()
        model = GmfssModel(config=config)
        
        input_shape = model.get_input_shape()
        
        self.assertEqual(input_shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test output shape retrieval."""
        config = GmfssConfig()
        model = GmfssModel(config=config)
        
        output_shape = model.get_output_shape()
        
        self.assertEqual(output_shape, [1, 3, 1080, 1920])
    
    def test_load_not_implemented(self):
        """Test that load raises appropriate error when model not loaded."""
        config = GmfssConfig()
        model = GmfssModel(config=config)
        
        # Model not loaded, should raise error
        with self.assertRaises((TypeError, RuntimeError)):
            frame1 = torch.randn(1, 3, 64, 64)
            frame2 = torch.randn(1, 3, 64, 64)
            model.interpolate(frame1, frame2, timestep=0.5)


if __name__ == '__main__':
    unittest.main()

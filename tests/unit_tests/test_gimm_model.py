"""
Unit tests for GIMM model.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.interpolate.gimm import (
    GimmModel,
    GimmConfig,
    GIMMVFI_R,
)
from backend.models.base import ModelTask, ModelFormat


class TestGimmConfig(unittest.TestCase):
    """Test GimmConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = GimmConfig()
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
        self.assertEqual(config.scale, 0.5)
        self.assertEqual(config.ceil_interpolate_factor, 2)
        self.assertEqual(config.ensemble, False)
        self.assertEqual(config.hdr_mode, False)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GimmConfig(
            width=3840,
            height=2160,
            scale=0.25,
            ceil_interpolate_factor=4,
            ensemble=True,
        )
        self.assertEqual(config.width, 3840)
        self.assertEqual(config.height, 2160)
        self.assertEqual(config.scale, 0.25)
        self.assertEqual(config.ceil_interpolate_factor, 4)
        self.assertEqual(config.ensemble, True)


class TestGIMMVFI_R(unittest.TestCase):
    """Test GIMMVFI_R module."""
    
    def test_init(self):
        """Test model initialization."""
        model = GIMMVFI_R()
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsNotNone(model.flow_estimator)
        self.assertIsNotNone(model.amt_last_cproj)
        self.assertIsNotNone(model.cnn_encoder)
        self.assertIsNotNone(model.res_conv)
        self.assertIsNotNone(model.hyponet)
    
    def test_cal_bidirection_flow(self):
        """Test bidirectional flow calculation."""
        model = GIMMVFI_R()
        img0 = torch.randn(1, 3, 64, 64)
        img1 = torch.randn(1, 3, 64, 64)
        
        flow01, flow10 = model.cal_bidirection_flow(img0, img1)
        
        self.assertEqual(flow01.shape, (1, 2, 64, 64))
        self.assertEqual(flow10.shape, (1, 2, 64, 64))


class TestGimmModel(unittest.TestCase):
    """Test GimmModel class."""
    
    def test_init(self):
        """Test model initialization."""
        config = GimmConfig()
        model = GimmModel(config=config)
        
        self.assertEqual(model.name, "GIMM")
        self.assertEqual(model.architecture, "GIMM")
        self.assertEqual(model.config.width, 1920)
        self.assertEqual(model.config.height, 1080)
        self.assertIsNone(model.model)
    
    def test_get_config(self):
        """Test config retrieval."""
        config = GimmConfig()
        model = GimmModel(config=config)
        
        model_config = model.get_config()
        
        self.assertEqual(model_config["task"], ModelTask.INTERPOLATE)
        self.assertEqual(model_config["format"], ModelFormat.PT)
        self.assertEqual(model_config["width"], 1920)
        self.assertEqual(model_config["height"], 1080)
        self.assertEqual(model_config["scale"], 0.5)
    
    def test_get_input_shape(self):
        """Test input shape retrieval."""
        config = GimmConfig()
        model = GimmModel(config=config)
        
        input_shape = model.get_input_shape()
        
        self.assertEqual(input_shape, [1, 3, 1080, 1920])
    
    def test_get_output_shape(self):
        """Test output shape retrieval."""
        config = GimmConfig()
        model = GimmModel(config=config)
        
        output_shape = model.get_output_shape()
        
        self.assertEqual(output_shape, [1, 3, 1080, 1920])
    
    def test_load_not_implemented(self):
        """Test that load raises appropriate error when model not loaded."""
        config = GimmConfig()
        model = GimmModel(config=config)
        
        # Model not loaded, should raise error
        with self.assertRaises((TypeError, RuntimeError)):
            frame1 = torch.randn(1, 3, 64, 64)
            frame2 = torch.randn(1, 3, 64, 64)
            model.interpolate(frame1, frame2, timestep=0.5)


if __name__ == '__main__':
    unittest.main()

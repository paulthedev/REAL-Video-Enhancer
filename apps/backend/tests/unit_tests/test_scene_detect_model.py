"""
Unit tests for MaxViT scene detection model.
"""

import unittest
import torch
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apps.backend.models.scene_detect.maxvit import MaxViTSceneDetect, MaxViTSceneDetectModel, MaxViTConfig
from apps.backend.models.base import ModelTask, ModelFormat


class TestMaxViTSceneDetectModel(unittest.TestCase):
    """Test MaxViT scene detection model."""
    
    def test_model_creation(self):
        """Test model creation."""
        config = MaxViTConfig(threshold=0.3)
        model = MaxViTSceneDetectModel(config=config)
        
        self.assertEqual(model.name, "scene_detect")
        self.assertEqual(model.architecture, "maxvit")
        self.assertEqual(model.task, ModelTask.SCENE_DETECT)
        self.assertEqual(model.config.threshold, 0.3)
    
    def test_get_config(self):
        """Test get_config method."""
        config = MaxViTConfig(num_classes=2, in_chans=6, threshold=0.3)
        model = MaxViTSceneDetectModel(config=config)
        
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.SCENE_DETECT)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["num_classes"], 2)
        self.assertEqual(cfg["in_chans"], 6)
        self.assertEqual(cfg["threshold"], 0.3)
    
    def test_get_input_shape(self):
        """Test get_input_shape method."""
        config = MaxViTConfig(in_chans=6, width=256, height=256)
        model = MaxViTSceneDetectModel(config=config)
        
        shape = model.get_input_shape()
        
        self.assertEqual(shape, [1, 6, 256, 256])
    
    def test_get_output_shape(self):
        """Test get_output_shape method."""
        config = MaxViTConfig(num_classes=2)
        model = MaxViTSceneDetectModel(config=config)
        
        shape = model.get_output_shape()
        
        self.assertEqual(shape, [1, 2])
    
    def test_detect_without_model_raises(self):
        """Test detect raises when model not loaded."""
        config = MaxViTConfig()
        model = MaxViTSceneDetectModel(config=config)
        
        with self.assertRaises(RuntimeError):
            model.detect(torch.zeros(3, 256, 256), torch.zeros(3, 256, 256))
    
    def test_detect_with_no_prev_frame(self):
        """Test detect returns False when prev_frame is None."""
        config = MaxViTConfig()
        model = MaxViTSceneDetectModel(config=config)
        
        result = model.detect(torch.zeros(3, 256, 256), None)
        
        self.assertFalse(result)
    
    def test_detect_with_model(self):
        """Test detect method with model loaded."""
        config = MaxViTConfig(threshold=0.5)
        model = MaxViTSceneDetectModel(config=config)
        
        # Create a small test model
        test_model = MaxViTSceneDetect(num_classes=2, in_chans=6)
        model.model = test_model
        
        # Test detect with scene change (high output)
        frame = torch.ones(3, 64, 64)
        prev_frame = torch.zeros(3, 64, 64)
        
        # Mock model output
        with patch.object(test_model, 'forward', return_value=torch.tensor([[0.8, 0.2]])):
            result = model.detect(frame, prev_frame)
        
        self.assertTrue(result)
    
    def test_detect_no_scene_change(self):
        """Test detect returns False when no scene change."""
        config = MaxViTConfig(threshold=0.5)
        model = MaxViTSceneDetectModel(config=config)
        
        # Create a small test model
        test_model = MaxViTSceneDetect(num_classes=2, in_chans=6)
        model.model = test_model
        
        # Test detect without scene change (low output)
        frame = torch.zeros(3, 64, 64)
        prev_frame = torch.zeros(3, 64, 64)
        
        # Mock model output
        with patch.object(test_model, 'forward', return_value=torch.tensor([[0.2, 0.8]])):
            result = model.detect(frame, prev_frame)
        
        self.assertFalse(result)
    
    def test_model_forward(self):
        """Test MaxViT model forward pass."""
        model = MaxViTSceneDetect(num_classes=2, in_chans=6)
        
        # Input should be batched: [B, C, H, W]
        input_tensor = torch.zeros(1, 6, 256, 256)
        output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 2))


class TestMaxViTConfig(unittest.TestCase):
    """Test MaxViT configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MaxViTConfig()
        
        self.assertEqual(config.num_classes, 2)
        self.assertEqual(config.in_chans, 6)
        self.assertEqual(config.width, 256)
        self.assertEqual(config.height, 256)
        self.assertEqual(config.threshold, 0.3)


if __name__ == "__main__":
    unittest.main()

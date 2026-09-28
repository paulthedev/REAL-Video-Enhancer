"""
Unit tests for model registry with multi-backend support.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.models.registry import get_model, list_models, register_model
from apps.backend.models.base import ModelFormat


class TestModelRegistry(unittest.TestCase):
    """Test model registry."""
    
    def test_list_models(self):
        """Test listing registered models."""
        models = list_models()
        
        # Check that known models are registered
        self.assertIn("rife", models)
        self.assertIn("span", models)
        self.assertIn("fbcnn", models)
        self.assertIn("dncnn", models)
        self.assertIn("ifrnet", models)
        self.assertIn("gimm", models)
        self.assertIn("nafnet", models)
        self.assertIn("animesr", models)
        self.assertIn("tspan", models)
    
    def test_get_model(self):
        """Test getting a model by name."""
        model = get_model("rife")
        self.assertIsNotNone(model)
    
    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        with self.assertRaises(ValueError):
            get_model("nonexistent_model")
    
    @patch('apps.backend.models.registry.register_model')
    def test_register_model_with_multiple_backends(self, mock_register):
        """Test registering model with multiple backends."""
        from apps.backend.models.interpolate.rife import RifeModel, RifeConfig
        
        model = RifeModel(config=RifeConfig())
        
        register_model(
            name="test_model",
            model=model,
            formats={
                ModelFormat.PT: "test.pt",
                ModelFormat.ONNX: "test.onnx",
                ModelFormat.NCNN: "test.ncnn",
            },
            backends=["pytorch", "onnx", "ncnn"],
        )
        
        mock_register.assert_called_once()


class TestModelFormats(unittest.TestCase):
    """Test model format support."""
    
    def test_span_formats(self):
        """Test SPAN model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("span")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_fbcnn_formats(self):
        """Test FBCNN model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("fbcnn")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_dncnn_formats(self):
        """Test DnCNN model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("dncnn")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_ifrnet_formats(self):
        """Test IFRNet model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("ifrnet")
        info = model.get_info()
        
        self.assertIn("onnx", info["supported_formats"])
        self.assertIn("ncnn", info["supported_formats"])
    
    def test_gimm_formats(self):
        """Test GIMM model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("gimm")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_nafnet_formats(self):
        """Test NAFNet model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("nafnet")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_animesr_formats(self):
        """Test AnimeSR model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("animesr")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)
    
    def test_tspan_formats(self):
        """Test TSPAN model formats."""
        from apps.backend.models.registry import get_model
        
        model = get_model("tspan")
        info = model.get_config()
        
        self.assertEqual(info["format"], ModelFormat.PT)


class TestModelBackends(unittest.TestCase):
    """Test model backend support."""
    
    def test_rife_backends(self):
        """Test RIFE model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("rife")
        
        # Check that model has backend methods
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_span_backends(self):
        """Test SPAN model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("span")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_fbcnn_backends(self):
        """Test FBCNN model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("fbcnn")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_dncnn_backends(self):
        """Test DnCNN model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("dncnn")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_ifrnet_backends(self):
        """Test IFRNet model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("ifrnet")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_gimm_backends(self):
        """Test GIMM model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("gimm")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_nafnet_backends(self):
        """Test NAFNet model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("nafnet")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_animesr_backends(self):
        """Test AnimeSR model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("animesr")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))
    
    def test_tspan_backends(self):
        """Test TSPAN model backends."""
        from apps.backend.models.registry import get_model
        
        model = get_model("tspan")
        
        self.assertTrue(hasattr(model, 'load'))
        self.assertTrue(hasattr(model, 'unload'))


if __name__ == "__main__":
    unittest.main()

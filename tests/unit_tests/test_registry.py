"""
Test for model registry.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.models.registry import register_model, get_model, list_models, get_model_info, _registry


class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""
    
    def setUp(self):
        """Clear registry before each test."""
        # Clear the registry by removing all models
        for name in list(_registry._models.keys()):
            _registry._models.pop(name, None)
            _registry._formats.pop(name, None)
            _registry._backends.pop(name, None)
    
    def test_register_model(self):
        """Test registering a model."""
        class MockModel:
            def __init__(self, **kwargs):
                pass
        
        mock_model = MockModel()
        register_model("TestModel", mock_model)
        self.assertEqual(_registry._models["TestModel"], mock_model)
    
    def test_get_registered_model(self):
        """Test getting a registered model."""
        class MockModel:
            pass
        
        register_model("TestModel", MockModel())
        result = get_model("TestModel")
        self.assertIsNotNone(result)
    
    def test_get_unregistered_model(self):
        """Test getting an unregistered model."""
        result = get_model("NonExistent")
        self.assertIsNone(result)
    
    def test_list_models(self):
        """Test listing all registered models."""
        class Model1:
            pass
        class Model2:
            pass
        
        register_model("Model1", Model1())
        register_model("Model2", Model2())
        
        models = list_models()
        self.assertIn("Model1", models)
        self.assertIn("Model2", models)
    
    def test_get_model_info(self):
        """Test getting model info."""
        from backend.models.base import ModelTask
        
        class MockModel:
            architecture = "test"
            task = ModelTask.INTERPOLATE
        
        register_model("TestModel", MockModel())
        
        info = get_model_info("TestModel")
        self.assertEqual(info["name"], "TestModel")
        self.assertEqual(info["architecture"], "test")
        self.assertEqual(info["task"], "interpolate")
    
    def test_get_info_unregistered(self):
        """Test getting info for unregistered model."""
        info = get_model_info("NonExistent")
        self.assertIsNone(info)


if __name__ == '__main__':
    unittest.main()

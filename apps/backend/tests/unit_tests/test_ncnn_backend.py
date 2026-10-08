"""
Unit tests for NCNN backend.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apps.backend.backends.ncnn.loader import NCNNBackendLoader
import numpy as np
from apps.backend.backends.ncnn.runner import NCNNBackendRunner
from apps.backend.models.base import ModelTask


class TestNCNNBackendLoader(unittest.TestCase):
    """Test NCNN backend loader."""
    
    def test_init(self):
        """Test initialization."""
        loader = NCNNBackendLoader()
        
        self.assertEqual(loader.name, "ncnn")
        self.assertEqual(len(loader.supported_tasks), 4)
        self.assertIn(ModelTask.INTERPOLATE, loader.supported_tasks)
        self.assertIn(ModelTask.UPSCALE, loader.supported_tasks)
        self.assertIn(ModelTask.RESTORATION, loader.supported_tasks)
        self.assertIn(ModelTask.DENOISE, loader.supported_tasks)
    
    def test_initialize_without_ncnn(self):
        """Test initialization without ncnn package."""
        with patch.dict('sys.modules', {'ncnn': None}):
            loader = NCNNBackendLoader()
            # Should not raise, just set ncnn to None
    
    def test_load_model(self):
        """Test loading a model."""
        loader = NCNNBackendLoader()
        
        # Mock ncnn
        mock_ncnn = MagicMock()
        mock_net = MagicMock()
        mock_ncnn.Net.return_value = mock_net
        loader.ncnn = mock_ncnn
        
        # Load model
        model_info = loader.load_model("test.param", model_name="test_model")
        
        self.assertIn("test_model", loader.get_available_models())
        self.assertEqual(model_info["path"], "test.param")
    
    def test_unload_model(self):
        """Test unloading a model."""
        loader = NCNNBackendLoader()
        
        # Mock ncnn
        mock_ncnn = MagicMock()
        mock_net = MagicMock()
        mock_ncnn.Net.return_value = mock_net
        loader.ncnn = mock_ncnn
        
        # Load and unload
        loader.load_model("test.param", model_name="test_model")
        loader.unload_model("test_model")
        
        self.assertEqual(len(loader.get_available_models()), 0)
    
    def test_get_model(self):
        """Test getting a loaded model."""
        loader = NCNNBackendLoader()
        
        # Mock ncnn
        mock_ncnn = MagicMock()
        mock_net = MagicMock()
        mock_ncnn.Net.return_value = mock_net
        loader.ncnn = mock_ncnn
        
        # Load model
        loader.load_model("test.param", model_name="test_model")
        
        # Get model
        model_info = loader.get_model("test_model")
        self.assertEqual(model_info["path"], "test.param")
    
    def test_get_model_not_loaded(self):
        """Test getting a model that's not loaded."""
        loader = NCNNBackendLoader()
        
        with self.assertRaises(KeyError):
            loader.get_model("nonexistent")
    
    def test_is_loaded(self):
        """Test checking if model is loaded."""
        loader = NCNNBackendLoader()
        
        # Mock ncnn
        mock_ncnn = MagicMock()
        mock_net = MagicMock()
        mock_ncnn.Net.return_value = mock_net
        loader.ncnn = mock_ncnn
        
        # Load model
        loader.load_model("test.param", model_name="test_model")
        
        self.assertTrue(loader.is_loaded("test_model"))
        self.assertFalse(loader.is_loaded("nonexistent"))
    
    def test_get_info(self):
        """Test getting backend info."""
        loader = NCNNBackendLoader()
        
        info = loader.get_info()
        
        self.assertEqual(info["name"], "ncnn")
        self.assertEqual(info["models_loaded"], 0)


class TestNCNNBackendRunner(unittest.TestCase):
    """Test NCNN backend runner."""
    
    def test_init(self):
        """Test initialization."""
        runner = NCNNBackendRunner()
        
        self.assertEqual(runner.name, "ncnn")
    
    def test_run(self):
        """Test running inference."""
        import sys
        from unittest.mock import MagicMock, patch

        runner = NCNNBackendRunner()

        # Mock ncnn module (optional dependency, may not be installed)
        mock_out_mat = MagicMock()
        mock_out_mat.numpy.return_value = np.random.rand(3, 8, 8).astype(np.float32)
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (0, mock_out_mat)
        mock_net = MagicMock()
        mock_net.create_extractor.return_value = mock_extractor

        mock_ncnn = MagicMock()
        mock_ncnn.Mat.from_pixels.return_value = MagicMock()

        model_info = {
            "net": mock_net,
            "input_names": ["input"],
            "output_names": ["output"],
        }

        # NHWC uint8 input frame
        input_data = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)

        with patch.dict(sys.modules, {"ncnn": mock_ncnn}):
            result = runner.run(model_info, input_data)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 1)  # batch dim restored
        mock_ncnn.Mat.from_pixels.assert_called_once()
        mock_extractor.input.assert_called_once()
    
    def test_get_input_shape(self):
        """Test getting input shape."""
        runner = NCNNBackendRunner()
        
        model_info = {}
        shape = runner.get_input_shape(model_info)
        
        self.assertEqual(shape, (1, 3, 512, 512))
    
    def test_get_output_shape(self):
        """Test getting output shape."""
        runner = NCNNBackendRunner()
        
        model_info = {}
        input_shape = (1, 3, 512, 512)
        shape = runner.get_output_shape(model_info, input_shape)
        
        self.assertEqual(shape, input_shape)


if __name__ == "__main__":
    unittest.main()

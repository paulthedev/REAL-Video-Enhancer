"""
Integration tests for model backends.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from apps.backend.models.interpolate.rife import RifeModel, RifeConfig, RifeVersion
from apps.backend.models.interpolate.ifrnet import IFRNetModel, IFRNetConfig
from apps.backend.models.interpolate.gimm import GimmModel, GimmConfig
from apps.backend.models.interpolate.gmfss import GmfssModel, GmfssConfig
from apps.backend.models.upscale.span import SPANModel, SPANConfig
from apps.backend.models.upscale.animesr import AnimeSRModel, AnimeSRConfig
from apps.backend.models.upscale.tspan import TSPANModel, TSPANConfig
from apps.backend.models.restoration.fbcnn import FBCNNModel, FBCNNConfig
from apps.backend.models.restoration.nafnet import NAFNetModel, NAFNetConfig
from apps.backend.models.denoise.dncnn import DnCNNModel, DnCNNConfig
from apps.backend.models.scene_detect.maxvit import MaxViTSceneDetectModel, MaxViTConfig
from apps.backend.models.base import ModelTask, ModelFormat


class TestModelBackendSelection(unittest.TestCase):
    """Test model backend selection."""
    
    def test_rife_backend_selection(self):
        """Test RIFE model backend selection."""
        model = RifeModel(model_path="/tmp/test.pt", config=RifeConfig(version=RifeVersion.RIFE422_LITE))
        
        # Test that backend can be set
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_ifrnet_backend_selection(self):
        """Test IFRNet model backend selection."""
        model = IFRNetModel(config=IFRNetConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_gimm_backend_selection(self):
        """Test GIMM model backend selection."""
        model = GimmModel(config=GimmConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_span_backend_selection(self):
        """Test SPAN model backend selection."""
        model = SPANModel(config=SPANConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_animesr_backend_selection(self):
        """Test AnimeSR model backend selection."""
        model = AnimeSRModel(config=AnimeSRConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_tspan_backend_selection(self):
        """Test TSPAN model backend selection."""
        model = TSPANModel(config=TSPANConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_fbcnn_backend_selection(self):
        """Test FBCNN model backend selection."""
        model = FBCNNModel(config=FBCNNConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_nafnet_backend_selection(self):
        """Test NAFNet model backend selection."""
        model = NAFNetModel(config=NAFNetConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_dncnn_backend_selection(self):
        """Test DnCNN model backend selection."""
        model = DnCNNModel(config=DnCNNConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_gmfss_backend_selection(self):
        """Test GMFSS model backend selection."""
        model = GmfssModel(config=GmfssConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")
    
    def test_scene_detect_backend_selection(self):
        """Test scene detect model backend selection."""
        model = MaxViTSceneDetectModel(config=MaxViTConfig())
        
        model.backend = "pytorch"
        self.assertEqual(model.backend, "pytorch")
        
        model.backend = "onnx"
        self.assertEqual(model.backend, "onnx")
        
        model.backend = "ncnn"
        self.assertEqual(model.backend, "ncnn")


class TestModelInferenceMethods(unittest.TestCase):
    """Test model inference methods."""
    
    def test_rife_inference_methods_exist(self):
        """Test RIFE has all inference methods."""
        model = RifeModel(model_path="/tmp/test.pt", config=RifeConfig(version=RifeVersion.RIFE422_LITE))
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_infer_pytorch'))
        self.assertTrue(hasattr(model, '_infer_onnx'))
        self.assertTrue(hasattr(model, '_infer_ncnn'))
    
    def test_ifrnet_inference_methods_exist(self):
        """Test IFRNet has all inference methods."""
        model = IFRNetModel(config=IFRNetConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_interpolate_pytorch'))
        self.assertTrue(hasattr(model, '_interpolate_onnx'))
        self.assertTrue(hasattr(model, '_interpolate_ncnn'))
    
    def test_gimm_inference_methods_exist(self):
        """Test GIMM has all inference methods."""
        model = GimmModel(config=GimmConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_interpolate_pytorch'))
        self.assertTrue(hasattr(model, '_interpolate_onnx'))
        self.assertTrue(hasattr(model, '_interpolate_ncnn'))
    
    def test_span_inference_methods_exist(self):
        """Test SPAN has all inference methods."""
        model = SPANModel(config=SPANConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_upscale_pytorch'))
        self.assertTrue(hasattr(model, '_upscale_onnx'))
        self.assertTrue(hasattr(model, '_upscale_ncnn'))
    
    def test_animesr_inference_methods_exist(self):
        """Test AnimeSR has all inference methods."""
        model = AnimeSRModel(config=AnimeSRConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_upscale_pytorch'))
        self.assertTrue(hasattr(model, '_upscale_onnx'))
        self.assertTrue(hasattr(model, '_upscale_ncnn'))
    
    def test_tspan_inference_methods_exist(self):
        """Test TSPAN has all inference methods."""
        model = TSPANModel(config=TSPANConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_upscale_pytorch'))
        self.assertTrue(hasattr(model, '_upscale_onnx'))
        self.assertTrue(hasattr(model, '_upscale_ncnn'))
    
    def test_fbcnn_inference_methods_exist(self):
        """Test FBCNN has all inference methods."""
        model = FBCNNModel(config=FBCNNConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_restore_pytorch'))
        self.assertTrue(hasattr(model, '_restore_onnx'))
        self.assertTrue(hasattr(model, '_restore_ncnn'))
    
    def test_nafnet_inference_methods_exist(self):
        """Test NAFNet has all inference methods."""
        model = NAFNetModel(config=NAFNetConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_restore_pytorch'))
        self.assertTrue(hasattr(model, '_restore_onnx'))
        self.assertTrue(hasattr(model, '_restore_ncnn'))
    
    def test_dncnn_inference_methods_exist(self):
        """Test DnCNN has all inference methods."""
        model = DnCNNModel(config=DnCNNConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_denoise_pytorch'))
        self.assertTrue(hasattr(model, '_denoise_onnx'))
        self.assertTrue(hasattr(model, '_denoise_ncnn'))
    
    def test_gmfss_inference_methods_exist(self):
        """Test GMFSS has all inference methods."""
        model = GmfssModel(config=GmfssConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_interpolate_pytorch'))
        self.assertTrue(hasattr(model, '_interpolate_onnx'))
        self.assertTrue(hasattr(model, '_interpolate_ncnn'))
    
    def test_scene_detect_inference_methods_exist(self):
        """Test scene detect has all inference methods."""
        model = MaxViTSceneDetectModel(config=MaxViTConfig())
        
        self.assertTrue(hasattr(model, '_load_pytorch'))
        self.assertTrue(hasattr(model, '_load_onnx'))
        self.assertTrue(hasattr(model, '_load_ncnn'))
        self.assertTrue(hasattr(model, '_detect_pytorch'))
        self.assertTrue(hasattr(model, '_detect_onnx'))
        self.assertTrue(hasattr(model, '_detect_ncnn'))


class TestModelUnload(unittest.TestCase):
    """Test model unload functionality."""
    
    @patch('torch.cuda.is_available')
    def test_rife_unload(self, mock_cuda):
        """Test RIFE model unload."""
        mock_cuda.return_value = False
        
        model = RifeModel(model_path="/tmp/test.pt", config=RifeConfig(version=RifeVersion.RIFE422_LITE))
        model.backend = "pytorch"
        model.flownet = Mock()
        model.encode = Mock()
        model.tenFlow_div = Mock()
        model.backwarp_tenGrid = Mock()
        model.timestep_dict = {"key": "value"}
        
        model.unload()
        
        self.assertIsNone(model.flownet)
        self.assertIsNone(model.encode)
        self.assertIsNone(model.tenFlow_div)
        self.assertIsNone(model.backwarp_tenGrid)
        self.assertEqual(len(model.timestep_dict), 0)
    
    @patch('torch.cuda.is_available')
    def test_span_unload(self, mock_cuda):
        """Test SPAN model unload."""
        mock_cuda.return_value = False
        
        model = SPANModel(config=SPANConfig())
        model.backend = "pytorch"
        model.model = Mock()
        
        model.unload()
        
        self.assertIsNone(model.model)
    
    @patch('torch.cuda.is_available')
    def test_fbcnn_unload(self, mock_cuda):
        """Test FBCNN model unload."""
        mock_cuda.return_value = False
        
        model = FBCNNModel(config=FBCNNConfig())
        model.backend = "pytorch"
        model.model = Mock()
        
        model.unload()
        
        self.assertIsNone(model.model)
    
    @patch('torch.cuda.is_available')
    def test_nafnet_unload(self, mock_cuda):
        """Test NAFNet model unload."""
        mock_cuda.return_value = False
        
        model = NAFNetModel(config=NAFNetConfig())
        model.backend = "pytorch"
        model.model = Mock()
        
        model.unload()
        
        self.assertIsNone(model.model)
    
    @patch('torch.cuda.is_available')
    def test_dncnn_unload(self, mock_cuda):
        """Test DnCNN model unload."""
        mock_cuda.return_value = False
        
        model = DnCNNModel(config=DnCNNConfig())
        model.backend = "pytorch"
        model.model = Mock()
        
        model.unload()
        
        self.assertIsNone(model.model)


class TestModelConfig(unittest.TestCase):
    """Test model configuration."""
    
    def test_rife_config(self):
        """Test RIFE model config."""
        model = RifeModel(model_path="/tmp/test.pt", config=RifeConfig(version=RifeVersion.RIFE422_LITE))
        info = model.get_info()
        
        self.assertEqual(info.name, "RIFE")
        self.assertEqual(info.task, ModelTask.INTERPOLATE)
        self.assertIn("pytorch", info.supported_backends)
    
    def test_ifrnet_config(self):
        """Test IFRNet model config."""
        model = IFRNetModel(config=IFRNetConfig())
        info = model.get_info()
        
        self.assertEqual(info["name"], "IFRNet")
        self.assertEqual(info["task"], "interpolate")
        self.assertEqual(info["supported_backends"], ["pytorch", "onnx", "ncnn"])
    
    def test_gimm_config(self):
        """Test GIMM model config."""
        model = GimmModel(config=GimmConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.INTERPOLATE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
    
    def test_span_config(self):
        """Test SPAN model config."""
        model = SPANModel(config=SPANConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.UPSCALE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["scale"], 4)
    
    def test_animesr_config(self):
        """Test AnimeSR model config."""
        model = AnimeSRModel(config=AnimeSRConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.UPSCALE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["scale"], 4)
    
    def test_tspan_config(self):
        """Test TSPAN model config."""
        model = TSPANModel(config=TSPANConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.UPSCALE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
        self.assertEqual(cfg["scale"], 2)
    
    def test_fbcnn_config(self):
        """Test FBCNN model config."""
        model = FBCNNModel(config=FBCNNConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.RESTORATION)
        self.assertEqual(cfg["format"], ModelFormat.PT)
    
    def test_nafnet_config(self):
        """Test NAFNet model config."""
        model = NAFNetModel(config=NAFNetConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.RESTORATION)
        self.assertEqual(cfg["format"], ModelFormat.PT)
    
    def test_dncnn_config(self):
        """Test DnCNN model config."""
        model = DnCNNModel(config=DnCNNConfig())
        cfg = model.get_config()
        
        self.assertEqual(cfg["task"], ModelTask.DENOISE)
        self.assertEqual(cfg["format"], ModelFormat.PT)
    
    def test_gmfss_config(self):
        """Test GMFSS model config."""
        model = GmfssModel(config=GmfssConfig())
        info = model.get_info()
        
        self.assertEqual(info["name"], "GMFSS")
        self.assertEqual(info["task"], "interpolate")
        self.assertIn("pytorch", info["supported_backends"])
        self.assertIn("onnx", info["supported_backends"])
        self.assertIn("ncnn", info["supported_backends"])
    
    def test_scene_detect_config(self):
        """Test scene detect model config."""
        model = MaxViTSceneDetectModel(config=MaxViTConfig())
        info = model.get_info()
        
        self.assertEqual(info["name"], "scene_detect")
        self.assertEqual(info["task"], "scene_detect")
        self.assertIn("pytorch", info["supported_backends"])
        self.assertIn("onnx", info["supported_backends"])
        self.assertIn("ncnn", info["supported_backends"])


if __name__ == "__main__":
    unittest.main()

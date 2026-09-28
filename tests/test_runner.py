"""
Test runner for backend-agnostic architecture tests.
"""

import unittest
import sys
import os

# Add test directory to path
sys.path.insert(0, os.path.dirname(__file__))


def load_tests(loader, standard_tests, pattern):
    """Load all test modules."""
    test_suite = unittest.TestSuite()
    
    # Unit tests
    from test_rife_model import TestRifeConfig, TestRifeHead, TestRifeResConv, TestRifeIFBlock, TestRifeIFNet, TestRifeModel, TestRifeVersion
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeConfig))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeHead))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeResConv))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeIFBlock))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeIFNet))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeModel))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRifeVersion))
    
    from test_pytorch_backend import TestPyTorchBackendLoader, TestPyTorchInterpolateRunner, TestPyTorchUpscaleRunner
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPyTorchBackendLoader))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPyTorchInterpolateRunner))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPyTorchUpscaleRunner))
    
    from test_onnx_backend import TestONNXBackendLoader, TestONNXInterpolateRunner, TestONNXUpscaleRunner
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestONNXBackendLoader))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestONNXInterpolateRunner))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestONNXUpscaleRunner))
    
    from test_converters import TestTorchToOnnxConverter, TestOnnxToNcnnConverter, TestConvertFunctions
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTorchToOnnxConverter))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOnnxToNcnnConverter))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConvertFunctions))
    
    # Integration tests
    from test_backend_pipeline import TestEndToEndPipeline, TestModelConversionPipeline, TestConfigLoading
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndPipeline))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelConversionPipeline))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfigLoading))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_tests(None, None, None))

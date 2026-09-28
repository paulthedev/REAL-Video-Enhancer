"""
NCNN backend package.
"""

from backend.backends.ncnn.loader import NCNNBackendLoader
from backend.backends.ncnn.runner import NCNNBackendRunner

__all__ = ["NCNNBackendLoader", "NCNNBackendRunner"]

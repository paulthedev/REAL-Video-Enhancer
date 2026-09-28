"""
NCNN backend package.
"""

from apps.backend.backends.ncnn.loader import NCNNBackendLoader
from apps.backend.backends.ncnn.runner import NCNNBackendRunner

__all__ = ["NCNNBackendLoader", "NCNNBackendRunner"]

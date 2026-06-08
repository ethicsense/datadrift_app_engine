"""
YOLO CAM (Class Activation Mapping) library for YOLO models.
"""

from .eigen_cam import EigenCAM
from .base_cam import BaseCAM

__all__ = ['EigenCAM', 'BaseCAM'] 
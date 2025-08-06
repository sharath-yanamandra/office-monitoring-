#!/usr/bin/env python3
"""
src/__init__.py
Video Monitoring System Package

Main package initialization for the video monitoring system.
"""

__version__ = "1.0.0"
__author__ = "Video Monitoring Team"
__description__ = "Flexible Video Monitoring System with AI Detection"

# Import main components for easy access
from .config import Config, DetectionModels, EventTypes
from .database import Database
from .model_manager import ModelManager
from .camera_manager import CameraManager
from .kalman_tracker import MultiObjectTracker
from .camera_model_base import CameraModelBase
from .db_writer import DatabaseWriter
from .storage_handler import StorageHandler
from .logger import setup_logger

__all__ = [
    'Config',
    'DetectionModels', 
    'EventTypes',
    'Database',
    'ModelManager',
    'CameraManager',
    'MultiObjectTracker',
    'CameraModelBase',
    'DatabaseWriter',
    'StorageHandler',
    'setup_logger'
]
#!/usr/bin/env python3
"""
detection_scripts/__init__.py
Detection Scripts Package

This package contains all standalone detection modules for the video monitoring system.
Each script can be used independently or combined with others.
"""

from .tailgating_detection import TailgatingDetector
from .people_counting import PeopleCounter
from .ppe_detection import PPEDetector
from .unauthorized_access_detection import UnauthorizedAccessDetector
from .intrusion_detection import IntrusionDetector
from .loitering_detection import LoiteringDetector

# Export all detector classes
__all__ = [
    'TailgatingDetector',
    'PeopleCounter', 
    'PPEDetector',
    'UnauthorizedAccessDetector',
    'IntrusionDetector',
    'LoiteringDetector'
]

# Detection model mapping for easy access
DETECTOR_CLASSES = {
    'tailgating': TailgatingDetector,
    'people_counting': PeopleCounter,
    'ppe_detection': PPEDetector,
    'unauthorized_access': UnauthorizedAccessDetector,
    'intrusion_detection': IntrusionDetector,
    'loitering_detection': LoiteringDetector
}

def get_detector_class(model_name: str):
    """Get detector class by model name"""
    return DETECTOR_CLASSES.get(model_name)
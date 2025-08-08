#!/usr/bin/env python3
"""
Datacenter Camera Models Module

This module contains all camera model implementations for datacenter monitoring use cases.
Each camera model inherits from CameraModelBase and implements specific detection logic.

Available Camera Models:
1. PeopleCountingMonitor - Count people in defined zones
2. PPEDetector - Detect PPE violations (hard hats, vests, etc.)
3. TailgatingZoneMonitor - Detect tailgating at entry points
4. IntrusionZoneMonitor - Detect unauthorized zone access
5. LoiteringZoneMonitor - Detect loitering behavior
6. UnauthorizedZoneMonitor - General unauthorized access detection (from SIB)
"""

# Import base classes
from .camera_model_base import CameraModelBase
from .kalman_track import Sort

# Import datacenter-specific use case monitors
from .people_count_monitoring import PeopleCountingMonitor
from .ppe_kit_monitoring import PPEDetector
from .tailgating_zone_monitoring import TailgatingZoneMonitor
from .intrusion_zone_monitoring import IntrusionZoneMonitor
from .loitering_zone_monitoring import LoiteringZoneMonitor


# Import multi-camera coordination (if available)
try:
    from .multi_camera_coordinator import MultiCameraCoordinator
    MULTI_CAMERA_AVAILABLE = True
except ImportError:
    MultiCameraCoordinator = None
    MULTI_CAMERA_AVAILABLE = False

# Define camera model mapping for datacenter use cases
DATACENTER_CAMERA_MODEL_MAPPING = {
    # Datacenter-specific use cases
    'people_counting': PeopleCountingMonitor,
    'ppe_detection': PPEDetector,
    'tailgating': TailgatingZoneMonitor,
    'intrusion': IntrusionZoneMonitor,
    'loitering': LoiteringZoneMonitor,
    

    # Aliases for backward compatibility
    'people_count_monitoring': PeopleCountingMonitor,
    'ppe_kit_monitoring': PPEDetector,
    'tailgating_zone_monitoring': TailgatingZoneMonitor,
    'intrusion_zone_monitoring': IntrusionZoneMonitor,
    'loitering_zone_monitoring': LoiteringZoneMonitor,
}

# Datacenter camera types enum-like mapping
class DatacenterCameraTypes:
    """Enum-like class for datacenter camera types"""
    PEOPLE_COUNTING = 'people_counting'
    PPE_DETECTION = 'ppe_detection'
    TAILGATING = 'tailgating'
    INTRUSION = 'intrusion'
    LOITERING = 'loitering'
    UNAUTHORIZED_ZONE = 'unauthorized_zone'
    
    @classmethod
    def get_all_types(cls):
        """Get all available camera types"""
        return [
            cls.PEOPLE_COUNTING,
            cls.PPE_DETECTION,
            cls.TAILGATING,
            cls.INTRUSION,
            cls.LOITERING,
            cls.UNAUTHORIZED_ZONE
        ]
    
    @classmethod
    def get_model_class(cls, camera_type: str):
        """Get model class for camera type"""
        return DATACENTER_CAMERA_MODEL_MAPPING.get(camera_type)

# Export all available camera models
__all__ = [
    # Base classes
    'CameraModelBase',
    'Sort',
    
    # Datacenter use case monitors
    'PeopleCountingMonitor',
    'PPEDetector',
    'TailgatingZoneMonitor', 
    'IntrusionZoneMonitor',
    'LoiteringZoneMonitor',
    
    # SIB compatibility
    'UnauthorizedZoneMonitor',
    
    # Multi-camera coordination
    'MultiCameraCoordinator',
    
    # Mappings and types
    'DATACENTER_CAMERA_MODEL_MAPPING',
    'DatacenterCameraTypes',
    
    # Flags
    'MULTI_CAMERA_AVAILABLE'
]

# Validate all imports on module load
def _validate_imports():
    """Validate that all camera models are properly imported"""
    missing_models = []
    
    for model_name, model_class in DATACENTER_CAMERA_MODEL_MAPPING.items():
        if model_class is None:
            missing_models.append(model_name)
    
    if missing_models:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Missing camera models: {missing_models}")
        return False
    
    return True

# Run validation
_validate_imports()
#!/usr/bin/env python3
"""
Configuration for Datacenter Monitoring System

This module contains all configuration settings for the datacenter monitoring application,
including database settings, camera parameters, AI model configurations, and system limits.
"""

import os
from typing import List, Dict, Any

class Config:
    """Configuration class for datacenter monitoring system"""
    
    # Database Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', '34.93.87.255')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'insighteye')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'insighteye0411')
    MYSQL_ROOT_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD', 'insighteye0411')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'dc_test')
    
    # Connection pooling
    MYSQL_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '32'))
    MYSQL_MAX_OVERFLOW = int(os.getenv('MYSQL_MAX_OVERFLOW', '20'))
    MYSQL_POOL_TIMEOUT = int(os.getenv('MYSQL_POOL_TIMEOUT', '30'))
    MYSQL_POOL_RECYCLE = int(os.getenv('MYSQL_POOL_RECYCLE', '3600'))
    
    # AI Model Configuration
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/yolov11l.pt')
    POSE_ESTIMATION_MODEL_PATH = os.getenv('POSE_ESTIMATION_MODEL_PATH', 'models/yolov11l-pose.pt')
    PPE_DETECTION_MODEL_PATH = os.getenv('PPE_DETECTION_MODEL_PATH', 'models/ppe_detection.pt')
    
    # Model settings
    DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('DETECTION_CONFIDENCE_THRESHOLD', '0.5'))
    NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', '0.4'))
    MODEL_INPUT_SIZE = int(os.getenv('MODEL_INPUT_SIZE', '640'))
    
    # GPU Configuration
    CUDA_DEVICE_ID = int(os.getenv('CUDA_DEVICE_ID', '0'))
    GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
    ENABLE_MIXED_PRECISION = os.getenv('ENABLE_MIXED_PRECISION', 'true').lower() == 'true'
    
    # Camera Configuration
    READER_FPS_LIMIT = int(os.getenv('READER_FPS_LIMIT', '15'))
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', '50'))
    FRAME_SKIP_THRESHOLD = int(os.getenv('FRAME_SKIP_THRESHOLD', '10'))
    
    # Activity level based FPS settings
    ACTIVITY_LEVEL_HIGH = int(os.getenv('ACTIVITY_LEVEL_HIGH', '15'))    # For high security areas
    ACTIVITY_LEVEL_MEDIUM = int(os.getenv('ACTIVITY_LEVEL_MEDIUM', '10')) # For moderate monitoring
    ACTIVITY_LEVEL_LOW = int(os.getenv('ACTIVITY_LEVEL_LOW', '5'))        # For background monitoring
    
    # Reconnection settings
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '5'))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))
    CONNECTION_TIMEOUT = int(os.getenv('CONNECTION_TIMEOUT', '30'))
    
    # Batch Processing Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
    BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', '0.1'))
    MAX_PARALLEL_CAMERAS = int(os.getenv('MAX_PARALLEL_CAMERAS', '8'))
    
    # Event Detection Configuration
    TRACKING_THRESHOLD = float(os.getenv('TRACKING_THRESHOLD', '0.3'))
    EVENT_COOLDOWN = int(os.getenv('EVENT_COOLDOWN', '30'))  # seconds
    
    # Storage Configuration
    FRAMES_OUTPUT_DIR = os.getenv('FRAMES_OUTPUT_DIR', 'frames')
    VIDEOS_OUTPUT_DIR = os.getenv('VIDEOS_OUTPUT_DIR', 'videos')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    
    # Video Recording Settings
    AUTO_RECORDING_ENABLED = os.getenv('AUTO_RECORDING_ENABLED', 'true').lower() == 'true'
    VIDEO_DURATION = int(os.getenv('VIDEO_DURATION', '30'))  # seconds
    VIDEO_BUFFER_SIZE = int(os.getenv('VIDEO_BUFFER_SIZE', '300'))  # frames
    MEDIA_PREFERENCE = os.getenv('MEDIA_PREFERENCE', 'both')  # 'frames', 'videos', 'both'
    
    # Cloud Storage (GCP)
    GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME', 'dc_bucket_test')
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'datacenter-monitoring')
    GCP_CREDENTIALS_PATH = os.getenv('GCP_CREDENTIALS_PATH', 'gcp-credentials.json')
    
    # Monitoring and Health Checks
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))  # seconds
    PERFORMANCE_LOG_INTERVAL = int(os.getenv('PERFORMANCE_LOG_INTERVAL', '300'))  # seconds
    MOTION_DETECTION_ENABLED = os.getenv('MOTION_DETECTION_ENABLED', 'false').lower() == 'true'
    
    # Database Writer Configuration
    DB_WRITER_BATCH_SIZE = int(os.getenv('DB_WRITER_BATCH_SIZE', '100'))
    DB_WRITER_FLUSH_INTERVAL = int(os.getenv('DB_WRITER_FLUSH_INTERVAL', '5'))  # seconds
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_MAX_SIZE = int(os.getenv('LOG_FILE_MAX_SIZE', '10485760'))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv('LOG_FILE_BACKUP_COUNT', '5'))
    
    # Security Settings
    API_SECRET_KEY = os.getenv('API_SECRET_KEY', 'datacenter-monitoring-secret-key')
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    
    # Performance Optimization
    ENABLE_FRAME_CACHING = os.getenv('ENABLE_FRAME_CACHING', 'true').lower() == 'true'
    CACHE_SIZE_MB = int(os.getenv('CACHE_SIZE_MB', '500'))
    
    # ----------- NEW: LOCAL TESTING CONFIGURATION -----------
    
    # Local testing mode settings
    SINGLE_CAMERA_MODE = False  # Set to True for local testing
    LOCAL_CAMERA_ID = "local_test_camera"
    LOCAL_RTSP_URL = "rtsp://admin:password@192.168.29.213:554/ch0_0.264"  # Update with your camera URL
    
    # Multi-use case settings for local testing
    LOCAL_USE_CASES = [
        'people_counting',
        'ppe_detection', 
        'tailgating',
        'intrusion',
        'loitering'
    ]
    
    # Performance optimization for multi-use case
    SHARED_MODEL_DETECTION = True  # Use same YOLO detection for all use cases
    MAX_USE_CASES_PER_CAMERA = 5
    
    # Local testing database settings (will be overridden with GCP settings)
    LOCAL_TEST_DB_HOST = "localhost"
    LOCAL_TEST_DB_USER = "test_user"
    LOCAL_TEST_DB_PASSWORD = "test_password"
    LOCAL_TEST_DB_NAME = "datacenter_local_test"
    
    # Local testing paths
    LOCAL_FRAMES_DIR = "local_testing/outputs/frames"
    LOCAL_VIDEOS_DIR = "local_testing/outputs/videos"
    LOCAL_LOGS_DIR = "local_testing/logs"
    LOCAL_MODELS_DIR = "local_testing/models"
    
    # Debug settings for local testing
    DEBUG_MODE = False
    VERBOSE_LOGGING = False
    SAVE_DEBUG_FRAMES = False
    
    # Local testing performance settings
    LOCAL_BATCH_SIZE = 1  # Smaller batches for single camera
    LOCAL_FPS_LIMIT = 10  # Lower FPS for testing
    LOCAL_QUEUE_SIZE = 10  # Smaller queue for testing


class DatacenterEventTypes:
    """Event types for datacenter monitoring"""
    
    # People counting events
    PEOPLE_COUNT_EXCEEDED = "people_count_exceeded"
    PEOPLE_COUNT_BELOW_THRESHOLD = "people_count_below_threshold"
    
    # PPE detection events
    PPE_VIOLATION = "ppe_violation"
    PPE_COMPLIANCE = "ppe_compliance"
    MISSING_HARD_HAT = "missing_hard_hat"
    MISSING_SAFETY_VEST = "missing_safety_vest"
    MISSING_SAFETY_GLASSES = "missing_safety_glasses"
    
    # Tailgating events
    TAILGATING_DETECTED = "tailgating_detected"
    UNAUTHORIZED_ENTRY = "unauthorized_entry"
    ENTRY_TIMEOUT = "entry_timeout"
    
    # Intrusion events
    ZONE_INTRUSION = "zone_intrusion"
    RESTRICTED_AREA_ACCESS = "restricted_area_access"
    PERIMETER_BREACH = "perimeter_breach"
    
    # Loitering events
    LOITERING_DETECTED = "loitering_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    EXTENDED_PRESENCE = "extended_presence"
    
    # System events
    CAMERA_OFFLINE = "camera_offline"
    CAMERA_ONLINE = "camera_online"
    SYSTEM_ERROR = "system_error"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"


class DatacenterCameraTypes:
    """Camera types for datacenter monitoring"""
    
    # Use case based types
    PEOPLE_COUNTING = "people_counting"
    PPE_DETECTION = "ppe_detection"
    TAILGATING = "tailgating"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    UNAUTHORIZED_ZONE = "unauthorized_zone"
    
    # Location based types (legacy)
    DC_ENTRY_MONITOR = "dc_entry_monitor"
    DC_SERVER_ROOM = "dc_server_room"
    DC_CORRIDOR = "dc_corridor"
    DC_PERIMETER = "dc_perimeter"
    DC_CRITICAL_ZONE = "dc_critical_zone"
    DC_COMMON_AREA = "dc_common_area"
    
    @classmethod
    def get_all_types(cls):
        """Get all available camera types"""
        return [
            cls.PEOPLE_COUNTING,
            cls.PPE_DETECTION,
            cls.TAILGATING,
            cls.INTRUSION,
            cls.LOITERING,
            cls.UNAUTHORIZED_ZONE,
            cls.DC_ENTRY_MONITOR,
            cls.DC_SERVER_ROOM,
            cls.DC_CORRIDOR,
            cls.DC_PERIMETER,
            cls.DC_CRITICAL_ZONE,
            cls.DC_COMMON_AREA
        ]


# FIXED: Camera model mapping using direct string values instead of .value
DATACENTER_CAMERA_MODEL_MAPPING = {
    "people_counting": "PeopleCountingMonitor",
    "ppe_detection": "PPEDetector",
    "tailgating": "TailgatingZoneMonitor",
    "intrusion": "IntrusionZoneMonitor",
    "loitering": "LoiteringZoneMonitor",
    "unauthorized_zone": "DatacenterUnauthorizedZoneMonitor",
}

# Zone types for datacenter monitoring
DATACENTER_ZONE_TYPES = [
    'entry_zone',
    'server_zone',
    'restricted_zone',
    'common_zone',
    'perimeter_zone',
    'critical_zone',
    'counting_zone',
    'loitering_zone',
    'unauthorized_zone',
    'ppe_zone'
]

# Severity levels
DATACENTER_SEVERITY_LEVELS = [
    'low',
    'medium', 
    'high',
    'critical'
]

# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required database settings
    if not Config.MYSQL_USER:
        errors.append("MYSQL_USER is required")
    if not Config.MYSQL_PASSWORD:
        errors.append("MYSQL_PASSWORD is required")
    if not Config.MYSQL_DATABASE:
        errors.append("MYSQL_DATABASE is required")
    
    # Check model paths
    if not os.path.exists(Config.DETECTION_MODEL_PATH):
        errors.append(f"Detection model not found: {Config.DETECTION_MODEL_PATH}")
    
    # Check batch size
    if Config.BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be greater than 0")
    
    # Check FPS settings
    if Config.ACTIVITY_LEVEL_HIGH <= 0:
        errors.append("ACTIVITY_LEVEL_HIGH must be greater than 0")
    
    return errors


# LOCAL TESTING CONFIGURATION CLASS
class LocalTestConfig(Config):
    """Configuration override for local testing with single camera and multiple use cases"""
    
    # Override for local testing
    SINGLE_CAMERA_MODE = True
    LOCAL_CAMERA_ID = "office_cam_001"
    LOCAL_RTSP_URL = "rtsp://admin:password@192.168.29.212:554/ch0_0.264"  # UPDATE WITH YOUR CAMERA
    
    # All use cases for single camera testing
    LOCAL_USE_CASES = [
        'people_counting',
        'ppe_detection', 
        'tailgating',
        'intrusion',
        'loitering'
    ]
    
    # Database settings
    MYSQL_HOST = "34.93.87.255"  
    MYSQL_USER = "insighteye"  
    MYSQL_PASSWORD = "insighteye0411" 
    MYSQL_DATABASE = "dc_test"
    
    # Performance settings optimized for local testing
    BATCH_SIZE = 1  # Single camera
    MAX_PARALLEL_CAMERAS = 1
    ACTIVITY_LEVEL_HIGH = 15  # Higher FPS for testing
    ACTIVITY_LEVEL_MEDIUM = 10
    ACTIVITY_LEVEL_LOW = 5
    
    # Local storage paths
    FRAMES_OUTPUT_DIR = "local_testing/outputs/frames"
    VIDEOS_OUTPUT_DIR = "local_testing/outputs/videos"
    LOGS_DIR = "local_testing/logs"
    
    # Model paths for local testing
    DETECTION_MODEL_PATH = "C:/Users/DELL/Desktop/Datacenter full integration/local_testing/models/yolo11l.pt"
    PPE_MODEL_PATH = "C:/Users/DELL/Desktop/Datacenter full integration/local_testing/models/ppe_detection.pt"
    
    # Debug settings
    DEBUG_MODE = True
    VERBOSE_LOGGING = True
    SAVE_DEBUG_FRAMES = True
    
    # Optimizations for single camera multi-use case
    SHARED_MODEL_DETECTION = True
    MAX_USE_CASES_PER_CAMERA = 5


# Export commonly used configurations
__all__ = [
    'DatacenterConfig',
    'LocalTestConfig',
    'DatacenterEventTypes', 
    'DatacenterCameraTypes',
    'DATACENTER_CAMERA_MODEL_MAPPING',
    'DATACENTER_ZONE_TYPES',
    'DATACENTER_SEVERITY_LEVELS',
    'validate_config'
]

#!/usr/bin/env python3
"""
Datacenter Monitoring System - Configuration Management

This module handles:
1. System configuration and environment variables
2. Datacenter-specific monitoring parameters
3. Detection model settings
4. Alert and notification configurations
5. Event type definitions for datacenter use cases
6. Motion detection and video buffer settings (from SIB)
"""

import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class DatacenterConfig:
    """Main configuration class for datacenter monitoring system"""
    
    # GCP Storage settings
    GCP_PROJECT = os.getenv('GCP_PROJECT')
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    STORAGE_BASE_URL = f"https://storage.googleapis.com/{BUCKET_NAME}" if os.getenv("GCP_BUCKET_NAME") else None
    
    # Detection Model Paths
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/yolov11l.pt')
    PPE_DETECTION_MODEL_PATH = os.getenv('PPE_DETECTION_MODEL_PATH', 'models/ppe_detection.pt')
    POSE_ENABLED = os.getenv('POSE_ENABLED', 'False').lower() == 'true'
    POSE_ESTIMATION_MODEL_PATH = os.getenv('POSE_ESTIMATION_MODEL_PATH', 'models/yolov11l-pose.pt')

    # MySQL Database Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'datacenter_monitoring')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))

    # Database Pool Configuration
    DB_POOL_NAME = 'datacenter_pool'
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 32))

    # Processing Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT', 3600))  # 1 hour
    FRAMES_OUTPUT_DIR = os.getenv('FRAMES_OUTPUT_DIR', 'frames')
    
    # Detection Confidence Thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))
    PERSON_DETECTION_CONFIDENCE = float(os.getenv('PERSON_DETECTION_CONFIDENCE', 0.5))
    GENERAL_DETECTION_CONFIDENCE = float(os.getenv('GENERAL_DETECTION_CONFIDENCE', 0.6))

    # Camera and Tracking Configuration
    TRACKING_THRESHOLD = int(os.getenv('TRACKING_THRESHOLD', 3))
    MAX_AGE = int(os.getenv('MAX_AGE', 30))
    MIN_MA = int(os.getenv('MIN_MA', 3))
    
    # Media Storage Configuration
    MEDIA_PREFERENCE = os.getenv('MEDIA_PREFERENCE', 'image')  # "image" or "video"
    EVENT_COOLDOWN = int(os.getenv('EVENT_COOLDOWN', 120))  # Seconds between similar events
    AUTO_RECORDING_ENABLED = os.getenv('AUTO_RECORDING_ENABLED', 'False').lower() == 'true'
    
    # Batch Processing Configuration
    NUMBER_OF_CAMERAS = int(os.getenv('NUMBER_OF_CAMERAS', 4))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', NUMBER_OF_CAMERAS))
    BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', 0.5))
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 200))
    
    # Multi-Camera Configuration
    NUM_CAMERA_READERS = NUMBER_OF_CAMERAS  # Number of camera reader threads
    READER_FPS_LIMIT = int(os.getenv('READER_FPS_LIMIT', 4))
    NUM_BATCH_THREADS = int(os.getenv('NUM_BATCH_THREADS', 2))
    
    # Activity Level FPS Settings (from SIB)
    ACTIVITY_LEVEL_HIGH = int(os.getenv('ACTIVITY_LEVEL_HIGH', 10))    # High security zones
    ACTIVITY_LEVEL_MEDIUM = int(os.getenv('ACTIVITY_LEVEL_MEDIUM', 4)) # Moderate monitoring
    ACTIVITY_LEVEL_LOW = int(os.getenv('ACTIVITY_LEVEL_LOW', 2))       # Low priority areas
    FPS_ACTIVITY_LEVEL = os.getenv('FPS_ACTIVITY_LEVEL', 'medium')
    
    # Parallel Processing Configuration
    MAX_PARALLEL_CAMERAS = int(os.getenv('MAX_PARALLEL_CAMERAS', 4))
    
    # Database Writer Configuration
    DB_WRITER_BATCH_SIZE = int(os.getenv('DB_WRITER_BATCH_SIZE', 5))
    
    # Video Recording Configuration (from SIB)
    VIDEO_BUFFER_SIZE = int(os.getenv('VIDEO_BUFFER_SIZE', 30))
    VIDEO_FPS = int(os.getenv('VIDEO_FPS', 4))
    PRE_EVENT_SECONDS = int(os.getenv('PRE_EVENT_SECONDS', 3))
    POST_EVENT_SECONDS = int(os.getenv('POST_EVENT_SECONDS', 7))
    VIDEO_EXTENSION = os.getenv('VIDEO_EXTENSION', 'mp4')
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'avc1')
    
    # Video buffer settings - past + future footage (from SIB)
    VIDEO_BUFFER_PAST_SECONDS = int(os.getenv('VIDEO_BUFFER_PAST_SECONDS', 10))  # Keep 10 seconds of past footage
    VIDEO_BUFFER_FUTURE_SECONDS = int(os.getenv('VIDEO_BUFFER_FUTURE_SECONDS', 50))  # Record 50 seconds after event
    VIDEO_BUFFER_TOTAL_SECONDS = VIDEO_BUFFER_PAST_SECONDS + VIDEO_BUFFER_FUTURE_SECONDS  # Total 60 seconds
    ORIGINAL_FPS = int(os.getenv('ORIGINAL_FPS', 25))  # Assumed original camera FPS
    VIDEO_RESOLUTION = (1280, 720)  # Target resolution for video buffer (width, height)
    
    # Motion Detection Configuration (from SIB)
    MOTION_DETECTION_ENABLED = os.getenv('MOTION_DETECTION_ENABLED', 'True').lower() == 'true'
    MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', 25))  # Background subtractor sensitivity
    MOTION_AREA_THRESHOLD = float(os.getenv('MOTION_AREA_THRESHOLD', 0.10))  # 10% of frame area
    MOTION_IDLE_TIMEOUT = int(os.getenv('MOTION_IDLE_TIMEOUT', 120))  # Stop GPU processing after 2 minutes
    MOTION_WARMUP_FRAMES = int(os.getenv('MOTION_WARMUP_FRAMES', 16))  # Frames to skip for background model
    
    # Configuration Management
    CAMERA_CONFIG_DIR = os.getenv('CAMERA_CONFIG_DIR', 'configs/cameras')
    USE_CONFIG_FILES = os.getenv('USE_CONFIG_FILES', 'False').lower() == 'true'
    
    # DATACENTER-SPECIFIC CONFIGURATIONS
    
    # PPE Detection Settings
    PPE_DETECTION_ENABLED = os.getenv('PPE_DETECTION_ENABLED', 'True').lower() == 'true'
    REQUIRED_PPE_CLASSES = ['hard_hat', 'safety_vest', 'safety_glasses']
    PPE_CONFIDENCE_THRESHOLD = float(os.getenv('PPE_CONFIDENCE_THRESHOLD', 0.7))
    
    # Tailgating Detection
    TAILGATING_TIME_WINDOW = int(os.getenv('TAILGATING_TIME_WINDOW', 10))  # seconds
    MAX_PEOPLE_PER_ENTRY = int(os.getenv('MAX_PEOPLE_PER_ENTRY', 1))
    ENTRY_ZONE_BUFFER = float(os.getenv('ENTRY_ZONE_BUFFER', 2.0))
    
    # Loitering Detection
    LOITERING_THRESHOLD = int(os.getenv('LOITERING_THRESHOLD', 300))  # 5 minutes in seconds
    MOVEMENT_THRESHOLD = float(os.getenv('MOVEMENT_THRESHOLD', 1.0))
    LOITERING_CHECK_INTERVAL = int(os.getenv('LOITERING_CHECK_INTERVAL', 30))
    
    # Intrusion Detection
    INTRUSION_SENSITIVITY = os.getenv('INTRUSION_SENSITIVITY', 'high')
    RESTRICTED_ZONE_BUFFER = float(os.getenv('RESTRICTED_ZONE_BUFFER', 0.5))
    INTRUSION_CONFIDENCE_THRESHOLD = float(os.getenv('INTRUSION_CONFIDENCE_THRESHOLD', 0.8))
    INTRUSION_THRESHOLD = int(os.getenv('INTRUSION_THRESHOLD', 1))  # Number of people to trigger
    
    # Unauthorized Zone Detection
    UNAUTHORIZED_THRESHOLD = int(os.getenv('UNAUTHORIZED_THRESHOLD', 1))  # Number of people to trigger
    
    # People Counting
    PEOPLE_COUNT_THRESHOLD = int(os.getenv('PEOPLE_COUNT_THRESHOLD', 10))  # Max people before alert
    
    # Camera Tamper Detection
    TAMPER_DETECTION_ENABLED = os.getenv('TAMPER_DETECTION_ENABLED', 'True').lower() == 'true'
    FRAME_DIFF_THRESHOLD = float(os.getenv('FRAME_DIFF_THRESHOLD', 0.8))
    OBSTRUCTION_THRESHOLD = float(os.getenv('OBSTRUCTION_THRESHOLD', 0.9))
    TAMPER_CHECK_INTERVAL = int(os.getenv('TAMPER_CHECK_INTERVAL', 60))
    
    # Health Check and Monitoring
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', 60))
    ENABLE_HEALTH_ENDPOINT = os.getenv('ENABLE_HEALTH_ENDPOINT', 'True').lower() == 'true'
    ENABLE_METRICS_COLLECTION = os.getenv('ENABLE_METRICS_COLLECTION', 'True').lower() == 'true'
    METRICS_EXPORT_INTERVAL = int(os.getenv('METRICS_EXPORT_INTERVAL', 300))
    
    # System Monitoring
    MONITOR_CPU = os.getenv('MONITOR_CPU', 'True').lower() == 'true'
    MONITOR_MEMORY = os.getenv('MONITOR_MEMORY', 'True').lower() == 'true'
    MONITOR_GPU = os.getenv('MONITOR_GPU', 'True').lower() == 'true'
    MONITOR_DISK = os.getenv('MONITOR_DISK', 'True').lower() == 'true'
    
    # Data Retention Policies
    EVENT_RETENTION_DAYS = int(os.getenv('EVENT_RETENTION_DAYS', 365))
    LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', 90))
    VIDEO_RETENTION_DAYS = int(os.getenv('VIDEO_RETENTION_DAYS', 30))
    IMAGE_RETENTION_DAYS = int(os.getenv('IMAGE_RETENTION_DAYS', 60))
    
    # Privacy and Compliance Settings
    ANONYMIZE_LOGS = os.getenv('ANONYMIZE_LOGS', 'False').lower() == 'true'
    ENCRYPT_SENSITIVE_DATA = os.getenv('ENCRYPT_SENSITIVE_DATA', 'True').lower() == 'true'
    GDPR_COMPLIANCE = os.getenv('GDPR_COMPLIANCE', 'False').lower() == 'true'
    HIPAA_COMPLIANCE = os.getenv('HIPAA_COMPLIANCE', 'False').lower() == 'true'
    SOC2_COMPLIANCE = os.getenv('SOC2_COMPLIANCE', 'True').lower() == 'true'
    
    # Deployment Settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    VERSION = os.getenv('VERSION', '1.0.0')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    
    # Create method to get DB pool config dict
    @classmethod
    def get_db_pool_config(cls):
        return {
            'pool_name': cls.DB_POOL_NAME,
            'pool_size': cls.DB_POOL_SIZE,
            'host': cls.MYSQL_HOST,
            'user': cls.MYSQL_USER,
            'password': cls.MYSQL_PASSWORD,
            'database': cls.MYSQL_DATABASE,
            'port': cls.MYSQL_PORT
        }


class DatacenterEventTypes(Enum):
    """Enumeration of datacenter event types"""
    TAILGATING = "tailgating"
    INTRUSION = "intrusion"
    PPE_VIOLATION = "ppe_violation"
    LOITERING = "loitering"
    PEOPLE_COUNTING = "people_counting"
    CAMERA_TAMPER = "camera_tamper"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class DatacenterCameraTypes(Enum):
    """Enumeration of datacenter camera types"""
    ENTRY_MONITOR = "dc_entry_monitor"
    SERVER_ROOM = "dc_server_room"
    CORRIDOR = "dc_corridor"
    PERIMETER = "dc_perimeter"
    CRITICAL_ZONE = "dc_critical_zone"
    COMMON_AREA = "dc_common_area"
    
    # Your friend's use case types
    PEOPLE_COUNTING = "people_counting"
    PPE_DETECTION = "ppe_detection"
    TAILGATING = "tailgating"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    UNAUTHORIZED_ZONE = "unauthorized_zone"


# Camera model mapping for your friend's use cases
DATACENTER_CAMERA_MODEL_MAPPING = {
    DatacenterCameraTypes.PEOPLE_COUNTING.value: "PeopleCountingMonitor",
    DatacenterCameraTypes.PPE_DETECTION.value: "PPEDetector",
    DatacenterCameraTypes.TAILGATING.value: "TailgatingZoneMonitor",
    DatacenterCameraTypes.INTRUSION.value: "IntrusionZoneMonitor",
    DatacenterCameraTypes.LOITERING.value: "LoiteringZoneMonitor",
    DatacenterCameraTypes.UNAUTHORIZED_ZONE.value: "DatacenterUnauthorizedZoneMonitor",
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
    if not DatacenterConfig.MYSQL_USER:
        errors.append("MYSQL_USER is required")
    if not DatacenterConfig.MYSQL_PASSWORD:
        errors.append("MYSQL_PASSWORD is required")
    if not DatacenterConfig.MYSQL_DATABASE:
        errors.append("MYSQL_DATABASE is required")
    
    # Check model paths
    if not os.path.exists(DatacenterConfig.DETECTION_MODEL_PATH):
        errors.append(f"Detection model not found: {DatacenterConfig.DETECTION_MODEL_PATH}")
    
    # Check batch size
    if DatacenterConfig.BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be greater than 0")
    
    # Check FPS settings
    if DatacenterConfig.ACTIVITY_LEVEL_HIGH <= 0:
        errors.append("ACTIVITY_LEVEL_HIGH must be greater than 0")
    
    return errors


# Export commonly used configurations
__all__ = [
    'DatacenterConfig',
    'DatacenterEventTypes', 
    'DatacenterCameraTypes',
    'DATACENTER_CAMERA_MODEL_MAPPING',
    'DATACENTER_ZONE_TYPES',
    'DATACENTER_SEVERITY_LEVELS',
    'validate_config'
]
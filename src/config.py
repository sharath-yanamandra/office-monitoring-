#!/usr/bin/env python3
"""
config.py
Flexible Video Monitoring System - Configuration Management

This module handles:
1. System configuration and environment variables
2. Detection model settings
3. Database configurations
4. Available detection models mapping
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Main configuration class for video monitoring system"""
    
    # Database Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'video_monitoring')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 16))

    # Model Paths
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/yolov11l.pt')
    PPE_DETECTION_MODEL_PATH = os.getenv('PPE_DETECTION_MODEL_PATH', 'models/ppe_detection.pt')
    POSE_ESTIMATION_MODEL_PATH = os.getenv('POSE_ESTIMATION_MODEL_PATH', 'models/yolov11l-pose.pt')

    # Processing Configuration
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
    BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', 0.5))
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 100))
    
    # Detection Thresholds
    PERSON_DETECTION_CONFIDENCE = float(os.getenv('PERSON_DETECTION_CONFIDENCE', 0.5))
    PPE_CONFIDENCE_THRESHOLD = float(os.getenv('PPE_CONFIDENCE_THRESHOLD', 0.7))
    GENERAL_DETECTION_CONFIDENCE = float(os.getenv('GENERAL_DETECTION_CONFIDENCE', 0.6))

    # Storage Configuration
    FRAMES_OUTPUT_DIR = os.getenv('FRAMES_OUTPUT_DIR', 'output_frames')
    
    # Google Cloud Storage (Optional)
    GCP_PROJECT = os.getenv('GCP_PROJECT')
    GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    # Video Recording
    VIDEO_FPS = int(os.getenv('VIDEO_FPS', 4))
    VIDEO_BUFFER_SIZE = int(os.getenv('VIDEO_BUFFER_SIZE', 30))
    PRE_EVENT_SECONDS = int(os.getenv('PRE_EVENT_SECONDS', 3))
    POST_EVENT_SECONDS = int(os.getenv('POST_EVENT_SECONDS', 7))

    # Hardware Settings
    GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', 0.8))
    MAX_CPU_USAGE = int(os.getenv('MAX_CPU_USAGE', 80))
    MAX_MEMORY_USAGE = int(os.getenv('MAX_MEMORY_USAGE', 85))


class DetectionModels:
    """Available detection models and their configurations"""
    
    AVAILABLE_MODELS = {
        'tailgating': {
            'name': 'Tailgating Detection',
            'description': 'Detect multiple people entering through single access point',
            'script': 'detection_scripts/tailgating_detection.py',
            'requires': ['person_detection'],
            'parameters': {
                'time_window': 10,
                'max_people_per_entry': 1,
                'entry_zone_buffer': 2.0
            }
        },
        'unauthorized_access': {
            'name': 'Unauthorized Access Detection',
            'description': 'Detect people in restricted areas',
            'script': 'detection_scripts/unauthorized_access_detection.py',
            'requires': ['person_detection'],
            'parameters': {
                'confidence_threshold': 0.8,
                'restricted_zone_buffer': 0.5
            }
        },
        'people_counting': {
            'name': 'People Counting',
            'description': 'Count people in specified zones',
            'script': 'detection_scripts/people_counting.py',
            'requires': ['person_detection'],
            'parameters': {
                'counting_line_threshold': 0.5,
                'tracking_enabled': True
            }
        },
        'ppe_detection': {
            'name': 'PPE Detection',
            'description': 'Detect Personal Protective Equipment compliance',
            'script': 'detection_scripts/ppe_detection.py',
            'requires': ['person_detection', 'ppe_model'],
            'parameters': {
                'required_ppe': ['hard_hat', 'safety_vest'],
                'grace_period': 30
            }
        },
        'intrusion_detection': {
            'name': 'Intrusion Detection',
            'description': 'Detect unauthorized entry into secured areas',
            'script': 'detection_scripts/intrusion_detection.py',
            'requires': ['person_detection'],
            'parameters': {
                'sensitivity': 'high',
                'motion_threshold': 0.3
            }
        },
        'loitering_detection': {
            'name': 'Loitering Detection',
            'description': 'Detect people staying in areas for extended periods',
            'script': 'detection_scripts/loitering_detection.py',
            'requires': ['person_detection'],
            'parameters': {
                'loitering_threshold': 300,  # 5 minutes
                'movement_threshold': 1.0
            }
        }
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a specific model"""
        return cls.AVAILABLE_MODELS.get(model_name, {})
    
    @classmethod
    def get_all_models(cls) -> dict:
        """Get all available models"""
        return cls.AVAILABLE_MODELS
    
    @classmethod
    def validate_model(cls, model_name: str) -> bool:
        """Check if model exists"""
        return model_name in cls.AVAILABLE_MODELS


class EventTypes:
    """Event types for different detections"""
    
    TAILGATING = "tailgating"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PEOPLE_COUNTING = "people_counting"
    PPE_VIOLATION = "ppe_violation"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    
    EVENT_METADATA = {
        TAILGATING: {
            'severity': 'high',
            'description': 'Multiple people entering through single access point'
        },
        UNAUTHORIZED_ACCESS: {
            'severity': 'critical',
            'description': 'Person detected in restricted area'
        },
        PEOPLE_COUNTING: {
            'severity': 'info',
            'description': 'People count update'
        },
        PPE_VIOLATION: {
            'severity': 'medium',
            'description': 'PPE compliance violation detected'
        },
        INTRUSION: {
            'severity': 'critical',
            'description': 'Unauthorized intrusion detected'
        },
        LOITERING: {
            'severity': 'medium',
            'description': 'Person loitering detected'
        }
    }
    
    @classmethod
    def get_severity(cls, event_type: str) -> str:
        """Get severity level for event type"""
        return cls.EVENT_METADATA.get(event_type, {}).get('severity', 'info')


# Export main classes
__all__ = [
    'Config',
    'DetectionModels', 
    'EventTypes'
]
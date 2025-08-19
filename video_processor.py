#!/usr/bin/env python3
"""
Datacenter Video Processor - Main Video Processing Orchestrator

This module orchestrates the entire datacenter video processing pipeline:
1. Coordinates camera managers, model managers, and camera models
2. Manages batch processing for GPU efficiency
3. Routes detection results to appropriate datacenter camera models
4. Handles multi-camera coordination and cross-camera logic
5. Manages database writing and cloud storage
6. Provides system monitoring and health checks for datacenter environments
"""

import os
import sys
import threading
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from functools import partial
from collections import defaultdict

# Import core components
from database import DatacenterDatabase
from logger import setup_datacenter_logger, audit_logger, performance_logger
from config import Config, DatacenterEventTypes, DatacenterCameraTypes
from camera_manager import DatacenterCameraManager
from model_manager import DatacenterSingleGPUModelManager
from db_writer import DatacenterDatabaseWriter
from storage_handler import DatacenterStorageHandler

# Import datacenter camera models - FIXED IMPORTS
# try:
from camera_models.people_count_monitoring import PeopleCountingMonitor
from camera_models.ppe_kit_monitoring import PPEDetector
from camera_models.tailgating_zone_monitoring import TailgatingZoneMonitor
from camera_models.intrusion_zone_monitoring import IntrusionZoneMonitor
from camera_models.loitering_zone_monitoring import LoiteringZoneMonitor

# Multi-camera coordination (if available)
try:
    from camera_models.multi_camera_coordinator import MultiCameraCoordinator
    multi_camera_available = True
except ImportError:
    multi_camera_available = False
    MultiCameraCoordinator = None
    
camera_models_available = True
# except ImportError as e:
#     camera_models_available = False
#     print(f"Warning: Camera models not available: {e}")

# FIXED: Datacenter Camera model mapping - only available use cases
DATACENTER_CAMERA_MODEL_MAPPING = {
    'people_counting': PeopleCountingMonitor,
    'ppe_detection': PPEDetector, 
    'tailgating': TailgatingZoneMonitor,
    'intrusion': IntrusionZoneMonitor,
    'loitering': LoiteringZoneMonitor,
}

class DatacenterVideoProcessor:
    """
    Main video processing orchestrator for datacenter monitoring system.
    Coordinates all components and manages the processing pipeline.
    """
    
    def __init__(self, datacenter_id: Optional[int] = None, config=None):
        """
        Initialize the datacenter video processor
        
        Args:
            datacenter_id: Optional datacenter ID for filtering cameras
            config: Optional configuration override for local testing
        """
        self.logger = setup_datacenter_logger(
            'datacenter_video_processor', 
            'datacenter_video_processor.log',
            datacenter_id=str(datacenter_id) if datacenter_id else None
        )
        self.logger.info("Initializing DatacenterVideoProcessor")
        
        # Store configuration (for local testing override)
        self.config = config if config else Config()
        
        # Store datacenter filter
        self.datacenter_id = datacenter_id
        if self.datacenter_id:
            self.logger.info(f"Filtering for datacenter ID: {self.datacenter_id}")
        
        # Initialize core components
        self.database = DatacenterDatabase()
        self.camera_manager = DatacenterCameraManager()
        self.model_manager = DatacenterSingleGPUModelManager()
        self.db_writer = DatacenterDatabaseWriter()
        self.storage_handler = DatacenterStorageHandler()
        
        # FIXED: Multi-use case tracking
        self.camera_models = {}  # Legacy compatibility
        self.camera_use_case_models = {}  # NEW: {camera_id: {use_case: model}}
        self.camera_metadata = {}
        self.camera_feeds = {}
        
        # Performance tracking for multi-use case
        self.performance_stats = defaultdict(dict)
        self.shared_detection_cache = {}  # Cache detection results to share across use cases
        
        # Multi-camera coordination for datacenters
        self.datacenter_coordinators = {}  # {datacenter_id: MultiCameraCoordinator}
        
        # Batch processing configuration
        self.batch_size = getattr(self.config, 'BATCH_SIZE', Config.BATCH_SIZE)
        self.batch_timeout = getattr(self.config, 'BATCH_TIMEOUT', Config.BATCH_TIMEOUT)
        self.max_parallel_cameras = getattr(self.config, 'MAX_PARALLEL_CAMERAS', Config.MAX_PARALLEL_CAMERAS)
        
        # Processing threads and state
        self.batch_processing_thread = None
        self.batch_processing_running = False
        self.processing_executor = None
        
        # System monitoring
        self.system_stats = {
            'total_frames_processed': 0,
            'total_events_detected': 0,
            'cameras_active': 0,
            'average_fps': 0,
            'uptime_seconds': 0,
            'start_time': time.time()
        }
        
        # Load camera configurations and initialize models
        self._load_camera_configurations()
        self._initialize_camera_models()
        if multi_camera_available and MultiCameraCoordinator is not None:
            self._initialize_multi_camera_coordinators()
        
        self.logger.info("DatacenterVideoProcessor initialization complete")
    
    def _load_camera_configurations(self):
        """Load camera configurations from database or local config"""
        try:
            # Check if we're in single camera mode (local testing)
            if hasattr(self.config, 'SINGLE_CAMERA_MODE') and self.config.SINGLE_CAMERA_MODE:
                self.logger.info("Loading single camera configuration for local testing")
                self._load_single_camera_config()
                return
            
            # Normal database loading
            self.logger.info("Loading camera configurations from database")
            
            query = """
                SELECT c.camera_id, c.name, c.stream_url, c.camera_type, c.location_details, 
                       c.status, c.metadata, c.datacenter_id,
                       d.name as datacenter_name
                FROM cameras c
                JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                WHERE c.status = 'active'
            """
            
            params = []
            if self.datacenter_id:
                query += " AND c.datacenter_id = %s"
                params.append(self.datacenter_id)
            
            cameras = self.database.execute_query(query, params)
            
            for camera in cameras:
                camera_id = str(camera['camera_id'])
                metadata = camera.get('metadata', {})
                
                # Parse metadata if it's a string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                self.camera_metadata[camera_id] = {
                    'name': camera['name'],
                    'stream_url': camera['stream_url'],
                    'camera_type': camera['camera_type'],
                    'location_details': camera['location_details'],
                    'datacenter_id': camera['datacenter_id'],
                    'datacenter_name': camera['datacenter_name'],
                    'metadata': metadata,
                    'use_cases': metadata.get('use_cases', [camera['camera_type']])  # NEW: Support use_cases array
                }
                
                # Determine activity level for FPS
                activity_level = self._determine_activity_level(camera['camera_type'], metadata)
                
                self.camera_feeds[camera_id] = (
                    camera['stream_url'],
                    activity_level
                )
            
            self.logger.info(f"Loaded {len(self.camera_metadata)} camera configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading camera configurations: {str(e)}", exc_info=True)
    
    def _load_single_camera_config(self):
        """Load single camera configuration for local testing"""
        try:
            camera_id = getattr(self.config, 'LOCAL_CAMERA_ID', 'test_camera')
            rtsp_url = getattr(self.config, 'LOCAL_RTSP_URL', 'rtsp://localhost:554/stream')
            use_cases = getattr(self.config, 'LOCAL_USE_CASES', ['people_counting'])
            
            self.camera_metadata[camera_id] = {
                'name': 'Local Test Camera',
                'stream_url': rtsp_url,
                'camera_type': 'multi_use_case',
                'location_details': {'floor': 'Test', 'location': 'Office'},
                'datacenter_id': 1,
                'datacenter_name': 'Local Test Datacenter',
                'metadata': {
                    'activity_level': 'high',
                    'use_cases': use_cases
                },
                'use_cases': use_cases
            }
            
            self.camera_feeds[camera_id] = (rtsp_url, 'high')
            
            self.logger.info(f"Loaded single camera config: {camera_id} with use cases: {use_cases}")
            
        except Exception as e:
            self.logger.error(f"Error loading single camera config: {e}")
    
    def _determine_activity_level(self, camera_type, metadata):
        """Determine activity level for camera based on type and metadata"""
        try:
            # Check metadata first
            if metadata and 'activity_level' in metadata:
                return metadata['activity_level']
            
            # Determine based on camera type - datacenter specific logic
            if camera_type in ['tailgating', 'intrusion']:
                return 'high'  # High security areas need more FPS
            elif camera_type in ['ppe_detection']:
                return 'medium'  # PPE detection needs moderate monitoring
            elif camera_type in ['people_counting', 'loitering']:
                return 'low'  # Can use lower FPS
            else:
                return 'medium'  # Default for unknown types
                
        except Exception as e:
            self.logger.error(f"Error determining activity level: {e}")
            return 'medium'
    
    def _initialize_camera_models(self):
        """
        FIXED: Initialize multiple camera models per camera for different use cases
        This addresses the critical issue identified by your friend
        """
        try:
            self.logger.info("Initializing MULTI-USE CASE camera models")
            
            if not camera_models_available:
                self.logger.warning("Camera models not available, using basic processing")
                return
            
            # Initialize multi-use case tracking
            self.camera_use_case_models = {}
            
            for camera_id, camera_info in self.camera_metadata.items():
                self.logger.info(f"Setting up camera {camera_id}")
                
                # Get use cases for this camera - NEW LOGIC
                use_cases = self._get_camera_use_cases(camera_id, camera_info)
                if not use_cases:
                    self.logger.warning(f"No use cases defined for camera {camera_id}")
                    continue
                
                self.logger.info(f" Use cases for camera {camera_id}: {use_cases}")
                
                # Initialize storage for this camera's models
                self.camera_use_case_models[camera_id] = {}
                
                # Initialize each use case model separately
                for use_case in use_cases:
                    if use_case in DATACENTER_CAMERA_MODEL_MAPPING:
                        self._initialize_single_use_case_model(camera_id, use_case, camera_info)
                    else:
                        self.logger.warning(f"Unknown use case: {use_case}")
                
                # Store first model in legacy format for compatibility
                if self.camera_use_case_models[camera_id]:
                    first_model = list(self.camera_use_case_models[camera_id].values())[0]
                    self.camera_models[camera_id] = first_model
                
                self.logger.info(f" Initialized {len(self.camera_use_case_models[camera_id])} models for camera {camera_id}")
            
            total_models = sum(len(models) for models in self.camera_use_case_models.values())
            self.logger.info(f"Total camera models initialized: {total_models}")
            
        except Exception as e:
            self.logger.error(f" Error initializing camera models: {str(e)}", exc_info=True)
    
    def _get_camera_use_cases(self, camera_id: str, camera_info: dict) -> list:
        """Get list of use cases for a camera"""
        # Check if use_cases is defined in camera info
        if 'use_cases' in camera_info:
            return camera_info['use_cases']
        
        # Check if defined in camera metadata
        metadata = camera_info.get('metadata', {})
        if 'use_cases' in metadata:
            return metadata['use_cases']
        
        # For local testing - all cameras get all use cases if configured
        if hasattr(self.config, 'SINGLE_CAMERA_MODE') and self.config.SINGLE_CAMERA_MODE:
            return getattr(self.config, 'LOCAL_USE_CASES', ['people_counting'])
        
        # Fallback to single camera_type
        camera_type = camera_info.get('camera_type')
        if camera_type and camera_type in DATACENTER_CAMERA_MODEL_MAPPING:
            return [camera_type]
        
        # Default use cases for unknown cameras
        return ['people_counting']
    
    def _initialize_single_use_case_model(self, camera_id: str, use_case: str, camera_info: dict):
        """Initialize a single use case model for a camera"""
        try:
            model_class = DATACENTER_CAMERA_MODEL_MAPPING[use_case]
            
            # Get use-case specific zones and rules
            zones = self._get_camera_zones_for_use_case(camera_id, use_case)
            rules = self._get_camera_rules_for_use_case(camera_id, use_case)
            settings = camera_info.get('metadata', {}).copy()
            
            # Add use case specific settings
            settings['use_case'] = use_case
            settings['camera_info'] = camera_info
            
            self.logger.info(f"Creating {use_case} model for camera {camera_id}")
            
            # Create the model instance
            camera_model = model_class(
                camera_id=camera_id,
                zones=zones,
                rules=rules,
                settings=settings,
                db=self.database,
                db_writer=self.db_writer,
                frames_base_dir=getattr(self.config, 'FRAMES_OUTPUT_DIR', Config.FRAMES_OUTPUT_DIR),
                camera_manager=self.camera_manager
            )
            
            # Store the model
            self.camera_use_case_models[camera_id][use_case] = camera_model
            
            self.logger.info(f"{use_case} model initialized for camera {camera_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {use_case} for camera {camera_id}: {e}")
    
    # def _get_camera_zones_for_use_case(self, camera_id: str, use_case: str) -> dict:
    #     """Get zones specific to a use case"""
    #     # Try to get from database first
    #     try:
    #         zones = self._get_camera_zones(camera_id)
    #         if zones:
    #             return zones
    #     except:
    #         pass

    def _get_camera_zones_for_use_case(self, camera_id: str, use_case: str) -> dict:
        """Get zones for a specific camera and use case"""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # FIX: Check what columns actually exist first
                cursor.execute("DESCRIBE zones")
                columns = [row['Field'] for row in cursor.fetchall()]
                
                # Build query based on available columns
                if 'coordinates' in columns:
                    coord_field = 'z.coordinates'
                elif 'polygon_coordinates' in columns:
                    coord_field = 'z.polygon_coordinates'
                elif 'zone_coordinates' in columns:
                    coord_field = 'z.zone_coordinates'
                else:
                    coord_field = 'NULL as coordinates'
                
                # Use dynamic query based on actual schema
                query = f"""
                    SELECT z.zone_id, z.name, {coord_field} as coordinates,
                        COALESCE(z.zone_type, %s) as type, z.metadata
                    FROM zones z 
                    WHERE z.camera_id = %s 
                    AND (z.zone_type = %s OR z.zone_type IS NULL OR z.zone_type = '')
                """
                
                cursor.execute(query, (use_case, camera_id, use_case))
                zones = cursor.fetchall()
                
                # Group zones by type
                zones_by_type = {}
                for zone in zones:
                    zone_type = zone.get('type', use_case)
                    if zone_type not in zones_by_type:
                        zones_by_type[zone_type] = []
                    zones_by_type[zone_type].append(zone)
                
                return zones_by_type
                
        except Exception as e:
            self.logger.error(f"Error getting zones for camera {camera_id}: {str(e)}")
            return {use_case: []}
        
        # Default zones for local testing
        default_zones = {
            'people_counting': {
                'counting': [{
                    'name': f'Camera {camera_id} Counting Zone',
                    'coordinates': [[100, 100], [500, 100], [500, 400], [100, 400]]
                }]
            },
            'ppe_detection': {
                'ppe_zone': [{
                    'name': f'Camera {camera_id} PPE Zone', 
                    'coordinates': [[50, 50], [550, 50], [550, 450], [50, 450]]
                }]
            },
            'tailgating': {
                'entry': [{
                    'name': f'Camera {camera_id} Entry Zone',
                    'coordinates': [[200, 300], [400, 300], [400, 500], [200, 500]]
                }]
            },
            'intrusion': {
                'restricted': [{
                    'name': f'Camera {camera_id} Restricted Zone',
                    'coordinates': [[300, 200], [500, 200], [500, 350], [300, 350]]
                }]
            },
            'loitering': {
                'monitoring': [{
                    'name': f'Camera {camera_id} Loitering Zone',
                    'coordinates': [[150, 150], [450, 150], [450, 300], [150, 300]]
                }]
            }
        }
        
        return default_zones.get(use_case, {})
    
    def _get_camera_rules_for_use_case(self, camera_id: str, use_case: str) -> dict:
        """Get rules specific to a use case"""
        # Try to get from database first
        try:
            rules = self._get_camera_rules(camera_id)
            if rules:
                return rules
        except:
            pass
        
        # Default rules for local testing
        default_rules = {
            'people_counting': {'people_count_threshold': 5},
            'ppe_detection': {'conf_threshold': 0.6, 'required_ppe': ['hard_hat', 'safety_vest']},
            'tailgating': {'tailgating_time_limit': 3.0, 'conf_threshold': 0.5},
            'intrusion': {'alert_threshold': 0.7, 'cooldown_period': 10},
            'loitering': {'loitering_time_threshold': 30, 'conf_threshold': 0.6}
        }
        
        return default_rules.get(use_case, {})
    
    async def process_frame_with_all_use_cases(self, frame, timestamp, camera_id):
        """
        NEW: Process a frame with ALL use cases for the camera
        This is the key optimization for multi-use case processing
        """
        results = {}
        
        if camera_id not in self.camera_use_case_models:
            return results
        
        # Get detection results once (shared across use cases for efficiency)
        base_detection_result = None
        detection_cache_key = f"{camera_id}_{timestamp}"
        
        if detection_cache_key in self.shared_detection_cache:
            base_detection_result = self.shared_detection_cache[detection_cache_key]
        else:
            try:
                if hasattr(self.model_manager, 'detect'):
                    base_detection_result = await self.model_manager.detect(frame)
                    # Cache for other use cases
                    self.shared_detection_cache[detection_cache_key] = base_detection_result
                    # Limit cache size
                    if len(self.shared_detection_cache) > 100:
                        oldest_key = next(iter(self.shared_detection_cache))
                        del self.shared_detection_cache[oldest_key]
            except Exception as e:
                self.logger.error(f"Error getting base detection result: {e}")
        
        # Process with each use case model
        for use_case, model in self.camera_use_case_models[camera_id].items():
            try:
                start_time = time.time()
                
                # Process frame with this use case model
                annotated_frame, detections = model.process_frame(
                    frame, timestamp, base_detection_result
                )
                
                processing_time = time.time() - start_time
                
                results[use_case] = {
                    'annotated_frame': annotated_frame,
                    'detections': detections,
                    'processing_time': processing_time
                }
                
                # Update performance stats
                if camera_id not in self.performance_stats:
                    self.performance_stats[camera_id] = {}
                
                self.performance_stats[camera_id][use_case] = {
                    'last_processing_time': processing_time,
                    'avg_processing_time': self._update_avg_processing_time(camera_id, use_case, processing_time)
                }
                
            except Exception as e:
                self.logger.error(f"Error processing {use_case} for camera {camera_id}: {e}")
                results[use_case] = {'error': str(e)}
        
        return results
    
    def _update_avg_processing_time(self, camera_id: str, use_case: str, processing_time: float) -> float:
        """Update average processing time for performance monitoring"""
        if 'processing_times' not in self.performance_stats[camera_id]:
            self.performance_stats[camera_id]['processing_times'] = defaultdict(list)
        
        times = self.performance_stats[camera_id]['processing_times'][use_case]
        times.append(processing_time)
        
        # Keep only last 100 measurements
        if len(times) > 100:
            times.pop(0)
        
        return sum(times) / len(times)
    
    def get_performance_report(self) -> dict:
        """Get performance report for all cameras and use cases"""
        report = {
            'timestamp': time.time(),
            'cameras': {}
        }
        
        for camera_id, stats in self.performance_stats.items():
            report['cameras'][camera_id] = {
                'use_cases': {},
                'total_avg_time': 0
            }
            
            total_time = 0
            use_case_count = 0
            
            for use_case, use_case_stats in stats.items():
                if use_case != 'processing_times':
                    avg_time = use_case_stats.get('avg_processing_time', 0)
                    report['cameras'][camera_id]['use_cases'][use_case] = {
                        'avg_processing_time_ms': round(avg_time * 1000, 2),
                        'last_processing_time_ms': round(use_case_stats.get('last_processing_time', 0) * 1000, 2)
                    }
                    total_time += avg_time
                    use_case_count += 1
            
            if use_case_count > 0:
                report['cameras'][camera_id]['total_avg_time'] = round((total_time / use_case_count) * 1000, 2)
        
        return report
    
    def get_system_stats(self):
        """Get system statistics"""
        return {
            'total_frames_processed': self.system_stats.get('total_frames_processed', 0),
            'total_events_detected': self.system_stats.get('total_events_detected', 0),
            'cameras_active': len(self.camera_feeds),
            'uptime_seconds': time.time() - self.system_stats.get('start_time', time.time())
        }


    # Keep all existing methods from original video_processor.py
    def _get_camera_zones(self, camera_id):
        """Get zones for a specific camera from the database"""
        try:
            query = """
                SELECT zone_id, name, zone_type, polygon_coordinates, metadata 
                FROM zones 
                WHERE camera_id = %s
            """
            zones_data = self.database.execute_query(query, (camera_id,))
            
            zones_by_type = {}
            for zone in zones_data:
                zone_type = zone['type']
                if zone_type not in zones_by_type:
                    zones_by_type[zone_type] = []
                
                zone_dict = {
                    'zone_id': zone['zone_id'],
                    'name': zone['name'],
                    'coordinates': zone['polygon_coordinates'],
                    'metadata': zone['metadata']
                }
                zones_by_type[zone_type].append(zone_dict)
            
            return zones_by_type
        except Exception as e:
            self.logger.error(f"Error getting zones for camera {camera_id}: {str(e)}")
            return {}
    
    def _get_camera_rules(self, camera_id):
        """Get rules for a specific camera from the database"""
        try:
            query = """
                SELECT rule_id, name, description, event_type, severity, parameters, enabled
                FROM rules 
                WHERE camera_id = %s AND enabled = TRUE
            """
            rules_data = self.database.execute_query(query, (camera_id,))
            
            rules = {}
            for rule in rules_data:
                parameters = rule.get('parameters', {})
                if isinstance(parameters, str):
                    try:
                        parameters = json.loads(parameters)
                    except:
                        parameters = {}
                
                rules[rule['event_type']] = parameters
            
            return rules
        except Exception as e:
            self.logger.error(f"Error getting rules for camera {camera_id}: {str(e)}")
            return {}
    
    def _initialize_multi_camera_coordinators(self):
        """Initialize multi-camera coordinators for datacenter monitoring"""
        if not multi_camera_available or MultiCameraCoordinator is None:
            self.logger.info("Multi-camera coordination not available - skipping coordinator initialization")
            return
            
        try:
            self.logger.info("Initializing multi-camera coordinators for datacenters")
            
            # Group cameras by datacenter_id
            datacenter_cameras = {}
            for camera_id, metadata in self.camera_metadata.items():
                datacenter_id = metadata.get('datacenter_id')
                if datacenter_id:
                    if datacenter_id not in datacenter_cameras:
                        datacenter_cameras[datacenter_id] = []
                    datacenter_cameras[datacenter_id].append(camera_id)
            
            # Create coordinators for datacenters with multiple cameras
            for datacenter_id, camera_ids in datacenter_cameras.items():
                if len(camera_ids) >= 2:  # Only create coordinator if multiple cameras
                    try:
                        # Pass camera models to coordinator for event handling
                        camera_models_for_coordinator = {}
                        for cam_id in camera_ids:
                            if cam_id in self.camera_use_case_models:
                                camera_models_for_coordinator[cam_id] = self.camera_use_case_models[cam_id]
                        
                        self.datacenter_coordinators[datacenter_id] = MultiCameraCoordinator(
                            datacenter_id, camera_ids, camera_models_for_coordinator, self.database
                        )
                        
                        self.logger.info(f"Created multi-camera coordinator for datacenter {datacenter_id} with {len(camera_ids)} cameras")
                        
                    except Exception as e:
                        self.logger.error(f"Error creating coordinator for datacenter {datacenter_id}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error initializing multi-camera coordinators: {str(e)}")
    
    async def start_monitoring(self):
        """Start the monitoring system with multi-use case support"""
        self.logger.info("Starting datacenter monitoring system")
        
        # Validate camera configurations
        if not self.camera_feeds:
            self.logger.error("No camera configurations found. Please ensure cameras are configured in the database.")
            return False
        
        # Make sure camera models are initialized
        if camera_models_available and not self.camera_use_case_models:
            self._initialize_camera_models()
        
        # Start batch processing
        success = await self.start_batch_processing(self.camera_feeds)
        
        if success:
            self.running = True
            self.logger.info(f"Datacenter monitoring started for {len(self.camera_feeds)} cameras")
            return True
        else:
            self.logger.error("Failed to start datacenter monitoring system")
            return False

    async def stop_monitoring(self):
        """Stop all monitoring"""
        self.logger.info("Stopping datacenter camera monitoring")
        
        # Stop batch processing
        self.stop_batch_processing()
        
        # Clear status
        self.running = False
        
        self.logger.info("Datacenter monitoring stopped")
        return True
    
    async def start_batch_processing(self, camera_sources, model_name='detection'):
        """Start batch processing of frames from multiple cameras"""
        if self.batch_processing_thread and self.batch_processing_thread.is_alive():
            self.logger.warning("Batch processing already running")
            return False
            
        self.logger.info(f"Starting batch processing for {len(camera_sources)} cameras")
        
        # Start camera readers using the camera manager
        self.camera_manager.start_cameras(camera_sources)
        
        # Store camera sources
        self.camera_sources = camera_sources
        
        # Start batch processing thread
        self.batch_processing_running = True
        self.batch_processing_thread = threading.Thread(
            target=self._batch_processing_worker,
            args=(model_name,),
            daemon=True
        )
        self.batch_processing_thread.start()
        
        self.logger.info("Batch processing started successfully")
        return True
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        self.logger.info("Stopping batch processing")
        
        # Signal to stop
        self.batch_processing_running = False
        
        # Wait for thread to finish
        if self.batch_processing_thread and self.batch_processing_thread.is_alive():
            self.batch_processing_thread.join(timeout=5)
        
        # Stop camera manager
        if self.camera_manager:
            self.camera_manager.stop_all_cameras()
        
        self.logger.info("Batch processing stopped")
    
    def _batch_processing_worker(self, model_name):
        """Worker thread for batch processing with multi-use case support"""
        self.logger.info("Batch processing worker started")
        
        # while self.batch_processing_running:
        #     try:
        #         # Collect frames from all cameras
        #         batch_frames = self.camera_manager.get_frame_batch(
        #             batch_size=self.batch_size,
        #             timeout=self.batch_timeout
        #         )
                
        #         if batch_frames:
        #             # Process batch asynchronously
        #             asyncio.run(self._process_camera_batch_multi_use_case(batch_frames))
                    
        #             # Update stats
        #             self.system_stats['total_frames_processed'] += len(batch_frames)
                
        #         # Small sleep to prevent CPU spinning
        #         time.sleep(0.001)
        
        while self.batch_processing_running:
            try: 
                time.sleep(1.0)
                continue
                
            except Exception as e:
                self.logger.error(f"Error in batch processing worker: {str(e)}", exc_info=True)
                time.sleep(1)  # Wait before retrying
        
        self.logger.info("Batch processing worker stopped")
    
    async def _process_camera_batch_multi_use_case(self, batch_frames):
        """Process a batch of frames with multi-use case support"""
        try:
            # Process each camera's frame with all its use cases
            for camera_id, frame_data in batch_frames.items():
                if camera_id in self.camera_use_case_models:
                    # Process frame with all use cases for this camera
                    results = await self.process_frame_with_all_use_cases(
                        frame_data['frame'], 
                        frame_data['timestamp'], 
                        camera_id
                    )
                    
                    # Handle results from each use case
                    for use_case, result in results.items():
                        if 'error' not in result:
                            # Process detections if any
                            detections = result.get('detections', [])
                            if detections:
                                self.system_stats['total_events_detected'] += len(detections)
                                
                                # Route to database writer
                                if self.db_writer:
                                    await self._write_detection_results(camera_id, use_case, detections, frame_data['timestamp'])
                        else:
                            self.logger.error(f"Error in {use_case} for camera {camera_id}: {result['error']}")
                
        except Exception as e:
            self.logger.error(f"Error processing camera batch: {str(e)}", exc_info=True)
    
    async def _write_detection_results(self, camera_id, use_case, detections, timestamp):
        """Write detection results to database"""
        try:
            # This method would handle writing detection results
            # Implementation depends on your specific database schema
            pass
        except Exception as e:
            self.logger.error(f"Error writing detection results: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        # Calculate uptime
        uptime = time.time() - self.system_stats['start_time']
        
        # Calculate average FPS
        if uptime > 0:
            avg_fps = self.system_stats['total_frames_processed'] / uptime
        else:
            avg_fps = 0
        
        # Get camera stats
        camera_stats = {}
        for camera_id in self.camera_metadata.keys():
            if camera_id in self.camera_use_case_models:
                use_case_count = len(self.camera_use_case_models[camera_id])
                camera_stats[camera_id] = {
                    'use_cases': list(self.camera_use_case_models[camera_id].keys()),
                    'use_case_count': use_case_count,
                    'status': 'active' if self.camera_manager.is_camera_active(camera_id) else 'inactive'
                }
        
        return {
            'uptime_seconds': uptime,
            'total_frames_processed': self.system_stats['total_frames_processed'],
            'total_events_detected': self.system_stats['total_events_detected'],
            'cameras_total': len(self.camera_metadata),
            'cameras_active': len([c for c in camera_stats.values() if c['status'] == 'active']),
            'average_fps': avg_fps,
            'model_info': self.model_manager.get_model_info() if hasattr(self.model_manager, 'get_model_info') else {},
            'camera_stats': camera_stats,
            'performance_stats': self.get_performance_report()
        }

    def shutdown(self):
        """Graceful shutdown of the video processor"""
        self.logger.info("Shutting down DatacenterVideoProcessor")
        
        # Stop monitoring
        if hasattr(self, 'running') and self.running:
            asyncio.run(self.stop_monitoring())
        
        # Shutdown components
        if self.camera_manager:
            self.camera_manager.stop_all_cameras()
        
        if self.model_manager:
            self.model_manager.shutdown()
        
        if self.db_writer:
            self.db_writer.shutdown()
        
        if self.storage_handler:
            self.storage_handler.shutdown()
        
        self.logger.info("DatacenterVideoProcessor shutdown complete")


# For testing and backwards compatibility
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Datacenter Video Processing System")
    parser.add_argument("--datacenter-id", type=int, help="Filter for specific datacenter")
    parser.add_argument("--single-camera", action="store_true", help="Run in single camera mode for testing")
    args = parser.parse_args()
    
    # Override config for single camera testing
    config = None
    if args.single_camera:
        from config import Config
        class TestConfig(Config):
            SINGLE_CAMERA_MODE = True
            LOCAL_CAMERA_ID = "test_cam_001"
            LOCAL_RTSP_URL = "rtsp://localhost:554/stream"  # Update with your camera
            LOCAL_USE_CASES = ['people_counting', 'ppe_detection', 'tailgating', 'intrusion', 'loitering']
        config = TestConfig()
    
    processor = DatacenterVideoProcessor(datacenter_id=args.datacenter_id, config=config)
    
    try:
        # Run monitoring
        asyncio.run(processor.start_monitoring())
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
        processor.shutdown()


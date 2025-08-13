


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
from .database import DatacenterDatabase
from .logger import setup_datacenter_logger, audit_logger, performance_logger
from .config import Config, DatacenterEventTypes, DatacenterCameraTypes
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
            self.logger.info("🔧 Initializing MULTI-USE CASE camera models")
            
            if not camera_models_available:
                self.logger.warning("Camera models not available, using basic processing")
                return
            
            # Initialize multi-use case tracking
            self.camera_use_case_models = {}
            
            for camera_id, camera_info in self.camera_metadata.items():
                self.logger.info(f"📹 Setting up camera {camera_id}")
                
                # Get use cases for this camera - NEW LOGIC
                use_cases = self._get_camera_use_cases(camera_id, camera_info)
                if not use_cases:
                    self.logger.warning(f"No use cases defined for camera {camera_id}")
                    continue
                
                self.logger.info(f"🎯 Use cases for camera {camera_id}: {use_cases}")
                
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
                
                self.logger.info(f"✅ Initialized {len(self.camera_use_case_models[camera_id])} models for camera {camera_id}")
            
            total_models = sum(len(models) for models in self.camera_use_case_models.values())
            self.logger.info(f"🎉 Total camera models initialized: {total_models}")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing camera models: {str(e)}", exc_info=True)
    
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
            
            self.logger.info(f"  🔨 Creating {use_case} model for camera {camera_id}")
            
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
            
            self.logger.info(f"  ✅ {use_case} model initialized for camera {camera_id}")
            
        except Exception as e:
            self.logger.error(f"  ❌ Failed to initialize {use_case} for camera {camera_id}: {e}")
    
    def _get_camera_zones_for_use_case(self, camera_id: str, use_case: str) -> dict:
        """Get zones specific to a use case"""
        # Try to get from database first
        try:
            zones = self._get_camera_zones(camera_id)
            if zones:
                return zones
        except:
            pass
        
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
    
    # Keep all existing methods from original video_processor.py
    def _get_camera_zones(self, camera_id):
        """Get zones for a specific camera from the database"""
        try:
            query = """
                SELECT zone_id, name, type, polygon_coordinates, metadata 
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
        
        while self.batch_processing_running:
            try:
                # Collect frames from all cameras
                batch_frames = self.camera_manager.get_frame_batch(
                    batch_size=self.batch_size,
                    timeout=self.batch_timeout
                )
                
                if batch_frames:
                    # Process batch asynchronously
                    asyncio.run(self._process_camera_batch_multi_use_case(batch_frames))
                    
                    # Update stats
                    self.system_stats['total_frames_processed'] += len(batch_frames)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
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

'''
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

# Import core components
from database import DatacenterDatabase
from logger import setup_datacenter_logger, audit_logger, performance_logger
from config import Config, DatacenterEventTypes, DatacenterCameraTypes
from camera_manager import DatacenterCameraManager
from model_manager import DatacenterSingleGPUModelManager
from db_writer import DatacenterDatabaseWriter
from storage_handler import DatacenterStorageHandler

# Import datacenter camera models
try:
    from camera_models import (
        DatacenterEntryMonitor,
        ServerRoomMonitor,
        CorridorMonitor,
        PerimeterMonitor,
        CriticalZoneMonitor,
        CommonAreaMonitor
    )
    # Import your specific use case models
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
except ImportError as e:
    camera_models_available = False
    print(f"Warning: Some camera models not available: {e}")

# Datacenter Camera model mapping - combines original datacenter models with new use cases
DATACENTER_CAMERA_MODEL_MAPPING = {
    # Original datacenter camera types
    'dc_entry_monitor': DatacenterEntryMonitor if 'DatacenterEntryMonitor' in globals() else None,
    'dc_server_room': ServerRoomMonitor if 'ServerRoomMonitor' in globals() else None,
    'dc_corridor': CorridorMonitor if 'CorridorMonitor' in globals() else None,
    'dc_perimeter': PerimeterMonitor if 'PerimeterMonitor' in globals() else None,
    'dc_critical_zone': CriticalZoneMonitor if 'CriticalZoneMonitor' in globals() else None,
    'dc_common_area': CommonAreaMonitor if 'CommonAreaMonitor' in globals() else None,
    
    # New use case models
    'people_counting': PeopleCountingMonitor,
    'ppe_detection': PPEDetector,
    'tailgating': TailgatingZoneMonitor,
    'intrusion': IntrusionZoneMonitor,
    'loitering': LoiteringZoneMonitor,

}

# Filter out None values (for models that aren't available)
DATACENTER_CAMERA_MODEL_MAPPING = {k: v for k, v in DATACENTER_CAMERA_MODEL_MAPPING.items() if v is not None}

class DatacenterVideoProcessor:
    """
    Main video processing orchestrator for datacenter monitoring system.
    Coordinates all components and manages the processing pipeline.
    """
    
    def __init__(self, datacenter_id: Optional[int] = None):
        """
        Initialize the datacenter video processor
        
        Args:
            datacenter_id: Optional datacenter ID for filtering cameras
        """
        self.logger = setup_datacenter_logger(
            'datacenter_video_processor', 
            'datacenter_video_processor.log',
            datacenter_id=str(datacenter_id) if datacenter_id else None
        )
        self.logger.info("Initializing DatacenterVideoProcessor")
        
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
        
        # Camera configuration and models
        self.camera_models = {}
        self.camera_metadata = {}
        self.camera_feeds = {}
        
        # Multi-camera coordination for datacenters
        self.datacenter_coordinators = {}  # {datacenter_id: MultiCameraCoordinator}
        
        # Batch processing configuration
        self.batch_size = Config.BATCH_SIZE
        self.batch_timeout = Config.BATCH_TIMEOUT
        self.max_parallel_cameras = Config.MAX_PARALLEL_CAMERAS
        
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
        
        # Set up result routing
        self._setup_result_routing()
        
        self.logger.info("DatacenterVideoProcessor initialization complete")
    
    def _load_camera_configurations(self):
        """Load camera configurations from database"""
        try:
            self.logger.info("Loading camera configurations from database")
            
            # Build query based on datacenter filter
            if self.datacenter_id:
                camera_query = """
                    SELECT 
                        c.camera_id, c.datacenter_id, d.user_id, d.name as datacenter_name, 
                        c.name, c.stream_url, c.camera_type, c.location_details, c.status, c.metadata,
                        d.location as datacenter_location
                    FROM cameras c
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                    WHERE c.status = 'active' AND d.status = 'active' AND c.datacenter_id = %s
                """
                cameras = self.database.execute_query(camera_query, (self.datacenter_id,))
            else:
                camera_query = """
                    SELECT 
                        c.camera_id, c.datacenter_id, d.user_id, d.name as datacenter_name, 
                        c.name, c.stream_url, c.camera_type, c.location_details, c.status, c.metadata,
                        d.location as datacenter_location
                    FROM cameras c
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                    WHERE c.status = 'active' AND d.status = 'active'
                """
                cameras = self.database.execute_query(camera_query)
            
            if not cameras:
                self.logger.warning("No active cameras found in database")
                return
            
            # Process each camera
            for camera in cameras:
                camera_id = camera['camera_id']
                datacenter_id = camera['datacenter_id']
                user_id = camera['user_id']
                datacenter_name = camera['datacenter_name']
                camera_name = camera['name']
                stream_url = camera['stream_url']
                camera_type = camera['camera_type']
                location_details = camera['location_details']
                datacenter_location = camera['datacenter_location']
                
                # Parse metadata
                metadata = camera['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid metadata JSON for camera {camera_id}")
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                
                # Determine activity level based on camera type and location
                activity_level = self._determine_activity_level(camera_type, location_details, metadata)
                
                # Store camera feed information
                self.camera_feeds[camera_id] = (stream_url, activity_level)
                
                # Store comprehensive camera metadata
                self.camera_metadata[camera_id] = {
                    'camera_id': camera_id,
                    'datacenter_id': datacenter_id,
                    'user_id': user_id,
                    'datacenter_name': datacenter_name,
                    'name': camera_name,
                    'stream_url': stream_url,
                    'camera_type': camera_type,
                    'location_details': location_details,
                    'activity_level': activity_level,
                    'datacenter_location': datacenter_location,
                    'frames_processed': 0,
                    'events_detected': 0,
                    'last_processed': None,
                    'metadata': metadata
                }
                
                self.logger.info(f"Loaded camera {camera_id} ({camera['name']}) - Type: {camera['camera_type']}")
            
            self.logger.info(f"Loaded {len(self.camera_feeds)} camera configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading camera configurations: {str(e)}", exc_info=True)
    
    def _determine_activity_level(self, camera_type: str, location_details: Dict, metadata: Dict) -> str:
        """Determine activity level based on camera type and location for datacenter monitoring"""
        try:
            # Check metadata first
            if metadata and 'activity_level' in metadata:
                return metadata['activity_level']
            
            # Determine based on camera type - datacenter specific logic
            if camera_type in ['dc_entry_monitor', 'dc_critical_zone', 'tailgating', 'intrusion']:
                return 'high'  # High security areas need more FPS
            elif camera_type in ['dc_server_room', 'ppe_detection', 'unauthorized_zone']:
                return 'medium'  # Server rooms need moderate monitoring
            elif camera_type in ['dc_corridor', 'dc_common_area', 'people_counting', 'loitering']:
                return 'low'  # Common areas can use lower FPS
            else:
                return 'medium'  # Default for unknown types
                
        except Exception as e:
            self.logger.error(f"Error determining activity level: {e}")
            return 'medium'
    
    def _initialize_camera_models(self):
        """Initialize camera models based on camera types"""
        try:
            self.logger.info("Initializing datacenter camera models")
            
            if not camera_models_available:
                self.logger.warning("Camera models not available, using basic processing")
                return
            
            for camera_id, camera_info in self.camera_metadata.items():
                camera_type = camera_info['camera_type']
                datacenter_id = camera_info['datacenter_id']
                
                # Get zones and rules for this camera
                zones = self._get_camera_zones(camera_id)
                rules = self._get_camera_rules(camera_id)
                
                # Get camera settings from metadata
                settings = camera_info.get('metadata', {})
                
                # Initialize appropriate camera model
                if camera_type in DATACENTER_CAMERA_MODEL_MAPPING:
                    model_class = DATACENTER_CAMERA_MODEL_MAPPING[camera_type]
                    
                    try:
                        camera_model = model_class(
                            camera_id=camera_id,
                            zones=zones,
                            rules=rules,
                            settings=settings,
                            db=self.database,
                            db_writer=self.db_writer,
                            frames_base_dir=Config.FRAMES_OUTPUT_DIR,
                            camera_manager=self.camera_manager
                        )
                        
                        self.camera_models[camera_id] = camera_model
                        self.logger.info(f"Initialized {camera_type} model for camera {camera_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error initializing {camera_type} model for camera {camera_id}: {str(e)}", exc_info=True)
                else:
                    self.logger.warning(f"No model mapping found for camera type: {camera_type} (camera {camera_id})")
            
            self.logger.info(f"Initialized {len(self.camera_models)} datacenter camera models")
            
        except Exception as e:
            self.logger.error(f"Error initializing camera models: {str(e)}", exc_info=True)
    
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
                        self.datacenter_coordinators[datacenter_id] = MultiCameraCoordinator(
                            datacenter_id, camera_ids, self.camera_models
                        )
                        
                        # Configure from metadata
                        self._configure_coordinator_from_metadata(datacenter_id)
                        
                        # Disable individual camera events for cameras in this coordinator
                        for camera_id in camera_ids:
                            if camera_id in self.camera_models:
                                if hasattr(self.camera_models[camera_id], 'set_individual_events_enabled'):
                                    self.camera_models[camera_id].set_individual_events_enabled(False)
                                    self.logger.info(f"Disabled individual events for camera {camera_id} (using datacenter coordinator)")
                        
                        self.logger.info(f"Created datacenter coordinator for datacenter {datacenter_id} with {len(camera_ids)} cameras")
                    except Exception as e:
                        self.logger.error(f"Error creating coordinator for datacenter {datacenter_id}: {str(e)}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"Error initializing multi-camera coordinators: {str(e)}", exc_info=True)
    
    def _configure_coordinator_from_metadata(self, datacenter_id):
        """Configure coordinator parameters from camera metadata"""
        if datacenter_id not in self.datacenter_coordinators:
            return
            
        try:
            coordinator = self.datacenter_coordinators[datacenter_id]
            
            # Look for configuration in first camera's metadata for this datacenter
            for camera_id, metadata in self.camera_metadata.items():
                if metadata.get('datacenter_id') == datacenter_id:
                    # Check for datacenter-level configuration in metadata
                    metadata_dict = metadata.get('metadata', {})
                    
                    # Extract configuration
                    occlusion_handling = metadata_dict.get('occlusion_handling', True)
                    event_cooldown = Config.EVENT_COOLDOWN
                    
                    coordinator.configure(
                        occlusion_handling=occlusion_handling,
                        event_cooldown=event_cooldown
                    )
                    break
        except Exception as e:
            self.logger.error(f"Error configuring coordinator for datacenter {datacenter_id}: {str(e)}", exc_info=True)
    
    def _setup_result_routing(self):
        """
        Setup the result routing from batch processing back to individual cameras
        by connecting the CameraManager with this processor
        """
        self.logger.info("Setting up result routing from batch processing to individual cameras")
        
        # Set up result callback for camera manager
        self.camera_manager.set_result_callback(self.process_camera_result)
        
        self.logger.info("Result routing setup complete")
    
    def process_camera_result(self, camera_id, frame, result, timestamp, enhanced_metadata=None):
        """
        Process a detection result for a specific camera
        This function will be called by the camera manager when a result is ready
        """
        try:
            # Update metadata
            if camera_id in self.camera_metadata:
                self.camera_metadata[camera_id]['last_processed'] = timestamp
                self.camera_metadata[camera_id]['frames_processed'] = self.camera_metadata[camera_id].get('frames_processed', 0) + 1
            
            # Check if this result has batch aggregation data (synchronized processing)
            batch_aggregation = enhanced_metadata.get('batch_aggregation') if enhanced_metadata else None
            
            # Process with camera model if available
            if camera_id in self.camera_models and camera_models_available:
                
                if batch_aggregation and batch_aggregation.get('tracking_completed', False):
                    # Tracking was already done at batch level, use the processed frame and data
                    processed_frame = batch_aggregation.get('processed_frame', frame)
                    people_by_zone = batch_aggregation.get('people_by_zone', {})
                    
                    self.logger.debug(f"Camera {camera_id} batch aggregation (tracking completed): "
                                   f"aggregated_count={batch_aggregation['aggregated_count']}, "
                                   f"trigger_event={batch_aggregation['trigger_event']}, "
                                   f"reason={batch_aggregation['reason']}, "
                                   f"synchronized={batch_aggregation.get('synchronized', False)}")
                    
                    # Events were already handled at batch level, just log
                    if batch_aggregation['trigger_event']:
                        self.logger.info(f"Event triggered for camera {camera_id} through batch aggregation")
                
                else:
                    # Perform individual camera processing (either no batch aggregation or tracking not completed)
                    processed_frame, people_by_zone = self.camera_models[camera_id].process_frame(frame, timestamp, result)
                    
                    if batch_aggregation:
                        # Use synchronized batch aggregation result but tracking was done individually
                        self.logger.debug(f"Camera {camera_id} batch aggregation (individual tracking): "
                                       f"aggregated_count={batch_aggregation['aggregated_count']}, "
                                       f"trigger_event={batch_aggregation['trigger_event']}, "
                                       f"reason={batch_aggregation['reason']}")
                        
                        # Events were already handled at batch level
                        if batch_aggregation['trigger_event']:
                            self.logger.info(f"Event triggered for camera {camera_id} through batch aggregation")
                    
                    else:
                        # Fallback to individual camera processing (for cameras not in datacenter coordinators)
                        people_count = self._get_camera_people_count(camera_id)
                        unauthorized_count = self._get_camera_unauthorized_count(camera_id)
                        datacenter_id = self.camera_metadata[camera_id].get('datacenter_id')
                        
                        if datacenter_id in self.datacenter_coordinators:
                            # Update datacenter-level coordinator with both counts (legacy async path)
                            coordinator = self.datacenter_coordinators[datacenter_id]
                            aggregated_result = coordinator.update_camera_count(
                                camera_id, people_count, unauthorized_count, timestamp, processed_frame, people_by_zone
                            )
                            
                            # Coordinator handles event triggering internally
                            self.logger.debug(f"Async aggregated result for datacenter {datacenter_id}: {aggregated_result}")
                        else:
                            # No coordinator for this datacenter, enable individual camera events
                            if hasattr(self.camera_models[camera_id], 'set_individual_events_enabled'):
                                self.camera_models[camera_id].set_individual_events_enabled(True)
                            
                            # Log individual processing
                            self.logger.debug(f"Individual camera processing for {camera_id}: {people_count} people, {unauthorized_count} unauthorized")
                
        except Exception as e:
            self.logger.error(f"Error processing result for camera {camera_id}: {str(e)}", exc_info=True)
    
    def _get_camera_people_count(self, camera_id):
        """Get current people count from camera model"""
        if camera_id in self.camera_models:
            camera_model = self.camera_models[camera_id]
            if hasattr(camera_model, 'get_current_people_count'):
                return camera_model.get_current_people_count()
            else:
                # Fallback: count tracked objects
                return len(getattr(camera_model, 'tracked_objects', {}))
        return 0
    
    def _get_camera_unauthorized_count(self, camera_id):
        """Get current unauthorized people count from camera model"""
        if camera_id in self.camera_models:
            camera_model = self.camera_models[camera_id]
            if hasattr(camera_model, 'get_current_unauthorized_count'):
                return camera_model.get_current_unauthorized_count()
        return 0

    async def start_monitoring(self):
        """Start monitoring all cameras"""
        self.logger.info("Starting datacenter camera monitoring")
        
        if not self.camera_feeds:
            self.logger.warning("No camera feeds found. Please ensure cameras are configured in the database.")
            return False
        
        # Make sure camera models are initialized
        if camera_models_available and not self.camera_models:
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
        """
        Start batch processing of frames from multiple cameras
        
        Args:
            camera_sources: Dict mapping camera_id to tuple of (stream_url, activity_level)
            model_name: Name of the model to use for detection
        """
        if self.batch_processing_thread and self.batch_processing_thread.is_alive():
            self.logger.warning("Batch processing already running")
            return False
            
        self.logger.info(f"Starting batch processing for {len(camera_sources)} cameras")
        
        # Start camera readers using the camera manager - Pass activity level info
        self.camera_manager.start_cameras(camera_sources)
        
        # Store camera metadata including activity level
        for camera_id, (stream_url, activity_level) in camera_sources.items():
            if camera_id not in self.camera_metadata:
                self.camera_metadata[camera_id] = {
                    'camera_id': camera_id,
                    'stream_url': stream_url,
                    'activity_level': activity_level,
                    'frames_processed': 0,
                    'last_processed': None,
                    'status': 'active'
                }
            else:
                # Update with new activity level if exists
                self.camera_metadata[camera_id]['activity_level'] = activity_level
        
        # Start batch processing thread
        self.batch_processing_running = True
        self.batch_processing_thread = threading.Thread(
            target=self._batch_processing_worker,
            args=(model_name,),
            daemon=True
        )
        self.batch_processing_thread.start()
        
        self.logger.info(f"Batch processing started for model: {model_name}")
        return True

    def stop_batch_processing(self):
        """Stop batch processing thread"""
        self.logger.info("Stopping batch processing")
        self.batch_processing_running = False
        
        if self.batch_processing_thread and self.batch_processing_thread.is_alive():
            # Wait for thread to terminate
            self.batch_processing_thread.join(timeout=5.0)
            if self.batch_processing_thread.is_alive():
                self.logger.warning("Batch processing thread did not terminate gracefully")
        
        # Stop camera readers using the camera manager
        self.camera_manager.stop_all_cameras()
        
        self.logger.info("Batch processing stopped")

    def _batch_processing_worker(self, model_name):
        """Worker thread for batch processing"""
        self.logger.info(f"Batch processing worker started for model: {model_name}")
        
        try:
            # Load model using synchronous method
            model_instance = self.model_manager.get_model(model_name)
            if not model_instance:
                self.logger.error(f"Failed to load model: {model_name}")
                return
            
            # Get number of classes for the model
            num_classes = self.get_num_classes(model_instance)
            self.logger.info(f"Model loaded with {num_classes} classes")
            
            # Initialize metrics
            frames_processed = 0
            batch_count = 0
            start_time = time.time()
            last_motion_log_time = time.time()
            
            # Process batches until stopped
            while self.batch_processing_running:
                # Get a batch of frames from the camera manager
                frames, metadata = self.camera_manager.get_batch(timeout=self.batch_timeout)
                
                if frames and metadata and len(frames) > 0:
                    batch_count += 1
                    frames_processed += len(frames)
                    
                    # Log batch processing info and motion status
                    current_time = time.time()
                    if batch_count % 10 == 0:  # Log every 10 batches
                        elapsed = current_time - start_time
                        processing_fps = frames_processed / max(0.1, elapsed)
                        self.logger.debug(
                            f"Processing batch #{batch_count} with {len(frames)} frames. "
                            f"Total FPS: {processing_fps:.1f}, Frames: {frames_processed}"
                        )
                    
                    # Log motion status periodically (every 60 seconds)
                    if Config.MOTION_DETECTION_ENABLED and (current_time - last_motion_log_time) > 60:
                        motion_summary = self.get_motion_summary()
                        self.logger.info(f"Motion Status: {motion_summary['cameras_with_motion']}/{motion_summary['total_cameras']} cameras with motion")
                        
                        # Log datacenter-specific motion status
                        for datacenter_id, datacenter_status in motion_summary['datacenters'].items():
                            if datacenter_status['has_motion']:
                                self.logger.info(f"  Datacenter {datacenter_id}: Motion detected ({datacenter_status['cameras_with_motion']}/{datacenter_status['total_cameras']} cameras)")
                            else:
                                self.logger.debug(f"  Datacenter {datacenter_id}: No motion")
                        
                        last_motion_log_time = current_time
                    
                    # Process batch - use the _process_batch_sync method
                    self._process_batch_sync(model_instance, frames, metadata)
                else:
                    # No frames available, sleep briefly to avoid CPU spinning
                    time.sleep(0.05)
                    
            # Make sure to release the model when done
            self.model_manager.release_model_sync(model_instance)
                    
        except Exception as e:
            self.logger.error(f"Error in batch processing worker: {str(e)}", exc_info=True)
        finally:
            self.logger.info("Batch processing worker stopped")

    def _process_batch_sync(self, model_instance, frames, frame_metadata):
        """Synchronous version of _process_batch"""
        try:
            # Run inference on batch
            results = model_instance.model(frames, conf=Config.PERSON_DETECTION_CONFIDENCE)
            
            # Perform batch-level aggregation for synchronized cameras
            batch_aggregation_results = self._perform_batch_aggregation(frames, results, frame_metadata)
            
            # Process each result with its metadata
            for i, (result, metadata) in enumerate(zip(results, frame_metadata)):
                camera_id = metadata['camera_id']
                timestamp = metadata['timestamp']
                
                # Update the camera metadata
                if camera_id in self.camera_metadata:
                    # Initialize frames_processed if it doesn't exist
                    if 'frames_processed' not in self.camera_metadata[camera_id]:
                        self.camera_metadata[camera_id]['frames_processed'] = 0
                    
                    # Now safely increment
                    self.camera_metadata[camera_id]['frames_processed'] += 1
                    self.camera_metadata[camera_id]['last_processed'] = timestamp
                else:
                    # Create metadata entry for this camera if it doesn't exist
                    self.camera_metadata[camera_id] = {
                        'camera_id': camera_id,
                        'frames_processed': 1,
                        'last_processed': timestamp
                    }
                
                # Include batch aggregation result for this camera
                enhanced_metadata = metadata.copy()
                if camera_id in batch_aggregation_results:
                    enhanced_metadata['batch_aggregation'] = batch_aggregation_results[camera_id]
                
                # Route result to camera through the camera manager
                self.camera_manager.route_result_to_camera(
                    camera_id=camera_id,
                    frame=frames[i],
                    result=result,
                    timestamp=timestamp,
                    enhanced_metadata=enhanced_metadata
                )
                
        except Exception as e:
            self.logger.error(f"Error processing batches: {str(e)}", exc_info=True)
    
    def _perform_batch_aggregation(self, frames, results, frame_metadata):
        """
        Perform synchronized aggregation at batch level with parallel tracking for datacenter monitoring
        
        Args:
            frames: List of frames in this batch
            results: Detection results for each frame
            frame_metadata: Metadata for each frame
            
        Returns:
            Dict mapping camera_id to aggregation results for this batch
        """
        batch_results = {}
        
        try:
            # Group cameras that need aggregation by datacenter_id
            aggregation_cameras = {}  # {datacenter_id: [(camera_id, frame_idx, frame, result, metadata), ...]}
            
            for i, (result, metadata) in enumerate(zip(results, frame_metadata)):
                camera_id = metadata['camera_id']
                datacenter_id = self.camera_metadata.get(camera_id, {}).get('datacenter_id')
                
                        # Only include cameras that have coordinators
                if (datacenter_id and datacenter_id in self.datacenter_coordinators and 
                    camera_id in self.camera_models and multi_camera_available):
                    if datacenter_id not in aggregation_cameras:
                        aggregation_cameras[datacenter_id] = []
                    
                    aggregation_cameras[datacenter_id].append((camera_id, i, frames[i], result, metadata))
            
            # Process cameras in parallel using ThreadPoolExecutor
            if aggregation_cameras:
                # Create executor for parallel camera processing
                with ThreadPoolExecutor(max_workers=self.max_parallel_cameras, thread_name_prefix="DatacenterCameraTracker") as executor:
                    
                    # Submit all camera tracking tasks in parallel
                    future_to_camera = {}
                    
                    for datacenter_id, camera_data_list in aggregation_cameras.items():
                        for camera_id, frame_idx, frame, result, metadata in camera_data_list:
                            # Submit parallel tracking task
                            future = executor.submit(
                                self._process_camera_tracking_parallel,
                                camera_id, frame_idx, frame, result, metadata
                            )
                            future_to_camera[future] = (datacenter_id, camera_id, frame_idx, metadata)
                    
                    # Collect results as they complete
                    datacenter_batches = {}
                    
                    for future in as_completed(future_to_camera):
                        datacenter_id, camera_id, frame_idx, metadata = future_to_camera[future]
                        
                        try:
                            # Get tracking result
                            tracking_result = future.result()
                            if not tracking_result:
                                continue
                                
                            processed_frame, people_count, unauthorized_count, people_by_zone = tracking_result
                            timestamp = metadata['timestamp']
                            
                            # Group by datacenter for aggregation
                            if datacenter_id not in datacenter_batches:
                                datacenter_batches[datacenter_id] = {
                                    'cameras': {},
                                    'timestamp': timestamp,
                                    'frame_indices': {},
                                    'processed_frames': {},
                                    'zone_name': None
                                }
                            
                            datacenter_batches[datacenter_id]['cameras'][camera_id] = {
                                'people_count': people_count,
                                'unauthorized_count': unauthorized_count,
                                'timestamp': timestamp,
                                'people_by_zone': people_by_zone
                            }
                            datacenter_batches[datacenter_id]['frame_indices'][camera_id] = frame_idx
                            datacenter_batches[datacenter_id]['processed_frames'][camera_id] = processed_frame
                            
                            # Store zone name (first zone found)
                            if people_by_zone and not datacenter_batches[datacenter_id]['zone_name']:
                                datacenter_batches[datacenter_id]['zone_name'] = list(people_by_zone.keys())[0]
                                
                        except Exception as e:
                            self.logger.error(f"Error processing camera {camera_id} tracking: {str(e)}", exc_info=True)
                
                # Now perform aggregation for each datacenter with completed tracking
                for datacenter_id, batch_data in datacenter_batches.items():
                    cameras = batch_data['cameras']
                    
                    # Only aggregate if we have multiple cameras in this batch
                    if len(cameras) >= 2:
                        self.logger.debug(f"Performing batch aggregation for datacenter {datacenter_id} with {len(cameras)} cameras (parallel tracking)")
                        
                        # Get the coordinator
                        coordinator = self.datacenter_coordinators[datacenter_id]
                        
                        # Apply cross-camera logic with synchronized data
                        aggregated_result = coordinator._apply_cross_camera_logic_batch(cameras, batch_data['timestamp'])
                        
                        # Store results for each camera in this datacenter
                        for camera_id in cameras:
                            batch_results[camera_id] = {
                                'aggregated_count': aggregated_result['aggregated_count'],
                                'trigger_event': aggregated_result['trigger_event'],
                                'reason': aggregated_result['reason'],
                                'individual_counts': aggregated_result.get('individual_counts', {}),
                                'synchronized': True,
                                'batch_timestamp': batch_data['timestamp'],
                                'processed_frame': batch_data['processed_frames'][camera_id],
                                'people_by_zone': cameras[camera_id]['people_by_zone'],
                                'tracking_completed': True,
                                'parallel_processed': True  # Flag indicating parallel processing
                            }
                        
                        # Handle events ONCE per batch (not per camera) if needed
                        if aggregated_result['trigger_event']:
                            # Determine which camera should handle the event based on who detected people
                            event_camera_id = self._determine_event_camera_for_batch(aggregated_result, batch_data)
                            event_frame = batch_data['processed_frames'].get(event_camera_id) if event_camera_id else None
                            
                            # Pass zone name and camera info to event handler
                            coordinator._handle_aggregated_event(
                                aggregated_result, 
                                batch_data['timestamp'], 
                                event_frame,
                                batch_data.get('zone_name'),
                                event_camera_id
                            )
                                
                        self.logger.debug(f"Parallel batch aggregation for datacenter {datacenter_id}: "
                                       f"aggregated_count={aggregated_result['aggregated_count']}, "
                                       f"trigger_event={aggregated_result['trigger_event']}, "
                                       f"reason={aggregated_result['reason']}")
        
        except Exception as e:
            self.logger.error(f"Error in parallel batch aggregation: {str(e)}", exc_info=True)
        
        return batch_results
    
    def _process_camera_tracking_parallel(self, camera_id, frame_idx, frame, result, metadata):
        """
        Process tracking for a single camera in parallel
        
        Args:
            camera_id: Camera identifier
            frame_idx: Frame index in batch
            frame: Video frame
            result: Detection result
            metadata: Frame metadata
            
        Returns:
            Tuple of (processed_frame, people_count, unauthorized_count, people_by_zone) or None
        """
        try:
            if camera_id not in self.camera_models:
                return None
                
            camera_model = self.camera_models[camera_id]
            
            # Temporarily disable individual events during parallel processing
            original_events_enabled = getattr(camera_model, 'enable_individual_events', True)
            if hasattr(camera_model, 'set_individual_events_enabled'):
                camera_model.set_individual_events_enabled(False)
            
            # Perform full tracking and detection using the appropriate method for each camera model
            if hasattr(camera_model, 'detect_people'):
                # For use case models that have detect_people method
                processed_frame, people_count, unauthorized_count, people_by_zone = camera_model.detect_people(frame, result)
            else:
                # For original datacenter models that might use different method names
                processed_frame, people_by_zone = camera_model.process_frame(frame, metadata['timestamp'], result)
                people_count = len(people_by_zone.get('all', []))
                unauthorized_count = sum(len(people) for zone, people in people_by_zone.items() if 'unauthorized' in zone.lower())
            
            # Restore individual events setting
            if hasattr(camera_model, 'set_individual_events_enabled'):
                camera_model.set_individual_events_enabled(original_events_enabled)
            
            return processed_frame, people_count, unauthorized_count, people_by_zone
            
        except Exception as e:
            self.logger.error(f"Error in parallel camera tracking for {camera_id}: {str(e)}", exc_info=True)
            return None
    
    def _determine_event_camera_for_batch(self, aggregated_result, batch_data):
        """Determine which camera should be used for saving the event in batch processing"""
        try:
            # Get individual counts from aggregated result
            individual_counts = aggregated_result.get('individual_counts', {})
            
            # Find camera with highest count (most likely to have detected the people)
            max_count = 0
            selected_camera = None
            
            for camera_id, count in individual_counts.items():
                if count > max_count and camera_id in batch_data['processed_frames']:
                    max_count = count
                    selected_camera = camera_id
            
            # If no camera found from counts, use first camera with a processed frame
            if not selected_camera:
                for camera_id in batch_data['processed_frames']:
                    selected_camera = camera_id
                    break
            
            self.logger.debug(f"Selected camera {selected_camera} for batch event (detected {max_count} people)")
            return selected_camera
            
        except Exception as e:
            self.logger.error(f"Error determining event camera for batch: {str(e)}")
            # Fallback to first camera
            camera_ids = list(batch_data.get('processed_frames', {}).keys())
            return camera_ids[0] if camera_ids else None

    def get_num_classes(self, model_instance):
        """Get number of classes from model"""
        try:
            return len(model_instance.model.names)
        except:
            self.logger.warning("Could not get number of classes from model, using default classes")
            return 0

    def get_motion_summary(self):
        """Get motion status summary for all datacenters"""
        motion_summary = {
            'datacenters': {},
            'total_cameras': 0,
            'cameras_with_motion': 0,
            'motion_detection_enabled': Config.MOTION_DETECTION_ENABLED
        }
        
        # Group cameras by datacenter
        datacenter_cameras = {}
        for camera_id, metadata in self.camera_metadata.items():
            datacenter_id = metadata.get('datacenter_id', 'unknown')
            if datacenter_id not in datacenter_cameras:
                datacenter_cameras[datacenter_id] = []
            datacenter_cameras[datacenter_id].append(camera_id)
            motion_summary['total_cameras'] += 1
        
        # Check motion status for each datacenter
        for datacenter_id, camera_ids in datacenter_cameras.items():
            datacenter_motion = False
            cameras_with_motion = 0
            
            for camera_id in camera_ids:
                reader = self.camera_manager.get_camera_reader(camera_id)
                if reader and reader.motion_detected:
                    cameras_with_motion += 1
                    datacenter_motion = True
                    motion_summary['cameras_with_motion'] += 1
            
            motion_summary['datacenters'][datacenter_id] = {
                'has_motion': datacenter_motion,
                'cameras_with_motion': cameras_with_motion,
                'total_cameras': len(camera_ids)
            }
        
        return motion_summary

    async def update_camera_activity(self, camera_id, activity_level):
        """
        Update the activity level for a specific camera.
        This will adjust the FPS based on the new activity level.
        
        Args:
            camera_id: The camera ID to update
            activity_level: New activity level ('high', 'medium', 'low')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if camera_id not in self.camera_metadata:
            self.logger.warning(f"Cannot update activity level: Camera {camera_id} not found in metadata")
            return False
        
        if activity_level not in ('high', 'medium', 'low'):
            self.logger.warning(f"Invalid activity level: {activity_level}. Must be 'high', 'medium', or 'low'.")
            return False
        
        # Update the camera's activity level in our camera manager
        success = self.camera_manager.update_camera_activity(camera_id, activity_level)
        
        if success:
            # Also update our own metadata
            self.camera_metadata[camera_id]['activity_level'] = activity_level
            self.logger.info(f"Updated activity level for camera {camera_id} to {activity_level}")
            
            # Update activity level in database
            try:
                # Get current metadata from database
                query = "SELECT metadata FROM cameras WHERE camera_id = %s"
                result = self.database.execute_query(query, (camera_id,))
                
                if result and len(result) > 0:
                    metadata = result[0]['metadata']
                    
                    # Parse metadata if it's a string
                    if isinstance(metadata, str):
                        try:
                            metadata_dict = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata_dict = {}
                    elif metadata is None:
                        metadata_dict = {}
                    else:
                        metadata_dict = metadata
                    
                    # Update activity level
                    metadata_dict['activity_level'] = activity_level
                    
                    # Save back to database
                    update_query = "UPDATE cameras SET metadata = %s WHERE camera_id = %s"
                    self.database.execute_query(update_query, (json.dumps(metadata_dict), camera_id))
                    
                    self.logger.info(f"Updated activity level in database for camera {camera_id}")
            except Exception as e:
                self.logger.error(f"Error updating activity level in database: {str(e)}", exc_info=True)
        
        return success

    def _get_camera_zones(self, camera_id):
        """
        Get zones for a specific camera from the database
        
        Args:
            camera_id: ID of the camera to get zones for
            
        Returns:
            dict: Dictionary of zones organized by type
        """
        try:
            query = """
                SELECT zone_id, name, type, polygon_coordinates, metadata 
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
        """
        Get rules for a specific camera from the database
        
        Args:
            camera_id: ID of the camera to get rules for
            
        Returns:
            list: List of rule dictionaries
        """
        try:
            query = """
                SELECT rule_id, name, description, event_type, severity, parameters, enabled
                FROM rules 
                WHERE camera_id = %s AND enabled = TRUE
            """
            rules_data = self.database.execute_query(query, (camera_id,))
            
            rules_list = []
            for rule in rules_data:
                rule_dict = {
                    'rule_id': rule['rule_id'],
                    'name': rule['name'],
                    'description': rule['description'],
                    'event_type': rule['event_type'],
                    'severity': rule['severity'],
                    'parameters': rule['parameters'],
                    'enabled': rule['enabled']
                }
                rules_list.append(rule_dict)
            
            return rules_list
        except Exception as e:
            self.logger.error(f"Error getting rules for camera {camera_id}: {str(e)}")
            return []
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        current_time = time.time()
        uptime = current_time - self.system_stats['start_time']
        
        # Get camera stats
        camera_stats = self.camera_manager.get_camera_stats()
        
        # Calculate totals
        total_frames = sum(self.camera_metadata.get(cam_id, {}).get('frames_processed', 0) for cam_id in self.camera_metadata)
        total_events = sum(model.stats.get('events_detected', 0) for model in self.camera_models.values())
        
        # Calculate average FPS
        active_cameras = len([stats for stats in camera_stats.values() if stats.get('connected', False)])
        avg_fps = sum(stats.get('fps', 0) for stats in camera_stats.values()) / max(1, active_cameras)
        
        return {
            'uptime_seconds': uptime,
            'total_frames_processed': total_frames,
            'total_events_detected': total_events,
            'cameras_active': active_cameras,
            'cameras_total': len(self.camera_metadata),
            'average_fps': avg_fps,
            'motion_summary': self.get_motion_summary() if Config.MOTION_DETECTION_ENABLED else None,
            'model_info': self.model_manager.get_model_info(),
            'storage_stats': self.storage_handler.get_storage_stats() if hasattr(self.storage_handler, 'get_storage_stats') else None,
            'camera_stats': camera_stats
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
    args = parser.parse_args()
    
    processor = DatacenterVideoProcessor(datacenter_id=args.datacenter_id)
    
    try:
        # Run monitoring
        asyncio.run(processor.start_monitoring())
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
        processor.shutdown()#!/usr/bin/env python3

'''

#!/usr/bin/env python3
"""
main.py
Main Application Entry Point for Video Monitoring System

This application provides:
1. Command-line interface for video monitoring
2. Camera feed selection and model assignment
3. Real-time detection processing
4. Event management and search
"""

import os
import sys
import argparse
import asyncio
import time
import cv2
import threading
from typing import Dict, List, Any, Optional
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import core components
from database import Database
from config import Config, DetectionModels
from model_manager import ModelManager
from camera_manager import CameraManager
from logger import setup_logger
from db_writer import DatabaseWriter
from storage_handler import StorageHandler

# Import detection scripts
from detection_scripts.tailgating_detection import TailgatingDetector
from detection_scripts.people_counting import PeopleCounter
from detection_scripts.ppe_detection import PPEDetector
from detection_scripts.unauthorized_access_detection import UnauthorizedAccessDetector
from detection_scripts.intrusion_detection import IntrusionDetector
from detection_scripts.loitering_detection import LoiteringDetector

# Global variables
running = False
monitoring_session = None

def setup_logging():
    """Setup application logging"""
    logger = setup_logger('main', 'main.log')
    return logger

class MonitoringSession:
    """Manages a monitoring session with selected cameras and models"""
    
    def __init__(self, camera_id: int, models: List[str], logger):
        self.camera_id = camera_id
        self.models = models
        self.logger = logger
        
        # Core components
        self.db = Database()
        self.model_manager = ModelManager()
        self.camera_manager = CameraManager()
        self.db_writer = DatabaseWriter()
        self.storage_handler = StorageHandler()
        
        # Detection models
        self.detectors = {}
        
        # Session tracking
        self.session_id = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'start_time': time.time()
        }
    
    def initialize(self) -> bool:
        """Initialize monitoring session"""
        try:
            self.logger.info(f"Initializing monitoring session for camera {self.camera_id}")
            self.logger.info(f"Selected models: {', '.join(self.models)}")
            
            # Get camera info
            camera_info = self.db.execute_query(
                "SELECT * FROM cameras WHERE camera_id = %s", 
                (self.camera_id,)
            )
            
            if not camera_info:
                self.logger.error(f"Camera {self.camera_id} not found in database")
                return False
            
            camera = camera_info[0]
            
            # Get camera zones
            zones = self.db.get_camera_zones(self.camera_id)
            
            # Add camera to camera manager
            self.camera_manager.add_camera(
                self.camera_id, 
                camera['stream_url'], 
                camera['name']
            )
            
            # Initialize selected detectors
            detector_settings = {
                'event_cooldown': 60,
                'frames_base_dir': Config.FRAMES_OUTPUT_DIR
            }
            
            for model_name in self.models:
                if model_name == 'tailgating':
                    self.detectors[model_name] = TailgatingDetector(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                elif model_name == 'people_counting':
                    self.detectors[model_name] = PeopleCounter(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                elif model_name == 'ppe_detection':
                    self.detectors[model_name] = PPEDetector(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                elif model_name == 'unauthorized_access':
                    self.detectors[model_name] = UnauthorizedAccessDetector(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                elif model_name == 'intrusion_detection':
                    self.detectors[model_name] = IntrusionDetector(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                elif model_name == 'loitering_detection':
                    self.detectors[model_name] = LoiteringDetector(
                        self.camera_id, zones, detector_settings, 
                        self.db, self.db_writer
                    )
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
            
            self.logger.info(f"Initialized {len(self.detectors)} detectors")
            
            # Start detection session in database
            self.session_id = self.db.start_detection_session(
                self.camera_id, self.models, f"Session_{int(time.time())}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing session: {e}")
            return False
    
    def start(self) -> bool:
        """Start monitoring session"""
        try:
            # Connect to camera
            if not self.camera_manager.connect_camera(self.camera_id):
                self.logger.error(f"Failed to connect to camera {self.camera_id}")
                return False
            
            # Start camera capture
            self.camera_manager.add_frame_callback(self._process_frame)
            if not self.camera_manager.start_camera_capture(self.camera_id):
                self.logger.error(f"Failed to start camera capture")
                return False
            
            self.is_running = True
            self.logger.info("Monitoring session started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting session: {e}")
            return False
    
    def stop(self):
        """Stop monitoring session"""
        try:
            self.is_running = False
            
            # Stop camera
            self.camera_manager.stop_all_captures()
            self.camera_manager.cleanup()
            
            # End database session
            if self.session_id:
                self.db.end_detection_session(self.session_id, 'completed')
            
            self.logger.info("Monitoring session stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping session: {e}")
    
    def _process_frame(self, camera_id: int, frame, timestamp: float):
        """Process frame with selected detection models"""
        if not self.is_running or camera_id != self.camera_id:
            return
        
        try:
            # Run person detection
            detections = self.model_manager.detect_persons(frame)
            
            # Process frame with each selected detector
            annotated_frame = frame.copy()
            all_events = []
            
            for model_name, detector in self.detectors.items():
                try:
                    # Process frame
                    model_frame, events = detector.process_frame(frame, timestamp, detections)
                    
                    # Combine events
                    all_events.extend(events)
                    
                    # Use the last annotated frame (could be improved to combine annotations)
                    annotated_frame = model_frame
                    
                except Exception as e:
                    self.logger.error(f"Error in {model_name} detector: {e}")
            
            # Update statistics
            self.stats['frames_processed'] += 1
            if all_events:
                self.stats['events_detected'] += len(all_events)
            
            # Display frame (optional - for local monitoring)
            if os.getenv('DISPLAY_VIDEO', 'False').lower() == 'true':
                cv2.imshow(f'Camera {camera_id} - Monitoring', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        detector_stats = {}
        for model_name, detector in self.detectors.items():
            detector_stats[model_name] = detector.get_statistics()
        
        return {
            'session_id': self.session_id,
            'camera_id': self.camera_id,
            'models': self.models,
            'runtime_seconds': runtime,
            'frames_processed': self.stats['frames_processed'],
            'events_detected': self.stats['events_detected'],
            'processing_fps': fps,
            'detector_stats': detector_stats,
            'is_running': self.is_running
        }

async def list_cameras():
    """List available cameras"""
    try:
        db = Database()
        cameras = db.get_cameras()
        
        if not cameras:
            print("No cameras found in the database.")
            return
        
        print("\nAvailable Cameras:")
        print("-" * 80)
        print(f"{'ID':<4} {'Name':<25} {'Location':<20} {'Status':<10}")
        print("-" * 80)
        
        for camera in cameras:
            print(f"{camera['camera_id']:<4} {camera['name']:<25} "
                  f"{camera.get('location', 'N/A'):<20} {camera['status']:<10}")
        
        print("-" * 80)
        print(f"Total: {len(cameras)} cameras")
        
    except Exception as e:
        print(f"Error listing cameras: {e}")

async def list_models():
    """List available detection models"""
    print("\nAvailable Detection Models:")
    print("-" * 80)
    
    models = DetectionModels.get_all_models()
    for model_name, model_info in models.items():
        print(f"\nModel: {model_name}")
        print(f"  Name: {model_info['name']}")
        print(f"  Description: {model_info['description']}")
        print(f"  Requirements: {', '.join(model_info['requires'])}")
        
        # Display parameters
        params = model_info.get('parameters', {})
        if params:
            print("  Parameters:")
            for param_name, param_value in params.items():
                print(f"    {param_name}: {param_value}")
    
    print("-" * 80)
    print(f"Total: {len(models)} models available")

async def show_camera_details(camera_id: int):
    """Show detailed information about a camera"""
    try:
        db = Database()
        
        # Get camera info
        camera_info = db.execute_query(
            "SELECT * FROM cameras WHERE camera_id = %s", 
            (camera_id,)
        )
        
        if not camera_info:
            print(f"Camera {camera_id} not found.")
            return
        
        camera = camera_info[0]
        
        # Get zones
        zones = db.get_camera_zones(camera_id)
        
        # Get recent events
        recent_events = db.search_events(camera_id=camera_id, limit=10)
        
        print(f"\nCamera Details - ID: {camera_id}")
        print("=" * 60)
        print(f"Name: {camera['name']}")
        print(f"Description: {camera.get('description', 'N/A')}")
        print(f"Location: {camera.get('location', 'N/A')}")
        print(f"Status: {camera['status']}")
        print(f"Stream URL: {camera['stream_url']}")
        print(f"Created: {camera['created_at']}")
        
        if camera.get('metadata'):
            print(f"Metadata: {camera['metadata']}")
        
        print(f"\nZones ({len(zones)}):")
        for zone in zones:
            print(f"  - {zone['name']} ({zone['zone_type']})")
            if zone.get('description'):
                print(f"    Description: {zone['description']}")
        
        print(f"\nRecent Events ({len(recent_events)}):")
        for event in recent_events:
            print(f"  - {event['timestamp']} | {event['event_type']} | {event['severity']} | {event['description']}")
        
    except Exception as e:
        print(f"Error getting camera details: {e}")

async def search_events(camera_id: int = None, event_type: str = None, 
                       severity: str = None, limit: int = 50):
    """Search events with filters"""
    try:
        db = Database()
        events = db.search_events(camera_id, event_type, severity, limit)
        
        if not events:
            print("No events found matching the criteria.")
            return
        
        print(f"\nFound {len(events)} events:")
        print("-" * 120)
        print(f"{'Timestamp':<20} {'Camera':<8} {'Type':<18} {'Severity':<10} {'Description':<50}")
        print("-" * 120)
        
        for event in events:
            print(f"{str(event['timestamp']):<20} {event['camera_id']:<8} "
                  f"{event['event_type']:<18} {event['severity']:<10} "
                  f"{event['description'][:48]:<50}")
        
        print("-" * 120)
        
    except Exception as e:
        print(f"Error searching events: {e}")

async def run_monitoring_session(camera_id: int, models: List[str]):
    """Run interactive monitoring session"""
    global running, monitoring_session
    
    logger = setup_logging()
    
    try:
        # Validate camera
        db = Database()
        camera_check = db.execute_query(
            "SELECT name FROM cameras WHERE camera_id = %s AND status = 'active'",
            (camera_id,)
        )
        
        if not camera_check:
            print(f"Error: Camera {camera_id} not found or inactive")
            return False
        
        camera_name = camera_check[0]['name']
        print(f"Starting monitoring for: {camera_name} (ID: {camera_id})")
        print(f"Selected models: {', '.join(models)}")
        
        # Create monitoring session
        monitoring_session = MonitoringSession(camera_id, models, logger)
        
        # Initialize session
        if not monitoring_session.initialize():
            print("Failed to initialize monitoring session")
            return False
        
        # Start monitoring
        if not monitoring_session.start():
            print("Failed to start monitoring session")
            return False
        
        running = True
        print("\nMonitoring started! Press 'q' to stop, 's' for statistics.")
        
        # Main monitoring loop
        while running:
            try:
                # Check for user input (non-blocking)
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.read(1).strip()
                    if user_input.lower() == 'q':
                        print("\nStopping monitoring...")
                        running = False
                        break
                    elif user_input.lower() == 's':
                        stats = monitoring_session.get_statistics()
                        print(f"\nSession Statistics:")
                        print(f"  Runtime: {stats['runtime_seconds']:.1f}s")
                        print(f"  Frames processed: {stats['frames_processed']}")
                        print(f"  Events detected: {stats['events_detected']}")
                        print(f"  Processing FPS: {stats['processing_fps']:.2f}")
                
                # Small delay
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\nReceived interrupt signal, stopping...")
                running = False
                break
        
        # Stop monitoring
        monitoring_session.stop()
        
        # Show final statistics
        final_stats = monitoring_session.get_statistics()
        print(f"\nSession completed:")
        print(f"  Total runtime: {final_stats['runtime_seconds']:.1f} seconds")
        print(f"  Frames processed: {final_stats['frames_processed']}")
        print(f"  Events detected: {final_stats['events_detected']}")
        print(f"  Average FPS: {final_stats['processing_fps']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in monitoring session: {e}")
        if monitoring_session:
            monitoring_session.stop()
        return False

async def health_check():
    """Perform system health check"""
    print("Performing system health check...")
    print("=" * 50)
    
    # Check database connection
    print("1. Testing database connection...")
    try:
        db = Database()
        test_result = db.execute_query("SELECT 1 as test")
        if test_result and test_result[0]['test'] == 1:
            print("   ✓ Database connection successful")
        else:
            print("   ✗ Database query failed")
    except Exception as e:
        print(f"   ✗ Database connection error: {e}")
    
    # Check model files
    print("2. Testing AI models...")
    try:
        if os.path.exists(Config.DETECTION_MODEL_PATH):
            print(f"   ✓ Person detection model found: {Config.DETECTION_MODEL_PATH}")
        else:
            print(f"   ✗ Person detection model missing: {Config.DETECTION_MODEL_PATH}")
        
        if os.path.exists(Config.PPE_DETECTION_MODEL_PATH):
            print(f"   ✓ PPE detection model found: {Config.PPE_DETECTION_MODEL_PATH}")
        else:
            print(f"   ⚠ PPE detection model missing: {Config.PPE_DETECTION_MODEL_PATH}")
    except Exception as e:
        print(f"   ✗ Model path error: {e}")
    
    # Test model loading
    print("3. Testing model loading...")
    try:
        model_manager = ModelManager()
        model_info = model_manager.get_model_info()
        print(f"   ✓ Model manager initialized on {model_info['device']}")
        print(f"   ✓ Person model available: {model_info['person_model_available']}")
        print(f"   ⚠ PPE model available: {model_info['ppe_model_available']}")
    except Exception as e:
        print(f"   ✗ Model loading error: {e}")
    
    # Check output directories
    print("4. Testing output directories...")
    try:
        os.makedirs(Config.FRAMES_OUTPUT_DIR, exist_ok=True)
        print(f"   ✓ Frames directory: {Config.FRAMES_OUTPUT_DIR}")
        
        os.makedirs("logs", exist_ok=True)
        print(f"   ✓ Logs directory: logs/")
    except Exception as e:
        print(f"   ✗ Directory creation error: {e}")
    
    # Test camera connections (sample)
    print("5. Testing sample camera connections...")
    try:
        db = Database()
        cameras = db.get_cameras()
        if cameras:
            print(f"   ✓ Found {len(cameras)} cameras in database")
            
            # Test connection to first camera
            test_camera = cameras[0]
            print(f"   Testing connection to: {test_camera['name']}")
            
            camera_manager = CameraManager()
            camera_manager.add_camera(test_camera['camera_id'], test_camera['stream_url'], test_camera['name'])
            
            if camera_manager.connect_camera(test_camera['camera_id']):
                print("   ✓ Sample camera connection successful")
                camera_manager.disconnect_camera(test_camera['camera_id'])
            else:
                print("   ⚠ Sample camera connection failed (check stream URL)")
            
            camera_manager.cleanup()
        else:
            print("   ⚠ No cameras found in database")
    except Exception as e:
        print(f"   ✗ Camera connection test error: {e}")
    
    print("=" * 50)
    print("Health check completed!")

async def init_sample_data():
    """Initialize sample data"""
    try:
        print("Initializing sample data...")
        
        # Check if data exists
        db = Database()
        existing_cameras = db.execute_query("SELECT COUNT(*) as count FROM cameras")
        
        if existing_cameras and existing_cameras[0]['count'] > 0:
            response = input("Sample data already exists. Recreate? (y/N): ")
            if response.lower() != 'y':
                print("Sample data initialization cancelled.")
                return
        
        # Load and execute setup script
        script_path = "db_setup.sql"
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                sql_script = f.read()
            
            db.execute_script(sql_script)
            print("✓ Sample data initialized successfully!")
            
            # Show what was created
            cameras = db.execute_query("SELECT COUNT(*) as count FROM cameras")
            zones = db.execute_query("SELECT COUNT(*) as count FROM zones") 
            events = db.execute_query("SELECT COUNT(*) as count FROM events")
            
            print(f"Created: {cameras[0]['count']} cameras, {zones[0]['count']} zones, {events[0]['count']} events")
        else:
            print(f"Setup script not found: {script_path}")
    
    except Exception as e:
        print(f"Error initializing sample data: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running, monitoring_session
    
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    
    if monitoring_session:
        monitoring_session.stop()
    
    sys.exit(0)

async def main():
    """Main application entry point"""
    import signal
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Video Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py list-cameras                    # List all cameras
  python main.py list-models                     # List detection models  
  python main.py monitor --camera 1 --models tailgating people_counting
  python main.py camera-details 1               # Show camera details
  python main.py search-events --camera 1       # Search events
  python main.py health-check                   # System health check
  python main.py init-data                      # Initialize sample data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List cameras
    list_cameras_parser = subparsers.add_parser('list-cameras', help='List available cameras')
    
    # List models
    list_models_parser = subparsers.add_parser('list-models', help='List detection models')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring')
    monitor_parser.add_argument('--camera', type=int, required=True, help='Camera ID to monitor')
    monitor_parser.add_argument('--models', nargs='+', required=True, 
                              choices=['tailgating', 'unauthorized_access', 'people_counting', 
                                     'ppe_detection', 'intrusion_detection', 'loitering_detection'],
                              help='Detection models to use')
    
    # Camera details
    details_parser = subparsers.add_parser('camera-details', help='Show camera details')
    details_parser.add_argument('camera_id', type=int, help='Camera ID')
    
    # Search events
    search_parser = subparsers.add_parser('search-events', help='Search events')
    search_parser.add_argument('--camera', type=int, help='Filter by camera ID')
    search_parser.add_argument('--type', type=str, help='Filter by event type')
    search_parser.add_argument('--severity', type=str, help='Filter by severity')
    search_parser.add_argument('--limit', type=int, default=50, help='Maximum results')
    
    # Health check
    health_parser = subparsers.add_parser('health-check', help='System health check')
    
    # Initialize data
    init_parser = subparsers.add_parser('init-data', help='Initialize sample data')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'list-cameras':
        await list_cameras()
    
    elif args.command == 'list-models':
        await list_models()
    
    elif args.command == 'monitor':
        await run_monitoring_session(args.camera, args.models)
    
    elif args.command == 'camera-details':
        await show_camera_details(args.camera_id)
    
    elif args.command == 'search-events':
        await search_events(args.camera, args.type, args.severity, args.limit)
    
    elif args.command == 'health-check':
        await health_check()
    
    elif args.command == 'init-data':
        await init_sample_data()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
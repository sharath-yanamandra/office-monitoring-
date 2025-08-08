#!/usr/bin/env python3
"""
Script 16: main.py
File Path: src/main.py

Datacenter Monitoring System - Main Application Entry Point

This module provides:
1. Main application entry point and command-line interface
2. System initialization and configuration
3. Monitoring system lifecycle management
4. Administrative commands and utilities
5. Health checks and system status reporting
6. Graceful shutdown handling
"""

import os
import sys
import argparse
import asyncio
import signal
import logging
import time
from typing import Optional, Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import core components
from video_processor import DatacenterVideoProcessor
from database import DatacenterDatabase
from config import DatacenterConfig
from logger import setup_datacenter_logger, audit_logger

# Global variables for signal handling
processor = None
running = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running, processor
    
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    
    if processor:
        try:
            processor.stop_monitoring()
        except Exception as e:
            print(f"Error during shutdown: {e}")
    
    # Log shutdown
    audit_logger.log_system_event(
        component='main_application',
        event='shutdown_signal_received',
        status='graceful_shutdown',
        details={'signal': signum}
    )
    
    sys.exit(0)

async def list_datacenters():
    """List all available datacenters for monitoring"""
    try:
        db = DatacenterDatabase()
        
        query = """
            SELECT 
                d.datacenter_id,
                d.name,
                d.description,
                d.location,
                d.facility_type,
                d.status,
                COUNT(c.camera_id) as camera_count
            FROM datacenters d
            LEFT JOIN cameras c ON d.datacenter_id = c.datacenter_id AND c.status = 'active'
            WHERE d.status = 'active'
            GROUP BY d.datacenter_id, d.name, d.description, d.location, d.facility_type, d.status
            ORDER BY d.datacenter_id
        """
        
        datacenters = db.execute_query(query)
        
        if not datacenters:
            print("No active datacenters found in the database")
            return True
        
        print("\nAvailable Datacenters for Monitoring:")
        print("=" * 100)
        print(f"{'ID':<4} {'Name':<30} {'Location':<25} {'Type':<10} {'Cameras':<8} {'Status':<10}")
        print("-" * 100)
        
        for dc in datacenters:
            print(f"{dc['datacenter_id']:<4} {dc['name']:<30} {dc['location']:<25} "
                  f"{dc['facility_type']:<10} {dc['camera_count']:<8} {dc['status']:<10}")
        
        print("-" * 100)
        print(f"Total: {len(datacenters)} active datacenters")
        print("\nTo monitor a specific datacenter, use:")
        print("python main.py monitor --datacenter-id <DATACENTER_ID>")
        print("\nTo monitor all datacenters, use:")
        print("python main.py monitor")
        
        return True
        
    except Exception as e:
        print(f"Error listing datacenters: {e}")
        return False

async def show_datacenter_details(datacenter_id: int):
    """Show detailed information about a specific datacenter"""
    try:
        db = DatacenterDatabase()
        
        # Get datacenter info
        dc_query = """
            SELECT 
                d.datacenter_id, d.name, d.description, d.location, d.address,
                d.facility_type, d.capacity_info, d.contact_info, d.status,
                d.created_at, d.updated_at
            FROM datacenters d
            WHERE d.datacenter_id = %s
        """
        
        datacenter = db.execute_query(dc_query, (datacenter_id,))
        
        if not datacenter:
            print(f"Datacenter with ID {datacenter_id} not found")
            return False
        
        dc = datacenter[0]
        
        # Get cameras
        cameras_query = """
            SELECT 
                c.camera_id, c.name, c.camera_type, c.stream_url, c.status,
                c.location_details, c.installation_date, c.last_maintenance
            FROM cameras c
            WHERE c.datacenter_id = %s
            ORDER BY c.camera_type, c.name
        """
        
        cameras = db.execute_query(cameras_query, (datacenter_id,))
        
        # Get recent events
        events_query = """
            SELECT 
                e.event_id, e.event_type, e.severity, e.timestamp, e.status,
                c.name as camera_name, z.name as zone_name
            FROM events e
            JOIN cameras c ON e.camera_id = c.camera_id
            LEFT JOIN zones z ON e.zone_id = z.zone_id
            WHERE c.datacenter_id = %s
            AND e.timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            ORDER BY e.timestamp DESC
            LIMIT 10
        """
        
        recent_events = db.execute_query(events_query, (datacenter_id,))
        
        # Display datacenter details
        print(f"\nDatacenter Details: {dc['name']}")
        print("=" * 60)
        print(f"ID: {dc['datacenter_id']}")
        print(f"Location: {dc['location']}")
        print(f"Address: {dc['address']}")
        print(f"Facility Type: {dc['facility_type']}")
        print(f"Status: {dc['status']}")
        print(f"Created: {dc['created_at']}")
        
        # Display capacity info if available
        if dc['capacity_info']:
            print(f"\nCapacity Information:")
            try:
                import json
                capacity = json.loads(dc['capacity_info']) if isinstance(dc['capacity_info'], str) else dc['capacity_info']
                for key, value in capacity.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            except:
                print(f"  {dc['capacity_info']}")
        
        # Display cameras
        print(f"\nCameras ({len(cameras)}):")
        print("-" * 60)
        if cameras:
            for camera in cameras:
                print(f"  {camera['camera_id']}: {camera['name']} ({camera['camera_type']})")
                print(f"    Status: {camera['status']}")
                print(f"    Stream: {camera['stream_url']}")
                if camera['installation_date']:
                    print(f"    Installed: {camera['installation_date']}")
                print()
        else:
            print("  No cameras configured")
        
        # Display recent events
        print(f"\nRecent Events (Last 24 Hours - {len(recent_events)}):")
        print("-" * 60)
        if recent_events:
            for event in recent_events:
                print(f"  {event['timestamp']}: {event['event_type']} ({event['severity']})")
                print(f"    Camera: {event['camera_name']}")
                if event['zone_name']:
                    print(f"    Zone: {event['zone_name']}")
                print(f"    Status: {event['status']}")
                print()
        else:
            print("  No recent events")
        
        return True
        
    except Exception as e:
        print(f"Error showing datacenter details: {e}")
        return False

async def test_system_health():
    """Test system health and connectivity"""
    try:
        print("Testing Datacenter Monitoring System Health...")
        print("=" * 50)
        
        # Test database connectivity
        print("1. Testing database connectivity...")
        try:
            db = DatacenterDatabase()
            result = db.execute_query("SELECT COUNT(*) as count FROM datacenters")
            datacenter_count = result[0]['count'] if result else 0
            print(f"   ✓ Database connected - {datacenter_count} datacenters found")
        except Exception as e:
            print(f"   ✗ Database connection failed: {e}")
            return False
        
        # Test configuration
        print("2. Testing configuration...")
        try:
            print(f"   ✓ Batch size: {DatacenterConfig.BATCH_SIZE}")
            print(f"   ✓ FPS limits: High={DatacenterConfig.ACTIVITY_LEVEL_HIGH}, "
                  f"Med={DatacenterConfig.ACTIVITY_LEVEL_MEDIUM}, Low={DatacenterConfig.ACTIVITY_LEVEL_LOW}")
            print(f"   ✓ PPE detection: {'Enabled' if DatacenterConfig.PPE_DETECTION_ENABLED else 'Disabled'}")
            print(f"   ✓ SMS alerts: {'Enabled' if DatacenterConfig.SMS_ENABLED else 'Disabled'}")
        except Exception as e:
            print(f"   ✗ Configuration error: {e}")
            return False
        
        # Test model paths
        print("3. Testing model files...")
        try:
            if os.path.exists(DatacenterConfig.DETECTION_MODEL_PATH):
                print(f"   ✓ Detection model found: {DatacenterConfig.DETECTION_MODEL_PATH}")
            else:
                print(f"   ✗ Detection model missing: {DatacenterConfig.DETECTION_MODEL_PATH}")
            
            if DatacenterConfig.PPE_DETECTION_ENABLED:
                if os.path.exists(DatacenterConfig.PPE_DETECTION_MODEL_PATH):
                    print(f"   ✓ PPE model found: {DatacenterConfig.PPE_DETECTION_MODEL_PATH}")
                else:
                    print(f"   ✗ PPE model missing: {DatacenterConfig.PPE_DETECTION_MODEL_PATH}")
        except Exception as e:
            print(f"   ✗ Model path error: {e}")
        
        # Test output directories
        print("4. Testing output directories...")
        try:
            os.makedirs(DatacenterConfig.FRAMES_OUTPUT_DIR, exist_ok=True)
            print(f"   ✓ Frames directory: {DatacenterConfig.FRAMES_OUTPUT_DIR}")
            
            os.makedirs("logs", exist_ok=True)
            print(f"   ✓ Logs directory: logs/")
        except Exception as e:
            print(f"   ✗ Directory creation error: {e}")
        
        # Test GPU availability
        print("5. Testing GPU availability...")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   ✓ GPU available: {gpu_name} (Count: {gpu_count})")
            else:
                print("   ⚠ GPU not available - will use CPU (slower)")
        except ImportError:
            print("   ⚠ PyTorch not installed - cannot check GPU")
        except Exception as e:
            print(f"   ✗ GPU check error: {e}")
        
        print("\n" + "=" * 50)
        print("Health check completed!")
        return True
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

async def initialize_sample_data():
    """Initialize sample data for testing"""
    try:
        print("Initializing sample data...")
        
        db = DatacenterDatabase()
        
        # Check if sample data already exists
        existing_data = db.execute_query("SELECT COUNT(*) as count FROM datacenters")
        if existing_data and existing_data[0]['count'] > 0:
            print("Sample data already exists. Use --force to recreate.")
            return True
        
        # Load and execute sample data script
        script_path = "setup_datacenter_config.sql"
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                sql_script = f.read()
            
            db.execute_script(sql_script)
            print("Sample data initialized successfully!")
            
            # Verify initialization
            datacenters = db.execute_query("SELECT COUNT(*) as count FROM datacenters")
            cameras = db.execute_query("SELECT COUNT(*) as count FROM cameras")
            zones = db.execute_query("SELECT COUNT(*) as count FROM zones")
            
            print(f"Created: {datacenters[0]['count']} datacenters, "
                  f"{cameras[0]['count']} cameras, {zones[0]['count']} zones")
            
            return True
        else:
            print(f"Sample data script not found: {script_path}")
            return False
        
    except Exception as e:
        print(f"Error initializing sample data: {e}")
        return False

async def main_monitor(args):
    """Main monitoring function"""
    global processor, running
    
    # Initialize logger
    logger = setup_datacenter_logger('datacenter_main', 'datacenter_main.log')
    
    try:
        # Log startup
        audit_logger.log_system_event(
            component='main_application',
            event='startup_initiated',
            status='starting',
            details={'datacenter_id': args.datacenter_id, 'args': vars(args)}
        )
        
        # Validate datacenter if specified
        if args.datacenter_id:
            db = DatacenterDatabase()
            datacenter_check = db.execute_query(
                "SELECT name FROM datacenters WHERE datacenter_id = %s AND status = 'active'",
                (args.datacenter_id,)
            )
            
            if not datacenter_check:
                print(f"Error: Datacenter {args.datacenter_id} not found or inactive")
                print("Use 'python main.py list-datacenters' to see available datacenters")
                return False
            
            datacenter_name = datacenter_check[0]['name']
            print(f"Monitoring datacenter: {datacenter_name} (ID: {args.datacenter_id})")
        else:
            print("Monitoring all active datacenters")
        
        # Initialize video processor
        print("Initializing video processor...")
        processor = DatacenterVideoProcessor(datacenter_id=args.datacenter_id)
        
        # Start monitoring
        print("Starting monitoring system...")
        success = await processor.start_monitoring()
        
        if not success:
            print("Failed to start monitoring system")
            return False
        
        # Log successful startup
        audit_logger.log_system_event(
            component='main_application',
            event='startup_completed',
            status='running',
            details={'datacenter_id': args.datacenter_id}
        )
        
        print("Monitoring system started successfully!")
        print("Press Ctrl+C to stop monitoring...")
        
        # Main monitoring loop
        running = True
        last_stats_time = time.time()
        
        while running:
            try:
                # Print periodic status updates
                current_time = time.time()
                if current_time - last_stats_time >= 30:  # Every 30 seconds
                    stats = processor.get_system_stats()
                    
                    print(f"\n--- System Status ---")
                    print(f"Uptime: {stats['system']['uptime_seconds']:.0f}s")
                    print(f"Frames processed: {stats['system']['total_frames_processed']}")
                    print(f"Events detected: {stats['system']['total_events_detected']}")
                    print(f"Active cameras: {stats['system']['cameras_active']}")
                    print(f"Average FPS: {stats['system']['average_fps']:.2f}")
                    
                    last_stats_time = current_time
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\nShutdown requested by user...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in main monitoring: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        return False
    
    finally:
        # Cleanup
        if processor:
            print("Stopping monitoring system...")
            processor.stop_monitoring()
            
        # Log shutdown
        audit_logger.log_system_event(
            component='main_application',
            event='shutdown_completed',
            status='stopped',
            details={}
        )
        
        print("Monitoring system stopped.")

async def main():
    """Main application entry point"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Datacenter Security Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py monitor                     # Monitor all datacenters
  python main.py monitor --datacenter-id 1  # Monitor specific datacenter
  python main.py list-datacenters          # List available datacenters
  python main.py health-check              # Test system health
  python main.py init-data                 # Initialize sample data
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring system')
    monitor_parser.add_argument('--datacenter-id', type=int, default=None,
                               help='Monitor specific datacenter (default: all)')
    monitor_parser.add_argument('--log-level', type=str, default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='Logging level')
    monitor_parser.add_argument('--batch-size', type=int, default=None,
                               help='Override batch size for GPU processing')
    monitor_parser.add_argument('--fps-limit', type=int, default=None,
                               help='Override FPS limit for cameras')
    
    # List datacenters command
    list_parser = subparsers.add_parser('list-datacenters', help='List available datacenters')
    
    # Datacenter details command
    details_parser = subparsers.add_parser('datacenter-details', help='Show datacenter details')
    details_parser.add_argument('datacenter_id', type=int, help='Datacenter ID')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Test system health')
    
    # Initialize data command
    init_parser = subparsers.add_parser('init-data', help='Initialize sample data')
    init_parser.add_argument('--force', action='store_true', help='Force recreate existing data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Handle commands
    if args.command == 'monitor' or args.command is None:
        # Default to monitor if no command specified
        if args.command is None:
            args.datacenter_id = None
            args.log_level = 'INFO'
            args.batch_size = None
            args.fps_limit = None
        
        # Override config with command line arguments
        if args.batch_size:
            DatacenterConfig.BATCH_SIZE = args.batch_size
        if args.fps_limit:
            DatacenterConfig.READER_FPS_LIMIT = args.fps_limit
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Start monitoring
        success = await main_monitor(args)
        sys.exit(0 if success else 1)
        
    elif args.command == 'list-datacenters':
        success = await list_datacenters()
        sys.exit(0 if success else 1)
        
    elif args.command == 'datacenter-details':
        success = await show_datacenter_details(args.datacenter_id)
        sys.exit(0 if success else 1)
        
    elif args.command == 'health-check':
        success = await test_system_health()
        sys.exit(0 if success else 1)
        
    elif args.command == 'init-data':
        success = await initialize_sample_data()
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
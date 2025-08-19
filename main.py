#!/usr/bin/env python3
"""
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
import json
import threading
from typing import Optional, Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import core components
from video_processor import DatacenterVideoProcessor
from database import DatacenterDatabase
from config import Config
from logger import setup_datacenter_logger, audit_logger, get_main_logger

# Global variables for signal handling
processor = None
running = False
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully - FIXED VERSION"""
    global running, processor, shutdown_event
    
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    shutdown_event.set()
    
    # Log shutdown
    audit_logger.log_system_event(
        component='main_application',
        event='shutdown_signal_received',
        status='graceful_shutdown',
        details={'signal': signum}
    )
    
    # Don't call processor.shutdown() here - let the main loop handle it
    # This avoids the "asyncio.run() cannot be called from a running event loop" error

async def graceful_shutdown(processor):
    """Perform graceful shutdown of the processor"""
    if processor:
        try:
            print("Stopping monitoring system...")
            # Use the async method directly since we're in an async context
            await processor.stop_monitoring()
            print("Monitoring system stopped.")
        except Exception as e:
            print(f"Error during shutdown: {e}")

async def get_datacenter_info(datacenter_id: int) -> Optional[Dict[str, Any]]:
    """Get datacenter information from database"""
    try:
        db = DatacenterDatabase()
        
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT datacenter_id, name, description, location, facility_type, status
                FROM datacenters 
                WHERE datacenter_id = %s AND status = 'active'
            """, (datacenter_id,))
            
            return cursor.fetchone()
            
    except Exception as e:
        logger = get_main_logger()
        logger.error(f"Error getting datacenter info: {e}")
        return None

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
        
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            datacenters = cursor.fetchall()
        
        if not datacenters:
            print("No active datacenters found.")
            return False
        
        print("\n=== Available Datacenters ===")
        print(f"{'ID':<4} {'Name':<25} {'Location':<20} {'Type':<15} {'Cameras':<8}")
        print("-" * 80)
        
        for dc in datacenters:
            print(f"{dc['datacenter_id']:<4} {dc['name']:<25} {dc['location']:<20} "
                  f"{dc['facility_type']:<15} {dc['camera_count']:<8}")
        
        print(f"\nTotal: {len(datacenters)} datacenters")
        return True
        
    except Exception as e:
        print(f"Error listing datacenters: {e}")
        return False

async def show_datacenter_details(datacenter_id: int):
    """Show detailed information about a specific datacenter"""
    try:
        db = DatacenterDatabase()
        
        # Get datacenter info
        datacenter_info = await get_datacenter_info(datacenter_id)
        if not datacenter_info:
            print(f"Datacenter {datacenter_id} not found")
            return False
        
        # Get camera details
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT camera_id, name, stream_url, camera_type, location_details, status
                FROM cameras 
                WHERE datacenter_id = %s
                ORDER BY camera_id
            """, (datacenter_id,))
            cameras = cursor.fetchall()
        
        print(f"\n=== Datacenter Details: {datacenter_info['name']} ===")
        print(f"ID: {datacenter_info['datacenter_id']}")
        print(f"Description: {datacenter_info['description']}")
        print(f"Location: {datacenter_info['location']}")
        print(f"Type: {datacenter_info['facility_type']}")
        print(f"Status: {datacenter_info['status']}")
        
        print(f"\n--- Cameras ({len(cameras)}) ---")
        if cameras:
            print(f"{'ID':<4} {'Name':<20} {'Type':<15} {'Status':<10} {'Location':<20}")
            print("-" * 80)
            for cam in cameras:
                print(f"{cam['camera_id']:<4} {cam['name']:<20} {cam['camera_type']:<15} "
                      f"{cam['status']:<10} {cam['location_details']:<20}")
        else:
            print("No cameras configured")
        
        return True
        
    except Exception as e:
        print(f"Error showing datacenter details: {e}")
        return False

async def test_system_health():
    """Test system health and connectivity"""
    print("=== System Health Check ===")
    
    # Test database connection
    try:
        db = DatacenterDatabase()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        print("✓ Database connection: OK")
    except Exception as e:
        print(f"✗ Database connection: FAILED - {e}")
        return False
    
    # Test configuration
    try:
        config_ok = True
        if not hasattr(Config, 'BATCH_SIZE'):
            print("✗ Configuration: Missing BATCH_SIZE")
            config_ok = False
        if not hasattr(Config, 'READER_FPS_LIMIT'):
            print("✗ Configuration: Missing READER_FPS_LIMIT")
            config_ok = False
        
        if config_ok:
            print("✓ Configuration: OK")
    except Exception as e:
        print(f"✗ Configuration: FAILED - {e}")
        return False
    
    # Test logging
    try:
        logger = get_main_logger()
        logger.info("Health check test log")
        print("✓ Logging system: OK")
    except Exception as e:
        print(f"✗ Logging system: FAILED - {e}")
        return False
    
    print("\n=== Health Check Complete ===")
    return True

async def initialize_sample_data(force=False):
    """Initialize sample datacenter and camera data"""
    try:
        db = DatacenterDatabase()
        
        # Check if data already exists
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM datacenters")
            dc_count = cursor.fetchone()[0]
            
            if dc_count > 0 and not force:
                print(f"Sample data already exists ({dc_count} datacenters). Use --force to recreate.")
                return True
        
        print("Initializing sample data...")
        
        # Sample datacenter data
        sample_datacenters = [
            {
                'name': 'Your Office Datacenter',
                'description': 'Main office datacenter facility',
                'location': 'Office Building A',
                'facility_type': 'office',
                'status': 'active'
            }
        ]
        
        # Sample camera data (using your RTSP URL)
        sample_cameras = [
            {
                'datacenter_id': 1,
                'name': 'Office Entrance',
                'stream_url': 'rtsp://admin:password@192.168.29.212:554/ch0_0.264',
                'camera_type': 'security',
                'location_details': 'Main entrance area',
                'status': 'active',
                'metadata': json.dumps({
                    'use_cases': ['people_counting'],
                    'activity_level': 'medium'
                })
            },
            {
                'datacenter_id': 1,
                'name': 'Server Room Monitor',
                'stream_url': 'rtsp://admin:password@192.168.29.212:554/ch0_0.264',
                'camera_type': 'security',
                'location_details': 'Server room access point',
                'status': 'active',
                'metadata': json.dumps({
                    'use_cases': ['people_counting', 'ppe_detection', 'tailgating', 'intrusion', 'loitering'],
                    'activity_level': 'high'
                })
            },
            {
                'datacenter_id': 1,
                'name': 'Storage Area',
                'stream_url': 'rtsp://admin:password@192.168.29.212:554/ch0_0.264',
                'camera_type': 'security',
                'location_details': 'Equipment storage area',
                'status': 'active',
                'metadata': json.dumps({
                    'use_cases': ['intrusion', 'loitering'],
                    'activity_level': 'medium'
                })
            }
        ]
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing data if force
            if force:
                cursor.execute("DELETE FROM cameras")
                cursor.execute("DELETE FROM datacenters")
                print("Cleared existing data")
            
            # Insert datacenters
            for dc in sample_datacenters:
                cursor.execute("""
                    INSERT INTO datacenters (name, description, location, facility_type, status, created_at)
                    VALUES (%(name)s, %(description)s, %(location)s, %(facility_type)s, %(status)s, NOW())
                """, dc)
            
            # Insert cameras
            for cam in sample_cameras:
                cursor.execute("""
                    INSERT INTO cameras (datacenter_id, name, stream_url, camera_type, location_details, status, metadata, created_at)
                    VALUES (%(datacenter_id)s, %(name)s, %(stream_url)s, %(camera_type)s, %(location_details)s, %(status)s, %(metadata)s, NOW())
                """, cam)
            
            conn.commit()
        
        print(f"Successfully initialized {len(sample_datacenters)} datacenters and {len(sample_cameras)} cameras")
        return True
        
    except Exception as e:
        print(f"Error initializing sample data: {e}")
        return False

async def main_monitor(args):
    """Main monitoring function with proper shutdown handling"""
    global processor, running
    
    logger = get_main_logger()
    
    try:
        # Log startup
        audit_logger.log_system_event(
            component='main_application',
            event='startup_initiated',
            status='starting',
            details={
                'datacenter_id': args.datacenter_id,
                'args': vars(args)
            }
        )
        
        # Initialize database
        db = DatacenterDatabase()
        
        # Get datacenter info if specific ID provided
        if args.datacenter_id:
            datacenter_info = await get_datacenter_info(args.datacenter_id)
            if not datacenter_info:
                logger.error(f"Datacenter {args.datacenter_id} not found")
                return False
            print(f"Monitoring datacenter: {datacenter_info['name']} (ID: {args.datacenter_id})")
        else:
            print("Monitoring all active datacenters")
        
        # Initialize video processor
        print("Initializing video processor...")
        processor = DatacenterVideoProcessor(datacenter_id=args.datacenter_id)
        
        # Start monitoring system
        print("Starting monitoring system...")
        await processor.start_monitoring()
        
        # Log successful startup
        audit_logger.log_system_event(
            component='main_application',
            event='startup_completed',
            status='running',
            details={'datacenter_id': args.datacenter_id}
        )
        
        print("Monitoring system started successfully!")
        print("Press Ctrl+C to stop monitoring...")
        
        # Main monitoring loop with proper shutdown handling
        running = True
        status_log_counter = 0
        
        while running and not shutdown_event.is_set():
            try:
                # Check for shutdown signal
                if shutdown_event.is_set():
                    break
                
                # Get system status with error handling
                try:
                    status = processor.get_system_status()
                    cameras_active = status.get('cameras_active', 0)
                    cameras_total = status.get('cameras_total', 0)
                    uptime = status.get('uptime_seconds', 0)
                    total_frames = status.get('total_frames_processed', 0)
                    total_events = status.get('total_events_detected', 0)
                    
                    # Log status every 30 seconds
                    status_log_counter += 1
                    if status_log_counter >= 30:
                        logger.info(f"System Status - Cameras: {cameras_active}/{cameras_total}, "
                                  f"Uptime: {uptime:.0f}s, Frames: {total_frames}, Events: {total_events}")
                        status_log_counter = 0
                except Exception as status_error:
                    logger.error(f"Error getting system status: {status_error}")
                
                # Sleep with cancellation check
                for _ in range(10):  # Sleep for 1 second total, but check every 100ms
                    if shutdown_event.is_set():
                        break
                    await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                print("Monitoring loop cancelled")
                break
            except KeyboardInterrupt:
                print("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(1)
        
        return True
        
    except KeyboardInterrupt:
        print("Stopping monitoring system...")
        running = False
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in monitoring: {str(e)}", exc_info=True)
        return False
        
    finally:
        # Graceful shutdown
        await graceful_shutdown(processor)
        
        # Log shutdown
        audit_logger.log_system_event(
            component='main_application',
            event='shutdown_completed',
            status='stopped',
            details={}
        )

async def main():
    """Main application entry point with proper signal handling"""
    
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
    monitor_parser.add_argument('--datacenter-id', type=int, help='Monitor specific datacenter ID')
    monitor_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               help='Set logging level')
    monitor_parser.add_argument('--batch-size', type=int, help='Override batch size for processing')
    monitor_parser.add_argument('--fps-limit', type=int, help='Override FPS limit for cameras')
    
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
    
    # Set up signal handlers for proper shutdown
    def setup_signal_handlers():
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # On Windows, also handle SIGBREAK
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    setup_signal_handlers()
    
    # Handle commands
    try:
        if args.command == 'monitor' or args.command is None:
            # Default to monitor if no command specified
            if args.command is None:
                args.datacenter_id = None
                args.log_level = 'INFO'
                args.batch_size = None
                args.fps_limit = None
            
            # Override config with command line arguments
            if args.batch_size:
                Config.BATCH_SIZE = args.batch_size
            if args.fps_limit:
                Config.READER_FPS_LIMIT = args.fps_limit
            
            # Set log level
            logging.getLogger().setLevel(getattr(logging, args.log_level))
            
            # Start monitoring
            success = await main_monitor(args)
            return 0 if success else 1
            
        elif args.command == 'list-datacenters':
            success = await list_datacenters()
            return 0 if success else 1
            
        elif args.command == 'datacenter-details':
            success = await show_datacenter_details(args.datacenter_id)
            return 0 if success else 1
            
        elif args.command == 'health-check':
            success = await test_system_health()
            return 0 if success else 1
            
        elif args.command == 'init-data':
            success = await initialize_sample_data(force=args.force)
            return 0 if success else 1
            
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except Exception as e:
        print(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal application error: {e}")
        sys.exit(1)
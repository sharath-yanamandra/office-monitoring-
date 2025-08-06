#!/usr/bin/env python3
"""
storage_handler.py
Cloud Storage Management for Video Monitoring

This module handles:
1. Google Cloud Storage integration
2. Frame and video upload
3. Storage optimization and cleanup
4. Local file management
"""

import os
import time
import threading
import queue
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta

from config import Config

# Optional Google Cloud Storage import
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

class StorageHandler:
    """Handles local and cloud storage operations"""
    
    def __init__(self):
        self.logger = logging.getLogger('storage_handler')
        
        # Cloud storage setup
        self.gcs_client = None
        self.bucket = None
        self.cloud_enabled = False
        
        # Local storage
        self.local_storage_dir = Config.FRAMES_OUTPUT_DIR
        self.ensure_local_directories()
        
        # Upload queue for async operations
        self.upload_queue = queue.Queue()
        self.upload_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'files_uploaded': 0,
            'upload_failures': 0,
            'bytes_uploaded': 0,
            'local_files_saved': 0
        }
        
        # Initialize cloud storage if available
        self._initialize_cloud_storage()
        
        self.logger.info(f"StorageHandler initialized - Cloud enabled: {self.cloud_enabled}")
    
    def _initialize_cloud_storage(self):
        """Initialize Google Cloud Storage if configured"""
        if not GCS_AVAILABLE:
            self.logger.warning("Google Cloud Storage not available - install google-cloud-storage")
            return
        
        if not Config.GCP_PROJECT or not Config.GCP_BUCKET_NAME:
            self.logger.info("GCP not configured - using local storage only")
            return
        
        try:
            # Initialize client
            if Config.GOOGLE_APPLICATION_CREDENTIALS:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Config.GOOGLE_APPLICATION_CREDENTIALS
            
            self.gcs_client = storage.Client(project=Config.GCP_PROJECT)
            self.bucket = self.gcs_client.bucket(Config.GCP_BUCKET_NAME)
            
            # Test bucket access
            self.bucket.reload()
            self.cloud_enabled = True
            self.logger.info(f"Connected to GCS bucket: {Config.GCP_BUCKET_NAME}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS: {e}")
            self.cloud_enabled = False
    
    def ensure_local_directories(self):
        """Ensure local storage directories exist"""
        try:
            os.makedirs(self.local_storage_dir, exist_ok=True)
            
            # Create subdirectories
            for subdir in ['frames', 'videos', 'events']:
                os.makedirs(os.path.join(self.local_storage_dir, subdir), exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to create storage directories: {e}")
    
    def start_upload_worker(self):
        """Start background upload worker"""
        if not self.is_running and self.cloud_enabled:
            self.is_running = True
            self.upload_thread = threading.Thread(target=self._upload_worker_loop)
            self.upload_thread.daemon = True
            self.upload_thread.start()
            self.logger.info("Upload worker started")
    
    def stop_upload_worker(self):
        """Stop background upload worker"""
        if self.is_running:
            self.is_running = False
            
            if self.upload_thread and self.upload_thread.is_alive():
                self.upload_thread.join(timeout=10)
            
            self.logger.info("Upload worker stopped")
    
    def _upload_worker_loop(self):
        """Background worker for uploading files to cloud storage"""
        while self.is_running:
            try:
                # Get upload task from queue
                try:
                    upload_task = self.upload_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process upload
                self._process_upload_task(upload_task)
                
            except Exception as e:
                self.logger.error(f"Error in upload worker: {e}")
                time.sleep(1)
    
    def _process_upload_task(self, task: Dict[str, Any]):
        """Process single upload task"""
        try:
            local_path = task['local_path']
            cloud_path = task['cloud_path']
            delete_local = task.get('delete_local', False)
            
            if not os.path.exists(local_path):
                self.logger.warning(f"Local file not found for upload: {local_path}")
                return
            
            # Upload to cloud storage
            blob = self.bucket.blob(cloud_path)
            
            with open(local_path, 'rb') as file_data:
                blob.upload_from_file(file_data)
            
            # Update statistics
            file_size = os.path.getsize(local_path)
            self.stats['files_uploaded'] += 1
            self.stats['bytes_uploaded'] += file_size
            
            # Delete local file if requested
            if delete_local:
                os.remove(local_path)
            
            self.logger.debug(f"Uploaded {local_path} to {cloud_path}")
            
        except Exception as e:
            self.stats['upload_failures'] += 1
            self.logger.error(f"Upload failed for {task.get('local_path', 'unknown')}: {e}")
    
    def save_frame(self, frame_data: bytes, camera_id: int, 
                   event_type: str = None, timestamp: float = None) -> str:
        """
        Save frame to local storage and optionally upload to cloud
        
        Args:
            frame_data: Frame image data (bytes)
            camera_id: Camera identifier
            event_type: Optional event type for organization
            timestamp: Frame timestamp
            
        Returns:
            Local file path
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Generate filename
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime('%Y%m%d')
        time_str = dt.strftime('%H%M%S_%f')[:-3]  # Include milliseconds
        
        if event_type:
            filename = f"{camera_id}_{event_type}_{date_str}_{time_str}.jpg"
            subdir = 'events'
        else:
            filename = f"{camera_id}_{date_str}_{time_str}.jpg"
            subdir = 'frames'
        
        # Create camera-specific directory
        camera_dir = os.path.join(self.local_storage_dir, subdir, f"camera_{camera_id}")
        os.makedirs(camera_dir, exist_ok=True)
        
        # Save to local storage
        local_path = os.path.join(camera_dir, filename)
        
        try:
            with open(local_path, 'wb') as f:
                f.write(frame_data)
            
            self.stats['local_files_saved'] += 1
            
            # Queue for cloud upload if enabled
            if self.cloud_enabled:
                cloud_path = f"frames/camera_{camera_id}/{date_str}/{filename}"
                self.queue_upload(local_path, cloud_path, delete_local=False)
            
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
            return ""
    
    def save_video_segment(self, video_data: bytes, camera_id: int, 
                          event_type: str, start_time: float, duration: float) -> str:
        """
        Save video segment to storage
        
        Args:
            video_data: Video data (bytes)
            camera_id: Camera identifier
            event_type: Event type
            start_time: Video start timestamp
            duration: Video duration in seconds
            
        Returns:
            Local file path
        """
        dt = datetime.fromtimestamp(start_time)
        date_str = dt.strftime('%Y%m%d')
        time_str = dt.strftime('%H%M%S')
        
        filename = f"{camera_id}_{event_type}_{date_str}_{time_str}_{int(duration)}s.mp4"
        
        # Create camera-specific video directory
        video_dir = os.path.join(self.local_storage_dir, 'videos', f"camera_{camera_id}")
        os.makedirs(video_dir, exist_ok=True)
        
        local_path = os.path.join(video_dir, filename)
        
        try:
            with open(local_path, 'wb') as f:
                f.write(video_data)
            
            # Queue for cloud upload
            if self.cloud_enabled:
                cloud_path = f"videos/camera_{camera_id}/{date_str}/{filename}"
                self.queue_upload(local_path, cloud_path, delete_local=True)  # Delete after upload
            
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")
            return ""
    
    def queue_upload(self, local_path: str, cloud_path: str, delete_local: bool = False):
        """Queue file for cloud upload"""
        if not self.cloud_enabled:
            return
        
        upload_task = {
            'local_path': local_path,
            'cloud_path': cloud_path,
            'delete_local': delete_local,
            'queued_at': time.time()
        }
        
        try:
            self.upload_queue.put(upload_task, block=False)
        except queue.Full:
            self.logger.warning("Upload queue full, dropping upload task")
    
    def get_file_url(self, file_path: str) -> Optional[str]:
        """Get public URL for cloud-stored file"""
        if not self.cloud_enabled:
            return None
        
        try:
            blob = self.bucket.blob(file_path)
            
            # Generate signed URL (valid for 1 hour)
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.utcnow() + timedelta(hours=1),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate URL for {file_path}: {e}")
            return None
    
    def cleanup_old_files(self, days_old: int = 7):
        """Clean up old local files"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            deleted_count = 0
            
            for root, dirs, files in os.walk(self.local_storage_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                            deleted_count += 1
                    except OSError:
                        pass  # File might be in use or already deleted
            
            self.logger.info(f"Cleaned up {deleted_count} old files")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get local storage usage statistics"""
        try:
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(self.local_storage_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                    except OSError:
                        pass
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count,
                'storage_dir': self.local_storage_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {e}")
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage handler statistics"""
        storage_usage = self.get_storage_usage()
        
        return {
            **self.stats,
            'cloud_enabled': self.cloud_enabled,
            'upload_queue_size': self.upload_queue.qsize(),
            'is_upload_worker_running': self.is_running,
            'storage_usage': storage_usage
        }
    
    def reset_statistics(self):
        """Reset storage statistics"""
        self.stats = {
            'files_uploaded': 0,
            'upload_failures': 0,
            'bytes_uploaded': 0,
            'local_files_saved': 0
        }
        self.logger.info("Storage statistics reset")


class LocalStorageHandler:
    """Simplified local-only storage handler"""
    
    def __init__(self, base_dir: str = None):
        self.logger = logging.getLogger('local_storage')
        self.base_dir = base_dir or Config.FRAMES_OUTPUT_DIR
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'events'), exist_ok=True)
    
    def save_frame(self, frame_data: bytes, camera_id: int, 
                   event_type: str = None, timestamp: float = None) -> str:
        """Save frame locally"""
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime('%Y%m%d')
        time_str = dt.strftime('%H%M%S_%f')[:-3]
        
        if event_type:
            filename = f"{camera_id}_{event_type}_{date_str}_{time_str}.jpg"
            subdir = 'events'
        else:
            filename = f"{camera_id}_{date_str}_{time_str}.jpg"
            subdir = 'frames'
        
        camera_dir = os.path.join(self.base_dir, subdir, f"camera_{camera_id}")
        os.makedirs(camera_dir, exist_ok=True)
        
        local_path = os.path.join(camera_dir, filename)
        
        try:
            with open(local_path, 'wb') as f:
                f.write(frame_data)
            return local_path
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
            return ""


# Export main classes
__all__ = ['StorageHandler', 'LocalStorageHandler']
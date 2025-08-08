#!/usr/bin/env python3
"""
Script 14: storage_handler.py
File Path: src/storage_handler.py

Datacenter Monitoring System - Cloud Storage Management

This module handles:
1. Cloud storage operations for frames and videos
2. Intelligent storage tiering based on event severity
3. Automatic cleanup and retention policies
4. Parallel upload optimization
5. Error handling and retry mechanisms
6. Storage cost optimization
7. Compliance and audit trail management
"""

import os
import time
import threading
import queue
import hashlib
import mimetypes
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from config import DatacenterConfig
from logger import setup_datacenter_logger, audit_logger, performance_logger

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    from google.api_core import exceptions as gcs_exceptions
    gcp_available = True
except ImportError:
    gcp_available = False

@dataclass
class DatacenterStorageItem:
    """Data class for items to be stored in cloud"""
    local_path: str
    cloud_path: str
    item_type: str  # 'frame', 'video', 'log'
    camera_id: int
    datacenter_id: int
    event_type: str
    severity: str
    timestamp: datetime
    metadata: Dict[str, Any]
    retention_days: int = 30
    priority: int = 1  # 1=high, 2=medium, 3=low
    delete_after_upload: bool = True
    compress: bool = False

@dataclass
class DatacenterStorageStats:
    """Storage operation statistics"""
    total_uploads: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    total_size_bytes: int = 0
    average_upload_time: float = 0.0
    queue_size: int = 0
    storage_cost_estimate: float = 0.0

class DatacenterStorageHandler:
    """
    Advanced cloud storage handler for datacenter monitoring
    with intelligent tiering, compression, and cost optimization
    """
    
    def __init__(self, bucket_name: Optional[str] = None):
        self.logger = setup_datacenter_logger('datacenter_storage', 'datacenter_storage.log')
        self.logger.info("Initializing DatacenterStorageHandler")
        
        # Configuration
        self.bucket_name = bucket_name or DatacenterConfig.BUCKET_NAME
        self.max_workers = 8  # Parallel upload threads
        self.max_queue_size = 5000
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for large files
        
        # Storage client and bucket
        self.storage_client = None
        self.bucket = None
        self._init_storage_client()
        
        # Upload queue with priority
        self.upload_queue = queue.PriorityQueue(maxsize=self.max_queue_size)
        
        # Thread management
        self.running = False
        self.worker_threads = []
        self.upload_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Statistics and monitoring
        self.stats = DatacenterStorageStats()
        self.upload_times = []
        self.error_count = 0
        
        # Retention policies by event type
        self.retention_policies = {
            'critical': 365,      # 1 year for critical events
            'high': 180,          # 6 months for high severity
            'medium': 90,         # 3 months for medium severity
            'low': 30,            # 1 month for low severity
            'tailgating': 180,    # 6 months for tailgating
            'intrusion': 365,     # 1 year for intrusion
            'ppe_violation': 90,  # 3 months for PPE violations
            'camera_tamper': 365, # 1 year for camera tampering
            'loitering': 30       # 1 month for loitering
        }
        
        # Storage classes by priority
        self.storage_classes = {
            1: 'STANDARD',        # High priority - immediate access
            2: 'NEARLINE',        # Medium priority - monthly access
            3: 'COLDLINE'         # Low priority - quarterly access
        }
        
        # Compression settings
        self.compression_enabled = True
        self.compression_threshold = 1024 * 1024  # 1MB
        
        # Start worker threads
        self.start_workers()
        
        self.logger.info("DatacenterStorageHandler initialized successfully")
    
    def _init_storage_client(self):
        """Initialize Google Cloud Storage client"""
        if not gcp_available:
            self.logger.warning("Google Cloud Storage not available, uploads will be disabled")
            return
        
        try:
            # Initialize client with credentials
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.storage_client = storage.Client(
                    credentials=credentials,
                    project=DatacenterConfig.GCP_PROJECT
                )
            else:
                self.storage_client = storage.Client(project=DatacenterConfig.GCP_PROJECT)
            
            # Initialize bucket
            if self.bucket_name:
                self.bucket = self.storage_client.bucket(self.bucket_name)
                
                # Verify bucket exists
                if self.bucket.exists():
                    self.logger.info(f"Connected to GCS bucket: {self.bucket_name}")
                else:
                    self.logger.error(f"Bucket {self.bucket_name} does not exist")
                    self.bucket = None
            else:
                self.logger.warning("No bucket name configured")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize storage client: {str(e)}")
            self.storage_client = None
            self.bucket = None
    
    def start_workers(self):
        """Start worker threads for upload processing"""
        if self.running:
            self.logger.warning("Workers already running")
            return
        
        self.running = True
        
        # Start multiple worker threads for parallel processing
        for i in range(3):  # 3 worker threads
            worker = threading.Thread(
                target=self._upload_worker,
                args=(i,),
                daemon=True,
                name=f"storage_worker_{i}"
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start cleanup worker
        cleanup_worker = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="storage_cleanup"
        )
        cleanup_worker.start()
        self.worker_threads.append(cleanup_worker)
        
        self.logger.info(f"Started {len(self.worker_threads)} storage worker threads")
    
    def upload_frame(self, local_path: str, camera_id: int, datacenter_id: int, 
                    event_type: str, severity: str, timestamp: datetime,
                    metadata: Optional[Dict] = None, priority: int = 1) -> bool:
        """
        Queue a frame for upload to cloud storage
        
        Args:
            local_path: Path to local frame file
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            event_type: Type of event that triggered this frame
            severity: Event severity level
            timestamp: Frame timestamp
            metadata: Additional metadata
            priority: Upload priority (1=high, 2=medium, 3=low)
            
        Returns:
            True if successfully queued
        """
        try:
            if not self.bucket:
                self.logger.warning("No storage bucket available")
                return False
            
            # Generate cloud path
            cloud_path = self._generate_cloud_path(
                item_type='frame',
                camera_id=camera_id,
                datacenter_id=datacenter_id,
                event_type=event_type,
                timestamp=timestamp,
                filename=os.path.basename(local_path)
            )
            
            # Determine retention period
            retention_days = self._get_retention_period(severity, event_type)
            
            # Create storage item
            storage_item = DatacenterStorageItem(
                local_path=local_path,
                cloud_path=cloud_path,
                item_type='frame',
                camera_id=camera_id,
                datacenter_id=datacenter_id,
                event_type=event_type,
                severity=severity,
                timestamp=timestamp,
                metadata=metadata or {},
                retention_days=retention_days,
                priority=priority,
                delete_after_upload=True,
                compress=self._should_compress(local_path)
            )
            
            # Queue for upload
            self.upload_queue.put((priority, time.time(), storage_item))
            
            self.logger.debug(f"Queued frame upload: {local_path} -> {cloud_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing frame upload: {str(e)}")
            return False
    
    def upload_video(self, local_path: str, camera_id: int, datacenter_id: int,
                    event_type: str, severity: str, timestamp: datetime,
                    duration: float, metadata: Optional[Dict] = None) -> bool:
        """
        Queue a video for upload to cloud storage
        
        Args:
            local_path: Path to local video file
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            event_type: Type of event
            severity: Event severity level
            timestamp: Video timestamp
            duration: Video duration in seconds
            metadata: Additional metadata
            
        Returns:
            True if successfully queued
        """
        try:
            if not self.bucket:
                self.logger.warning("No storage bucket available")
                return False
            
            # Generate cloud path
            cloud_path = self._generate_cloud_path(
                item_type='video',
                camera_id=camera_id,
                datacenter_id=datacenter_id,
                event_type=event_type,
                timestamp=timestamp,
                filename=os.path.basename(local_path)
            )
            
            # Videos get higher priority and longer retention
            priority = 1 if severity in ['critical', 'high'] else 2
            retention_days = self._get_retention_period(severity, event_type)
            
            # Enhanced metadata for videos
            video_metadata = (metadata or {}).copy()
            video_metadata.update({
                'duration': duration,
                'file_size': os.path.getsize(local_path),
                'mime_type': mimetypes.guess_type(local_path)[0] or 'video/mp4'
            })
            
            # Create storage item
            storage_item = DatacenterStorageItem(
                local_path=local_path,
                cloud_path=cloud_path,
                item_type='video',
                camera_id=camera_id,
                datacenter_id=datacenter_id,
                event_type=event_type,
                severity=severity,
                timestamp=timestamp,
                metadata=video_metadata,
                retention_days=retention_days,
                priority=priority,
                delete_after_upload=True,
                compress=False  # Don't compress videos
            )
            
            # Queue for upload
            self.upload_queue.put((priority, time.time(), storage_item))
            
            self.logger.info(f"Queued video upload: {local_path} -> {cloud_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing video upload: {str(e)}")
            return False
    
    def _generate_cloud_path(self, item_type: str, camera_id: int, datacenter_id: int,
                            event_type: str, timestamp: datetime, filename: str) -> str:
        """Generate cloud storage path with intelligent organization"""
        
        # Organize by datacenter, then by date, then by type
        date_str = timestamp.strftime('%Y/%m/%d')
        hour_str = timestamp.strftime('%H')
        
        # Create hierarchical path
        path_parts = [
            'datacenters',
            f'dc_{datacenter_id}',
            'cameras',
            f'cam_{camera_id}',
            item_type + 's',  # frames, videos
            event_type,
            date_str,
            hour_str,
            filename
        ]
        
        return '/'.join(path_parts)
    
    def _get_retention_period(self, severity: str, event_type: str) -> int:
        """Get retention period based on severity and event type"""
        
        # Check event type first
        if event_type in self.retention_policies:
            return self.retention_policies[event_type]
        
        # Fall back to severity-based retention
        if severity in self.retention_policies:
            return self.retention_policies[severity]
        
        # Default retention
        return 30
    
    def _should_compress(self, file_path: str) -> bool:
        """Determine if file should be compressed"""
        try:
            if not self.compression_enabled:
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < self.compression_threshold:
                return False
            
            # Check file type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                # Don't compress already compressed formats
                if mime_type.startswith('video/') or mime_type in ['image/jpeg', 'image/png']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking compression: {e}")
            return False
    
    def _upload_worker(self, worker_id: int):
        """Worker thread for processing upload queue"""
        self.logger.info(f"Storage worker {worker_id} started")
        
        while self.running:
            try:
                # Get item from queue with timeout
                try:
                    priority, queued_time, storage_item = self.upload_queue.get(timeout=1.0)
                    
                    # Process the upload
                    success = self._process_upload(storage_item, worker_id)
                    
                    # Update statistics
                    if success:
                        self.stats.successful_uploads += 1
                        
                        # Track upload time
                        upload_time = time.time() - queued_time
                        self.upload_times.append(upload_time)
                        if len(self.upload_times) > 100:
                            self.upload_times = self.upload_times[-100:]
                        
                        self.stats.average_upload_time = sum(self.upload_times) / len(self.upload_times)
                        
                    else:
                        self.stats.failed_uploads += 1
                        self.error_count += 1
                    
                    self.stats.total_uploads += 1
                    self.upload_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in upload worker {worker_id}: {str(e)}", exc_info=True)
                time.sleep(1)
        
        self.logger.info(f"Storage worker {worker_id} stopped")
    
    def _process_upload(self, storage_item: DatacenterStorageItem, worker_id: int) -> bool:
        """Process a single upload item"""
        try:
            start_time = time.time()
            
            # Check if local file exists
            if not os.path.exists(storage_item.local_path):
                self.logger.warning(f"Local file not found: {storage_item.local_path}")
                return False
            
            # Get file information
            file_size = os.path.getsize(storage_item.local_path)
            self.stats.total_size_bytes += file_size
            
            # Create blob
            blob = self.bucket.blob(storage_item.cloud_path)
            
            # Set storage class based on priority
            storage_class = self.storage_classes.get(storage_item.priority, 'STANDARD')
            blob.storage_class = storage_class
            
            # Upload file
            self._upload_file_with_retry(blob, storage_item)
            
            # Set metadata
            self._set_blob_metadata(blob, storage_item)
            
            # Set lifecycle policy
            self._set_lifecycle_policy(blob, storage_item)
            
            # Clean up local file if requested
            if storage_item.delete_after_upload:
                try:
                    os.remove(storage_item.local_path)
                    self.logger.debug(f"Deleted local file: {storage_item.local_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete local file: {e}")
            
            # Log successful upload
            upload_time = time.time() - start_time
            self.logger.info(f"Worker {worker_id} uploaded {storage_item.item_type}: "
                           f"{storage_item.cloud_path} ({file_size} bytes) in {upload_time:.2f}s")
            
            # Log performance metrics
            performance_logger.log_processing_stats(
                camera_id=str(storage_item.camera_id),
                fps=0,  # Not applicable for uploads
                batch_size=1,
                processing_time=upload_time,
                queue_size=self.upload_queue.qsize()
            )
            
            # Log audit trail
            audit_logger.log_system_event(
                component='storage_handler',
                event='file_uploaded',
                status='success',
                details={
                    'cloud_path': storage_item.cloud_path,
                    'file_size': file_size,
                    'camera_id': storage_item.camera_id,
                    'datacenter_id': storage_item.datacenter_id,
                    'event_type': storage_item.event_type,
                    'storage_class': storage_class,
                    'upload_time': upload_time
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {storage_item.local_path}: {str(e)}", exc_info=True)
            return False
    
    def _upload_file_with_retry(self, blob: storage.Blob, storage_item: DatacenterStorageItem):
        """Upload file with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Upload file
                blob.upload_from_filename(storage_item.local_path)
                return
                
            except gcs_exceptions.GoogleCloudError as e:
                if attempt == max_retries - 1:
                    raise
                
                self.logger.warning(f"Upload attempt {attempt + 1} failed for {storage_item.local_path}: {e}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                self.logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)
    
    def _set_blob_metadata(self, blob: storage.Blob, storage_item: DatacenterStorageItem):
        """Set comprehensive metadata on uploaded blob"""
        try:
            # Base metadata
            metadata = {
                'datacenter_id': str(storage_item.datacenter_id),
                'camera_id': str(storage_item.camera_id),
                'event_type': storage_item.event_type,
                'severity': storage_item.severity,
                'timestamp': storage_item.timestamp.isoformat(),
                'upload_time': datetime.utcnow().isoformat(),
                'retention_days': str(storage_item.retention_days),
                'item_type': storage_item.item_type,
                'version': '1.0'
            }
            
            # Add custom metadata
            if storage_item.metadata:
                for key, value in storage_item.metadata.items():
                    metadata[f'custom_{key}'] = str(value)
            
            # Set metadata
            blob.metadata = metadata
            blob.patch()
            
        except Exception as e:
            self.logger.error(f"Failed to set blob metadata: {e}")
    
    def _set_lifecycle_policy(self, blob: storage.Blob, storage_item: DatacenterStorageItem):
        """Set lifecycle policy for automatic deletion"""
        try:
            # Calculate deletion date
            deletion_date = storage_item.timestamp + timedelta(days=storage_item.retention_days)
            
            # Note: Individual blob lifecycle policies are not supported in GCS
            # This would need to be implemented at the bucket level
            # For now, we'll store the deletion date in metadata for custom cleanup
            
            if blob.metadata:
                blob.metadata['deletion_date'] = deletion_date.isoformat()
                blob.patch()
            
        except Exception as e:
            self.logger.error(f"Failed to set lifecycle policy: {e}")
    
    def _cleanup_worker(self):
        """Worker thread for cleanup operations"""
        self.logger.info("Storage cleanup worker started")
        
        while self.running:
            try:
                # Run cleanup every hour
                time.sleep(3600)
                
                if self.running:
                    self._cleanup_expired_files()
                    
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {str(e)}", exc_info=True)
        
        self.logger.info("Storage cleanup worker stopped")
    
    def _cleanup_expired_files(self):
        """Clean up expired files based on retention policies"""
        try:
            if not self.bucket:
                return
            
            self.logger.info("Starting cleanup of expired files")
            
            # List all blobs with metadata
            blobs = self.bucket.list_blobs()
            
            deleted_count = 0
            total_size_deleted = 0
            
            for blob in blobs:
                try:
                    # Check if blob has deletion date
                    if blob.metadata and 'deletion_date' in blob.metadata:
                        deletion_date = datetime.fromisoformat(blob.metadata['deletion_date'])
                        
                        if datetime.utcnow() > deletion_date:
                            # File is expired, delete it
                            blob_size = blob.size or 0
                            blob.delete()
                            
                            deleted_count += 1
                            total_size_deleted += blob_size
                            
                            self.logger.info(f"Deleted expired file: {blob.name}")
                            
                            # Log audit trail
                            audit_logger.log_system_event(
                                component='storage_handler',
                                event='file_deleted',
                                status='success',
                                details={
                                    'cloud_path': blob.name,
                                    'reason': 'retention_policy',
                                    'file_size': blob_size,
                                    'deletion_date': deletion_date.isoformat()
                                }
                            )
                    
                except Exception as e:
                    self.logger.error(f"Error processing blob {blob.name}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"Cleanup completed: deleted {deleted_count} files, "
                               f"freed {total_size_deleted} bytes")
            else:
                self.logger.debug("Cleanup completed: no expired files found")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics"""
        return {
            'queue_size': self.upload_queue.qsize(),
            'total_uploads': self.stats.total_uploads,
            'successful_uploads': self.stats.successful_uploads,
            'failed_uploads': self.stats.failed_uploads,
            'success_rate': (self.stats.successful_uploads / max(self.stats.total_uploads, 1)) * 100,
            'total_size_bytes': self.stats.total_size_bytes,
            'average_upload_time': self.stats.average_upload_time,
            'error_count': self.error_count,
            'worker_threads': len([t for t in self.worker_threads if t.is_alive()]),
            'bucket_name': self.bucket_name,
            'storage_available': self.bucket is not None
        }
    
    def shutdown(self):
        """Graceful shutdown of storage handler"""
        self.logger.info("Shutting down DatacenterStorageHandler")
        
        self.running = False
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=30)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate gracefully")
        
        # Shutdown thread pool
        self.upload_executor.shutdown(wait=True)
        
        # Log final statistics
        final_stats = self.get_storage_stats()
        self.logger.info(f"Storage handler shutdown complete. Final stats: {final_stats}")

# Export main classes
__all__ = [
    'DatacenterStorageHandler',
    'DatacenterStorageItem',
    'DatacenterStorageStats'
]
        
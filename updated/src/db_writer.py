#!/usr/bin/env python3
"""
Datacenter Database Writer - Database Writing and Batch Operations

This module handles:
1. Batch writing of events, frames, and metadata to database
2. Cloud storage upload coordination
3. Audit trail logging for compliance
4. Performance optimization for high-volume writes
5. Error handling and retry mechanisms
6. Data validation and sanitization
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
import mysql.connector
from queue import Queue, Empty
from datetime import datetime
from threading import Thread
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
import uuid

from config import DatacenterConfig
from database import DatacenterDatabase
from logger import setup_datacenter_logger

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    gcp_available = True
except ImportError:
    gcp_available = False

@dataclass
class DatacenterProcessedFrame:
    """Data class for processed frame with datacenter context"""
    camera_id: int
    datacenter_id: int
    event_id: str
    timestamp: datetime
    local_path: str
    frame_path: str  # Will be set after cloud upload
    event_type: str = "general"
    severity: str = "low"
    zone_name: str = "unknown"

@dataclass
class DatacenterEvent:
    """Data class for datacenter security events"""
    event_id: str
    camera_id: int
    datacenter_id: int
    event_type: str
    severity: str
    timestamp: datetime
    zone_name: Optional[str] = None
    detection_data: Optional[Dict] = None
    snapshot_url: Optional[str] = None
    video_clip_url: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class DatacenterVideoClip:
    """Data class for datacenter video clips"""
    event_id: str
    camera_id: int
    datacenter_id: int
    local_path: str
    cloud_path: str
    duration: float
    timestamp: datetime
    event_type: str
    metadata: Dict

class DatacenterDatabaseWriter:
    """Database writer optimized for datacenter monitoring workloads"""
    
    def __init__(self):
        self.logger = setup_datacenter_logger('datacenter_db_writer', 'datacenter_db_writer.log')
        self.logger.info("Initializing DatacenterDatabaseWriter")
        
        # Batch configuration
        self.batch_size = DatacenterConfig.DB_WRITER_BATCH_SIZE
        self.max_queue_size = 2000  # Prevent memory issues
        
        # Processing queues
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        self.event_queue = Queue(maxsize=self.max_queue_size)
        self.video_queue = Queue(maxsize=self.max_queue_size)
        self.access_log_queue = Queue(maxsize=self.max_queue_size)
        
        # Database connection
        self.db = DatacenterDatabase()
        
        # Store orphaned frames (frames without events) for later processing
        self.orphaned_frames = {}

        # Initialize Cloud Storage client
        self.storage_client = None
        self.bucket = None
        if gcp_available and hasattr(DatacenterConfig, 'GCP_PROJECT') and DatacenterConfig.GCP_PROJECT:
            try:
                self.storage_client = storage.Client.from_service_account_json(
                    './gcp-service-account.json',
                    project=DatacenterConfig.GCP_PROJECT
                )
                self.bucket = self.storage_client.bucket(DatacenterConfig.BUCKET_NAME)
                self.logger.info("Cloud storage initialized successfully")
            except Exception as e:
                self.logger.warning(f"Cloud storage initialization failed: {e}")
                self.storage_client = None
                self.bucket = None

        # Thread pool for parallel uploads
        self.upload_executor = ThreadPoolExecutor(max_workers=self.batch_size)

        # Worker thread
        self.running = True
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.max_retries = 3
        self.retry_delay = 1
        self.worker_thread.start()

    def _upload_frame_to_storage(self, local_path: str, cloud_path: str) -> bool:
        """Upload a single frame to Cloud Storage"""
        if not self.bucket:
            return False
        try:
            blob = self.bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            self.logger.error(f"Error uploading frame {local_path}: {str(e)}")
            return False
    
    def _upload_frames_batch(self, frames: List[DatacenterProcessedFrame]) -> List[DatacenterProcessedFrame]:
        """Upload a batch of frames to Cloud Storage in parallel"""
        if not self.bucket:
            return frames  # Return frames without upload if no storage
            
        upload_tasks = []
        last_valid_date = None  # Track the last valid date for consistency

        for frame in frames:
            frame_filename = os.path.basename(frame.local_path)
            
            # Handle timestamp properly whether it's a string or datetime
            if isinstance(frame.timestamp, str):
                # Parse the timestamp string to extract date
                try:
                    # Try ISO format first
                    dt = datetime.fromisoformat(frame.timestamp.replace('Z', '+00:00'))
                    recorded_date = dt.strftime('%Y-%m-%d')
                    last_valid_date = recorded_date  # Update last valid date
                except ValueError:
                    # Fallback to just using the first 10 characters if it's in YYYY-MM-DD format
                    if len(frame.timestamp) >= 10 and frame.timestamp[4] == '-' and frame.timestamp[7] == '-':
                        recorded_date = frame.timestamp[:10]
                        last_valid_date = recorded_date  # Update last valid date
                    else:
                        # Use the last valid date if available, otherwise use current date
                        if last_valid_date:
                            recorded_date = last_valid_date
                            self.logger.warning(f"Using previous date {recorded_date} for timestamp: {frame.timestamp}")
                        else:
                            recorded_date = datetime.now().strftime('%Y-%m-%d')
                            self.logger.warning(f"No previous date available, using current date for timestamp: {frame.timestamp}")
            else:
                # It's already a datetime object
                recorded_date = frame.timestamp.strftime('%Y-%m-%d')
                last_valid_date = recorded_date  # Update last valid date
            
            # Use datacenter_id in cloud path
            cloud_path = f"datacenter_{frame.datacenter_id}/camera_{frame.camera_id}/{recorded_date}/frames/{frame_filename}"
            frame.frame_path = cloud_path # update frame path to Cloud Storage path

            # submit upload task to thread pool
            future = self.upload_executor.submit(
                self._upload_frame_to_storage,
                frame.local_path,
                cloud_path
            )
            upload_tasks.append((frame, future))
        
        # Wait for all uploads to complete
        successful_frames = []
        for frame, future in upload_tasks:
            try:
                if future.result(): # check if upload was successful
                    successful_frames.append(frame)
                    # Clean up local file after successful upload
                    try:
                        os.remove(frame.local_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete local file {frame.local_path}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Frame upload failed: {str(e)}")

        return successful_frames
    
    def _upload_video_to_storage(self, local_path: str, event_id: str, camera_id: int, 
                                 datacenter_id: int, timestamp, metadata: Dict[str, Any]) -> str:
        """Upload a video to Cloud Storage with retry logic"""
        if not self.bucket:
            return local_path  # Return local path if no cloud storage
            
        # Log the current working directory for debugging
        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(f"Attempting to upload video from path: {local_path}")
        
        # Check if path is absolute or relative
        if not os.path.isabs(local_path):
            self.logger.info(f"Path is relative. Absolute path would be: {os.path.abspath(local_path)}")
        
        if not os.path.exists(local_path):
            self.logger.error(f"Video file not found at {local_path}")
            # Try to list files in the expected directory
            video_dir = os.path.dirname(local_path)
            if os.path.exists(video_dir):
                self.logger.error(f"Files in {video_dir}: {os.listdir(video_dir)}")
            else:
                self.logger.error(f"Directory {video_dir} does not exist")
            return ""
            
        video_filename = os.path.basename(local_path)
        
        # Format the date for the path
        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                recorded_date = dt.strftime('%Y-%m-%d')
            except ValueError:
                # Fallback to current date
                recorded_date = datetime.now().strftime('%Y-%m-%d')
        else:
            # It's a timestamp
            recorded_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        
        # Create cloud path with datacenter context
        cloud_path = f"datacenter_{datacenter_id}/camera_{camera_id}/{recorded_date}/videos/{video_filename}"
        
        # Try uploading with retries
        for attempt in range(self.max_retries):
            try:
                # Upload to cloud storage
                blob = self.bucket.blob(cloud_path)
                blob.upload_from_filename(local_path)
                
                # Set metadata on the blob
                blob_metadata = {
                    'event_id': event_id,
                    'camera_id': str(camera_id),
                    'datacenter_id': str(datacenter_id),
                    'timestamp': str(timestamp),
                    'detection_type': metadata.get('detection_type', 'unknown'),
                    'duration': str(metadata.get('duration', 0)),
                    'camera_type': metadata.get('camera_type', 'unknown')
                }
                blob.metadata = blob_metadata
                blob.update()
                
                self.logger.info(f"Uploaded video {local_path} to {cloud_path}")
                
                # Delete local file after successful upload
                try:
                    os.remove(local_path)
                    self.logger.info(f"Deleted local video file {local_path} after successful upload")
                except Exception as e:
                    self.logger.warning(f"Failed to delete local video file {local_path}: {str(e)}")
                
                return cloud_path
                
            except Exception as e:
                self.logger.error(f"Error uploading video {local_path} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Final attempt failed
                    self.logger.error(f"Failed to upload video {local_path} after {self.max_retries} attempts")
        
        return ""
    
    def _process_queue(self):
        """Process items from queue with proper transaction handling"""
        while self.running or not self.frame_queue.empty():
            items = []
            try:
                # Get a single item with timeout to avoid busy waiting
                try:
                    item = self.frame_queue.get(timeout=1.0)
                    
                    # Check if this is a video item that should be processed immediately
                    if isinstance(item, tuple) and item[0] == 'video':
                        # Process video immediately without batching
                        self.logger.info("Processing video upload immediately without batching")
                        items = [item]
                    else:
                        items.append(item)
                        
                        # Try to get more items if available without blocking
                        # Only collect up to batch_size items to avoid processing too many at once
                        try:
                            while len(items) < self.batch_size:
                                # Use get_nowait to avoid blocking if queue is empty
                                next_item = self.frame_queue.get_nowait()
                                
                                # If we find a video, stop batching and process current batch first
                                if isinstance(next_item, tuple) and next_item[0] == 'video':
                                    # Put the video back in queue to process it separately
                                    self.frame_queue.put(next_item)
                                    break
                                    
                                items.append(next_item)
                        except Empty:
                            # No more items available right now, which is fine
                            pass
                    
                except Empty:
                    # Queue was empty, continue the outer loop
                    continue
                
                # Process whatever items we have, even if it's just one
                if items:
                    retry_count = 0
                    while retry_count < self.max_retries:
                        try:
                            self._process_batch(items)
                            break
                        except Exception as e:
                            retry_count += 1
                            self.logger.error(
                                f"Batch processing failed (attempt {retry_count}/{self.max_retries}): {str(e)}"
                            )
                            if retry_count == self.max_retries:
                                self.logger.error(f"Fatal error in queue processing: {str(e)}", exc_info=True)
                            else:
                                time.sleep(self.retry_delay * retry_count)

            except Exception as e:
                self.logger.error(f"Error processing queue batch: {str(e)}", exc_info=True)
    
    def _process_batch(self, items: List[Any]):
        """Process a batch of items with proper transaction handling"""
        with self.db.get_connection() as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                # start transaction for entire batch
                conn.start_transaction()

                # Group items by type
                frames_obj = []
                events = []
                videos = []
                status_updates = []

                for item in items:
                    if isinstance(item, DatacenterProcessedFrame):
                        frames_obj.append(item)
                    elif isinstance(item, tuple) and item[0] == 'event':
                        events.append(item[1])
                    elif isinstance(item, tuple) and item[0] == 'video':
                        videos.append(item[1])
                    elif isinstance(item, tuple) and item[0] == 'status_update':
                        status_updates.append((item[1], item[2]))
                
                # First process all frames in a single upload batch
                # And create a mapping of event_id to frame_path for use later
                event_frame_map = {}
                successful_frames = []
                if frames_obj:
                    successful_frames = self._upload_frames_batch(frames_obj)
                    if successful_frames:
                        # Create a mapping of event_id to frame_path
                        for frame in successful_frames:
                            event_frame_map[frame.event_id] = frame.frame_path
                            self.logger.info(f"Mapped event {frame.event_id} to snapshot URL {frame.frame_path}")

                # Process all events with the updated snapshot URLs
                if events:
                    for event_data in events:
                        # Check if this event has an associated frame
                        event_id = event_data.get('event_id')
                        if event_id in event_frame_map:
                            # Update the event data with the frame path
                            event_data['snapshot_url'] = event_frame_map[event_id]
                            self.logger.info(f"Adding snapshot URL {event_frame_map[event_id]} to event {event_id} before storing")
                        elif event_id in self.orphaned_frames:
                            # Use a previously orphaned frame for this event
                            event_data['snapshot_url'] = self.orphaned_frames[event_id]
                            self.logger.info(f"Adding previously orphaned snapshot URL {self.orphaned_frames[event_id]} to event {event_id} before storing")
                            # Remove from orphaned frames map as it's now used
                            del self.orphaned_frames[event_id]
                            
                        self._store_event(cursor, event_data)
                
                # Handle orphaned frames (frames without matching event in this batch)
                # These might be frames for events that were already stored in a previous batch
                # or for events that will be processed in a future batch
                orphaned_frames = []
                if successful_frames:
                    for frame in successful_frames:
                        event_id = frame.event_id
                        # Check if this frame's event was NOT processed in this batch
                        if not any(event_data.get('event_id') == event_id for event_data in events):
                            orphaned_frames.append(frame)
                
                # First try to update existing events in the database with frame paths for orphaned frames
                if orphaned_frames:
                    newly_orphaned_frames = []
                    for frame in orphaned_frames:
                        # Check if the event exists before updating
                        cursor.execute("SELECT 1 FROM events WHERE event_id = %s", (frame.event_id,))
                        if cursor.fetchone():
                            cursor.execute("""
                                UPDATE events 
                                SET snapshot_url = %s
                                WHERE event_id = %s AND (snapshot_url IS NULL OR snapshot_url = '')
                            """, (frame.frame_path, frame.event_id))
                            
                            if cursor.rowcount > 0:
                                self.logger.info(f"Updated existing event {frame.event_id} with snapshot URL {frame.frame_path}")
                            else:
                                self.logger.info(f"Event {frame.event_id} already has a snapshot URL or could not be updated")
                        else:
                            # Store frame in orphaned_frames map for future events
                            self.orphaned_frames[frame.event_id] = frame.frame_path
                            newly_orphaned_frames.append(frame)
                            self.logger.info(f"Storing frame for event {frame.event_id} for future processing")
                    
                    # Only log warning for frames that are actually orphaned
                    for frame in newly_orphaned_frames:
                        self.logger.warning(f"Event {frame.event_id} not found in database, snapshot will be orphaned, but stored for later use")
                
                # Process all videos
                if videos:
                    for video_metadata in videos:
                        self._update_video_metadata(cursor, video_metadata)
                
                # Process all status updates
                if status_updates:
                    for camera_id, status in status_updates:
                        self._update_camera_status(cursor, camera_id, status)
                
                # commit the entire batch
                conn.commit()
                self.logger.info(f"Batch processed successfully: {len(frames_obj)} frames, {len(events)} events, {len(videos)} videos.")

            except Exception as e:
                conn.rollback()
                self.logger.error(f"Batch processing error: {str(e)}", exc_info=True)
                raise
            finally:
                if cursor:
                    cursor.close()
    
    def _store_event(self, cursor, event_data: Dict[str, Any]):
        """Store an event in the database using datacenter schema"""
        try:
            # Extract event data
            event_id = event_data.get('event_id', str(uuid.uuid4()))
            camera_id = event_data.get('camera_id')
            datacenter_id = event_data.get('datacenter_id')
            timestamp = event_data.get('timestamp')
            event_type = event_data.get('event_type')
            metadata = event_data.get('metadata', [])
            snapshot_url = event_data.get('snapshot_url', '')
            video_path = event_data.get('video_path', '')
            video_duration = event_data.get('video_duration', 0)
            
            # Standardize timestamp format
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            # Find matching rule for this detection in datacenter schema
            cursor.execute("""
                SELECT rule_id, severity FROM rules 
                WHERE camera_id = %s 
                AND event_type = %s 
                AND enabled = TRUE
                LIMIT 1
            """, (camera_id, event_type))
            
            rule_result = cursor.fetchone()
            if not rule_result:
                self.logger.warning(f"No matching rule found for {event_type} on camera {camera_id}")
                return
                
            rule_id, severity = rule_result
            
            # Find zone if applicable
            zone_id = None
            zone_name = event_data.get('zone_name') or event_data.get('zone')
            if zone_name:
                cursor.execute("""
                    SELECT zone_id FROM zones 
                    WHERE camera_id = %s AND name = %s 
                    LIMIT 1
                """, (camera_id, zone_name))
                
                zone_result = cursor.fetchone()
                if zone_result:
                    zone_id = zone_result[0]
            
            # Handle video upload if video_path is provided
            video_clip_url = ''
            if video_path:
                # Upload to cloud storage if configured
                if self.bucket:
                    video_clip_url = self._upload_video_to_storage(
                        video_path, 
                        event_id, 
                        camera_id,
                        datacenter_id,
                        timestamp, 
                        {'detection_type': event_type, 'duration': video_duration}
                    )
                    # Update metadata with video information
                    if isinstance(metadata, list):
                        metadata = {'detections': metadata}
                    metadata['video_duration'] = video_duration
                    metadata['video_detection_type'] = event_type
                else:
                    # No cloud storage, use local path
                    video_clip_url = video_path
            
            # Insert event into datacenter events table
            cursor.execute("""
                INSERT INTO events (
                    event_id, rule_id, camera_id, zone_id, timestamp, 
                    detection_data, snapshot_url, video_clip_url, status 
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s 
                )
            """, (
                event_id,
                rule_id,
                camera_id,
                zone_id,
                timestamp,
                json.dumps(metadata),
                snapshot_url,
                video_clip_url,
                'new'
            ))
            
            self.logger.info(f"Stored event {event_id} for camera {camera_id}" + 
                           (f" with video {video_clip_url}" if video_clip_url else ""))
                
        except Exception as e:
            self.logger.error(f"Error storing event: {str(e)}", exc_info=True)
    
    def _update_video_metadata(self, cursor, video_metadata: Dict[str, Any]):
        """Update event with video metadata"""
        try:
            event_id = video_metadata.get('event_id')
            video_path = video_metadata.get('video_path')
            duration = video_metadata.get('duration', 0)
            camera_id = video_metadata.get('camera_id')
            datacenter_id = video_metadata.get('datacenter_id')
            timestamp = video_metadata.get('timestamp')
            
            # Upload to cloud storage if configured
            cloud_path = ""
            if self.bucket:
                cloud_path = self._upload_video_to_storage(
                    video_path, 
                    event_id, 
                    camera_id,
                    datacenter_id,
                    timestamp, 
                    video_metadata
                )
            
            # Only update database if upload was successful or if no bucket is configured
            if cloud_path or not self.bucket:
                # Update the event record with the video path
                cursor.execute("""
                    UPDATE events 
                    SET video_clip_url = %s,
                        detection_data = JSON_SET(
                            COALESCE(detection_data, '{}'), 
                            '$.video_duration', %s,
                            '$.video_detection_type', %s
                        ) 
                    WHERE event_id = %s
                """, (
                    cloud_path if cloud_path else video_path,
                    duration,
                    video_metadata.get('detection_type', 'unknown'),
                    event_id
                ))
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Updated event {event_id} with video path {cloud_path if cloud_path else video_path}")
                else:
                    self.logger.warning(f"No rows updated for event {event_id}")
            else:
                self.logger.error(f"Failed to upload video for event {event_id}, database not updated")
            
        except Exception as e:
            self.logger.error(f"Error updating video metadata: {str(e)}", exc_info=True)
    
    def _update_camera_status(self, cursor, camera_id: int, status: str):
        """Update camera status in the database"""
        try:
            cursor.execute("""
                UPDATE cameras 
                SET status = %s,
                    updated_at = %s 
                WHERE camera_id = %s
            """, (
                status,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                camera_id
            ))
            self.logger.info(f"Updated camera {camera_id} status to {status}")
        except mysql.connector.Error as e:
            self.logger.error(f"Error updating camera status: {str(e)}", exc_info=True)
    
    def queue_frame(self, frame_data: DatacenterProcessedFrame):
        """Queue a frame for batch processing"""
        try:
            self.frame_queue.put(frame_data)
        except Exception as e:
            self.logger.error(f"Error queueing frame: {str(e)}")

    def queue_event(self, event_data: Dict[str, Any]):
        """Queue an event for batch processing"""
        try:
            self.event_queue.put(('event', event_data))
        except Exception as e:
            self.logger.error(f"Error queueing event: {str(e)}")

    def queue_video_metadata(self, video_metadata: Dict[str, Any]):
        """Queue video metadata for batch processing"""
        try:
            self.video_queue.put(('video', video_metadata))
        except Exception as e:
            self.logger.error(f"Error queueing video metadata: {str(e)}")

    def queue_status_update(self, camera_id: int, status: str):
        """Queue camera status update for batch processing"""
        try:
            self.frame_queue.put(('status_update', camera_id, status))
        except Exception as e:
            self.logger.error(f"Error queueing status update: {str(e)}")

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down DatacenterDatabaseWriter")
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=30)
        if not self.frame_queue.empty():
            self.logger.warning(f"{self.frame_queue.qsize()} items left in queue during shutdown")
        
        # Shutdown upload executor
        self.upload_executor.shutdown(wait=True)
        
        self.logger.info("DatacenterDatabaseWriter shutdown complete")


# Legacy compatibility - alias the new class names
ProcessedFrame = DatacenterProcessedFrame
DatabaseWriter = DatacenterDatabaseWriter
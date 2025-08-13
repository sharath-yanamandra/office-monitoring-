#!/usr/bin/env python3
"""
Base class for all datacenter camera models. Provides common functionality and standardizes the interface.
"""

import cv2
import numpy as np
import os
import uuid
import time
import threading
from datetime import datetime
import pytz

from .kalman_track import Sort
from ..logger import setup_datacenter_logger
from ..config import Config
from ..database import DatacenterDatabase
from ..db_writer import DatacenterProcessedFrame

class DatacenterCameraModelBase:
    """
    Base class for all datacenter camera-specific models.
    Handles common functionality like tracking, database operations, and frame saving.
    """
    
    def __init__(self, camera_id, zones=None, rules=None, settings=None, db=None, 
                 db_writer=None, frames_base_dir='frames', camera_manager=None):
        """
        Initialize the datacenter camera model base class
        
        Args:
            camera_id: Camera identifier
            zones: List of zone objects for this camera
            rules: Dict of rules for this camera
            settings: Additional settings specific to this camera
            db: Database instance or None to create a new one
            db_writer: DatabaseWriter instance for database operations
            frames_base_dir: Base directory for saving frames
            camera_manager: CameraManager instance for accessing video buffers
        """
        # Initialize logger with datacenter context
        self.logger = setup_datacenter_logger(f'datacenter_camera_model_{camera_id}', f'datacenter_camera_model_{camera_id}.log')
        self.logger.info(f"Initializing datacenter camera model base for camera {camera_id}")
        
        # Store parameters
        self.camera_id = camera_id
        self.zones = zones or {}
        self.rules = rules or []
        self.settings = settings or {}
        self.camera_manager = camera_manager
        
        # Get datacenter_id from database based on camera_id
        self.datacenter_id = self._get_datacenter_id()
        
        # Initialize database
        self.db = db if db is not None else DatacenterDatabase()
        
        # Storage and database writer
        self.db_writer = db_writer
        self.frames_base_dir = frames_base_dir
        
        # Create directory for this camera with datacenter context
        self.camera_output_dir = os.path.join(self.frames_base_dir, f"datacenter_{self.datacenter_id}", f"camera_{camera_id}")
        os.makedirs(self.camera_output_dir, exist_ok=True)
        
        # Create directory for video clips
        self.video_output_dir = os.path.join(self.camera_output_dir, "clips")
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Initialize tracker
        self.object_tracker = Sort(
            max_age=Config.MAX_AGE,
            min_ma=Config.MIN_MA
        )
        
        # Tracking state
        self.tracked_objects = {}
        self.total_object_count = 0  # Total count of objects seen by the tracker
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'videos_saved': 0,
            'object_count': 0,
            'last_processed_time': None,
            'start_time': time.time()
        }
        
        # Video buffer is now managed by CameraManager
        # We'll fetch frames from there when needed for video saving
        
        # Video saving configuration
        self.video_fps = Config.VIDEO_FPS
        self.video_extension = Config.VIDEO_EXTENSION
        self.video_codec = Config.VIDEO_CODEC
        
        # Video recording thresholds
        # Flag to prevent multiple video recordings of the same event in quick succession
        self.recording_cooldown = Config.EVENT_COOLDOWN  # seconds
        self.last_recording_time = 0
        
        # Lock for thread-safe access to frame buffer
        self.buffer_lock = threading.Lock()
        
        # Background thread for video saving
        self.video_saving_threads = []
        
        # Media storage configuration - extracted from settings
        self.recent_events = {}
        self.tracking_threshold = self.settings.get('tracking_threshold', Config.TRACKING_THRESHOLD)
        self.media_preference = self.settings.get('media_preference', Config.MEDIA_PREFERENCE)
        self.event_cooldown = self.settings.get('event_cooldown', Config.EVENT_COOLDOWN)
        self.auto_recording_enabled = self.settings.get('auto_recording_enabled', Config.AUTO_RECORDING_ENABLED)
        
        self.logger.info(f"Datacenter camera model base initialized for camera {camera_id} in datacenter {self.datacenter_id}")
    
    def _get_datacenter_id(self):
        """Get datacenter_id for this camera from the database"""
        try:
            if hasattr(self, 'db') and self.db:
                db_instance = self.db
            else:
                db_instance = DatacenterDatabase()
            
            result = db_instance.execute_query(
                "SELECT datacenter_id FROM cameras WHERE camera_id = %s", 
                (self.camera_id,)
            )
            if result:
                return result[0]['datacenter_id']
            else:
                self.logger.warning(f"No datacenter found for camera {self.camera_id}, using default datacenter_id=1")
                return 1
        except Exception as e:
            self.logger.error(f"Error getting datacenter_id for camera {self.camera_id}: {e}")
            return 1  # Default datacenter_id
    
    def process_frame(self, frame, timestamp, detection_result=None):
        """
        Process a frame with detection results.
        This is the main entry point that should be called from the video processor.
        
        Args:
            frame: The frame to process
            timestamp: The timestamp of the frame
            detection_result: The detection result from the model (optional)
            
        Returns:
            Tuple of (annotated_frame, detection_list)
        """
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['last_processed_time'] = timestamp
        
        # Frame buffering is now handled by CameraManager at original FPS
        
        # Perform custom processing in subclasses
        annotated_frame, detections = self._process_frame_impl(frame, timestamp, detection_result)
        
        return annotated_frame, detections
    
    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Implement frame processing in subclasses.
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Detection model result
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        # This should be overridden by subclasses
        return frame, []
    
    def update_tracker(self, detection_array):
        """
        Update object tracker with new detections
        
        Args:
            detection_array: Numpy array of detections
            
        Returns:
            List of updated detections with tracking information
        """
        try:
            # Update tracker using the Sort tracker
            self.logger.debug(f"Detection array shape: {detection_array.shape if len(detection_array) > 0 else 'empty'}")
            if len(detection_array) > 0:
                tracked_objects_array, object_count = self.object_tracker.update(detection_array)
            else:
                tracked_objects_array, object_count = self.object_tracker.update()
            
            # Store the total object count from the tracker
            self.total_object_count = object_count
            
            # Create a list for updated detections
            updated_detections = []
            
            # Process tracker output and update our tracked_objects dictionary
            for i in range(tracked_objects_array.shape[0]):
                track = tracked_objects_array[i]
                x, y, aspect_ratio, height, track_id = track
                track_id = int(track_id)  # Convert from float to int
                
                # Convert back to bbox format
                width = aspect_ratio * height
                x1 = x - width/2
                y1 = y - height/2
                x2 = x + width/2
                y2 = y + height/2
                
                # Create or update entry in tracked_objects dictionary
                if track_id not in self.tracked_objects:
                    # New track - initialize with basic information
                    self.tracked_objects[track_id] = {
                        'track_id': track_id,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int(x), int(y)),
                        'frames_tracked': 1
                    }
                else:
                    # Existing track - update with new position and increment counter
                    self.tracked_objects[track_id].update({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int(x), int(y)),
                        'frames_tracked': self.tracked_objects[track_id].get('frames_tracked', 0) + 1
                    })
                
                # The Sort tracker already did the association work for us in its update method
                # Create a detection object with tracking information
                detection_copy = {
                    'track_id': track_id,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'frames_tracked': self.tracked_objects[track_id].get('frames_tracked', 1)
                }
                
                # Add to updated detections
                updated_detections.append(detection_copy)
                
            return updated_detections
            
        except Exception as e:
            self.logger.error(f"Error updating tracker: {str(e)}", exc_info=True)
            # Return empty list on error
            return []
    
    def _save_frame_with_detections(self, event_id, frame, timestamp):
        """
        Save a frame with detections to disk and queue for database storage
        
        Args:
            event_id: The ID of the event to associate this frame with
            frame: The annotated frame to save
            timestamp: The timestamp of the frame
            
        Returns:
            Path to the saved frame
        """
        try:
            # Generate timestamp string
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')
                
            frame_filename = f"datacenter_{self.datacenter_id}_cam{self.camera_id}_{timestamp_str}.jpg"
            frame_path = os.path.join(self.camera_output_dir, frame_filename)
            
            # Save the frame
            try:
                cv2.imwrite(frame_path, frame)
            except Exception as e:
                self.logger.error(f"Error saving frame: {str(e)}", exc_info=True)
            
            # Queue frame for database storage if db_writer is available
            if self.db_writer:
                # Make sure event_id is valid
                if not event_id:
                    self.logger.warning(f"No event_id provided for frame {frame_path}, generating random ID")
                    event_id = str(uuid.uuid4())
                
                frame_info = DatacenterProcessedFrame(
                    camera_id=self.camera_id,
                    datacenter_id=self.datacenter_id,
                    event_id=event_id,
                    timestamp=datetime.fromtimestamp(timestamp),
                    local_path=frame_path,
                    frame_path="",  # Will be set by db_writer during upload
                )
                self.db_writer.queue_frame(frame_info)
                self.logger.info(f"Queued frame {frame_path} for database storage with event ID {event_id}")
            
            self.logger.debug(f"Saved frame to {frame_path}")
            return frame_path
            
        except Exception as e:
            self.logger.error(f"Error saving frame with detections: {str(e)}", exc_info=True)
            return None
    
    def _save_event_video(self, event_id, detection, event_timestamp):
        """
        Queue video recording for event (past + future seconds)
        This method will return immediately and video will be saved asynchronously
        
        Args:
            event_id: The ID of the event
            detection: The detection that triggered the event
            event_timestamp: The timestamp of the event
            
        Returns:
            Path where the video will be saved (actual saving happens asynchronously)
        """
        try:
            # Create the video path
            if isinstance(event_timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(event_timestamp).strftime('%Y%m%d_%H%M%S')
            else:
                timestamp_str = datetime.fromisoformat(event_timestamp).strftime('%Y%m%d_%H%M%S')
            
            # Create a descriptive filename
            detection_type = detection.get('type', 'unknown')
            video_filename = f"datacenter_{self.datacenter_id}_cam{self.camera_id}_{timestamp_str}_{detection_type}_{event_id[:8]}.{self.video_extension}"
            video_path = os.path.join(self.video_output_dir, video_filename)
            
            # Queue the video recording request with the camera manager
            if hasattr(self.camera_manager, 'queue_event_recording'):
                self.logger.info(f"Queueing video recording for event {event_id}")
                
                # Create callback to update database when video is ready
                def video_completion_callback(evt_id, vid_path, duration):
                    self.logger.info(f"Video recording completed for event {evt_id}: {vid_path} ({duration:.1f}s)")
                    if self.db_writer:
                        # Queue video metadata for upload
                        video_metadata = {
                            'event_id': evt_id,
                            'video_path': vid_path,
                            'duration': duration,
                            'detection_type': detection_type,
                            'camera_id': self.camera_id,
                            'datacenter_id': self.datacenter_id,
                            'timestamp': event_timestamp
                        }
                        self.db_writer.queue_video_metadata(video_metadata)
                        self.logger.info(f"Queued video metadata for completed recording {evt_id}")
                
                recording_queued = self.camera_manager.queue_event_recording(
                    camera_id=self.camera_id,
                    event_id=event_id,
                    event_timestamp=event_timestamp,
                    video_path=video_path,
                    past_seconds=Config.VIDEO_BUFFER_PAST_SECONDS,
                    future_seconds=Config.VIDEO_BUFFER_FUTURE_SECONDS,
                    completion_callback=video_completion_callback
                )
                
                if recording_queued:
                    self.logger.info(f"Successfully queued video recording for event {event_id}")
                    return video_path
                else:
                    self.logger.error(f"Failed to queue video recording for event {event_id}")
                    return None
            else:
                # Fallback to old method - just save the current buffer
                self.logger.warning("Camera manager doesn't support event recording, falling back to buffer save")
                return self._save_buffer_video(event_id, detection, event_timestamp)
                
        except Exception as e:
            self.logger.error(f"Error in _save_event_video: {str(e)}", exc_info=True)
            return None
    
    def _save_buffer_video(self, event_id, detection, event_timestamp):
        """
        Fallback method to save just the current buffer (past footage)
        
        Args:
            event_id: The ID of the event
            detection: The detection that triggered the event
            event_timestamp: The timestamp of the event
            
        Returns:
            Path to the saved video or None on failure
        """
        try:
            # Get frame buffer from CameraManager
            if not self.camera_manager:
                self.logger.error(f"No camera manager available for video saving for event {event_id}")
                return None
                
            frames_and_timestamps = self.camera_manager.get_camera_video_buffer(self.camera_id)
            
            if not frames_and_timestamps:
                self.logger.warning(f"No frames in buffer for event {event_id}")
                return None
            
            # Separate frames and timestamps
            frames = [item[0] for item in frames_and_timestamps]
            timestamps = [item[1] for item in frames_and_timestamps]
            
            # Create a timestamp string for the filename
            if isinstance(event_timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(event_timestamp).strftime('%Y%m%d_%H%M%S')
            else:
                timestamp_str = datetime.fromisoformat(event_timestamp).strftime('%Y%m%d_%H%M%S')
            
            # Create a descriptive filename
            detection_type = detection.get('type', 'unknown')
            video_filename = f"datacenter_{self.datacenter_id}_cam{self.camera_id}_{timestamp_str}_{detection_type}_{event_id[:8]}.{self.video_extension}"
            video_path = os.path.join(self.video_output_dir, video_filename)
            
            # Get the dimensions of the first frame (already resized to VIDEO_RESOLUTION)
            if not frames:
                self.logger.error(f"No frames available for video recording for event {event_id}")
                return None
                
            height, width = frames[0].shape[:2]
            self.logger.info(f"Creating buffer video ({len(frames)} frames) with resolution {width}x{height} at {self.video_fps} FPS")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
            
            # Check if video writer was initialized successfully
            if not video_writer.isOpened():
                self.logger.error(f"Failed to initialize video writer with codec {self.video_codec}")
                return None
            
            # Write all frames from the buffer
            for i, frame in enumerate(frames):
                # Ensure frame is contiguous and in correct format
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                # Ensure frame is in BGR format and uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Write frame
                video_writer.write(frame)
                
                # Flush periodically to prevent buffer issues
                if i % 10 == 0:  # Every 10 frames
                    cv2.waitKey(1)
            
            self.logger.info(f"Successfully wrote {len(frames)} frames")
            
            # Get actual start and end times from the buffer
            actual_start_time = timestamps[0] if timestamps else None
            actual_end_time = timestamps[-1] if timestamps else None
            
            # Release video writer and ensure final flush
            video_writer.release()
            cv2.waitKey(1)  # Final flush
            
            # Verify the video file exists before proceeding
            if not os.path.exists(video_path):
                self.logger.error(f"Video file was not created at {video_path} for event {event_id}")
                # List files in the directory to debug
                if os.path.exists(self.video_output_dir):
                    self.logger.error(f"Files in {self.video_output_dir}: {os.listdir(self.video_output_dir)}")
                return None
                
            # Get absolute path to ensure consistency across threads
            video_path = os.path.abspath(video_path)
            
            # Log video duration and file size
            duration = 0
            if actual_start_time and actual_end_time:
                duration = actual_end_time - actual_start_time
                
                # Get file size
                try:
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
                    self.logger.info(f"Saved video clip ({duration:.1f}s, {file_size:.1f}MB, {width}x{height}) "
                                   f"for event {event_id} to {video_path}")
                except Exception as e:
                    self.logger.info(f"Saved video clip ({duration:.1f}s, {width}x{height}) for event {event_id} to {video_path}")
                    self.logger.warning(f"Could not get file size: {str(e)}")
            else:
                self.logger.warning(f"Could not determine video duration for event {event_id}")
            
            # Update statistics
            self.stats['videos_saved'] += 1
            
            # Return the video path and duration - don't queue here, it will be done in save_event_to_db
            return video_path
            
        except Exception as e:
            self.logger.error(f"Error saving event video for event {event_id}: {str(e)}", exc_info=True)
            return None
    
    def _queue_event(self, event_type, detections, timestamp, zone_name=None, event_id=None, video_path=None, video_duration=None):
        """
        Queue an event for database storage
        
        Args:
            event_type: Type of event
            detections: Detection object with event information
            timestamp: The timestamp of the event
            zone_name: Name of the zone where event occurred
            event_id: Optional pre-generated event ID
            video_path: Optional local path to video file
            video_duration: Optional video duration
            
        Returns:
            event_id: Generated event ID if successful, None otherwise
        """
        try:
            # Generate unique event ID
            event_id = event_id or str(uuid.uuid4())
            
            # Format timestamp for database
            dt_timestamp = datetime.fromtimestamp(timestamp).isoformat() if isinstance(timestamp, (int, float)) else timestamp
            
            # Prepare event data with datacenter context
            event_data = {
                'event_id': event_id,
                'camera_id': self.camera_id,
                'datacenter_id': self.datacenter_id,
                'timestamp': dt_timestamp,
                'event_type': event_type,
                'metadata': detections,
                'zone_name': zone_name
            }
            
            # Add video information if available
            if video_path:
                event_data['video_path'] = video_path
                event_data['video_duration'] = video_duration or Config.VIDEO_BUFFER_PAST_SECONDS + Config.VIDEO_BUFFER_FUTURE_SECONDS

            # Queue event for database storage if db_writer is available
            if self.db_writer:
                self.db_writer.queue_event(event_data)
                self.logger.info(f"Queued event {event_id} for camera {self.camera_id} in datacenter {self.datacenter_id}")
                self.stats['events_detected'] += 1
                return event_id
            else:
                self.logger.warning("No db_writer available to queue event")
                return None
                
        except Exception as e:
            self.logger.error(f"Error queueing event: {str(e)}", exc_info=True)
            return None
    
    def _should_record_event(self, event_type, zone, track_id):
        """Determine if an event should be recorded"""
        current_time = time.time()
        
        if track_id not in self.tracked_objects:
            self.logger.debug(f"Not recording {event_type} event in {zone}: track_id {track_id} not in tracked objects")
            return False
            
        frames_tracked = self.tracked_objects[track_id].get('frames_tracked', 0)
        if frames_tracked < self.tracking_threshold:
            self.logger.debug(f"Not recording {event_type} event in {zone}: track_id {track_id} " +
                             f"only tracked for {frames_tracked}/{self.tracking_threshold} frames")
            return False
            
        event_key = f"{event_type}_{zone}"
        if event_key in self.recent_events:
            time_since_last = current_time - self.recent_events[event_key]
            if time_since_last <= self.event_cooldown:
                self.logger.debug(f"Not recording {event_type} event in {zone}: " +
                                 f"cooldown active ({time_since_last:.1f}s/{self.event_cooldown}s)")
                return False
        
        self.logger.info(f"Will record {event_type} event in {zone} for track_id {track_id}")
        self.recent_events[event_key] = current_time
        return True
    
    def _save_event_media(self, event_type, detections, frame, timestamp, zone_name):
        """Save media for an event"""
        # Generate event ID first
        event_id = str(uuid.uuid4())
        
        if self.media_preference == "image":
            self.logger.info(f"Saving frame for {event_type} event {event_id}")
            # Save frame first so we can include its path in the event data
            frame_path = self._save_frame_with_detections(event_id, frame, timestamp)
            
            if frame_path:
                self.stats["frames_saved"] = self.stats.get("frames_saved", 0) + 1
                self.logger.info(f"Saved frame to {frame_path} for event {event_id}")
            else:
                self.logger.error(f"Failed to save frame for {event_type} event {event_id}")
            
            # Queue event after saving the frame
            self._queue_event(event_type, detections, timestamp, zone_name, event_id)
        else:
            # For video, save the video first and get the path
            self.logger.info(f"Saving video for {event_type} event {event_id}")
            
            # Prepare detection data for video saving
            detection_data = {
                'type': event_type,
                'detections': detections,
                'zone_name': zone_name
            }
            
            # Queue event immediately (without video path)
            self._queue_event(event_type, detections, timestamp, zone_name, event_id)
            
            # Start video recording (this will update the event when complete)
            video_path = self._save_event_video(event_id, detection_data, timestamp)
            
            if video_path:
                self.logger.info(f"Video recording queued for event {event_id}, will be saved to {video_path}")
            else:
                self.logger.error(f"Failed to queue video recording for {event_type} event {event_id}")
            
        return event_id
    
    def is_in_zone(self, point, zone):
        """
        Check if a point is inside a zone
        
        Args:
            point: (x, y) tuple
            zone: Zone coordinates as polygon
            
        Returns:
            True if point is in zone, False otherwise
        """
        # Extract zone coordinates
        if isinstance(zone, (list, tuple)) and len(zone) >= 4:
            # If zone is passed as (x1, y1, x2, y2, ...)
            polygon = np.array(zone[:4]).reshape((-1, 2))
        elif isinstance(zone, dict) and 'coordinates' in zone:
            # If zone is passed as {'coordinates': [[x1, y1], [x2, y2], ...]}
            polygon = np.array(zone['coordinates'])
        else:
            return False
            
        # Check if point is inside polygon
        return cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False) >= 0
    
    def object_zone_overlap(self, bbox, zone, min_overlap_ratio=0.5):
        """
        Calculate how much of an object overlaps with a zone
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            zone: Zone coordinates as polygon
            min_overlap_ratio: Minimum ratio of overlap to consider object in zone
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        x1, y1, x2, y2 = bbox
        
        # Create a mask for the bounding box
        min_x, min_y = int(min(x1, x2)), int(min(y1, y2))
        max_x, max_y = int(max(x1, x2)), int(max(y1, y2))
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            return 0.0
            
        # Create mask
        bbox_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Create mask for zone
        if isinstance(zone, (list, tuple)) and len(zone) >= 4:
            # If zone is passed as (x1, y1, x2, y2, ...)
            polygon = np.array(zone[:4]).reshape((-1, 2))
        elif isinstance(zone, dict) and 'coordinates' in zone:
            # If zone is passed as {'coordinates': [[x1, y1], [x2, y2], ...]}
            polygon = np.array(zone['coordinates'])
        else:
            return 0.0
            
        # Adjust polygon to mask coordinates
        polygon_adj = polygon - np.array([min_x, min_y])
        
        # Draw zone on mask
        zone_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [polygon_adj.astype(np.int32)], 255)
        
        # Calculate overlap
        overlap = cv2.bitwise_and(bbox_mask, zone_mask)
        overlap_area = cv2.countNonZero(overlap)
        bbox_area = cv2.countNonZero(bbox_mask)
        
        if bbox_area == 0:
            return 0.0
            
        overlap_ratio = overlap_area / bbox_area
        
        return overlap_ratio
    
    def get_current_people_count(self):
        """Get current number of people detected by this camera"""
        return len(self.tracked_objects)
    
    def get_stats(self):
        """
        Get statistics for this camera model
        
        Returns:
            Dict with statistics
        """
        # Calculate uptime
        uptime = time.time() - self.stats.get('start_time', time.time())
        
        # Calculate processing rate
        frames_processed = self.stats.get('frames_processed', 0)
        fps = frames_processed / max(1, uptime)
        
        stats = {
            'camera_id': self.camera_id,
            'datacenter_id': self.datacenter_id,
            'camera_type': self.__class__.__name__,
            'frames_processed': frames_processed,
            'events_detected': self.stats.get('events_detected', 0),
            'videos_saved': self.stats.get('videos_saved', 0),
            'tracked_objects': len(self.tracked_objects),
            'total_object_count': self.total_object_count,
            'uptime_seconds': uptime,
            'fps': fps,
            'last_processed': self.stats.get('last_processed_time')
        }
        
        # Add camera model specific stats
        stats.update(self.stats)
        
        return stats


# Legacy compatibility - alias the new class name
CameraModelBase = DatacenterCameraModelBase
#!/usr/bin/env python3
"""
Camera Manager for Datacenter Monitoring

This module is responsible for:
1. Reading frames from multiple RTSP camera sources
2. Managing camera connection status
3. Queueing frames for batch processing
4. Providing statistics about camera health
5. Routing processed results back to individual cameras
6. Motion detection and video buffering for events
"""

import cv2
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
from logger import setup_datacenter_logger
from config import DatacenterConfig

class CameraFrame:
    """Data class for a single camera frame with metadata"""
    def __init__(self, camera_id: str, frame: np.ndarray, timestamp: float, frame_number: int):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.processed_result = None  # Store the processed result here
        
    def set_processed_result(self, result):
        """Set the processed result for this frame"""
        self.processed_result = result


class CameraReader:
    """Manages a single camera stream"""
    def __init__(self, camera_id: str, stream_url: str, frame_queue: queue.Queue, result_callback=None, 
                 logger=None, target_fps=None, video_buffer_callback=None):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.frame_queue = frame_queue
        self.result_callback = result_callback
        self.video_buffer_callback = video_buffer_callback  # Callback to send frames to video buffer
        self.logger = logger or setup_datacenter_logger(f'camera_reader_{camera_id}', f'camera_reader_{camera_id}.log')
        
        # Camera status
        self.running = False
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = DatacenterConfig.MAX_RETRIES
        self.reconnect_delay = 5  # seconds
        
        # Frame counters
        self.frame_count = 0
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = 0
        self.fps = 0
        self.original_fps = 25  # Will be updated when camera connects
        
        # Allow per-camera FPS control for processing
        self.target_fps = target_fps if target_fps is not None else DatacenterConfig.READER_FPS_LIMIT
        
        # Thread management
        self.reader_thread = None
        self.streaming_thread = None
        
        # Result queue for processed frames
        self.result_queue = queue.Queue(maxsize=DatacenterConfig.MAX_QUEUE_SIZE)
        self.result_thread = None
        
        # Streaming queue - separate from processing queue
        self.streaming_queue = queue.Queue(maxsize=10)
        
        # Video buffer for this camera - stores frames at original FPS
        self.video_buffer = queue.Queue(maxsize=DatacenterConfig.VIDEO_BUFFER_SIZE)
        self.buffer_lock = threading.Lock()
        
        # Motion detection
        self.motion_detector = None
        self.motion_detected = False
        self.last_motion_time = 0
        self.enable_motion_detection = DatacenterConfig.MOTION_DETECTION_ENABLED
        self.motion_warmup_frames = 0  # Counter for warmup frames
        self.motion_stats = {
            'total_checks': 0,
            'motion_frames': 0,
            'idle_periods': 0,
            'last_motion_area': 0
        }

    def start_result_processor(self):
        """Start the result processor thread"""
        if not self.result_thread or not self.result_thread.is_alive():
            self.result_thread = threading.Thread(
                target=self._result_processor_worker,
                daemon=True,
                name=f"result_processor_{self.camera_id}"
            )
            self.result_thread.start()
            self.logger.info(f"Started result processor for camera {self.camera_id}")
            return True
        return False

    def _result_processor_worker(self):
        """Process results from the GPU"""
        self.logger.info(f"Result processor worker started for camera {self.camera_id}")
        
        results_processed = 0
        callback_errors = 0
        last_stats_time = time.time()
        total_callback_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Log periodic stats
                if current_time - last_stats_time >= 60:
                    avg_callback_time = 0
                    if results_processed > 0:
                        avg_callback_time = total_callback_time / results_processed
                        
                    self.logger.info(f"Result processor stats for camera {self.camera_id}: "
                                   f"processed {results_processed} results with {callback_errors} errors in last 60s, "
                                   f"avg callback time: {avg_callback_time:.3f}s")
                    results_processed = 0
                    callback_errors = 0
                    total_callback_time = 0
                    last_stats_time = current_time
                
                # Get result from queue with timeout
                try:
                    queue_start = time.time()
                    result_tuple = self.result_queue.get(timeout=0.5)
                    self.result_queue.task_done()
                    queue_time = time.time() - queue_start
                    
                    # Handle both formats: with and without enhanced metadata
                    if len(result_tuple) == 4:
                        frame, result, timestamp, enhanced_metadata = result_tuple
                    else:
                        frame, result, timestamp = result_tuple
                        enhanced_metadata = None
                    
                    self.logger.debug(f"Got result from queue for camera {self.camera_id} in {queue_time:.3f}s, "
                                    f"result is {type(result).__name__}, timestamp: {timestamp:.3f}")
                    
                except queue.Empty:
                    continue
                
                # Process the result if callback is provided
                if self.result_callback:
                    try:
                        callback_start = time.time()
                        # Pass enhanced metadata if available
                        if enhanced_metadata:
                            self.result_callback(self.camera_id, frame, result, timestamp, enhanced_metadata)
                        else:
                            self.result_callback(self.camera_id, frame, result, timestamp)
                        callback_time = time.time() - callback_start
                        
                        total_callback_time += callback_time
                        results_processed += 1
                        
                        self.logger.debug(f"Result callback completed for camera {self.camera_id} in {callback_time:.3f}s")
                        
                    except Exception as e:
                        callback_errors += 1
                        self.logger.error(f"Error in result callback for camera {self.camera_id}: {str(e)}", exc_info=True)
                else:
                    self.logger.warning(f"No result callback defined for camera {self.camera_id}, result discarded")
                
            except Exception as e:
                self.logger.error(f"Error in result processor for camera {self.camera_id}: {str(e)}", exc_info=True)
                time.sleep(0.1)
        
        self.logger.info(f"Result processor worker stopped for camera {self.camera_id}")
        
    def queue_result(self, frame, result, timestamp, enhanced_metadata=None):
        """Queue a processed result for this camera"""
        try:
            if self.result_queue.full():
                self.logger.warning(f"Result queue full for camera {self.camera_id}, dropping result")
                return False
            
            # Include enhanced metadata in the result tuple if provided
            if enhanced_metadata:
                self.result_queue.put((frame, result, timestamp, enhanced_metadata))
            else:
                self.result_queue.put((frame, result, timestamp))
            return True
        except Exception as e:
            self.logger.error(f"Error queueing result for camera {self.camera_id}: {str(e)}")
            return False
    
    def _init_motion_detector(self):
        """Initialize motion detector with background subtraction"""
        try:
            # Using MOG2 background subtractor - adaptive and robust
            self.motion_detector = cv2.createBackgroundSubtractorMOG2(
                detectShadows=False,
                varThreshold=DatacenterConfig.MOTION_THRESHOLD
            )
            self.motion_warmup_frames = 0
            self.logger.info(f"Motion detector initialized for camera {self.camera_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize motion detector: {str(e)}")
            self.enable_motion_detection = False
    
    def _detect_motion(self, frame):
        """
        Detect motion in frame using background subtraction
        Returns True if motion detected, False otherwise
        """
        if not self.enable_motion_detection:
            return True  # Always return motion if disabled
        
        try:
            # Initialize detector on first use
            if self.motion_detector is None:
                self._init_motion_detector()
            
            # Apply background subtraction
            fg_mask = self.motion_detector.apply(frame)
            
            # Skip warmup frames to let background model stabilize
            self.motion_warmup_frames += 1
            if self.motion_warmup_frames < DatacenterConfig.MOTION_WARMUP_FRAMES:
                return True  # Allow processing during warmup
            
            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate motion area
            motion_pixels = cv2.countNonZero(fg_mask)
            total_pixels = frame.shape[0] * frame.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            # Update stats
            self.motion_stats['total_checks'] += 1
            self.motion_stats['last_motion_area'] = motion_ratio
            
            # Check if motion exceeds threshold (10% of frame area)
            motion_detected = motion_ratio > DatacenterConfig.MOTION_AREA_THRESHOLD
            
            if motion_detected:
                self.motion_detected = True
                self.last_motion_time = time.time()
                self.motion_stats['motion_frames'] += 1
                
                if self.motion_stats['total_checks'] % 300 == 0:  # Log every 300 frames
                    self.logger.debug(f"Camera {self.camera_id} motion detected: {motion_ratio:.2%} of frame")
            
            return motion_detected
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {str(e)}")
            return True  # Default to processing on error
        
    def start(self):
        """Start the camera reader thread"""
        if self.running:
            self.logger.warning(f"Camera {self.camera_id} reader already running")
            return False
            
        self.running = True
        self.reader_thread = threading.Thread(
            target=self._reader_worker,
            daemon=True,
            name=f"camera_reader_{self.camera_id}"
        )
        self.reader_thread.start()
        
        # Also start the result processor
        self.start_result_processor()
        
        self.logger.info(f"Started camera reader for {self.camera_id} ({self.stream_url}) at {self.target_fps} FPS")
        return True
    
    def stop(self):
        """Stop the camera reader thread"""
        self.logger.info(f"Stopping camera reader for {self.camera_id}")
        self.running = False
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=3.0)
            if self.reader_thread.is_alive():
                self.logger.warning(f"Camera reader thread for {self.camera_id} did not terminate gracefully")
                
        if self.result_thread and self.result_thread.is_alive():
            self.result_thread.join(timeout=3.0)
            if self.result_thread.is_alive():
                self.logger.warning(f"Result processor thread for {self.camera_id} did not terminate gracefully")
                
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=3.0)
            if self.streaming_thread.is_alive():
                self.logger.warning(f"Streaming thread for {self.camera_id} did not terminate gracefully")
    
    def _reader_worker(self):
        """Camera reader worker thread"""
        self.logger.info(f"Camera reader worker started for {self.camera_id}")
        
        frames_processed = 0
        frames_read = 0
        frames_dropped = 0
        last_stats_time = time.time()
        connection_attempts = 0
        
        while self.running:
            try:
                # Open camera connection
                connection_start = time.time()
                connection_attempts += 1
                if not self._connect_to_camera():
                    connection_time = time.time() - connection_start
                    self.logger.warning(f"Connection attempt {connection_attempts} failed for camera {self.camera_id} "
                                     f"after {connection_time:.3f}s, will retry in {self.reconnect_delay}s")
                    # Reconnection will be attempted after delay
                    time.sleep(self.reconnect_delay)
                    continue
                
                connection_time = time.time() - connection_start
                self.logger.info(f"Connected to camera {self.camera_id} after {connection_time:.3f}s")
                connection_attempts = 0
                
                # Initialize FPS control
                processing_interval = 1.0 / max(1, self.target_fps)  # Interval for GPU processing
                last_processing_time = time.time()
                last_frame_time = time.time()
                actual_fps = 0
                frames_buffered = 0
                
                # Read frames while connected
                while self.running and self.is_connected:
                    # Periodic stats logging
                    current_time = time.time()
                    if current_time - last_stats_time >= 60:
                        motion_info = ""
                        if self.enable_motion_detection:
                            motion_pct = (self.motion_stats['motion_frames'] / max(1, self.motion_stats['total_checks'])) * 100
                            motion_info = f", motion: {motion_pct:.1f}% ({self.motion_stats['motion_frames']}/{self.motion_stats['total_checks']} frames)"
                        
                        self.logger.info(f"Camera {self.camera_id} stats: read {frames_read} frames, "
                                      f"processed {frames_processed}, buffered {frames_buffered}, dropped {frames_dropped} in last 60s, "
                                      f"actual FPS: {actual_fps:.2f}, processing FPS: {self.target_fps}{motion_info}")
                        frames_read = 0
                        frames_processed = 0
                        frames_buffered = 0
                        frames_dropped = 0
                        last_stats_time = current_time
                    
                    # Read frame at original camera FPS (no throttling)
                    read_start = time.time()
                    ret, frame = self.cap.read()
                    read_time = time.time() - read_start
                    
                    if not ret:
                        self.logger.warning(f"Failed to read frame from camera {self.camera_id} after {read_time:.3f}s")
                        self.is_connected = False
                        self.frames_dropped += 1
                        frames_dropped += 1
                        break
                    
                    frames_read += 1
                    
                    # Update frame stats
                    self.frame_count += 1
                    self.frames_captured += 1
                    frame_timestamp = time.time()
                    frame_interval = frame_timestamp - last_frame_time
                    self.fps = 1.0 / frame_interval if frame_interval > 0 else 0
                    actual_fps = self.fps  # Store for logging
                    last_frame_time = frame_timestamp
                    self.last_frame_time = frame_timestamp
                    
                    # Always add frame to video buffer (circular buffer at original FPS)
                    try:
                        # Resize frame for video buffer to save storage
                        target_width, target_height = DatacenterConfig.VIDEO_RESOLUTION
                        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                        
                        with self.buffer_lock:
                            if self.video_buffer.full():
                                # Remove oldest frame to make room
                                try:
                                    self.video_buffer.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            # Add resized frame with timestamp
                            self.video_buffer.put((resized_frame, frame_timestamp))
                            frames_buffered += 1
                            
                        # Also send to video buffer callback if provided (send resized frame)
                        if self.video_buffer_callback:
                            self.video_buffer_callback(self.camera_id, resized_frame, frame_timestamp)
                            
                    except Exception as e:
                        self.logger.error(f"Error buffering frame: {str(e)}")
                    
                    # Check if we should send this frame for GPU processing
                    time_since_last_processing = current_time - last_processing_time
                    
                    if time_since_last_processing >= processing_interval:
                        # Check for motion before GPU processing
                        motion_detected = self._detect_motion(frame)
                        
                        # Handle motion timeout
                        if self.motion_detected and not motion_detected:
                            time_since_motion = current_time - self.last_motion_time
                            if time_since_motion > DatacenterConfig.MOTION_IDLE_TIMEOUT:
                                self.motion_detected = False
                                self.motion_stats['idle_periods'] += 1
                                self.logger.info(f"Camera {self.camera_id}: No motion for {DatacenterConfig.MOTION_IDLE_TIMEOUT}s, "
                                               f"pausing GPU processing (motion area: {self.motion_stats['last_motion_area']:.2%})")
                        
                        # Only process if motion detected or still in motion window
                        if self.motion_detected:
                            # Send frame for processing at target FPS
                            self.logger.debug(f"Read frame {self.frame_count} from {self.camera_id} in {read_time:.3f}s, "
                                            f"interval: {frame_interval:.3f}s, FPS: {self.fps:.2f}, sending for processing")
                            
                            # Create frame object for processing pipeline
                            frame_obj = CameraFrame(
                                camera_id=self.camera_id,
                                frame=frame,
                                timestamp=frame_timestamp,
                                frame_number=self.frame_count
                            )
                            
                            # Check if processing queue is full
                            if self.frame_queue.full():
                                self.logger.warning(f"Frame queue full for camera {self.camera_id}, dropping frame {self.frame_count}")
                                self.frames_dropped += 1
                                frames_dropped += 1
                            else:
                                # Add frame to processing queue
                                queue_start = time.time()
                                self.frame_queue.put(frame_obj)
                                queue_time = time.time() - queue_start
                                
                                frames_processed += 1
                                last_processing_time = current_time
                                self.logger.debug(f"Added frame {self.frame_count} to processing queue in {queue_time:.3f}s")
                        else:
                            # No motion detected, skip GPU processing
                            last_processing_time = current_time  # Update time to maintain interval
                    
                    # Also add to streaming queue (replacing oldest frame if full)
                    streaming_start = time.time()
                    streaming_dropped = 0
                    
                    if self.streaming_queue.full():
                        try:
                            self.streaming_queue.get_nowait()  # Remove oldest frame
                            self.streaming_queue.task_done()
                            streaming_dropped += 1
                        except queue.Empty:
                            pass
                    
                    try:
                        self.streaming_queue.put_nowait(frame.copy())
                        streaming_time = time.time() - streaming_start
                        
                        if streaming_dropped > 0:
                            self.logger.debug(f"Added frame {self.frame_count} to streaming queue in {streaming_time:.3f}s "
                                           f"(dropped {streaming_dropped} old frames)")
                        else:
                            self.logger.debug(f"Added frame {self.frame_count} to streaming queue in {streaming_time:.3f}s")
                            
                    except queue.Full:
                        self.logger.debug(f"Streaming queue unexpectedly full for camera {self.camera_id}, frame {self.frame_count} not queued")
                
                # If we're here, we lost connection
                if self.running:
                    self.logger.warning(f"Lost connection to camera {self.camera_id}, will reconnect")
                    self._release_camera()
                
            except Exception as e:
                self.logger.error(f"Error in camera reader for {self.camera_id}: {str(e)}", exc_info=True)
                self._release_camera()
                time.sleep(self.reconnect_delay)
        
        # Cleanup before exiting
        self._release_camera()
        self.logger.info(f"Camera reader worker stopped for {self.camera_id}")
    
    def _connect_to_camera(self):
        """Attempt to connect to the camera"""
        try:
            connection_start = time.time()
            self.logger.info(f"Connecting to camera {self.camera_id} at {self.stream_url}")
            
            # Set OpenCV options for RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Open camera connection
            self.cap = cv2.VideoCapture(self.stream_url)
            
            # Set additional OpenCV properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            
            # Check if connection is successful
            connection_time = time.time() - connection_start
            
            if not self.cap.isOpened():
                self.reconnect_attempts += 1
                self.logger.warning(
                    f"Failed to connect to camera {self.camera_id} after {connection_time:.3f}s, "
                    f"attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}"
                )
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached for camera {self.camera_id}, stopping")
                    self.running = False
                
                return False
            
            # Successfully connected
            self.is_connected = True
            prev_attempts = self.reconnect_attempts
            self.reconnect_attempts = 0
            
            # Get camera information if available
            try:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                # Update original FPS if valid
                if fps > 0:
                    self.original_fps = fps
                    # Update video buffer size based on actual FPS (only past seconds)
                    new_buffer_size = int(DatacenterConfig.VIDEO_BUFFER_PAST_SECONDS * fps)
                    self.video_buffer = queue.Queue(maxsize=new_buffer_size)
                    
                    self.logger.info(f"Updated video buffer size to {new_buffer_size} frames for {DatacenterConfig.VIDEO_BUFFER_PAST_SECONDS}s past footage at {fps:.2f} FPS")
                
                target_width, target_height = DatacenterConfig.VIDEO_RESOLUTION
                self.logger.info(f"Successfully connected to camera {self.camera_id} after {connection_time:.3f}s "
                               f"({prev_attempts} previous attempts) - Original: {width}x{height}, "
                               f"Video Buffer: {target_width}x{target_height}, Source FPS: {fps:.2f}")
            except:
                self.logger.info(f"Successfully connected to camera {self.camera_id} after {connection_time:.3f}s "
                               f"({prev_attempts} previous attempts)")
            
            return True
            
        except Exception as e:
            connection_time = time.time() - connection_start
            self.logger.error(f"Error connecting to camera {self.camera_id} after {connection_time:.3f}s: {str(e)}", exc_info=True)
            self.reconnect_attempts += 1
            return False
    
    def _release_camera(self):
        """Release camera resources"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error releasing camera {self.camera_id}: {str(e)}")
    
    def get_status(self):
        """Get camera status information"""
        return {
            'camera_id': self.camera_id,
            'connected': self.is_connected,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'last_frame_time': self.last_frame_time,
            'fps': self.fps,
            'target_fps': self.target_fps,
            'original_fps': self.original_fps,
            'reconnect_attempts': self.reconnect_attempts,
            'video_buffer_size': self.video_buffer.qsize(),
            'motion_detected': self.motion_detected,
            'motion_stats': self.motion_stats.copy() if self.enable_motion_detection else None
        }
    
    def get_video_buffer(self):
        """Get a copy of the current video buffer frames and timestamps"""
        with self.buffer_lock:
            frames_and_timestamps = []
            temp_storage = []
            
            # Extract all frames from queue
            while not self.video_buffer.empty():
                try:
                    item = self.video_buffer.get_nowait()
                    frames_and_timestamps.append(item)
                    temp_storage.append(item)
                except queue.Empty:
                    break
            
            # Put them back to maintain the buffer
            for item in temp_storage:
                try:
                    self.video_buffer.put_nowait(item)
                except queue.Full:
                    break
                    
            return frames_and_timestamps
    
    def get_motion_status(self):
        """Get current motion detection status"""
        return {
            'motion_detected': self.motion_detected,
            'last_motion_time': self.last_motion_time,
            'motion_enabled': self.enable_motion_detection,
            'motion_stats': self.motion_stats.copy()
        }


class DatacenterCameraManager:
    """Manages multiple camera streams and provides batched frames for datacenter monitoring"""
    def __init__(self):
        self.logger = setup_datacenter_logger('datacenter_camera_manager', 'datacenter_camera_manager.log')
        self.logger.info("Initializing Datacenter Camera Manager")
        
        # Log configuration settings
        self.logger.info(f"Configuration settings:")
        self.logger.info(f"- Max retries: {DatacenterConfig.MAX_RETRIES}")
        self.logger.info(f"- Reader FPS limit: {DatacenterConfig.READER_FPS_LIMIT}")
        self.logger.info(f"- Batch size: {DatacenterConfig.BATCH_SIZE}")
        self.logger.info(f"- Max queue size: {DatacenterConfig.MAX_QUEUE_SIZE}")
        self.logger.info(f"- Activity levels - High: {DatacenterConfig.ACTIVITY_LEVEL_HIGH}, Medium: {DatacenterConfig.ACTIVITY_LEVEL_MEDIUM}, Low: {DatacenterConfig.ACTIVITY_LEVEL_LOW}")
        
        # Camera-specific frame queues
        self.camera_queues = {}
        
        # Batch queue for GPU processing
        self.batch_queue = queue.Queue(maxsize=20)  # Queue of ready-to-process batches
        
        # Camera readers
        self.camera_readers = {}
        
        # Batch collection 
        self.batch_collection_running = False
        self.batch_collection_threads = []
        self.num_batch_threads = DatacenterConfig.NUM_BATCH_THREADS if hasattr(DatacenterConfig, 'NUM_BATCH_THREADS') else 2
        self.logger.info(f"Using {self.num_batch_threads} batch collection threads")
        
        # Activity levels for cameras - set in start_camera
        self.camera_activity_levels = {}
        
        # Result callback function
        self.result_callback = None
        
        self.logger.info("Datacenter Camera Manager initialization complete")
    
    def set_result_callback(self, callback_fn):
        """Set a callback function to be called when results are processed"""
        self.result_callback = callback_fn
        
        # Update existing camera readers with the new callback
        for camera_id, reader in self.camera_readers.items():
            reader.result_callback = callback_fn
            
        self.logger.info("Set result callback function")
        
    def route_result_to_camera(self, camera_id: str, frame: np.ndarray, result: Any, timestamp: float, enhanced_metadata: dict = None) -> bool:
        """Route a processed result back to the appropriate camera"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot route result: Camera {camera_id} not found")
            return False
        
        return self.camera_readers[camera_id].queue_result(frame, result, timestamp, enhanced_metadata)
        
    def start_camera(self, camera_id: str, stream_url: str, activity_level: str = 'medium'):
        """
        Start a camera reader for the given camera ID and stream URL with activity-based FPS
        
        Args:
            camera_id: Unique identifier for the camera
            stream_url: RTSP or file URL for the camera stream
            activity_level: 'high', 'medium', or 'low' to determine frame rate
        """
        if camera_id in self.camera_readers:
            self.logger.warning(f"Camera {camera_id} already started")
            return False
        
        # Set FPS based on activity level
        target_fps = self._get_fps_for_activity_level(activity_level)
        self.camera_activity_levels[camera_id] = activity_level
        
        # Create a dedicated queue for this camera
        camera_queue = queue.Queue(maxsize=DatacenterConfig.MAX_QUEUE_SIZE)
        self.camera_queues[camera_id] = camera_queue
        
        # Create and start the camera reader
        reader = CameraReader(
            camera_id=camera_id, 
            stream_url=stream_url, 
            frame_queue=camera_queue, 
            result_callback=self.result_callback, 
            logger=self.logger,
            target_fps=target_fps
        )
        
        success = reader.start()
        
        if success:
            self.camera_readers[camera_id] = reader
            self.logger.info(f"Started camera {camera_id} at {target_fps} FPS (activity level: {activity_level})")
        
        return success
    
    def _get_fps_for_activity_level(self, activity_level: str) -> int:
        """Get the appropriate FPS based on activity level"""
        if activity_level == 'high':
            return DatacenterConfig.ACTIVITY_LEVEL_HIGH  # High activity areas get 10 FPS
        elif activity_level == 'low':
            return DatacenterConfig.ACTIVITY_LEVEL_LOW  # Low activity areas get 2 FPS
        else:  # medium or any other value
            return DatacenterConfig.ACTIVITY_LEVEL_MEDIUM  # Medium activity gets 4 FPS
    
    def start_cameras(self, camera_sources: Dict[str, Tuple[str, str]]):
        """
        Start multiple camera readers from a dictionary mapping camera_id -> (stream_url, activity_level)
        
        Args:
            camera_sources: Dict mapping camera_id to a tuple of (stream_url, activity_level)
        """
        self.logger.info(f"Starting {len(camera_sources)} cameras")
        start_time = time.time()
        
        successful_starts = 0
        failed_starts = 0
        
        for camera_id, (stream_url, activity_level) in camera_sources.items():
            camera_start_time = time.time()
            success = self.start_camera(camera_id, stream_url, activity_level)
            camera_start_duration = time.time() - camera_start_time
            
            if success:
                successful_starts += 1
                self.logger.info(f"Camera {camera_id} started successfully in {camera_start_duration:.3f}s")
            else:
                failed_starts += 1
                self.logger.error(f"Failed to start camera {camera_id} after {camera_start_duration:.3f}s")
        
        # Start the batch collection threads if not already running
        if camera_sources and not self.batch_collection_running:
            batch_start_time = time.time()
            self.start_batch_collection()
            batch_start_duration = time.time() - batch_start_time
            self.logger.info(f"Batch collection started in {batch_start_duration:.3f}s")
        
        total_duration = time.time() - start_time
        self.logger.info(f"Camera startup complete: {successful_starts} started, {failed_starts} failed in {total_duration:.3f}s")
        
        return len(self.camera_readers)
    
    def stop_camera(self, camera_id: str):
        """Stop the camera reader for the given camera ID"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Camera {camera_id} not found")
            return False
        
        reader = self.camera_readers[camera_id]
        reader.stop()
        del self.camera_readers[camera_id]
        
        # Also remove the camera's queue
        if camera_id in self.camera_queues:
            del self.camera_queues[camera_id]
            
        # Remove activity level
        if camera_id in self.camera_activity_levels:
            del self.camera_activity_levels[camera_id]
            
        self.logger.info(f"Stopped camera {camera_id}")
        return True
    
    def stop_all_cameras(self):
        """Stop all camera readers"""
        self.logger.info(f"Stopping all cameras ({len(self.camera_readers)})")
        
        camera_ids = list(self.camera_readers.keys())
        for camera_id in camera_ids:
            self.stop_camera(camera_id)
        
        # Stop the batch collection threads
        self.stop_batch_collection()
        
        self.logger.info("All cameras stopped")
    
    def start_batch_collection(self):
        """Start multiple batch collection threads"""
        if self.batch_collection_running:
            self.logger.warning("Batch collection already running")
            return False
        
        self.batch_collection_running = True
        self.batch_collection_threads = []
        
        for i in range(self.num_batch_threads):
            thread = threading.Thread(
                target=self._batch_collection_worker,
                args=(i,),
                daemon=True,
                name=f"batch_collection_{i}"
            )
            thread.start()
            self.batch_collection_threads.append(thread)
        
        self.logger.info(f"Started {self.num_batch_threads} batch collection threads")
        return True
    
    def stop_batch_collection(self):
        """Stop all batch collection threads"""
        self.logger.info("Stopping batch collection threads")
        self.batch_collection_running = False
        
        for thread in self.batch_collection_threads:
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    self.logger.warning(f"Batch collection thread {thread.name} did not terminate gracefully")
        
        self.logger.info("Batch collection stopped")
    
    def _batch_collection_worker(self, worker_id: int):
        """
        Worker thread to collect frames from camera queues into batches.
        Each worker is assigned a subset of cameras to handle.
        
        Args:
            worker_id: ID of this worker thread
        """
        self.logger.info(f"Batch collection worker {worker_id} started")
        
        pending_frames = []
        last_frame_time = time.time()
        last_camera_idx = 0
        frames_collected = 0
        batches_created = 0
        cameras_checked = 0
        last_stats_time = time.time()
        
        while self.batch_collection_running:
            try:
                current_time = time.time()
                
                # Log stats periodically
                if current_time - last_stats_time >= 60:
                    self.logger.info(f"Worker {worker_id} stats: collected {frames_collected} frames, "
                                   f"created {batches_created} batches, checked {cameras_checked} cameras in last 60s, "
                                   f"current pending: {len(pending_frames)}")
                    frames_collected = 0
                    batches_created = 0
                    cameras_checked = 0
                    last_stats_time = current_time
                
                # Check if we have enough frames for a batch or timeout occurred
                timeout_occurred = (current_time - last_frame_time) >= DatacenterConfig.BATCH_TIMEOUT
                
                if len(pending_frames) >= DatacenterConfig.BATCH_SIZE or (pending_frames and timeout_occurred):
                    # Create a batch from pending frames
                    batch_size = min(DatacenterConfig.BATCH_SIZE, len(pending_frames))
                    batch_frames = pending_frames[:batch_size]
                    pending_frames = pending_frames[batch_size:]
                    
                    batch_start = time.time()
                    
                    # Add batch to batch queue
                    frames = [frame.frame for frame in batch_frames]
                    metadata = [{
                        'camera_id': frame.camera_id,
                        'timestamp': frame.timestamp,
                        'frame_number': frame.frame_number
                    } for frame in batch_frames]
                    
                    batch_prep_time = time.time() - batch_start
                    
                    # Log batch creation with camera IDs
                    camera_ids = [frame.camera_id for frame in batch_frames]
                    self.logger.debug(f"Worker {worker_id} created batch of {len(frames)} frames in {batch_prep_time:.3f}s from cameras: {camera_ids}")
                    
                    # If batch queue is full, wait a bit
                    if self.batch_queue.full():
                        self.logger.warning(f"Batch queue full, worker {worker_id} waiting...")
                        time.sleep(0.1)
                    
                    queue_start = time.time()
                    
                    # Add batch to queue
                    self.batch_queue.put((frames, metadata))
                    
                    queue_time = time.time() - queue_start
                    self.logger.debug(f"Worker {worker_id} added batch to queue in {queue_time:.3f}s")
                    
                    last_frame_time = current_time
                    batches_created += 1
                
                # Determine which cameras to process based on worker ID
                # Distribute cameras across worker threads
                camera_ids = list(self.camera_queues.keys())
                if not camera_ids:
                    self.logger.debug(f"Worker {worker_id}: No cameras configured, waiting...")
                    time.sleep(0.1)
                    continue
                
                # Use a round-robin approach to process all camera queues fairly
                cameras_to_check = []
                for i in range(len(camera_ids)):
                    idx = (last_camera_idx + i) % len(camera_ids)
                    camera_id = camera_ids[idx]
                    # Only process cameras assigned to this worker
                    if idx % self.num_batch_threads == worker_id:
                        cameras_to_check.append(camera_id)
                
                # Update the last processed camera index
                last_camera_idx = (last_camera_idx + 1) % len(camera_ids)
                
                # Check each assigned camera for frames
                frames_added = 0
                cameras_checked += len(cameras_to_check)
                
                for camera_id in cameras_to_check:
                    try:
                        queue_check_start = time.time()
                        
                        # Get a frame from this camera if available (non-blocking)
                        if camera_id in self.camera_queues and not self.camera_queues[camera_id].empty():
                            frame = self.camera_queues[camera_id].get(block=False)
                            pending_frames.append(frame)
                            self.camera_queues[camera_id].task_done()
                            frames_added += 1
                            frames_collected += 1
                            
                            queue_check_time = time.time() - queue_check_start
                            self.logger.debug(f"Worker {worker_id}: Got frame from camera {camera_id} in {queue_check_time:.3f}s")
                    except queue.Empty:
                        continue
                    except KeyError:
                        # Camera might have been removed
                        self.logger.debug(f"Worker {worker_id}: Camera {camera_id} no longer exists, skipping")
                        continue
                
                # If no frames were added from any camera, sleep briefly
                if not frames_added:
                    time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in batch collection worker {worker_id}: {str(e)}", exc_info=True)
                time.sleep(0.1)
        
        self.logger.info(f"Batch collection worker {worker_id} stopped")
    
    def get_batch(self, timeout=1.0):
        """Get a batch of frames for processing"""
        try:
            start_time = time.time()
            
            batch = self.batch_queue.get(timeout=timeout)
            wait_duration = time.time() - start_time
            
            frames, metadata = batch
            if frames and metadata:
                camera_ids = [meta['camera_id'] for meta in metadata]
                camera_count = len(set(camera_ids))
                frame_count = len(frames)
                
                self.logger.debug(f"Retrieved batch of {frame_count} frames from {camera_count} unique cameras in {wait_duration:.3f}s")
            else:
                self.logger.debug(f"Retrieved empty batch after {wait_duration:.3f}s wait")
                
            self.batch_queue.task_done()
            return batch
            
        except queue.Empty:
            wait_duration = time.time() - start_time
            self.logger.debug(f"No batch available after {wait_duration:.3f}s timeout")
            return None, None
    
    def get_camera_stats(self):
        """Get status information for all cameras"""
        stats = {}
        for camera_id, reader in self.camera_readers.items():
            stats[camera_id] = reader.get_status()
            # Add activity level to stats
            stats[camera_id]['activity_level'] = self.camera_activity_levels.get(camera_id, 'medium')
        return stats
    
    def update_camera_activity(self, camera_id: str, activity_level: str):
        """
        Update the activity level and frame rate for a camera
        
        Args:
            camera_id: The camera ID to update
            activity_level: New activity level ('high', 'medium', 'low')
        """
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot update activity: Camera {camera_id} not found")
            return False
        
        # Calculate new FPS based on activity level
        new_fps = self._get_fps_for_activity_level(activity_level)
        
        # Update camera's target FPS
        reader = self.camera_readers[camera_id]
        old_fps = reader.target_fps
        reader.target_fps = new_fps
        
        # Update activity level tracking
        self.camera_activity_levels[camera_id] = activity_level
        
        self.logger.info(f"Updated camera {camera_id} activity to {activity_level} (FPS: {old_fps} → {new_fps})")
        return True
    
    def get_camera_video_buffer(self, camera_id: str):
        """Get the video buffer for a specific camera"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot get video buffer: Camera {camera_id} not found")
            return []
        
        return self.camera_readers[camera_id].get_video_buffer()
    
    def get_camera_reader(self, camera_id: str):
        """Get the camera reader instance for a specific camera"""
        return self.camera_readers.get(camera_id)
    
    def queue_event_recording(self, camera_id: str, event_id: str, event_timestamp, 
                            video_path: str, past_seconds: int = 10, future_seconds: int = 50,
                            completion_callback=None):
        """
        Queue a video recording for an event
        This will save the past N seconds from buffer and continue recording for future M seconds
        
        Args:
            camera_id: Camera ID
            event_id: Event ID
            event_timestamp: Timestamp when event was triggered
            video_path: Path where video should be saved
            past_seconds: Seconds of past footage to include (default 10)
            future_seconds: Seconds of future footage to record (default 50)
            completion_callback: Function to call when recording is complete (optional)
            
        Returns:
            bool: True if recording was queued successfully
        """
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot queue recording: Camera {camera_id} not found")
            return False
        
        # Create a recording task
        recording_task = {
            'camera_id': camera_id,
            'event_id': event_id,
            'event_timestamp': event_timestamp,
            'video_path': video_path,
            'past_seconds': past_seconds,
            'future_seconds': future_seconds,
            'start_time': time.time(),
            'completion_callback': completion_callback
        }
        
        # Start recording thread
        recording_thread = threading.Thread(
            target=self._record_event_video,
            args=(recording_task,),
            daemon=True,
            name=f"event_recorder_{event_id}"
        )
        recording_thread.start()
        
        self.logger.info(f"Started event recording thread for event {event_id} on camera {camera_id}")
        return True
    
    def _record_event_video(self, recording_task):
        """
        Thread function to record event video
        """
        try:
            camera_id = recording_task['camera_id']
            event_id = recording_task['event_id']
            video_path = recording_task['video_path']
            past_seconds = recording_task['past_seconds']
            future_seconds = recording_task['future_seconds']
            
            self.logger.info(f"Starting event recording for {event_id}: {past_seconds}s past + {future_seconds}s future")
            
            # Get past frames from buffer
            camera_reader = self.camera_readers.get(camera_id)
            if not camera_reader:
                self.logger.error(f"Camera reader not found for {camera_id}")
                return
            
            past_frames_and_timestamps = camera_reader.get_video_buffer()
            
            # Collect future frames
            future_frames = []
            future_timestamps = []
            start_time = time.time()
            
            # Create a temporary queue for collecting frames
            frame_collector = queue.Queue()
            
            # Register a callback to collect frames
            def collect_frame(cam_id, frame, timestamp):
                if cam_id == camera_id:
                    frame_collector.put((frame, timestamp))
            
            # Temporarily set the callback
            original_callback = camera_reader.video_buffer_callback
            camera_reader.video_buffer_callback = collect_frame
            
            # Collect frames for the specified duration
            while (time.time() - start_time) < future_seconds:
                try:
                    # Get frame with timeout
                    frame_data = frame_collector.get(timeout=1.0)
                    future_frames.append(frame_data[0])
                    future_timestamps.append(frame_data[1])
                except queue.Empty:
                    # No frame available, continue waiting
                    pass
            
            # Restore original callback
            camera_reader.video_buffer_callback = original_callback
            
            self.logger.info(f"Collected {len(future_frames)} future frames for event {event_id}")
            
            # Combine past and future frames
            all_frames = [f[0] for f in past_frames_and_timestamps] + future_frames
            all_timestamps = [f[1] for f in past_frames_and_timestamps] + future_timestamps
            
            if not all_frames:
                self.logger.error(f"No frames collected for event {event_id}")
                return
            
            # Save video
            success = self._save_video(video_path, all_frames, all_timestamps, event_id)
            
            # Call completion callback if provided
            if recording_task.get('completion_callback') and success:
                try:
                    recording_task['completion_callback'](event_id, video_path, len(all_frames)/DatacenterConfig.VIDEO_FPS)
                except Exception as e:
                    self.logger.error(f"Error calling completion callback: {str(e)}", exc_info=True)
            
        except Exception as e:
            self.logger.error(f"Error recording event video: {str(e)}", exc_info=True)
    
    def _save_video(self, video_path, frames, timestamps, event_id):
        """Save frames to video file"""
        try:
            if not frames:
                self.logger.error(f"No frames to save for event {event_id}")
                return False
            
            # Get video parameters
            height, width = frames[0].shape[:2]
            fps = DatacenterConfig.VIDEO_FPS
            codec = DatacenterConfig.VIDEO_CODEC
            
            self.logger.info(f"Saving video with {len(frames)} frames at {width}x{height} {fps}fps")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                self.logger.error(f"Failed to open video writer for {video_path}")
                return False
            
            # Write frames
            for i, frame in enumerate(frames):
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                writer.write(frame)
                
                # Periodic flush
                if i % 20 == 0:
                    cv2.waitKey(1)
            
            writer.release()
            cv2.waitKey(1)
            
            # Verify file exists
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
                duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
                self.logger.info(f"Successfully saved video for event {event_id}: "
                               f"{len(frames)} frames, {duration:.1f}s, {file_size:.1f}MB")
                return True
            else:
                self.logger.error(f"Video file not created: {video_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving video: {str(e)}", exc_info=True)
            return False


# For testing
if __name__ == "__main__":
    # Test with a local video file
    manager = DatacenterCameraManager()
    
    # Start cameras with different activity levels
    manager.start_camera("high_activity", "test1.mp4", "high")  # 10 FPS
    manager.start_camera("medium_activity", "test2.mp4", "medium")  # 4 FPS
    manager.start_camera("low_activity", "test3.mp4", "low")  # 2 FPS
    
    # Process a few batches
    try:
        for _ in range(5):
            frames, metadata = manager.get_batch(timeout=5.0)
            if frames:
                print(f"Got batch of {len(frames)} frames")
                for meta in metadata:
                    print(f"  Camera: {meta['camera_id']}, Timestamp: {meta['timestamp']}")
            else:
                print("No batch available")
            time.sleep(1)
    finally:
        manager.stop_all_cameras()
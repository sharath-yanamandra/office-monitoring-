#!/usr/bin/env python3
"""
camera_manager.py
Camera Stream Management for Video Monitoring

This module handles:
1. RTSP camera stream management
2. Frame capture and preprocessing
3. Connection handling and recovery
4. Multiple camera coordination
"""

import cv2
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any
import logging
import numpy as np

from config import Config

class CameraStream:
    """Individual camera stream handler"""
    
    def __init__(self, camera_id: int, stream_url: str, name: str = None):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.name = name or f"Camera_{camera_id}"
        
        self.logger = logging.getLogger(f'camera_{camera_id}')
        
        # Stream objects
        self.cap = None
        self.is_connected = False
        self.is_running = False
        
        # Frame handling
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0
        
        # Threading
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
        # Connection parameters
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Frame queue for buffering
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Callbacks
        self.frame_callback = None
        
    def connect(self) -> bool:
        """Connect to camera stream"""
        try:
            self.logger.info(f"Connecting to camera {self.name}: {self.stream_url}")
            
            self.cap = cv2.VideoCapture(self.stream_url)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test connection
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_connected = True
                self.reconnect_attempts = 0
                self.logger.info(f"Successfully connected to camera {self.name}")
                
                # Store first frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.last_frame_time = time.time()
                
                return True
            else:
                self.logger.error(f"Failed to read frame from camera {self.name}")
                self.disconnect()
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to camera {self.name}: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Disconnect from camera stream"""
        self.is_connected = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.logger.info(f"Disconnected from camera {self.name}")
    
    def start_capture(self, frame_callback: Callable = None):
        """Start frame capture in separate thread"""
        if not self.is_connected:
            self.logger.error(f"Camera {self.name} not connected")
            return False
        
        self.frame_callback = frame_callback
        self.is_running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.logger.info(f"Started capture for camera {self.name}")
        return True
    
    def stop_capture(self):
        """Stop frame capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        self.logger.info(f"Stopped capture for camera {self.name}")
    
    def _capture_loop(self):
        """Main frame capture loop"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running and self.is_connected:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    current_time = time.time()
                    
                    # Update current frame
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.frame_count += 1
                        self.last_frame_time = current_time
                    
                    # Add to queue (non-blocking)
                    try:
                        self.frame_queue.put({
                            'frame': frame,
                            'timestamp': current_time,
                            'frame_id': self.frame_count
                        }, block=False)
                    except queue.Full:
                        # Remove oldest frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put({
                                'frame': frame,
                                'timestamp': current_time,
                                'frame_id': self.frame_count
                            }, block=False)
                        except queue.Empty:
                            pass
                    
                    # Call frame callback if provided
                    if self.frame_callback:
                        try:
                            self.frame_callback(self.camera_id, frame, current_time)
                        except Exception as e:
                            self.logger.error(f"Error in frame callback: {e}")
                
                else:
                    consecutive_failures += 1
                    self.logger.warning(f"Failed to read frame from camera {self.name} ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        self.logger.error(f"Too many consecutive failures for camera {self.name}, attempting reconnection")
                        self._attempt_reconnection()
                        consecutive_failures = 0
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop for camera {self.name}: {e}")
                time.sleep(1)
    
    def _attempt_reconnection(self):
        """Attempt to reconnect to camera"""
        self.disconnect()
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self.logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} for camera {self.name}")
            
            time.sleep(self.reconnect_delay)
            
            if self.connect():
                self.logger.info(f"Successfully reconnected to camera {self.name}")
            else:
                self.logger.error(f"Reconnection failed for camera {self.name}")
        else:
            self.logger.error(f"Max reconnection attempts reached for camera {self.name}")
            self.is_running = False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get the latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_info(self) -> Dict[str, Any]:
        """Get camera frame information"""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'last_frame_time': self.last_frame_time,
            'queue_size': self.frame_queue.qsize()
        }


class CameraManager:
    """Manages multiple camera streams"""
    
    def __init__(self):
        self.logger = logging.getLogger('camera_manager')
        self.logger.info("Initializing CameraManager")
        
        # Camera streams
        self.cameras: Dict[int, CameraStream] = {}
        self.active_cameras: List[int] = []
        
        # Frame callbacks
        self.frame_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'total_cameras': 0,
            'active_cameras': 0,
            'total_frames': 0,
            'failed_connections': 0
        }
    
    def add_camera(self, camera_id: int, stream_url: str, name: str = None) -> bool:
        """Add camera to manager"""
        if camera_id in self.cameras:
            self.logger.warning(f"Camera {camera_id} already exists")
            return False
        
        camera = CameraStream(camera_id, stream_url, name)
        self.cameras[camera_id] = camera
        self.stats['total_cameras'] += 1
        
        self.logger.info(f"Added camera {camera_id}: {name or stream_url}")
        return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove camera from manager"""
        if camera_id not in self.cameras:
            self.logger.warning(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        camera.stop_capture()
        camera.disconnect()
        
        del self.cameras[camera_id]
        
        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
        
        self.stats['total_cameras'] -= 1
        self.logger.info(f"Removed camera {camera_id}")
        return True
    
    def connect_camera(self, camera_id: int) -> bool:
        """Connect specific camera"""
        if camera_id not in self.cameras:
            self.logger.error(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        if camera.connect():
            if camera_id not in self.active_cameras:
                self.active_cameras.append(camera_id)
            self.stats['active_cameras'] = len(self.active_cameras)
            return True
        else:
            self.stats['failed_connections'] += 1
            return False
    
    def disconnect_camera(self, camera_id: int) -> bool:
        """Disconnect specific camera"""
        if camera_id not in self.cameras:
            self.logger.error(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        camera.stop_capture()
        camera.disconnect()
        
        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
        
        self.stats['active_cameras'] = len(self.active_cameras)
        return True
    
    def connect_all_cameras(self) -> Dict[int, bool]:
        """Connect all cameras"""
        results = {}
        
        for camera_id in self.cameras:
            results[camera_id] = self.connect_camera(camera_id)
        
        self.logger.info(f"Connected {sum(results.values())}/{len(results)} cameras")
        return results
    
    def start_camera_capture(self, camera_id: int) -> bool:
        """Start capture for specific camera"""
        if camera_id not in self.cameras:
            self.logger.error(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        return camera.start_capture(self._frame_callback)
    
    def start_all_captures(self) -> Dict[int, bool]:
        """Start capture for all connected cameras"""
        results = {}
        
        for camera_id in self.active_cameras:
            results[camera_id] = self.start_camera_capture(camera_id)
        
        self.logger.info(f"Started capture for {sum(results.values())}/{len(results)} cameras")
        return results
    
    def stop_all_captures(self):
        """Stop capture for all cameras"""
        for camera_id in self.active_cameras:
            camera = self.cameras[camera_id]
            camera.stop_capture()
        
        self.logger.info("Stopped all camera captures")
    
    def add_frame_callback(self, callback: Callable):
        """Add frame callback function"""
        self.frame_callbacks.append(callback)
    
    def _frame_callback(self, camera_id: int, frame: np.ndarray, timestamp: float):
        """Internal frame callback that calls all registered callbacks"""
        self.stats['total_frames'] += 1
        
        for callback in self.frame_callbacks:
            try:
                callback(camera_id, frame, timestamp)
            except Exception as e:
                self.logger.error(f"Error in frame callback: {e}")
    
    def get_camera_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get current frame from specific camera"""
        if camera_id not in self.cameras:
            return None
        
        return self.cameras[camera_id].get_current_frame()
    
    def get_all_camera_frames(self) -> Dict[int, np.ndarray]:
        """Get current frames from all active cameras"""
        frames = {}
        
        for camera_id in self.active_cameras:
            frame = self.get_camera_frame(camera_id)
            if frame is not None:
                frames[camera_id] = frame
        
        return frames
    
    def get_camera_info(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Get information about specific camera"""
        if camera_id not in self.cameras:
            return None
        
        return self.cameras[camera_id].get_frame_info()
    
    def get_all_cameras_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all cameras"""
        info = {}
        
        for camera_id in self.cameras:
            info[camera_id] = self.get_camera_info(camera_id)
        
        return info
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get camera manager statistics"""
        return {
            **self.stats,
            'cameras': list(self.cameras.keys()),
            'active_cameras': self.active_cameras.copy()
        }
    
    def cleanup(self):
        """Cleanup all resources"""
        self.stop_all_captures()
        
        for camera_id in list(self.cameras.keys()):
            self.remove_camera(camera_id)
        
        self.logger.info("Camera manager cleanup completed")


# Export main classes
__all__ = ['CameraManager', 'CameraStream']
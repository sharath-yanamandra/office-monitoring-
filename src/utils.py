#!/usr/bin/env python3
"""
utils.py
Utility Functions for Video Monitoring System

This module provides:
1. Common utility functions
2. Image and video processing helpers
3. Geometry and math utilities
4. File and path utilities
"""

import os
import cv2
import numpy as np
import time
import json
import hashlib
import threading
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import logging

# Geometry utilities
def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm
    
    Args:
        point: (x, y) coordinates
        polygon: List of [x, y] coordinates defining the polygon
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    if len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center point of bounding box
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        
    Returns:
        (center_x, center_y) tuple
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2] first bounding box
        bbox2: [x1, y1, x2, y2] second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Check if there's an intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def expand_bbox(bbox: List[float], expansion_factor: float) -> List[float]:
    """
    Expand bounding box by given factor
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        expansion_factor: Factor to expand (e.g., 1.1 for 10% expansion)
        
    Returns:
        Expanded bounding box
    """
    x1, y1, x2, y2 = bbox
    center_x, center_y = calculate_bbox_center(bbox)
    
    width = x2 - x1
    height = y2 - y1
    
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    return [new_x1, new_y1, new_x2, new_y2]

# Image processing utilities
def resize_frame(frame: np.ndarray, target_width: int = None, 
                target_height: int = None, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize frame while optionally maintaining aspect ratio
    
    Args:
        frame: Input image frame
        target_width: Target width (optional)
        target_height: Target height (optional)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if target_width is None and target_height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if maintain_aspect:
        if target_width and target_height:
            # Calculate scale to fit within both dimensions
            scale_w = target_width / w
            scale_h = target_height / h
            scale = min(scale_w, scale_h)
            
            new_width = int(w * scale)
            new_height = int(h * scale)
        elif target_width:
            scale = target_width / w
            new_width = target_width
            new_height = int(h * scale)
        else:  # target_height
            scale = target_height / h
            new_width = int(w * scale)
            new_height = target_height
    else:
        new_width = target_width or w
        new_height = target_height or h
    
    return cv2.resize(frame, (new_width, new_height))

def apply_blur(frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply Gaussian blur to frame"""
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def enhance_contrast(frame: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    Enhance frame contrast and brightness
    
    Args:
        frame: Input frame
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced frame
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale"""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def crop_frame(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Crop frame using bounding box coordinates
    
    Args:
        frame: Input frame
        bbox: [x1, y1, x2, y2] crop coordinates
        
    Returns:
        Cropped frame
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are within frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return frame[y1:y2, x1:x2]

# File utilities
def ensure_directory_exists(directory_path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(directory_path, exist_ok=True)

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except OSError:
        return ""

def cleanup_old_files(directory: str, max_age_seconds: int):
    """
    Clean up files older than specified age
    
    Args:
        directory: Directory to clean
        max_age_seconds: Maximum age in seconds
    """
    if not os.path.exists(directory):
        return
    
    current_time = time.time()
    deleted_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if current_time - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1
            except OSError:
                pass
    
    return deleted_count

# Time utilities
def format_timestamp(timestamp: float, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format timestamp as string"""
    return datetime.fromtimestamp(timestamp).strftime(format_str)

def get_current_timestamp() -> float:
    """Get current timestamp"""
    return time.time()

def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert timestamp to datetime object"""
    return datetime.fromtimestamp(timestamp)

# JSON utilities
def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely load JSON string with default fallback"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string with default fallback"""
    try:
        return json.dumps(obj, default=str, indent=2)
    except (TypeError, ValueError):
        return default

# Performance utilities
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger('timer')
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.debug(f"{self.operation_name} took {duration:.3f} seconds")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0

class FPSCalculator:
    """Calculate frames per second"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.lock = threading.Lock()
    
    def update(self):
        """Update with new frame"""
        current_time = time.time()
        
        with self.lock:
            self.frame_times.append(current_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS"""
        with self.lock:
            if len(self.frame_times) < 2:
                return 0.0
            
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff <= 0:
                return 0.0
            
            return (len(self.frame_times) - 1) / time_diff

# Validation utilities
def validate_bbox(bbox: List[float], frame_shape: Tuple[int, int]) -> List[float]:
    """
    Validate and clamp bounding box coordinates to frame boundaries
    
    Args:
        bbox: [x1, y1, x2, y2] coordinates
        frame_shape: (height, width) of frame
        
    Returns:
        Validated bounding box
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    
    return [x1, y1, x2, y2]

def validate_polygon(polygon: List[List[float]], frame_shape: Tuple[int, int]) -> List[List[float]]:
    """
    Validate polygon coordinates
    
    Args:
        polygon: List of [x, y] coordinates
        frame_shape: (height, width) of frame
        
    Returns:
        Validated polygon
    """
    h, w = frame_shape[:2]
    validated_polygon = []
    
    for point in polygon:
        x, y = point
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        validated_polygon.append([x, y])
    
    return validated_polygon

def is_valid_confidence(confidence: float) -> bool:
    """Check if confidence value is valid (0.0 to 1.0)"""
    return 0.0 <= confidence <= 1.0

# Configuration utilities
def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.getLogger('utils').error(f"Failed to load config from {config_path}: {e}")
        return {}

def save_json_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to JSON file"""
    try:
        ensure_directory_exists(os.path.dirname(config_path))
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return True
    except Exception as e:
        logging.getLogger('utils').error(f"Failed to save config to {config_path}: {e}")
        return False

# Color utilities
def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors

def get_color_by_id(object_id: int) -> Tuple[int, int, int]:
    """Get consistent color for object ID"""
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
    ]
    return colors[object_id % len(colors)]

# Statistical utilities
def calculate_running_average(values: List[float], window_size: int = 10) -> float:
    """Calculate running average of values"""
    if not values:
        return 0.0
    
    recent_values = values[-window_size:] if len(values) > window_size else values
    return sum(recent_values) / len(recent_values)

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {'count': 0, 'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
    
    values_array = np.array(values)
    
    return {
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'std': float(np.std(values_array))
    }

# Debug utilities
def draw_debug_info(frame: np.ndarray, info: Dict[str, Any], 
                   position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw debug information on frame"""
    y_offset = position[1]
    
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (position[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    return frame

def log_detection_results(detections: List[Dict[str, Any]], logger: logging.Logger):
    """Log detection results for debugging"""
    if detections:
        logger.debug(f"Detected {len(detections)} objects:")
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class_name', 'unknown')
            logger.debug(f"  {i}: {class_name} ({confidence:.2f}) at {bbox}")
    else:
        logger.debug("No detections found")

# Export all utilities
__all__ = [
    # Geometry utilities
    'point_in_polygon', 'calculate_distance', 'calculate_bbox_center', 
    'calculate_bbox_area', 'calculate_iou', 'expand_bbox',
    
    # Image processing utilities
    'resize_frame', 'apply_blur', 'enhance_contrast', 'convert_to_grayscale', 'crop_frame',
    
    # File utilities
    'ensure_directory_exists', 'get_file_size', 'calculate_file_hash', 'cleanup_old_files',
    
    # Time utilities
    'format_timestamp', 'get_current_timestamp', 'timestamp_to_datetime',
    
    # JSON utilities
    'safe_json_loads', 'safe_json_dumps',
    
    # Performance utilities
    'Timer', 'FPSCalculator',
    
    # Validation utilities
    'validate_bbox', 'validate_polygon', 'is_valid_confidence',
    
    # Configuration utilities
    'load_json_config', 'save_json_config',
    
    # Color utilities
    'generate_colors', 'get_color_by_id',
    
    # Statistical utilities
    'calculate_running_average', 'calculate_statistics',
    
    # Debug utilities
    'draw_debug_info', 'log_detection_results'
]
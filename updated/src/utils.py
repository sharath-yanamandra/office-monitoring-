#!/usr/bin/env python3
"""
Script 7: utils.py
File Path: src/utils.py

Datacenter Monitoring System - Common Utilities and Helper Functions

This module provides:
1. Computer vision utilities for detection and tracking
2. Zone management and polygon operations
3. Drawing and visualization functions
4. Time and formatting utilities
5. PPE detection helpers
6. Event processing utilities
7. Performance monitoring helpers
"""

import cv2
import numpy as np
import os
import json
import time
import math
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import uuid
from pathlib import Path

from logger import setup_datacenter_logger

# Initialize logger
logger = setup_datacenter_logger('datacenter_utils', 'datacenter_utils.log')

# Detection class mappings for datacenter monitoring
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# PPE Detection Classes (custom classes for datacenter monitoring)
PPE_CLASSES = {
    80: 'hard_hat',      # Hard hat / helmet
    81: 'safety_vest',   # High-visibility safety vest
    82: 'safety_glasses',# Safety glasses/goggles
    83: 'gloves',        # Safety gloves
    84: 'boots',         # Safety boots
    85: 'mask',          # Face mask
    86: 'ear_protection' # Ear protection
}

# Vehicle classes for perimeter monitoring
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 8: 'boat'}

# Datacenter zone colors for visualization
DATACENTER_ZONE_COLORS = {
    'entry_zone': (0, 255, 255),      # Yellow - Entry points
    'server_zone': (0, 0, 255),       # Red - Server rooms (high security)
    'restricted_zone': (255, 0, 0),   # Blue - Restricted areas
    'common_zone': (0, 255, 0),       # Green - Common areas
    'perimeter_zone': (255, 255, 0),  # Cyan - Perimeter
    'critical_zone': (128, 0, 128)    # Purple - Critical infrastructure
}

# Event severity colors
SEVERITY_COLORS = {
    'low': (0, 255, 0),        # Green
    'medium': (0, 255, 255),   # Yellow
    'high': (0, 165, 255),     # Orange
    'critical': (0, 0, 255)    # Red
}

def format_duration(seconds: Union[int, float]) -> str:
    """
    Format a duration in seconds to a human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2h 30m 15s")
    """
    try:
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except (ValueError, TypeError):
        return "0s"

def format_timestamp(timestamp: Union[float, datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Timestamp as float, datetime, or string
        format_str: Format string for output
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            # Try to parse ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
            
        return dt.strftime(format_str)
    except Exception as e:
        logger.warning(f"Error formatting timestamp {timestamp}: {e}")
        return str(timestamp)

def generate_event_id() -> str:
    """Generate a unique event ID"""
    return f"evt-{uuid.uuid4().hex[:12]}"

def generate_track_id() -> str:
    """Generate a unique tracking ID"""
    return f"track-{uuid.uuid4().hex[:8]}"

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except (TypeError, IndexError):
        return float('inf')

def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center point of bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Center point (x, y)
    """
    try:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    except (ValueError, TypeError, IndexError):
        return (0.0, 0.0)

def calculate_bbox_area(bbox: List[float]) -> float:
    """
    Calculate area of bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Area of bounding box
    """
    try:
        x1, y1, x2, y2 = bbox
        return abs(x2 - x1) * abs(y2 - y1)
    except (ValueError, TypeError, IndexError):
        return 0.0

def bbox_intersection_over_union(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    try:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = calculate_bbox_area(bbox1)
        area2 = calculate_bbox_area(bbox2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    except Exception:
        return 0.0

def is_point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm
    
    Args:
        point: Point to check (x, y)
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...]
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    try:
        x, y = point
        polygon_array = np.array(polygon, dtype=np.float32)
        
        # Use OpenCV's pointPolygonTest for robust polygon checking
        result = cv2.pointPolygonTest(polygon_array, (float(x), float(y)), False)
        return result >= 0
    except Exception as e:
        logger.warning(f"Error in point-in-polygon test: {e}")
        return False

def calculate_polygon_area(polygon: List[List[float]]) -> float:
    """
    Calculate area of polygon using shoelace formula
    
    Args:
        polygon: List of polygon vertices [[x1, y1], [x2, y2], ...]
        
    Returns:
        Area of polygon
    """
    try:
        if len(polygon) < 3:
            return 0.0
            
        area = 0.0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
            
        return abs(area) / 2.0
    except Exception:
        return 0.0

def calculate_bbox_polygon_overlap(bbox: List[float], polygon: List[List[float]], 
                                 min_overlap_ratio: float = 0.5) -> float:
    """
    Calculate overlap ratio between bounding box and polygon
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        polygon: Polygon vertices [[x1, y1], [x2, y2], ...]
        min_overlap_ratio: Minimum overlap ratio to consider
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Create bounding box polygon
        bbox_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        zone_polygon = np.array(polygon, dtype=np.int32)
        
        # Create masks for intersection calculation
        frame_size = (int(max(x2, max([p[0] for p in polygon])) + 10),
                     int(max(y2, max([p[1] for p in polygon])) + 10))
        
        bbox_mask = np.zeros(frame_size, dtype=np.uint8)
        zone_mask = np.zeros(frame_size, dtype=np.uint8)
        
        cv2.fillPoly(bbox_mask, [bbox_polygon], 255)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(bbox_mask, zone_mask)
        intersection_area = cv2.countNonZero(intersection)
        bbox_area = cv2.countNonZero(bbox_mask)
        
        if bbox_area == 0:
            return 0.0
            
        overlap_ratio = intersection_area / bbox_area
        return overlap_ratio
        
    except Exception as e:
        logger.warning(f"Error calculating bbox-polygon overlap: {e}")
        return 0.0

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], 
                            font_scale: float = 0.6, color: Tuple[int, int, int] = (255, 255, 255),
                            thickness: int = 2, bg_color: Tuple[int, int, int] = (0, 0, 0),
                            bg_alpha: float = 0.7, padding: int = 5) -> np.ndarray:
    """
    Draw text with semi-transparent background
    
    Args:
        frame: Image frame to draw on
        text: Text to display
        position: (x, y) position for text
        font_scale: Font scale factor
        color: Text color (B, G, R)
        thickness: Text thickness
        bg_color: Background color (B, G, R)
        bg_alpha: Background transparency (0.0 to 1.0)
        padding: Padding around text
        
    Returns:
        Frame with text drawn
    """
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        
        # Calculate background rectangle
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_h - padding
        bg_x2 = x + text_w + padding
        bg_y2 = y + padding
        
        # Create overlay for semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        
        return frame
        
    except Exception as e:
        logger.warning(f"Error drawing text: {e}")
        return frame

def draw_datacenter_zone(frame: np.ndarray, zone: Dict[str, Any], 
                        alpha: float = 0.3, label: Optional[str] = None) -> np.ndarray:
    """
    Draw a datacenter zone on the frame
    
    Args:
        frame: Frame to draw on
        zone: Zone dictionary with coordinates and type
        alpha: Transparency for zone overlay (0.0 to 1.0)
        label: Optional label override
        
    Returns:
        Frame with zone drawn
    """
    try:
        # Extract zone information
        if 'coordinates' not in zone:
            return frame
            
        coordinates = zone['coordinates']
        zone_type = zone.get('zone_type', zone.get('type', 'unknown'))
        zone_name = label or zone.get('name', zone_type)
        
        # Get zone color
        color = DATACENTER_ZONE_COLORS.get(zone_type, (128, 128, 128))  # Default gray
        
        # Convert coordinates to numpy array
        polygon = np.array(coordinates, dtype=np.int32)
        
        # Create overlay for semi-transparent zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw zone outline
        cv2.polylines(frame, [polygon], True, color, 2)
        
        # Draw zone label
        if zone_name:
            # Find centroid for label placement
            centroid = np.mean(polygon, axis=0).astype(int)
            label_pos = tuple(centroid)
            
            draw_text_with_background(
                frame, zone_name, label_pos, 
                font_scale=0.5, color=(255, 255, 255), bg_color=color
            )
        
        return frame
        
    except Exception as e:
        logger.warning(f"Error drawing zone: {e}")
        return frame

def draw_detection_box(frame: np.ndarray, bbox: List[float], 
                      class_name: str, confidence: float,
                      track_id: Optional[int] = None,
                      color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw detection bounding box with labels
    
    Args:
        frame: Frame to draw on
        bbox: Bounding box [x1, y1, x2, y2]
        class_name: Detected class name
        confidence: Detection confidence
        track_id: Optional tracking ID
        color: Optional custom color
        
    Returns:
        Frame with detection box drawn
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine color based on class
        if color is None:
            if class_name == 'person':
                color = (0, 255, 0)  # Green for persons
            elif class_name in PPE_CLASSES.values():
                color = (0, 255, 255)  # Yellow for PPE
            else:
                color = (255, 0, 0)  # Blue for other objects
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label_parts = [class_name, f"{confidence:.2f}"]
        if track_id is not None:
            label_parts.insert(0, f"ID:{track_id}")
        
        label = " | ".join(label_parts)
        
        # Draw label with background
        label_pos = (x1, y1 - 10)
        draw_text_with_background(frame, label, label_pos, color=(255, 255, 255), bg_color=color)
        
        return frame
        
    except Exception as e:
        logger.warning(f"Error drawing detection box: {e}")
        return frame

def draw_ppe_status(frame: np.ndarray, bbox: List[float], ppe_status: Dict[str, bool],
                   position: str = 'top') -> np.ndarray:
    """
    Draw PPE compliance status for a person
    
    Args:
        frame: Frame to draw on
        bbox: Person's bounding box
        ppe_status: Dictionary of PPE item -> detected status
        position: Where to draw status ('top', 'bottom', 'side')
        
    Returns:
        Frame with PPE status drawn
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create PPE status text
        ppe_items = []
        for ppe_item, detected in ppe_status.items():
            status_char = "✓" if detected else "✗"
            color_name = "green" if detected else "red"
            ppe_items.append(f"{ppe_item}: {status_char}")
        
        ppe_text = " | ".join(ppe_items)
        
        # Determine position
        if position == 'top':
            text_pos = (x1, y1 - 30)
        elif position == 'bottom':
            text_pos = (x1, y2 + 20)
        else:  # side
            text_pos = (x2 + 10, y1)
        
        # Draw PPE status
        overall_compliant = all(ppe_status.values())
        bg_color = (0, 255, 0) if overall_compliant else (0, 0, 255)  # Green if compliant, red if not
        
        draw_text_with_background(
            frame, ppe_text, text_pos,
            color=(255, 255, 255), bg_color=bg_color, font_scale=0.4
        )
        
        return frame
        
    except Exception as e:
        logger.warning(f"Error drawing PPE status: {e}")
        return frame

def is_object_stationary(positions: List[Tuple[float, float]], 
                        time_threshold: float = 5.0,
                        movement_threshold: float = 10.0) -> bool:
    """
    Check if an object is stationary based on position history
    
    Args:
        positions: List of (x, y) positions over time
        time_threshold: Minimum time to be considered stationary
        movement_threshold: Maximum movement distance to be considered stationary
        
    Returns:
        True if object is stationary, False otherwise
    """
    try:
        if len(positions) < 2:
            return False
        
        # Calculate maximum displacement from first position
        first_pos = positions[0]
        max_displacement = 0
        
        for pos in positions[1:]:
            displacement = calculate_distance(first_pos, pos)
            max_displacement = max(max_displacement, displacement)
        
        return max_displacement <= movement_threshold
        
    except Exception as e:
        logger.warning(f"Error checking if object is stationary: {e}")
        return False

def detect_ppe_in_detection(detection_result: Any, required_ppe: List[str],
                          confidence_threshold: float = 0.7) -> Dict[str, bool]:
    """
    Detect PPE equipment in detection results
    
    Args:
        detection_result: Detection result from YOLO model
        required_ppe: List of required PPE items
        confidence_threshold: Minimum confidence for PPE detection
        
    Returns:
        Dictionary mapping PPE item to detection status
    """
    try:
        ppe_status = {item: False for item in required_ppe}
        
        if not detection_result or not hasattr(detection_result, 'boxes'):
            return ppe_status
        
        # Check detections for PPE items
        for box in detection_result.boxes:
            class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
            confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
            
            if confidence >= confidence_threshold:
                if class_id in PPE_CLASSES:
                    ppe_item = PPE_CLASSES[class_id]
                    if ppe_item in required_ppe:
                        ppe_status[ppe_item] = True
        
        return ppe_status
        
    except Exception as e:
        logger.warning(f"Error detecting PPE: {e}")
        return {item: False for item in required_ppe}

def create_event_metadata(event_type: str, camera_id: str, 
                         detections: List[Dict], zone_name: Optional[str] = None,
                         additional_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create standardized event metadata
    
    Args:
        event_type: Type of event
        camera_id: Camera that detected the event
        detections: List of detection objects
        zone_name: Name of zone where event occurred
        additional_info: Additional metadata
        
    Returns:
        Event metadata dictionary
    """
    try:
        metadata = {
            'event_type': event_type,
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'detection_count': len(detections),
            'detections': detections
        }
        
        if zone_name:
            metadata['zone_name'] = zone_name
            
        if additional_info:
            metadata.update(additional_info)
            
        return metadata
        
    except Exception as e:
        logger.warning(f"Error creating event metadata: {e}")
        return {'event_type': event_type, 'camera_id': camera_id, 'error': str(e)}

def validate_zone_coordinates(coordinates: List[List[float]]) -> bool:
    """
    Validate zone coordinates
    
    Args:
        coordinates: List of [x, y] coordinate pairs
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    try:
        if not coordinates or len(coordinates) < 3:
            return False
            
        for coord in coordinates:
            if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                return False
            if not all(isinstance(x, (int, float)) for x in coord):
                return False
                
        return True
        
    except Exception:
        return False

def normalize_bbox(bbox: List[float], frame_width: int, frame_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        Normalized bounding box
    """
    try:
        x1, y1, x2, y2 = bbox
        return [
            x1 / frame_width,
            y1 / frame_height,
            x2 / frame_width,
            y2 / frame_height
        ]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

def denormalize_bbox(normalized_bbox: List[float], frame_width: int, frame_height: int) -> List[float]:
    """
    Denormalize bounding box coordinates from [0, 1] range to pixel coordinates
    
    Args:
        normalized_bbox: Normalized bounding box [x1, y1, x2, y2]
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        Pixel coordinate bounding box
    """
    try:
        x1, y1, x2, y2 = normalized_bbox
        return [
            x1 * frame_width,
            y1 * frame_height,
            x2 * frame_width,
            y2 * frame_height
        ]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_system_timestamp() -> float:
    """Get current system timestamp"""
    return time.time()

def convert_timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert timestamp to datetime object"""
    try:
        return datetime.fromtimestamp(timestamp)
    except Exception:
        return datetime.now()

# Export main utility functions
__all__ = [
    'format_duration',
    'format_timestamp', 
    'generate_event_id',
    'generate_track_id',
    'calculate_distance',
    'calculate_bbox_center',
    'calculate_bbox_area',
    'bbox_intersection_over_union',
    'is_point_in_polygon',
    'calculate_polygon_area',
    'calculate_bbox_polygon_overlap',
    'draw_text_with_background',
    'draw_datacenter_zone',
    'draw_detection_box',
    'draw_ppe_status',
    'is_object_stationary',
    'detect_ppe_in_detection',
    'create_event_metadata',
    'validate_zone_coordinates',
    'normalize_bbox',
    'denormalize_bbox',
    'ensure_directory_exists',
    'get_system_timestamp',
    'convert_timestamp_to_datetime',
    'COCO_CLASSES',
    'PPE_CLASSES',
    'DATACENTER_ZONE_COLORS',
    'SEVERITY_COLORS'
]
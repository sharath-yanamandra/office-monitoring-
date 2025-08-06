#!/usr/bin/env python3
"""
camera_model_base.py
Base Camera Model for Video Monitoring

This module provides:
1. Base class for all camera models
2. Common functionality for zone management
3. Event handling and annotation utilities
4. Frame processing pipeline structure
"""

import cv2
import numpy as np
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from kalman_tracker import MultiObjectTracker
from config import EventTypes

class CameraModelBase(ABC):
    """Base class for all camera models"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize base camera model
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions for this camera
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer instance
            frames_base_dir: Base directory for frame output
        """
        self.camera_id = camera_id
        self.zones = zones or {}
        self.settings = settings or {}
        self.db = db
        self.db_writer = db_writer
        self.frames_base_dir = frames_base_dir
        
        # Setup logging
        self.logger = logging.getLogger(f'camera_{camera_id}')
        
        # Initialize tracker
        self.tracker = MultiObjectTracker(
            max_disappeared=self.settings.get('max_disappeared', 30),
            min_hits=self.settings.get('min_hits', 3)
        )
        
        # Zone management
        self.parsed_zones = self._parse_zones()
        
        # Event management
        self.last_event_times = {}  # Track last event times for cooldown
        self.event_cooldown = self.settings.get('event_cooldown', 60)  # seconds
        
        # Frame processing
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'people_detected': 0,
            'processing_fps': 0.0
        }
        
        # Colors for visualization
        self.zone_colors = {
            'entry_zone': (0, 255, 0),      # Green
            'restricted_zone': (0, 0, 255),  # Red  
            'common_zone': (255, 255, 0),    # Yellow
            'counting_zone': (255, 0, 255),  # Magenta
            'perimeter_zone': (0, 255, 255) # Cyan
        }
        
        self.logger.info(f"Initialized camera model for camera {camera_id}")
    
    def _parse_zones(self) -> Dict[str, List[Dict]]:
        """Parse and organize zones by type"""
        parsed_zones = {}
        
        if not self.zones:
            return parsed_zones
        
        # Handle different zone data formats
        if isinstance(self.zones, list):
            # Database format: list of zone objects
            for zone in self.zones:
                zone_type = zone.get('zone_type', 'unknown')
                if zone_type not in parsed_zones:
                    parsed_zones[zone_type] = []
                
                # Parse coordinates if they're JSON strings
                coordinates = zone.get('coordinates', [])
                if isinstance(coordinates, str):
                    try:
                        coordinates = json.loads(coordinates)
                    except json.JSONDecodeError:
                        coordinates = []
                
                parsed_zone = {
                    'zone_id': zone.get('zone_id'),
                    'name': zone.get('name', f'Zone_{zone.get("zone_id", "unknown")}'),
                    'coordinates': coordinates,
                    'description': zone.get('description', ''),
                    'type': zone_type
                }
                parsed_zones[zone_type].append(parsed_zone)
        
        elif isinstance(self.zones, dict):
            # Config format: dict of zone types
            for zone_type, zone_list in self.zones.items():
                parsed_zones[zone_type] = zone_list
        
        return parsed_zones
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting"""
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
    
    def _is_person_in_zone(self, person: Dict[str, Any], zone: Dict[str, Any]) -> bool:
        """Check if person is inside a zone"""
        if not zone.get('coordinates'):
            return False
        
        # Get person center point
        bbox = person['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Check if center is in zone
        return self._point_in_polygon((center_x, center_y), zone['coordinates'])
    
    def _get_people_in_zone(self, people: List[Dict[str, Any]], zone: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all people inside a specific zone"""
        people_in_zone = []
        
        for person in people:
            if self._is_person_in_zone(person, zone):
                people_in_zone.append(person)
        
        return people_in_zone
    
    def _check_event_cooldown(self, event_type: str, zone_id: int = None) -> bool:
        """Check if event is in cooldown period"""
        key = f"{event_type}_{zone_id}" if zone_id else event_type
        current_time = time.time()
        
        if key in self.last_event_times:
            time_since_last = current_time - self.last_event_times[key]
            if time_since_last < self.event_cooldown:
                return True  # Still in cooldown
        
        self.last_event_times[key] = current_time
        return False  # Not in cooldown
    
    def _create_event(self, event_type: str, description: str, 
                     detection_data: Dict[str, Any], zone_id: int = None,
                     severity: str = None, media_path: str = None) -> Dict[str, Any]:
        """Create event dictionary"""
        if severity is None:
            severity = EventTypes.get_severity(event_type)
        
        event = {
            'camera_id': self.camera_id,
            'zone_id': zone_id,
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'detection_data': detection_data,
            'media_path': media_path,
            'timestamp': datetime.now()
        }
        
        return event
    
    def _save_event(self, event: Dict[str, Any]):
        """Save event to database"""
        try:
            if self.db_writer:
                self.db_writer.add_event(event)
                self.stats['events_detected'] += 1
                self.logger.info(f"Event saved: {event['event_type']} - {event['description']}")
            elif self.db:
                self.db.add_event(
                    camera_id=event['camera_id'],
                    event_type=event['event_type'],
                    severity=event['severity'],
                    description=event['description'],
                    detection_data=event['detection_data'],
                    media_path=event['media_path'],
                    zone_id=event['zone_id']
                )
                self.stats['events_detected'] += 1
                self.logger.info(f"Event saved: {event['event_type']} - {event['description']}")
        except Exception as e:
            self.logger.error(f"Error saving event: {e}")
    
    def _draw_zone(self, frame: np.ndarray, zone: Dict[str, Any], 
                   color: Tuple[int, int, int], alpha: float = 0.3, 
                   label: str = None):
        """Draw zone on frame"""
        if not zone.get('coordinates'):
            return
        
        coordinates = zone['coordinates']
        if len(coordinates) < 3:
            return
        
        # Convert coordinates to numpy array
        points = np.array(coordinates, dtype=np.int32)
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Fill polygon
        cv2.fillPoly(overlay, [points], color)
        
        # Draw polygon outline
        cv2.polylines(frame, [points], True, color, 2)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add label
        if label:
            # Calculate center of polygon for label placement
            moments = cv2.moments(points)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (cx - label_size[0]//2 - 5, cy - label_size[1] - 5),
                            (cx + label_size[0]//2 + 5, cy + 5), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (cx - label_size[0]//2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_person(self, frame: np.ndarray, person: Dict[str, Any], 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        """Draw person bounding box on frame"""
        bbox = person['bbox']
        track_id = person.get('track_id', 'N/A')
        confidence = person.get('confidence', 0.0)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), color, thickness)
        
        # Draw label
        label = f"ID:{track_id} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                     (int(bbox[0]) + label_size[0] + 10, int(bbox[1])), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (int(bbox[0]) + 5, int(bbox[1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_frame(self, frame: np.ndarray, suffix: str = '') -> str:
        """Save frame to disk and return path"""
        try:
            import os
            
            # Create output directory
            output_dir = os.path.join(self.frames_base_dir, f'camera_{self.camera_id}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'{timestamp}_{suffix}.jpg' if suffix else f'{timestamp}.jpg'
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return ""
    
    def _update_statistics(self, people_count: int = 0):
        """Update processing statistics"""
        current_time = time.time()
        
        self.stats['frames_processed'] += 1
        if people_count > 0:
            self.stats['people_detected'] += people_count
        
        # Calculate FPS
        time_diff = current_time - self.last_process_time
        if time_diff > 0:
            self.stats['processing_fps'] = 1.0 / time_diff
        
        self.last_process_time = current_time
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame with detections - must be implemented by subclasses
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
            detection_result: Detection results from model
            
        Returns:
            Tuple of (annotated_frame, list_of_events)
        """
        pass
    
    def get_zones_by_type(self, zone_type: str) -> List[Dict[str, Any]]:
        """Get zones of specific type"""
        return self.parsed_zones.get(zone_type, [])
    
    def get_all_zones(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all zones organized by type"""
        return self.parsed_zones.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        tracker_stats = self.tracker.get_tracker_stats()
        
        return {
            **self.stats,
            'tracker_stats': tracker_stats,
            'zones_count': sum(len(zones) for zones in self.parsed_zones.values()),
            'event_cooldown': self.event_cooldown
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'people_detected': 0,
            'processing_fps': 0.0
        }
        self.tracker.reset()
        self.last_event_times.clear()
        self.logger.info(f"Statistics reset for camera {self.camera_id}")


class GenericCameraModel(CameraModelBase):
    """Generic camera model for basic monitoring"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """Initialize generic camera model"""
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        self.logger.info("Initialized generic camera model")
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame with basic person detection and tracking
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp  
            detection_result: Person detection results
            
        Returns:
            Tuple of (annotated_frame, list_of_events)
        """
        self.frame_count += 1
        events = []
        
        # Convert detection result to tracker format
        detections = []
        if detection_result:
            for detection in detection_result:
                detections.append({
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class_name': detection.get('class_name', 'person')
                })
        
        # Update tracker
        tracked_people = self.tracker.update(detections, timestamp)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw zones
        for zone_type, zones in self.parsed_zones.items():
            color = self.zone_colors.get(zone_type, (128, 128, 128))
            for zone in zones:
                self._draw_zone(annotated_frame, zone, color, alpha=0.2, 
                               label=zone.get('name', f'{zone_type}'))
        
        # Draw tracked people
        for person in tracked_people:
            self._draw_person(annotated_frame, person, (0, 255, 0), 2)
        
        # Basic people counting event
        if tracked_people and not self._check_event_cooldown('people_counting'):
            event = self._create_event(
                event_type='people_counting',
                description=f'{len(tracked_people)} people detected',
                detection_data={
                    'count': len(tracked_people),
                    'people': [{'track_id': p['track_id'], 'confidence': p['confidence']} 
                             for p in tracked_people]
                },
                severity='info'
            )
            events.append(event)
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        # Add frame info
        info_text = f"Camera {self.camera_id} | People: {len(tracked_people)} | Frame: {self.frame_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame, events


# Export main classes
__all__ = ['CameraModelBase', 'GenericCameraModel']
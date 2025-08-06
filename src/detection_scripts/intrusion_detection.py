#!/usr/bin/env python3
"""
detection_scripts/intrusion_detection.py
Intrusion Detection Script

This script detects general intrusions including motion-based detection,
perimeter breaches, and suspicious behavior patterns.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from camera_model_base import CameraModelBase
from config import EventTypes

class IntrusionDetector(CameraModelBase):
    """Intrusion detection using motion analysis and person detection"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize intrusion detector
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (perimeter_zones, restricted_zones)
            settings: Detection settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # Intrusion detection parameters
        self.sensitivity = settings.get('sensitivity', 'medium')
        self.motion_threshold = self._get_motion_threshold(self.sensitivity)
        self.min_contour_area = settings.get('min_contour_area', 500)
        self.detection_confidence = settings.get('detection_confidence', 0.7)
        
        # Time-based detection
        self.enhanced_after_hours = settings.get('enhanced_after_hours', True)
        self.business_start_hour = settings.get('business_start_hour', 8)
        self.business_end_hour = settings.get('business_end_hour', 18)
        
        # Get monitoring zones
        self.perimeter_zones = self.get_zones_by_type('perimeter_zone')
        self.restricted_zones = self.get_zones_by_type('restricted_zone')
        self.all_monitored_zones = self.perimeter_zones + self.restricted_zones
        
        # Motion detection setup
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Tracking state
        self.intrusion_events = {}     # Active intrusion tracking
        self.motion_history = []       # Motion detection history
        self.suspicious_tracks = {}    # Suspicious movement patterns
        
        # Previous frame for optical flow
        self.prev_frame = None
        
        # Statistics
        self.intrusion_stats = {
            'total_intrusions': 0,
            'motion_based_detections': 0,
            'person_based_detections': 0,
            'false_positives': 0,
            'after_hours_detections': 0
        }
        
        self.logger.info(f"Intrusion detector initialized - Sensitivity: {self.sensitivity}")
        self.logger.info(f"Monitoring {len(self.all_monitored_zones)} zones ({len(self.perimeter_zones)} perimeter, {len(self.restricted_zones)} restricted)")
    
    def _get_motion_threshold(self, sensitivity: str) -> float:
        """Get motion threshold based on sensitivity setting"""
        thresholds = {
            'low': 0.5,
            'medium': 0.3,
            'high': 0.1,
            'very_high': 0.05
        }
        return thresholds.get(sensitivity, 0.3)
    
    def _is_after_hours(self, timestamp: float) -> bool:
        """Check if current time is after business hours"""
        dt = datetime.fromtimestamp(timestamp)
        current_hour = dt.hour
        return current_hour < self.business_start_hour or current_hour >= self.business_end_hour
    
    def _detect_motion(self, frame: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """
        Detect motion in frame using background subtraction
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp
            
        Returns:
            List of motion detections
        """
        motion_detections = []
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Noise reduction
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.min_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate motion properties
                    center = (x + w // 2, y + h // 2)
                    
                    motion_detection = {
                        'type': 'motion',
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'center': center,
                        'contour': contour,
                        'confidence': min(1.0, area / 2000),  # Normalize confidence
                        'timestamp': timestamp
                    }
                    motion_detections.append(motion_detection)
            
            # Store motion history
            self.motion_history.append({
                'timestamp': timestamp,
                'motion_count': len(motion_detections),
                'total_motion_area': sum(d['area'] for d in motion_detections)
            })
            
            # Keep only recent history
            if len(self.motion_history) > 100:
                self.motion_history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
        
        return motion_detections
    
    def _analyze_movement_patterns(self, tracked_people: List[Dict[str, Any]], 
                                 timestamp: float) -> Dict[int, Dict[str, Any]]:
        """Analyze movement patterns for suspicious behavior"""
        pattern_analysis = {}
        
        for person in tracked_people:
            track_id = person['track_id']
            center = person['center']
            velocity = person.get('velocity', (0, 0))
            
            if track_id not in self.suspicious_tracks:
                self.suspicious_tracks[track_id] = {
                    'positions': [center],
                    'velocities': [velocity],
                    'first_seen': timestamp,
                    'direction_changes': 0,
                    'loitering_time': 0,
                    'max_speed': 0,
                    'zone_visits': set()
                }
            else:
                track_data = self.suspicious_tracks[track_id]
                track_data['positions'].append(center)
                track_data['velocities'].append(velocity)
                
                # Analyze patterns
                positions = track_data['positions']
                velocities = track_data['velocities']
                
                # Calculate speed
                if len(velocities) > 1:
                    speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                    track_data['max_speed'] = max(track_data['max_speed'], speed)
                
                # Detect direction changes (zigzag movement)
                if len(velocities) > 2:
                    prev_vel = velocities[-2]
                    curr_vel = velocities[-1]
                    
                    # Calculate angle change
                    if np.linalg.norm(prev_vel) > 0 and np.linalg.norm(curr_vel) > 0:
                        dot_product = np.dot(prev_vel, curr_vel)
                        norms = np.linalg.norm(prev_vel) * np.linalg.norm(curr_vel)
                        angle_change = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
                        
                        if angle_change > np.pi / 2:  # > 90 degrees
                            track_data['direction_changes'] += 1
                
                # Detect loitering (small movement area over time)
                if len(positions) > 30:  # At least 30 frames of history
                    recent_positions = positions[-30:]
                    x_coords = [pos[0] for pos in recent_positions]
                    y_coords = [pos[1] for pos in recent_positions]
                    
                    movement_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                    if movement_area < 2000:  # Small area threshold
                        track_data['loitering_time'] += 1
                
                # Keep limited history
                if len(positions) > 50:
                    track_data['positions'] = positions[-50:]
                    track_data['velocities'] = velocities[-50:]
            
            # Determine suspicion level
            track_data = self.suspicious_tracks[track_id]
            suspicion_score = 0
            suspicious_behaviors = []
            
            # High speed movement
            if track_data['max_speed'] > 100:  # pixels per frame
                suspicion_score += 2
                suspicious_behaviors.append('fast_movement')
            
            # Frequent direction changes (evasive movement)
            if track_data['direction_changes'] > 5:
                suspicion_score += 3
                suspicious_behaviors.append('evasive_movement')
            
            # Extended loitering
            if track_data['loitering_time'] > 50:  # frames
                suspicion_score += 2
                suspicious_behaviors.append('loitering')
            
            # Duration in area
            duration = timestamp - track_data['first_seen']
            if duration > 300:  # 5 minutes
                suspicion_score += 1
                suspicious_behaviors.append('extended_presence')
            
            pattern_analysis[track_id] = {
                'suspicion_score': suspicion_score,
                'suspicious_behaviors': suspicious_behaviors,
                'is_suspicious': suspicion_score >= 3,
                'duration': duration,
                'max_speed': track_data['max_speed'],
                'direction_changes': track_data['direction_changes'],
                'loitering_time': track_data['loitering_time']
            }
        
        return pattern_analysis
    
    def _detect_zone_intrusions(self, tracked_people: List[Dict[str, Any]], 
                               motion_detections: List[Dict[str, Any]], 
                               timestamp: float) -> List[Dict[str, Any]]:
        """Detect intrusions in monitored zones"""
        intrusion_events = []
        
        is_after_hours = self._is_after_hours(timestamp)
        
        for zone in self.all_monitored_zones:
            zone_id = zone.get('zone_id', 0)
            zone_type = zone.get('zone_type', 'unknown')
            zone_name = zone.get('name', f'{zone_type} {zone_id}')
            
            # Check for people in zone
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            
            # Check for motion in zone
            motion_in_zone = []
            for motion in motion_detections:
                motion_center = motion['center']
                if self._point_in_polygon(motion_center, zone.get('coordinates', [])):
                    motion_in_zone.append(motion)
            
            # Determine if this is an intrusion
            is_intrusion = False
            detection_method = []
            confidence = 0.0
            
            # Person-based detection
            if people_in_zone:
                is_intrusion = True
                detection_method.append('person_detection')
                confidence = max(confidence, max(p['confidence'] for p in people_in_zone))
                self.intrusion_stats['person_based_detections'] += 1
            
            # Motion-based detection (especially useful for perimeter zones)
            if motion_in_zone and zone_type == 'perimeter_zone':
                total_motion_area = sum(m['area'] for m in motion_in_zone)
                if total_motion_area > self.min_contour_area * 2:
                    is_intrusion = True
                    detection_method.append('motion_detection')
                    confidence = max(confidence, 0.7)
                    self.intrusion_stats['motion_based_detections'] += 1
            
            # Enhanced detection after hours
            if is_after_hours and self.enhanced_after_hours:
                if motion_in_zone:  # Any motion after hours is suspicious
                    is_intrusion = True
                    detection_method.append('after_hours_motion')
                    confidence = max(confidence, 0.8)
                    self.intrusion_stats['after_hours_detections'] += 1
            
            if is_intrusion and confidence >= self.detection_confidence:
                intrusion_event = {
                    'zone_id': zone_id,
                    'zone_name': zone_name,
                    'zone_type': zone_type,
                    'people_count': len(people_in_zone),
                    'motion_detections': len(motion_in_zone),
                    'detection_method': detection_method,
                    'confidence': confidence,
                    'is_after_hours': is_after_hours,
                    'timestamp': timestamp,
                    'people_in_zone': people_in_zone,
                    'motion_in_zone': motion_in_zone,
                    'severity': 'high' if is_after_hours else 'medium'
                }
                intrusion_events.append(intrusion_event)
                
                self.logger.warning(f"Intrusion detected in {zone_name}: {len(people_in_zone)} people, {len(motion_in_zone)} motion areas")
        
        return intrusion_events
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for intrusion detection
        
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
                if detection['confidence'] >= self.detection_confidence:
                    detections.append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class_name': detection.get('class_name', 'person')
                    })
        
        # Update tracker
        tracked_people = self.tracker.update(detections, timestamp)
        
        # Detect motion
        motion_detections = self._detect_motion(frame, timestamp)
        
        # Analyze movement patterns
        pattern_analysis = self._analyze_movement_patterns(tracked_people, timestamp)
        
        # Detect zone intrusions
        intrusion_events = self._detect_zone_intrusions(tracked_people, motion_detections, timestamp)
        
        # Create intrusion events
        for intrusion in intrusion_events:
            if not self._check_event_cooldown('intrusion', intrusion['zone_id']):
                event = self._create_event(
                    event_type='intrusion',
                    description=f"Intrusion detected in {intrusion['zone_name']}: {intrusion['people_count']} people, detection method: {', '.join(intrusion['detection_method'])}",
                    detection_data={
                        'zone_id': intrusion['zone_id'],
                        'zone_name': intrusion['zone_name'],
                        'zone_type': intrusion['zone_type'],
                        'people_count': intrusion['people_count'],
                        'motion_detections': intrusion['motion_detections'],
                        'detection_method': intrusion['detection_method'],
                        'is_after_hours': intrusion['is_after_hours'],
                        'confidence': intrusion['confidence'],
                        'pattern_analysis': {k: v for k, v in pattern_analysis.items() 
                                          if any(p['track_id'] == k for p in intrusion['people_in_zone'])}
                    },
                    zone_id=intrusion['zone_id'],
                    severity=intrusion['severity']
                )
                events.append(event)
                self._save_event(event)
                
                self.intrusion_stats['total_intrusions'] += 1
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, motion_detections, intrusion_events, pattern_analysis, timestamp
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               motion_detections: List[Dict[str, Any]],
                               intrusion_events: List[Dict[str, Any]],
                               pattern_analysis: Dict[int, Dict[str, Any]],
                               timestamp: float) -> np.ndarray:
        """Create annotated frame with intrusion detection visualization"""
        annotated_frame = frame.copy()
        
        # Draw monitored zones
        for zone in self.all_monitored_zones:
            zone_id = zone.get('zone_id', 0)
            zone_type = zone.get('zone_type', 'unknown')
            
            # Check if zone has intrusion
            has_intrusion = any(i['zone_id'] == zone_id for i in intrusion_events)
            
            # Color based on zone type and intrusion status
            if has_intrusion:
                color = (0, 0, 255)  # Red for intrusion
                alpha = 0.4
            elif zone_type == 'perimeter_zone':
                color = (255, 165, 0)  # Orange for perimeter
                alpha = 0.2
            else:  # restricted_zone
                color = (255, 255, 0)  # Yellow for restricted
                alpha = 0.2
            
            label = f"{zone.get('name', f'{zone_type} {zone_id}')}"
            if has_intrusion:
                label += " - INTRUSION!"
            
            self._draw_zone(annotated_frame, zone, color, alpha, label)
        
        # Draw motion detections
        for motion in motion_detections:
            bbox = motion['bbox']
            area = motion['area']
            confidence = motion['confidence']
            
            # Color based on motion intensity
            intensity = min(255, int(area / 10))
            color = (0, intensity, 255 - intensity)  # Blue to red gradient
            
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw motion info
            motion_text = f"Motion {confidence:.2f}"
            cv2.putText(annotated_frame, motion_text,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw tracked people with pattern analysis
        for person in tracked_people:
            track_id = person['track_id']
            bbox = person['bbox']
            
            # Get pattern analysis for this person
            pattern = pattern_analysis.get(track_id, {})
            is_suspicious = pattern.get('is_suspicious', False)
            suspicion_score = pattern.get('suspicion_score', 0)
            
            # Color based on suspicion level
            if is_suspicious:
                color = (0, 0, 255)  # Red for suspicious
                thickness = 3
            elif suspicion_score > 0:
                color = (0, 255, 255)  # Yellow for somewhat suspicious
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            self._draw_person(annotated_frame, person, color, thickness)
            
            # Draw suspicion info
            if suspicion_score > 0:
                suspicious_behaviors = pattern.get('suspicious_behaviors', [])
                suspicion_text = f"Suspicion: {suspicion_score} ({', '.join(suspicious_behaviors)})"
                cv2.putText(annotated_frame, suspicion_text,
                           (int(bbox[0]), int(bbox[3]) + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw intrusion alerts
        alert_y = 60
        for intrusion in intrusion_events:
            alert_text = f"INTRUSION ALERT: {intrusion['zone_name']} - {intrusion['people_count']} people - {', '.join(intrusion['detection_method'])}"
            cv2.putText(annotated_frame, alert_text, (10, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        
        # Draw time-based status
        is_after_hours = self._is_after_hours(timestamp)
        time_status = "AFTER HOURS" if is_after_hours else "BUSINESS HOURS"
        time_color = (0, 165, 255) if is_after_hours else (0, 255, 0)
        cv2.putText(annotated_frame, time_status, (10, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
        
        # Draw motion activity level
        recent_motion = [m for m in self.motion_history if timestamp - m['timestamp'] < 10]
        avg_motion = sum(m['motion_count'] for m in recent_motion) / len(recent_motion) if recent_motion else 0
        
        activity_level = "HIGH" if avg_motion > 5 else "MEDIUM" if avg_motion > 2 else "LOW"
        activity_color = (0, 0, 255) if avg_motion > 5 else (0, 255, 255) if avg_motion > 2 else (0, 255, 0)
        
        activity_text = f"Motion Activity: {activity_level} ({avg_motion:.1f})"
        cv2.putText(annotated_frame, activity_text, (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, activity_color, 2)
        
        # Draw statistics
        stats_text = f"Intrusions: {self.intrusion_stats['total_intrusions']} | Motion: {len(motion_detections)} | Suspicious: {sum(1 for p in pattern_analysis.values() if p.get('is_suspicious', False))}"
        cv2.putText(annotated_frame, stats_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Main info
        info_text = f"Camera {self.camera_id} - Intrusion Detection | People: {len(tracked_people)} | Sensitivity: {self.sensitivity}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detailed detection statistics"""
        recent_motion = [m for m in self.motion_history if time.time() - m['timestamp'] < 60]
        
        return {
            **self.intrusion_stats,
            'recent_motion_average': sum(m['motion_count'] for m in recent_motion) / len(recent_motion) if recent_motion else 0,
            'active_suspicious_tracks': sum(1 for track in self.suspicious_tracks.values() if time.time() - track.get('first_seen', 0) < 300),
            'zones_monitored': len(self.all_monitored_zones),
            'perimeter_zones': len(self.perimeter_zones),
            'restricted_zones': len(self.restricted_zones),
            'sensitivity': self.sensitivity,
            'motion_threshold': self.motion_threshold
        }
    
    def update_sensitivity(self, new_sensitivity: str):
        """Update detection sensitivity"""
        if new_sensitivity in ['low', 'medium', 'high', 'very_high']:
            self.sensitivity = new_sensitivity
            self.motion_threshold = self._get_motion_threshold(new_sensitivity)
            self.logger.info(f"Sensitivity updated to: {new_sensitivity} (threshold: {self.motion_threshold})")
    
    def reset_background_model(self):
        """Reset background subtraction model (useful for lighting changes)"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.logger.info("Background subtraction model reset")


# Export main class
__all__ = ['IntrusionDetector']
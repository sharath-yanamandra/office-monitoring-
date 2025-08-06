#!/usr/bin/env python3
"""
detection_scripts/unauthorized_access_detection.py
Unauthorized Access Detection Script

This script detects people in restricted areas where they should not be,
including access control integration and time-based restrictions.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, time as dt_time

from camera_model_base import CameraModelBase
from config import EventTypes

class UnauthorizedAccessDetector(CameraModelBase):
    """Unauthorized access detection for restricted areas"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize unauthorized access detector
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (should include restricted_zones)
            settings: Detection settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # Access control parameters
        self.confidence_threshold = settings.get('confidence_threshold', 0.8)
        self.time_based_restrictions = settings.get('time_based_restrictions', True)
        self.access_control_enabled = settings.get('access_control_enabled', True)
        self.unauthorized_tolerance = settings.get('unauthorized_tolerance', 5)  # seconds
        
        # Business hours for time-based access
        self.business_start_hour = settings.get('business_start_hour', 8)  # 8 AM
        self.business_end_hour = settings.get('business_end_hour', 18)    # 6 PM
        self.weekend_access = settings.get('weekend_access', False)
        
        # Get restricted zones
        self.restricted_zones = self.get_zones_by_type('restricted_zone')
        
        # Access tracking
        self.unauthorized_people = {}  # Track unauthorized people and duration
        self.access_attempts = {}      # Track access attempt history
        self.zone_occupancy = {}       # Current occupancy per zone
        
        # Authorized personnel simulation (in real system, integrate with access control)
        self.authorized_personnel = self._load_authorized_personnel()
        
        # Zone-specific access levels
        self.zone_access_levels = {
            'public': 0,
            'restricted': 1,
            'high_security': 2,
            'critical': 3
        }
        
        self.logger.info(f"Unauthorized access detector initialized for {len(self.restricted_zones)} zones")
        self.logger.info(f"Time-based restrictions: {self.time_based_restrictions}")
        self.logger.info(f"Business hours: {self.business_start_hour}:00 - {self.business_end_hour}:00")
    
    def _load_authorized_personnel(self) -> Dict[str, Any]:
        """
        Load authorized personnel database
        In real implementation, this would connect to access control system
        """
        # Simulation of authorized personnel database
        return {
            'person_1': {
                'name': 'John Doe',
                'access_level': 2,
                'authorized_zones': ['restricted_zone_1', 'restricted_zone_2'],
                'time_restrictions': None,  # No time restrictions
                'active': True
            },
            'person_2': {
                'name': 'Jane Smith',
                'access_level': 1,
                'authorized_zones': ['restricted_zone_1'],
                'time_restrictions': {'start': 9, 'end': 17},  # 9 AM - 5 PM
                'active': True
            }
        }
    
    def _is_business_hours(self, timestamp: float) -> bool:
        """Check if current time is within business hours"""
        if not self.time_based_restrictions:
            return True
        
        dt = datetime.fromtimestamp(timestamp)
        current_hour = dt.hour
        current_weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
        
        # Check weekend access
        if not self.weekend_access and current_weekday >= 5:  # Saturday or Sunday
            return False
        
        # Check business hours
        return self.business_start_hour <= current_hour < self.business_end_hour
    
    def _check_person_authorization(self, track_id: int, zone: Dict[str, Any], 
                                  timestamp: float) -> Dict[str, Any]:
        """
        Check if person is authorized to access zone
        In real system, this would integrate with access control/badge system
        
        Args:
            track_id: Person track ID
            zone: Zone information
            timestamp: Current timestamp
            
        Returns:
            Authorization result
        """
        # Simulation: randomly authorize some people for demo
        # In real system, this would check badge readers, biometrics, etc.
        
        import random
        
        zone_id = zone.get('zone_id', 0)
        zone_name = zone.get('name', f'Zone {zone_id}')
        zone_security_level = zone.get('security_level', 'restricted')
        
        # Simulate authorization check
        is_authorized = random.random() < 0.3  # 30% chance of being authorized
        
        # Check time-based restrictions
        is_business_hours = self._is_business_hours(timestamp)
        time_authorized = True
        
        if zone_security_level in ['high_security', 'critical'] and not is_business_hours:
            time_authorized = False
        
        # Combine authorization checks
        final_authorized = is_authorized and time_authorized
        
        authorization_result = {
            'track_id': track_id,
            'zone_id': zone_id,
            'zone_name': zone_name,
            'is_authorized': final_authorized,
            'authorization_level': 'authorized' if is_authorized else 'unauthorized',
            'time_authorized': time_authorized,
            'business_hours': is_business_hours,
            'zone_security_level': zone_security_level,
            'timestamp': timestamp,
            'reason': self._get_authorization_reason(is_authorized, time_authorized, is_business_hours)
        }
        
        return authorization_result
    
    def _get_authorization_reason(self, is_authorized: bool, time_authorized: bool, 
                                is_business_hours: bool) -> str:
        """Get human-readable authorization reason"""
        if is_authorized and time_authorized:
            return "Access granted"
        elif not is_authorized:
            return "Person not authorized for this zone"
        elif not time_authorized:
            if not is_business_hours:
                return "Access denied - outside business hours"
            else:
                return "Access denied - time restrictions apply"
        else:
            return "Access denied - unknown reason"
    
    def _detect_unauthorized_access(self, tracked_people: List[Dict[str, Any]], 
                                   timestamp: float) -> List[Dict[str, Any]]:
        """
        Detect unauthorized access attempts in restricted zones
        
        Args:
            tracked_people: List of tracked people
            timestamp: Current timestamp
            
        Returns:
            List of unauthorized access events
        """
        unauthorized_events = []
        
        for zone in self.restricted_zones:
            zone_id = zone.get('zone_id', 0)
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            
            # Update zone occupancy
            self.zone_occupancy[zone_id] = len(people_in_zone)
            
            for person in people_in_zone:
                track_id = person['track_id']
                
                # Check authorization for this person in this zone
                auth_result = self._check_person_authorization(track_id, zone, timestamp)
                
                if not auth_result['is_authorized']:
                    # Track unauthorized person
                    key = f"{track_id}_{zone_id}"
                    
                    if key not in self.unauthorized_people:
                        self.unauthorized_people[key] = {
                            'track_id': track_id,
                            'zone_id': zone_id,
                            'zone_name': zone.get('name', f'Zone {zone_id}'),
                            'first_detected': timestamp,
                            'last_seen': timestamp,
                            'duration': 0,
                            'auth_result': auth_result
                        }
                    else:
                        # Update existing unauthorized access
                        self.unauthorized_people[key]['last_seen'] = timestamp
                        self.unauthorized_people[key]['duration'] = timestamp - self.unauthorized_people[key]['first_detected']
                    
                    # Check if unauthorized duration exceeds tolerance
                    if self.unauthorized_people[key]['duration'] >= self.unauthorized_tolerance:
                        unauthorized_event = {
                            'track_id': track_id,
                            'zone_id': zone_id,
                            'zone_name': zone.get('name', f'Zone {zone_id}'),
                            'person_bbox': person['bbox'],
                            'duration': self.unauthorized_people[key]['duration'],
                            'auth_result': auth_result,
                            'timestamp': timestamp,
                            'severity': self._determine_severity(zone, auth_result)
                        }
                        unauthorized_events.append(unauthorized_event)
                        
                        self.logger.warning(f"Unauthorized access: Person {track_id} in {zone.get('name')} for {self.unauthorized_people[key]['duration']:.1f}s")
                else:
                    # Remove from unauthorized list if now authorized
                    key = f"{track_id}_{zone_id}"
                    if key in self.unauthorized_people:
                        del self.unauthorized_people[key]
        
        # Clean up old unauthorized entries
        self._cleanup_unauthorized_tracking(timestamp)
        
        return unauthorized_events
    
    def _determine_severity(self, zone: Dict[str, Any], auth_result: Dict[str, Any]) -> str:
        """Determine event severity based on zone and authorization details"""
        zone_security_level = zone.get('security_level', 'restricted')
        
        # Map security levels to severities
        severity_map = {
            'public': 'info',
            'restricted': 'medium',
            'high_security': 'high',
            'critical': 'critical'
        }
        
        base_severity = severity_map.get(zone_security_level, 'medium')
        
        # Escalate severity if after hours
        if not auth_result.get('business_hours', True):
            severity_escalation = {
                'info': 'medium',
                'medium': 'high',
                'high': 'critical',
                'critical': 'critical'
            }
            return severity_escalation.get(base_severity, base_severity)
        
        return base_severity
    
    def _cleanup_unauthorized_tracking(self, timestamp: float):
        """Clean up old unauthorized access tracking entries"""
        cleanup_threshold = 300  # 5 minutes
        keys_to_remove = []
        
        for key, data in self.unauthorized_people.items():
            if timestamp - data['last_seen'] > cleanup_threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.unauthorized_people[key]
    
    def _analyze_access_patterns(self, unauthorized_events: List[Dict[str, Any]], 
                                timestamp: float) -> Dict[str, Any]:
        """Analyze access patterns for suspicious behavior"""
        analysis = {
            'multiple_zone_attempts': 0,
            'repeat_offenders': 0,
            'after_hours_attempts': 0,
            'high_security_breaches': 0,
            'suspicious_patterns': []
        }
        
        # Count different types of violations
        for event in unauthorized_events:
            auth_result = event['auth_result']
            zone_security = event.get('severity', 'medium')
            
            # After hours attempts
            if not auth_result.get('business_hours', True):
                analysis['after_hours_attempts'] += 1
            
            # High security breaches
            if zone_security in ['high', 'critical']:
                analysis['high_security_breaches'] += 1
        
        # Check for repeat offenders
        track_ids = [event['track_id'] for event in unauthorized_events]
        unique_tracks = set(track_ids)
        if len(track_ids) > len(unique_tracks):
            analysis['repeat_offenders'] = len(track_ids) - len(unique_tracks)
        
        # Check for multiple zone attempts by same person
        person_zones = {}
        for event in unauthorized_events:
            track_id = event['track_id']
            zone_id = event['zone_id']
            
            if track_id not in person_zones:
                person_zones[track_id] = set()
            person_zones[track_id].add(zone_id)
        
        analysis['multiple_zone_attempts'] = sum(1 for zones in person_zones.values() if len(zones) > 1)
        
        # Identify suspicious patterns
        if analysis['after_hours_attempts'] > 2:
            analysis['suspicious_patterns'].append('Multiple after-hours access attempts')
        
        if analysis['high_security_breaches'] > 0:
            analysis['suspicious_patterns'].append('High security zone breach attempts')
        
        if analysis['multiple_zone_attempts'] > 0:
            analysis['suspicious_patterns'].append('Multiple zone access attempts by single person')
        
        return analysis
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for unauthorized access detection
        
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
                if detection['confidence'] >= self.confidence_threshold:
                    detections.append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class_name': detection.get('class_name', 'person')
                    })
        
        # Update tracker
        tracked_people = self.tracker.update(detections, timestamp)
        
        # Detect unauthorized access
        unauthorized_events = self._detect_unauthorized_access(tracked_people, timestamp)
        
        # Analyze access patterns
        pattern_analysis = self._analyze_access_patterns(unauthorized_events, timestamp)
        
        # Create unauthorized access events
        for unauthorized in unauthorized_events:
            if not self._check_event_cooldown('unauthorized_access', unauthorized['zone_id']):
                event = self._create_event(
                    event_type='unauthorized_access',
                    description=f"Unauthorized access: Person {unauthorized['track_id']} in {unauthorized['zone_name']} for {unauthorized['duration']:.1f}s",
                    detection_data={
                        'track_id': unauthorized['track_id'],
                        'zone_id': unauthorized['zone_id'],
                        'zone_name': unauthorized['zone_name'],
                        'duration': unauthorized['duration'],
                        'person_bbox': unauthorized['person_bbox'],
                        'auth_result': unauthorized['auth_result'],
                        'pattern_analysis': pattern_analysis,
                        'confidence': 0.9
                    },
                    zone_id=unauthorized['zone_id'],
                    severity=unauthorized['severity']
                )
                events.append(event)
                self._save_event(event)
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, unauthorized_events, pattern_analysis, timestamp
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               unauthorized_events: List[Dict[str, Any]],
                               pattern_analysis: Dict[str, Any],
                               timestamp: float) -> np.ndarray:
        """Create annotated frame with unauthorized access visualization"""
        annotated_frame = frame.copy()
        
        # Draw restricted zones
        for zone in self.restricted_zones:
            zone_id = zone.get('zone_id', 0)
            occupancy = self.zone_occupancy.get(zone_id, 0)
            security_level = zone.get('security_level', 'restricted')
            
            # Color based on security level and violations
            has_violations = any(e['zone_id'] == zone_id for e in unauthorized_events)
            
            if has_violations:
                color = (0, 0, 255)  # Red for violations
                alpha = 0.4
            else:
                # Color by security level
                security_colors = {
                    'restricted': (255, 255, 0),      # Yellow
                    'high_security': (255, 165, 0),  # Orange
                    'critical': (255, 0, 255)        # Magenta
                }
                color = security_colors.get(security_level, (255, 255, 0))
                alpha = 0.2
            
            label = f"{zone.get('name', f'Zone {zone_id}')} [{security_level}] ({occupancy})"
            self._draw_zone(annotated_frame, zone, color, alpha, label)
        
        # Draw tracked people with authorization status
        for person in tracked_people:
            track_id = person['track_id']
            bbox = person['bbox']
            
            # Check if person is unauthorized in any zone
            is_unauthorized = any(e['track_id'] == track_id for e in unauthorized_events)
            
            # Color based on authorization status
            if is_unauthorized:
                color = (0, 0, 255)  # Red for unauthorized
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for authorized/not in restricted area
                thickness = 2
            
            self._draw_person(annotated_frame, person, color, thickness)
            
            # Draw authorization status
            if is_unauthorized:
                violation = next(e for e in unauthorized_events if e['track_id'] == track_id)
                status_text = f"UNAUTHORIZED - {violation['duration']:.1f}s"
                cv2.putText(annotated_frame, status_text,
                           (int(bbox[0]), int(bbox[3]) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw unauthorized access alerts
        alert_y = 60
        for unauthorized in unauthorized_events:
            auth_result = unauthorized['auth_result']
            alert_text = f"UNAUTHORIZED ACCESS: {unauthorized['zone_name']} - ID:{unauthorized['track_id']} - {auth_result['reason']}"
            cv2.putText(annotated_frame, alert_text, (10, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        
        # Draw business hours status
        is_business_hours = self._is_business_hours(timestamp)
        hours_text = f"Business Hours: {'YES' if is_business_hours else 'NO'}"
        hours_color = (0, 255, 0) if is_business_hours else (0, 165, 255)
        cv2.putText(annotated_frame, hours_text, (10, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hours_color, 2)
        
        # Draw pattern analysis
        if pattern_analysis['suspicious_patterns']:
            pattern_text = f"Suspicious: {', '.join(pattern_analysis['suspicious_patterns'])}"
            cv2.putText(annotated_frame, pattern_text, (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw statistics
        stats_text = f"Violations: {len(unauthorized_events)} | After-hours: {pattern_analysis['after_hours_attempts']} | High-security: {pattern_analysis['high_security_breaches']}"
        cv2.putText(annotated_frame, stats_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Main info
        info_text = f"Camera {self.camera_id} - Unauthorized Access Detection | People: {len(tracked_people)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_current_violations(self) -> List[Dict[str, Any]]:
        """Get current active unauthorized access violations"""
        current_time = time.time()
        active_violations = []
        
        for key, data in self.unauthorized_people.items():
            if current_time - data['last_seen'] < 30:  # Active within last 30 seconds
                active_violations.append({
                    'track_id': data['track_id'],
                    'zone_id': data['zone_id'],
                    'zone_name': data['zone_name'],
                    'duration': current_time - data['first_detected'],
                    'auth_result': data['auth_result']
                })
        
        return active_violations
    
    def update_authorized_personnel(self, personnel_data: Dict[str, Any]):
        """Update authorized personnel database (for real-time access control integration)"""
        self.authorized_personnel.update(personnel_data)
        self.logger.info("Authorized personnel database updated")


# Export main class
__all__ = ['UnauthorizedAccessDetector']
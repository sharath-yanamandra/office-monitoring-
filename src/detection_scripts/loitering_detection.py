#!/usr/bin/env python3
"""
detection_scripts/loitering_detection.py
Loitering Detection Script

This script detects people who stay in areas for extended periods,
including movement analysis and zone-specific loitering rules.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from camera_model_base import CameraModelBase
from config import EventTypes

class LoiteringDetector(CameraModelBase):
    """Loitering detection for monitoring extended stays in areas"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize loitering detector
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (common_zones, restricted_zones)
            settings: Detection settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # Loitering detection parameters
        self.loitering_threshold = settings.get('loitering_threshold', 300)  # 5 minutes default
        self.movement_threshold = settings.get('movement_threshold', 50)     # pixels
        self.check_interval = settings.get('check_interval', 30)             # seconds
        self.confidence_threshold = settings.get('confidence_threshold', 0.7)
        
        # Zone-specific thresholds
        self.zone_thresholds = settings.get('zone_thresholds', {})
        
        # Time-based rules
        self.time_based_rules = settings.get('time_based_rules', True)
        self.business_hours_threshold = settings.get('business_hours_threshold', 600)  # 10 minutes
        self.after_hours_threshold = settings.get('after_hours_threshold', 120)        # 2 minutes
        
        # Get monitoring zones
        self.common_zones = self.get_zones_by_type('common_zone')
        self.restricted_zones = self.get_zones_by_type('restricted_zone')
        self.monitored_zones = self.common_zones + self.restricted_zones
        
        # Tracking state
        self.person_positions = {}      # Track position history per person
        self.loitering_timers = {}      # Track loitering duration per person per zone
        self.movement_analysis = {}     # Movement pattern analysis
        self.zone_occupancy_history = {} # Historical occupancy data
        
        # Statistics
        self.loitering_stats = {
            'total_loitering_events': 0,
            'active_loiterers': 0,
            'zones_with_loitering': set(),
            'average_loitering_duration': 0,
            'max_loitering_duration': 0
        }
        
        self.logger.info(f"Loitering detector initialized - Threshold: {self.loitering_threshold}s")
        self.logger.info(f"Monitoring {len(self.monitored_zones)} zones for loitering behavior")
    
    def _get_zone_threshold(self, zone: Dict[str, Any]) -> float:
        """Get loitering threshold for specific zone"""
        zone_id = zone.get('zone_id', 0)
        zone_type = zone.get('zone_type', 'common_zone')
        zone_name = zone.get('name', f'Zone {zone_id}')
        
        # Check for zone-specific threshold
        if zone_name in self.zone_thresholds:
            return self.zone_thresholds[zone_name]
        
        if f"zone_{zone_id}" in self.zone_thresholds:
            return self.zone_thresholds[f"zone_{zone_id}"]
        
        # Default thresholds by zone type
        zone_type_thresholds = {
            'common_zone': self.loitering_threshold,
            'restricted_zone': self.loitering_threshold // 2,  # Stricter for restricted areas
            'entry_zone': self.loitering_threshold // 4,       # Very strict for entries
            'perimeter_zone': self.loitering_threshold // 3    # Strict for perimeter
        }
        
        return zone_type_thresholds.get(zone_type, self.loitering_threshold)
    
    def _is_business_hours(self, timestamp: float) -> bool:
        """Check if current time is business hours"""
        if not self.time_based_rules:
            return True
        
        dt = datetime.fromtimestamp(timestamp)
        current_hour = dt.hour
        current_weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
        
        # Weekends are generally non-business hours
        if current_weekday >= 5:
            return False
        
        # Business hours: 8 AM to 6 PM
        return 8 <= current_hour < 18
    
    def _update_position_history(self, tracked_people: List[Dict[str, Any]], timestamp: float):
        """Update position history for movement analysis"""
        current_tracks = set()
        
        for person in tracked_people:
            track_id = person['track_id']
            center = person['center']
            bbox = person['bbox']
            
            current_tracks.add(track_id)
            
            if track_id not in self.person_positions:
                self.person_positions[track_id] = {
                    'positions': [],
                    'timestamps': [],
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'total_movement': 0,
                    'stationary_periods': []
                }
            
            position_data = self.person_positions[track_id]
            position_data['positions'].append(center)
            position_data['timestamps'].append(timestamp)
            position_data['last_seen'] = timestamp
            
            # Calculate movement since last position
            if len(position_data['positions']) > 1:
                prev_pos = position_data['positions'][-2]
                movement_distance = np.sqrt(
                    (center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2
                )
                position_data['total_movement'] += movement_distance
            
            # Keep limited history (last 100 positions)
            if len(position_data['positions']) > 100:
                position_data['positions'] = position_data['positions'][-100:]
                position_data['timestamps'] = position_data['timestamps'][-100:]
        
        # Clean up old tracks
        tracks_to_remove = []
        for track_id in self.person_positions:
            if track_id not in current_tracks:
                # Keep data for 5 minutes after person disappears
                if timestamp - self.person_positions[track_id]['last_seen'] > 300:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.person_positions[track_id]
            # Also clean up loitering timers
            keys_to_remove = [k for k in self.loitering_timers.keys() if k.startswith(f"{track_id}_")]
            for key in keys_to_remove:
                del self.loitering_timers[key]
    
    def _analyze_movement_patterns(self, track_id: int, timestamp: float) -> Dict[str, Any]:
        """Analyze movement patterns for a person"""
        if track_id not in self.person_positions:
            return {}
        
        position_data = self.person_positions[track_id]
        positions = position_data['positions']
        timestamps = position_data['timestamps']
        
        if len(positions) < 2:
            return {}
        
        # Calculate recent movement (last 30 seconds)
        recent_cutoff = timestamp - 30
        recent_positions = []
        recent_timestamps = []
        
        for pos, ts in zip(positions, timestamps):
            if ts >= recent_cutoff:
                recent_positions.append(pos)
                recent_timestamps.append(ts)
        
        analysis = {
            'total_duration': timestamp - position_data['first_seen'],
            'total_positions': len(positions),
            'total_movement': position_data['total_movement'],
            'recent_movement': 0,
            'movement_rate': 0,
            'is_stationary': False,
            'stationary_duration': 0,
            'movement_pattern': 'unknown'
        }
        
        # Calculate recent movement
        if len(recent_positions) > 1:
            for i in range(1, len(recent_positions)):
                curr_pos = recent_positions[i]
                prev_pos = recent_positions[i-1]
                movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                analysis['recent_movement'] += movement
        
        # Calculate movement rate (pixels per second)
        if analysis['total_duration'] > 0:
            analysis['movement_rate'] = analysis['total_movement'] / analysis['total_duration']
        
        # Determine if stationary
        analysis['is_stationary'] = analysis['recent_movement'] < self.movement_threshold
        
        # Calculate stationary duration
        if analysis['is_stationary']:
            # Find when person became stationary
            stationary_start = timestamp
            for i in range(len(positions) - 1, 0, -1):
                if i < len(positions) - 1:
                    curr_pos = positions[i]
                    next_pos = positions[i + 1]
                    movement = np.sqrt((next_pos[0] - curr_pos[0])**2 + (next_pos[1] - curr_pos[1])**2)
                    
                    if movement > self.movement_threshold:
                        stationary_start = timestamps[i + 1]
                        break
                else:
                    stationary_start = timestamps[i]
            
            analysis['stationary_duration'] = timestamp - stationary_start
        
        # Classify movement pattern
        if analysis['movement_rate'] < 5:
            analysis['movement_pattern'] = 'stationary'
        elif analysis['movement_rate'] < 20:
            analysis['movement_pattern'] = 'slow_walking'
        elif analysis['movement_rate'] < 50:
            analysis['movement_pattern'] = 'normal_walking'
        else:
            analysis['movement_pattern'] = 'fast_movement'
        
        return analysis
    
    def _detect_zone_loitering(self, tracked_people: List[Dict[str, Any]], 
                              timestamp: float) -> List[Dict[str, Any]]:
        """Detect loitering in monitored zones"""
        loitering_events = []
        is_business_hours = self._is_business_hours(timestamp)
        
        for zone in self.monitored_zones:
            zone_id = zone.get('zone_id', 0)
            zone_threshold = self._get_zone_threshold(zone)
            
            # Adjust threshold based on time of day
            if self.time_based_rules:
                if is_business_hours:
                    zone_threshold = max(zone_threshold, self.business_hours_threshold)
                else:
                    zone_threshold = min(zone_threshold, self.after_hours_threshold)
            
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            
            for person in people_in_zone:
                track_id = person['track_id']
                timer_key = f"{track_id}_{zone_id}"
                
                # Initialize or update loitering timer
                if timer_key not in self.loitering_timers:
                    self.loitering_timers[timer_key] = {
                        'start_time': timestamp,
                        'zone_id': zone_id,
                        'zone_name': zone.get('name', f'Zone {zone_id}'),
                        'track_id': track_id,
                        'alerted': False,
                        'movement_analysis': None
                    }
                
                timer_data = self.loitering_timers[timer_key]
                duration = timestamp - timer_data['start_time']
                
                # Get movement analysis
                movement_analysis = self._analyze_movement_patterns(track_id, timestamp)
                timer_data['movement_analysis'] = movement_analysis
                
                # Check if loitering threshold exceeded
                is_loitering = (
                    duration >= zone_threshold and
                    movement_analysis.get('is_stationary', False) and
                    movement_analysis.get('stationary_duration', 0) >= zone_threshold * 0.7
                )
                
                if is_loitering and not timer_data['alerted']:
                    # Mark as alerted to prevent repeated alerts
                    timer_data['alerted'] = True
                    
                    loitering_event = {
                        'track_id': track_id,
                        'zone_id': zone_id,
                        'zone_name': timer_data['zone_name'],
                        'duration': duration,
                        'threshold': zone_threshold,
                        'person_bbox': person['bbox'],
                        'movement_analysis': movement_analysis,
                        'is_business_hours': is_business_hours,
                        'timestamp': timestamp,
                        'severity': self._determine_loitering_severity(zone, duration, is_business_hours)
                    }
                    loitering_events.append(loitering_event)
                    
                    self.logger.warning(f"Loitering detected: Person {track_id} in {timer_data['zone_name']} for {duration:.1f}s")
        
        # Clean up timers for people no longer in zones
        active_timer_keys = set()
        for zone in self.monitored_zones:
            zone_id = zone.get('zone_id', 0)
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            for person in people_in_zone:
                active_timer_keys.add(f"{person['track_id']}_{zone_id}")
        
        timers_to_remove = []
        for timer_key in self.loitering_timers:
            if timer_key not in active_timer_keys:
                # Keep timer for a short period in case person temporarily moves out
                timer_data = self.loitering_timers[timer_key]
                if timestamp - timer_data.get('last_active', timestamp) > 60:  # 1 minute grace period
                    timers_to_remove.append(timer_key)
                else:
                    timer_data['last_active'] = timestamp
        
        for timer_key in timers_to_remove:
            del self.loitering_timers[timer_key]
        
        return loitering_events
    
    def _determine_loitering_severity(self, zone: Dict[str, Any], duration: float, 
                                    is_business_hours: bool) -> str:
        """Determine severity of loitering event"""
        zone_type = zone.get('zone_type', 'common_zone')
        zone_threshold = self._get_zone_threshold(zone)
        
        # Base severity by zone type
        base_severity = {
            'common_zone': 'low',
            'restricted_zone': 'medium',
            'entry_zone': 'high',
            'perimeter_zone': 'high'
        }.get(zone_type, 'low')
        
        # Escalate based on duration
        duration_multiplier = duration / zone_threshold
        if duration_multiplier > 3:  # More than 3x threshold
            severity_escalation = {'low': 'medium', 'medium': 'high', 'high': 'critical'}
            base_severity = severity_escalation.get(base_severity, base_severity)
        
        # Escalate for after-hours
        if not is_business_hours:
            severity_escalation = {'low': 'medium', 'medium': 'high', 'high': 'critical'}
            base_severity = severity_escalation.get(base_severity, base_severity)
        
        return base_severity
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for loitering detection
        
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
        
        # Update position history
        self._update_position_history(tracked_people, timestamp)
        
        # Detect loitering
        loitering_events = self._detect_zone_loitering(tracked_people, timestamp)
        
        # Update statistics
        self.loitering_stats['active_loiterers'] = len([t for t in self.loitering_timers.values() 
                                                       if timestamp - t['start_time'] >= self._get_zone_threshold({'zone_id': t['zone_id']})])
        
        if loitering_events:
            self.loitering_stats['zones_with_loitering'].update(e['zone_id'] for e in loitering_events)
            durations = [e['duration'] for e in loitering_events]
            self.loitering_stats['max_loitering_duration'] = max(self.loitering_stats['max_loitering_duration'], 
                                                               max(durations))
        
        # Create loitering events
        for loitering in loitering_events:
            if not self._check_event_cooldown('loitering', loitering['zone_id']):
                event = self._create_event(
                    event_type='loitering',
                    description=f"Loitering detected: Person {loitering['track_id']} in {loitering['zone_name']} for {loitering['duration']:.1f}s",
                    detection_data={
                        'track_id': loitering['track_id'],
                        'zone_id': loitering['zone_id'],
                        'zone_name': loitering['zone_name'],
                        'duration': loitering['duration'],
                        'threshold': loitering['threshold'],
                        'person_bbox': loitering['person_bbox'],
                        'movement_analysis': loitering['movement_analysis'],
                        'is_business_hours': loitering['is_business_hours'],
                        'confidence': 0.8
                    },
                    zone_id=loitering['zone_id'],
                    severity=loitering['severity']
                )
                events.append(event)
                self._save_event(event)
                
                self.loitering_stats['total_loitering_events'] += 1
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, loitering_events, timestamp
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               loitering_events: List[Dict[str, Any]],
                               timestamp: float) -> np.ndarray:
        """Create annotated frame with loitering detection visualization"""
        annotated_frame = frame.copy()
        
        # Draw monitored zones
        for zone in self.monitored_zones:
            zone_id = zone.get('zone_id', 0)
            zone_threshold = self._get_zone_threshold(zone)
            
            # Check if zone has loitering
            has_loitering = any(e['zone_id'] == zone_id for e in loitering_events)
            
            # Color based on zone type and loitering status
            if has_loitering:
                color = (0, 0, 255)  # Red for active loitering
                alpha = 0.4
            else:
                zone_type = zone.get('zone_type', 'common_zone')
                zone_colors = {
                    'common_zone': (0, 255, 255),     # Cyan
                    'restricted_zone': (255, 255, 0), # Yellow
                    'entry_zone': (255, 0, 255),      # Magenta
                    'perimeter_zone': (255, 165, 0)   # Orange
                }
                color = zone_colors.get(zone_type, (0, 255, 255))
                alpha = 0.2
            
            label = f"{zone.get('name', f'Zone {zone_id}')} (Threshold: {zone_threshold}s)"
            if has_loitering:
                label += " - LOITERING!"
            
            self._draw_zone(annotated_frame, zone, color, alpha, label)
        
        # Draw tracked people with loitering status and movement trails
        for person in tracked_people:
            track_id = person['track_id']
            bbox = person['bbox']
            
            # Check if person is loitering in any zone
            is_loitering = any(e['track_id'] == track_id for e in loitering_events)
            
            # Get movement analysis
            movement_analysis = self._analyze_movement_patterns(track_id, timestamp)
            is_stationary = movement_analysis.get('is_stationary', False)
            movement_pattern = movement_analysis.get('movement_pattern', 'unknown')
            
            # Color based on loitering status and movement
            if is_loitering:
                color = (0, 0, 255)  # Red for loitering
                thickness = 3
            elif is_stationary:
                color = (0, 255, 255)  # Yellow for stationary (potential loiterer)
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for moving normally
                thickness = 2
            
            self._draw_person(annotated_frame, person, color, thickness)
            
            # Draw movement trail
            if track_id in self.person_positions:
                positions = self.person_positions[track_id]['positions'][-20:]  # Last 20 positions
                if len(positions) > 1:
                    trail_color = color
                    for i in range(1, len(positions)):
                        pt1 = tuple(map(int, positions[i-1]))
                        pt2 = tuple(map(int, positions[i]))
                        cv2.line(annotated_frame, pt1, pt2, trail_color, 1)
            
            # Draw loitering timer and movement info
            timer_info = []
            for zone in self.monitored_zones:
                zone_id = zone.get('zone_id', 0)
                timer_key = f"{track_id}_{zone_id}"
                
                if timer_key in self.loitering_timers:
                    duration = timestamp - self.loitering_timers[timer_key]['start_time']
                    zone_name = zone.get('name', f'Zone {zone_id}')
                    timer_info.append(f"{zone_name}: {duration:.0f}s")
            
            # Draw timer information
            info_y = int(bbox[3]) + 15
            if timer_info:
                timer_text = ", ".join(timer_info)
                cv2.putText(annotated_frame, timer_text,
                           (int(bbox[0]), info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                info_y += 15
            
            # Draw movement pattern
            if movement_analysis:
                pattern_text = f"{movement_pattern} ({movement_analysis.get('movement_rate', 0):.1f} px/s)"
                cv2.putText(annotated_frame, pattern_text,
                           (int(bbox[0]), info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw loitering alerts
        alert_y = 60
        for loitering in loitering_events:
            movement_info = loitering['movement_analysis']
            alert_text = f"LOITERING ALERT: {loitering['zone_name']} - ID:{loitering['track_id']} - {loitering['duration']:.0f}s/{loitering['threshold']:.0f}s"
            cv2.putText(annotated_frame, alert_text, (10, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Additional movement details
            if movement_info:
                movement_text = f"  Pattern: {movement_info.get('movement_pattern', 'unknown')}, Stationary: {movement_info.get('stationary_duration', 0):.0f}s"
                cv2.putText(annotated_frame, movement_text, (10, alert_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                alert_y += 40
            else:
                alert_y += 25
        
        # Draw business hours status
        is_business_hours = self._is_business_hours(timestamp)
        hours_text = f"Hours: {'Business' if is_business_hours else 'After Hours'}"
        hours_color = (0, 255, 0) if is_business_hours else (0, 165, 255)
        cv2.putText(annotated_frame, hours_text, (10, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hours_color, 2)
        
        # Draw active timers summary
        active_timers = len([t for t in self.loitering_timers.values() 
                           if timestamp - t['start_time'] >= 30])  # Show timers > 30s
        
        if active_timers > 0:
            timers_text = f"Active Timers: {active_timers}"
            cv2.putText(annotated_frame, timers_text, (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw statistics
        stats_text = f"Loitering Events: {self.loitering_stats['total_loitering_events']} | Active: {self.loitering_stats['active_loiterers']} | Max Duration: {self.loitering_stats['max_loitering_duration']:.0f}s"
        cv2.putText(annotated_frame, stats_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Main info
        info_text = f"Camera {self.camera_id} - Loitering Detection | People: {len(tracked_people)} | Threshold: {self.loitering_threshold}s"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_active_loiterers(self) -> List[Dict[str, Any]]:
        """Get information about currently active loiterers"""
        current_time = time.time()
        active_loiterers = []
        
        for timer_key, timer_data in self.loitering_timers.items():
            duration = current_time - timer_data['start_time']
            zone_threshold = self._get_zone_threshold({'zone_id': timer_data['zone_id']})
            
            if duration >= zone_threshold * 0.5:  # At least 50% of threshold
                track_id = timer_data['track_id']
                movement_analysis = self._analyze_movement_patterns(track_id, current_time)
                
                loiterer_info = {
                    'track_id': track_id,
                    'zone_id': timer_data['zone_id'],
                    'zone_name': timer_data['zone_name'],
                    'duration': duration,
                    'threshold': zone_threshold,
                    'is_violating': duration >= zone_threshold,
                    'movement_analysis': movement_analysis,
                    'start_time': timer_data['start_time']
                }
                active_loiterers.append(loiterer_info)
        
        # Sort by duration (longest first)
        active_loiterers.sort(key=lambda x: x['duration'], reverse=True)
        return active_loiterers
    
    def get_zone_occupancy_stats(self) -> Dict[str, Any]:
        """Get occupancy statistics for monitored zones"""
        zone_stats = {}
        current_time = time.time()
        
        for zone in self.monitored_zones:
            zone_id = zone.get('zone_id', 0)
            zone_name = zone.get('name', f'Zone {zone_id}')
            
            # Count active timers in this zone
            active_in_zone = len([t for t in self.loitering_timers.values() 
                                if t['zone_id'] == zone_id])
            
            # Count loitering violations in this zone
            violations_in_zone = len([t for t in self.loitering_timers.values() 
                                    if t['zone_id'] == zone_id and 
                                    (current_time - t['start_time']) >= self._get_zone_threshold(zone)])
            
            zone_stats[zone_name] = {
                'zone_id': zone_id,
                'current_occupancy': active_in_zone,
                'loitering_violations': violations_in_zone,
                'threshold': self._get_zone_threshold(zone),
                'zone_type': zone.get('zone_type', 'unknown')
            }
        
        return zone_stats
    
    def update_zone_thresholds(self, new_thresholds: Dict[str, float]):
        """Update zone-specific loitering thresholds"""
        self.zone_thresholds.update(new_thresholds)
        self.logger.info(f"Zone thresholds updated: {new_thresholds}")
    
    def reset_person_tracking(self, track_id: int = None):
        """Reset tracking data for specific person or all people"""
        if track_id is not None:
            # Reset specific person
            if track_id in self.person_positions:
                del self.person_positions[track_id]
            
            # Remove loitering timers for this person
            keys_to_remove = [k for k in self.loitering_timers.keys() 
                            if k.startswith(f"{track_id}_")]
            for key in keys_to_remove:
                del self.loitering_timers[key]
            
            self.logger.info(f"Reset tracking data for person {track_id}")
        else:
            # Reset all tracking
            self.person_positions.clear()
            self.loitering_timers.clear()
            self.movement_analysis.clear()
            self.logger.info("Reset all person tracking data")


# Export main class
__all__ = ['LoiteringDetector']
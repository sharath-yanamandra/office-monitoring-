#!/usr/bin/env python3
"""
detection_scripts/tailgating_detection.py
Tailgating Detection Script

This script detects when multiple people enter through a single access point
within a short time window, indicating potential tailgating behavior.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from camera_model_base import CameraModelBase
from config import EventTypes

class TailgatingDetector(CameraModelBase):
    """Tailgating detection for entry points"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize tailgating detector
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (should include entry_zones)
            settings: Detection settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # Tailgating specific parameters
        self.time_window = settings.get('time_window', 10)  # seconds
        self.max_people_per_entry = settings.get('max_people_per_entry', 1)
        self.entry_zone_buffer = settings.get('entry_zone_buffer', 2.0)  # meters
        
        # Entry tracking
        self.entry_events = []  # Track recent entry events
        self.zone_occupancy = {}  # Track current occupancy per zone
        self.entry_directions = {}  # Track movement directions
        
        # Get entry zones
        self.entry_zones = self.get_zones_by_type('entry_zone')
        
        self.logger.info(f"Tailgating detector initialized for {len(self.entry_zones)} entry zones")
        self.logger.info(f"Parameters: time_window={self.time_window}s, max_people={self.max_people_per_entry}")
    
    def _detect_zone_entries(self, tracked_people: List[Dict[str, Any]], 
                           timestamp: float) -> List[Dict[str, Any]]:
        """
        Detect people entering zones
        
        Args:
            tracked_people: List of tracked people
            timestamp: Current timestamp
            
        Returns:
            List of entry events
        """
        entry_events = []
        
        for zone in self.entry_zones:
            zone_id = zone.get('zone_id', 0)
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            
            # Update current occupancy
            current_occupancy = len(people_in_zone)
            previous_occupancy = self.zone_occupancy.get(zone_id, 0)
            self.zone_occupancy[zone_id] = current_occupancy
            
            # Detect new entries (occupancy increased)
            if current_occupancy > previous_occupancy:
                new_entries = current_occupancy - previous_occupancy
                
                # Create entry event
                entry_event = {
                    'zone_id': zone_id,
                    'zone_name': zone.get('name', f'Entry Zone {zone_id}'),
                    'timestamp': timestamp,
                    'people_count': new_entries,
                    'total_occupancy': current_occupancy,
                    'people': [p['track_id'] for p in people_in_zone]
                }
                entry_events.append(entry_event)
                
                self.logger.debug(f"Entry detected in {zone['name']}: {new_entries} people")
        
        return entry_events
    
    def _check_tailgating(self, new_entries: List[Dict[str, Any]], 
                         timestamp: float) -> List[Dict[str, Any]]:
        """
        Check for tailgating based on recent entries
        
        Args:
            new_entries: New entry events
            timestamp: Current timestamp
            
        Returns:
            List of tailgating events
        """
        tailgating_events = []
        
        # Add new entries to history
        self.entry_events.extend(new_entries)
        
        # Clean old entries outside time window
        cutoff_time = timestamp - self.time_window
        self.entry_events = [e for e in self.entry_events if e['timestamp'] >= cutoff_time]
        
        # Group entries by zone
        zone_entries = {}
        for entry in self.entry_events:
            zone_id = entry['zone_id']
            if zone_id not in zone_entries:
                zone_entries[zone_id] = []
            zone_entries[zone_id].append(entry)
        
        # Check each zone for tailgating
        for zone_id, entries in zone_entries.items():
            if len(entries) < 2:  # Need at least 2 entries for tailgating
                continue
            
            # Calculate total people in time window
            total_people = sum(entry['people_count'] for entry in entries)
            
            # Check if too many people entered in time window
            if total_people > self.max_people_per_entry:
                # Find the zone info
                zone_info = None
                for zone in self.entry_zones:
                    if zone.get('zone_id') == zone_id:
                        zone_info = zone
                        break
                
                # Create tailgating event
                tailgating_event = {
                    'zone_id': zone_id,
                    'zone_name': zone_info.get('name', f'Entry Zone {zone_id}') if zone_info else f'Zone {zone_id}',
                    'timestamp': timestamp,
                    'people_count': total_people,
                    'time_window': self.time_window,
                    'entries': entries,
                    'severity': 'high' if total_people > 2 else 'medium'
                }
                tailgating_events.append(tailgating_event)
                
                self.logger.warning(f"Tailgating detected in {tailgating_event['zone_name']}: "
                                  f"{total_people} people in {self.time_window}s")
        
        return tailgating_events
    
    def _analyze_entry_patterns(self, tracked_people: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entry patterns for additional context"""
        analysis = {
            'simultaneous_entries': 0,
            'rapid_succession_entries': 0,
            'suspicious_behavior': False
        }
        
        # Count people currently in entry zones
        people_in_entries = 0
        for zone in self.entry_zones:
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            people_in_entries += len(people_in_zone)
            
            # Check for simultaneous entries (multiple people in same zone)
            if len(people_in_zone) > 1:
                analysis['simultaneous_entries'] += len(people_in_zone)
        
        # Check for rapid succession (multiple entries in short time)
        if len(self.entry_events) > 1:
            recent_entries = [e for e in self.entry_events 
                            if time.time() - e['timestamp'] < 5]  # Last 5 seconds
            if len(recent_entries) > 1:
                analysis['rapid_succession_entries'] = len(recent_entries)
        
        # Determine if behavior is suspicious
        analysis['suspicious_behavior'] = (
            analysis['simultaneous_entries'] > self.max_people_per_entry or
            analysis['rapid_succession_entries'] > 1
        )
        
        return analysis
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for tailgating detection
        
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
        
        # Detect new entries
        new_entries = self._detect_zone_entries(tracked_people, timestamp)
        
        # Check for tailgating
        tailgating_events = self._check_tailgating(new_entries, timestamp)
        
        # Analyze entry patterns
        pattern_analysis = self._analyze_entry_patterns(tracked_people)
        
        # Create tailgating events
        for tailgating in tailgating_events:
            if not self._check_event_cooldown('tailgating', tailgating['zone_id']):
                event = self._create_event(
                    event_type='tailgating',
                    description=f"Tailgating detected: {tailgating['people_count']} people in {tailgating['time_window']}s",
                    detection_data={
                        'zone_id': tailgating['zone_id'],
                        'zone_name': tailgating['zone_name'],
                        'people_count': tailgating['people_count'],
                        'time_window': tailgating['time_window'],
                        'entries': tailgating['entries'],
                        'pattern_analysis': pattern_analysis,
                        'confidence': 0.9 if tailgating['people_count'] > 3 else 0.7
                    },
                    zone_id=tailgating['zone_id'],
                    severity=tailgating['severity']
                )
                events.append(event)
                self._save_event(event)
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, new_entries, tailgating_events, pattern_analysis
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               new_entries: List[Dict[str, Any]],
                               tailgating_events: List[Dict[str, Any]],
                               pattern_analysis: Dict[str, Any]) -> np.ndarray:
        """Create annotated frame with tailgating visualization"""
        annotated_frame = frame.copy()
        
        # Draw entry zones
        for zone in self.entry_zones:
            zone_id = zone.get('zone_id', 0)
            occupancy = self.zone_occupancy.get(zone_id, 0)
            
            # Color based on occupancy and tailgating status
            is_tailgating = any(t['zone_id'] == zone_id for t in tailgating_events)
            if is_tailgating:
                color = (0, 0, 255)  # Red for tailgating
                alpha = 0.4
            elif occupancy > self.max_people_per_entry:
                color = (0, 165, 255)  # Orange for warning
                alpha = 0.3
            else:
                color = self.zone_colors.get('entry_zone', (0, 255, 0))  # Green for normal
                alpha = 0.2
            
            label = f"{zone.get('name', f'Zone {zone_id}')} ({occupancy})"
            self._draw_zone(annotated_frame, zone, color, alpha, label)
        
        # Draw tracked people
        for person in tracked_people:
            # Color based on zone location
            person_color = (0, 255, 0)  # Default green
            
            # Check if person is in entry zone
            for zone in self.entry_zones:
                if self._is_person_in_zone(person, zone):
                    zone_id = zone.get('zone_id', 0)
                    is_tailgating = any(t['zone_id'] == zone_id for t in tailgating_events)
                    person_color = (0, 0, 255) if is_tailgating else (255, 255, 0)  # Red if tailgating, yellow in entry
                    break
            
            self._draw_person(annotated_frame, person, person_color, 2)
        
        # Draw tailgating alerts
        alert_y = 60
        for tailgating in tailgating_events:
            alert_text = f"TAILGATING ALERT: {tailgating['zone_name']} - {tailgating['people_count']} people"
            cv2.putText(annotated_frame, alert_text, (10, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        
        # Draw statistics
        stats_y = 30
        info_texts = [
            f"Camera {self.camera_id} - Tailgating Detection",
            f"People: {len(tracked_people)} | Recent Entries: {len(self.entry_events)}",
            f"Simultaneous: {pattern_analysis['simultaneous_entries']} | Rapid: {pattern_analysis['rapid_succession_entries']}"
        ]
        
        for text in info_texts:
            cv2.putText(annotated_frame, text, (10, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            stats_y += 25
        
        # Draw entry history visualization
        self._draw_entry_history(annotated_frame)
        
        return annotated_frame
    
    def _draw_entry_history(self, frame: np.ndarray):
        """Draw entry history timeline"""
        if not self.entry_events:
            return
        
        # Draw timeline at bottom of frame
        timeline_y = frame.shape[0] - 50
        timeline_start_x = 50
        timeline_width = min(400, frame.shape[1] - 100)
        
        # Draw timeline background
        cv2.rectangle(frame, (timeline_start_x - 10, timeline_y - 20),
                     (timeline_start_x + timeline_width + 10, timeline_y + 20),
                     (50, 50, 50), -1)
        
        # Draw timeline axis
        cv2.line(frame, (timeline_start_x, timeline_y),
                (timeline_start_x + timeline_width, timeline_y), (255, 255, 255), 2)
        
        # Draw entry events on timeline
        current_time = time.time()
        for entry in self.entry_events:
            time_diff = current_time - entry['timestamp']
            if time_diff > self.time_window:
                continue
            
            # Calculate position on timeline
            pos_ratio = 1.0 - (time_diff / self.time_window)
            x_pos = int(timeline_start_x + pos_ratio * timeline_width)
            
            # Draw entry marker
            color = (0, 255, 255) if entry['people_count'] > 1 else (0, 255, 0)
            cv2.circle(frame, (x_pos, timeline_y), 5, color, -1)
            
            # Draw count
            cv2.putText(frame, str(entry['people_count']), (x_pos - 5, timeline_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw timeline labels
        cv2.putText(frame, f"{self.time_window}s ago", (timeline_start_x, timeline_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Now", (timeline_start_x + timeline_width - 20, timeline_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# Export main class
__all__ = ['TailgatingDetector']
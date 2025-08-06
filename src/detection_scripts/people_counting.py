#!/usr/bin/env python3
"""
detection_scripts/people_counting.py
People Counting Script

This script counts people crossing counting lines or entering/exiting zones,
providing occupancy monitoring and traffic flow analysis.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from camera_model_base import CameraModelBase
from config import EventTypes

class PeopleCounter(CameraModelBase):
    """People counting for traffic flow and occupancy monitoring"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize people counter
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (counting_zones)
            settings: Counting settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # Counting parameters
        self.counting_line_threshold = settings.get('counting_line_threshold', 0.5)
        self.direction_tracking = settings.get('direction_tracking', True)
        self.occupancy_monitoring = settings.get('occupancy_monitoring', True)
        
        # Counting state
        self.total_count_in = 0
        self.total_count_out = 0
        self.current_occupancy = 0
        self.track_positions = {}  # Track previous positions for direction
        self.track_counted = {}    # Track which people have been counted
        
        # Zone-specific counters
        self.zone_counters = {}
        
        # Get counting zones
        self.counting_zones = self.get_zones_by_type('counting_zone')
        
        # Initialize zone counters
        for zone in self.counting_zones:
            zone_id = zone.get('zone_id', 0)
            self.zone_counters[zone_id] = {
                'count_in': 0,
                'count_out': 0,
                'current_occupancy': 0,
                'people_inside': set(),
                'counting_line': self._extract_counting_line(zone)
            }
        
        self.logger.info(f"People counter initialized for {len(self.counting_zones)} counting zones")
    
    def _extract_counting_line(self, zone: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract counting line from zone coordinates"""
        coordinates = zone.get('coordinates', [])
        
        if len(coordinates) < 2:
            return None
        
        # For line zones, use first and last points
        if len(coordinates) == 2:
            return {
                'start': coordinates[0],
                'end': coordinates[1],
                'type': 'line'
            }
        
        # For polygon zones, calculate center line
        if len(coordinates) > 2:
            # Calculate bounding box center line
            xs = [p[0] for p in coordinates]
            ys = [p[1] for p in coordinates]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Vertical line through center
            center_x = (min_x + max_x) / 2
            return {
                'start': [center_x, min_y],
                'end': [center_x, max_y],
                'type': 'center_line'
            }
        
        return None
    
    def _point_line_distance(self, point: Tuple[float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
        """Calculate shortest distance from point to line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate line length
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if line_length == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Calculate distance
        distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length
        return distance
    
    def _check_line_crossing(self, track_id: int, current_pos: Tuple[float, float], 
                           previous_pos: Tuple[float, float], 
                           counting_line: Dict[str, Any]) -> Optional[str]:
        """
        Check if person crossed counting line and determine direction
        
        Args:
            track_id: Person track ID
            current_pos: Current center position
            previous_pos: Previous center position
            counting_line: Counting line definition
            
        Returns:
            Direction ('in' or 'out') if crossed, None otherwise
        """
        if not counting_line:
            return None
        
        line_start = counting_line['start']
        line_end = counting_line['end']
        
        # Calculate distances to line
        current_dist = self._point_line_distance(current_pos, line_start, line_end)
        previous_dist = self._point_line_distance(previous_pos, line_start, line_end)
        
        # Check if crossed the line (distance changed from one side to other)
        threshold = 20.0  # pixels
        
        if previous_dist > threshold and current_dist <= threshold:
            # Determine direction based on line orientation and movement
            line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
            movement_vector = (current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1])
            
            # Cross product to determine which side
            cross_product = line_vector[0] * movement_vector[1] - line_vector[1] * movement_vector[0]
            
            return 'in' if cross_product > 0 else 'out'
        
        return None
    
    def _update_zone_occupancy(self, tracked_people: List[Dict[str, Any]]):
        """Update occupancy for each zone"""
        for zone in self.counting_zones:
            zone_id = zone.get('zone_id', 0)
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            
            # Update current people in zone
            current_people_ids = {p['track_id'] for p in people_in_zone}
            
            if zone_id in self.zone_counters:
                previous_people_ids = self.zone_counters[zone_id]['people_inside']
                
                # People who entered
                new_people = current_people_ids - previous_people_ids
                
                # People who left
                left_people = previous_people_ids - current_people_ids
                
                # Update counters
                if new_people:
                    self.zone_counters[zone_id]['count_in'] += len(new_people)
                    self.total_count_in += len(new_people)
                
                if left_people:
                    self.zone_counters[zone_id]['count_out'] += len(left_people)
                    self.total_count_out += len(left_people)
                
                # Update occupancy
                self.zone_counters[zone_id]['current_occupancy'] = len(current_people_ids)
                self.zone_counters[zone_id]['people_inside'] = current_people_ids
    
    def _process_line_counting(self, tracked_people: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        """Process counting line crossings"""
        counting_events = []
        
        for person in tracked_people:
            track_id = person['track_id']
            current_pos = person['center']
            
            # Check if we have previous position
            if track_id not in self.track_positions:
                self.track_positions[track_id] = current_pos
                continue
            
            previous_pos = self.track_positions[track_id]
            
            # Check line crossings for each zone
            for zone in self.counting_zones:
                zone_id = zone.get('zone_id', 0)
                counting_line = self.zone_counters[zone_id]['counting_line']
                
                if counting_line:
                    direction = self._check_line_crossing(track_id, current_pos, previous_pos, counting_line)
                    
                    if direction and track_id not in self.track_counted:
                        # Record the counting
                        self.track_counted[track_id] = {
                            'zone_id': zone_id,
                            'direction': direction,
                            'timestamp': timestamp
                        }
                        
                        # Update counters
                        if direction == 'in':
                            self.zone_counters[zone_id]['count_in'] += 1
                            self.total_count_in += 1
                        else:
                            self.zone_counters[zone_id]['count_out'] += 1
                            self.total_count_out += 1
                        
                        # Create counting event
                        counting_event = {
                            'zone_id': zone_id,
                            'zone_name': zone.get('name', f'Zone {zone_id}'),
                            'track_id': track_id,
                            'direction': direction,
                            'timestamp': timestamp,
                            'position': current_pos
                        }
                        counting_events.append(counting_event)
                        
                        self.logger.debug(f"Person {track_id} crossed line in {zone['name']}: {direction}")
            
            # Update position
            self.track_positions[track_id] = current_pos
        
        # Clean up old track data
        active_track_ids = {p['track_id'] for p in tracked_people}
        self.track_positions = {k: v for k, v in self.track_positions.items() if k in active_track_ids}
        self.track_counted = {k: v for k, v in self.track_counted.items() if k in active_track_ids}
        
        return counting_events
    
    def _calculate_traffic_stats(self) -> Dict[str, Any]:
        """Calculate traffic flow statistics"""
        total_traffic = self.total_count_in + self.total_count_out
        net_count = self.total_count_in - self.total_count_out
        
        # Zone-specific stats
        zone_stats = {}
        for zone_id, counter in self.zone_counters.items():
            zone_total = counter['count_in'] + counter['count_out']
            zone_net = counter['count_in'] - counter['count_out']
            
            zone_stats[zone_id] = {
                'count_in': counter['count_in'],
                'count_out': counter['count_out'],
                'net_count': zone_net,
                'total_traffic': zone_total,
                'current_occupancy': counter['current_occupancy']
            }
        
        return {
            'total_count_in': self.total_count_in,
            'total_count_out': self.total_count_out,
            'net_count': net_count,
            'total_traffic': total_traffic,
            'current_occupancy': self.current_occupancy,
            'zones': zone_stats
        }
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for people counting
        
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
        
        # Update occupancy
        if self.occupancy_monitoring:
            self._update_zone_occupancy(tracked_people)
        
        # Process line counting
        counting_events = self._process_line_counting(tracked_people, timestamp)
        
        # Calculate current occupancy
        self.current_occupancy = sum(counter['current_occupancy'] for counter in self.zone_counters.values())
        
        # Calculate traffic statistics
        traffic_stats = self._calculate_traffic_stats()
        
        # Create counting events
        for count_event in counting_events:
            if not self._check_event_cooldown('people_counting', count_event['zone_id']):
                event = self._create_event(
                    event_type='people_counting',
                    description=f"Person {count_event['direction']}: {count_event['zone_name']}",
                    detection_data={
                        'zone_id': count_event['zone_id'],
                        'zone_name': count_event['zone_name'],
                        'track_id': count_event['track_id'],
                        'direction': count_event['direction'],
                        'position': count_event['position'],
                        'traffic_stats': traffic_stats,
                        'confidence': 0.9
                    },
                    zone_id=count_event['zone_id'],
                    severity='info'
                )
                events.append(event)
                self._save_event(event)
        
        # Periodic occupancy update
        if self.frame_count % 30 == 0:  # Every 30 frames
            if not self._check_event_cooldown('people_counting'):
                event = self._create_event(
                    event_type='people_counting',
                    description=f"Occupancy update: {self.current_occupancy} people",
                    detection_data={
                        'occupancy': self.current_occupancy,
                        'traffic_stats': traffic_stats,
                        'zone_occupancy': {zid: counter['current_occupancy'] 
                                         for zid, counter in self.zone_counters.items()},
                        'confidence': 0.8
                    },
                    severity='info'
                )
                events.append(event)
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, counting_events, traffic_stats
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               counting_events: List[Dict[str, Any]],
                               traffic_stats: Dict[str, Any]) -> np.ndarray:
        """Create annotated frame with counting visualization"""
        annotated_frame = frame.copy()
        
        # Draw counting zones and lines
        for zone in self.counting_zones:
            zone_id = zone.get('zone_id', 0)
            counter = self.zone_counters.get(zone_id, {})
            occupancy = counter.get('current_occupancy', 0)
            
            # Draw zone
            color = self.zone_colors.get('counting_zone', (255, 0, 255))
            label = f"{zone.get('name', f'Zone {zone_id}')} ({occupancy})"
            self._draw_zone(annotated_frame, zone, color, alpha=0.2, label=label)
            
            # Draw counting line
            counting_line = counter.get('counting_line')
            if counting_line:
                start_point = tuple(map(int, counting_line['start']))
                end_point = tuple(map(int, counting_line['end']))
                
                # Draw line
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 255), 3)
                
                # Draw direction arrows
                line_center = (
                    (start_point[0] + end_point[0]) // 2,
                    (start_point[1] + end_point[1]) // 2
                )
                
                # Draw 'IN' and 'OUT' labels
                cv2.putText(annotated_frame, "IN", (line_center[0] - 30, line_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "OUT", (line_center[0] + 10, line_center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw tracked people
        for person in tracked_people:
            track_id = person['track_id']
            
            # Color based on recent counting
            person_color = (0, 255, 0)  # Default green
            if track_id in self.track_counted:
                direction = self.track_counted[track_id]['direction']
                person_color = (0, 255, 0) if direction == 'in' else (0, 0, 255)  # Green for in, red for out
            
            self._draw_person(annotated_frame, person, person_color, 2)
            
            # Draw trail/path
            if track_id in self.track_positions:
                current_pos = person['center']
                prev_pos = self.track_positions[track_id]
                cv2.line(annotated_frame, 
                        tuple(map(int, prev_pos)), 
                        tuple(map(int, current_pos)), 
                        person_color, 1)
        
        # Draw counting events
        event_y = 60
        for count_event in counting_events:
            direction_text = "↑ IN" if count_event['direction'] == 'in' else "↓ OUT"
            event_text = f"{direction_text} {count_event['zone_name']} - ID:{count_event['track_id']}"
            color = (0, 255, 0) if count_event['direction'] == 'in' else (0, 0, 255)
            cv2.putText(annotated_frame, event_text, (10, event_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            event_y += 25
        
        # Draw traffic statistics
        self._draw_traffic_stats(annotated_frame, traffic_stats)
        
        # Main info
        info_text = f"Camera {self.camera_id} - People Counting | Current: {len(tracked_people)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_traffic_stats(self, frame: np.ndarray, traffic_stats: Dict[str, Any]):
        """Draw traffic flow statistics on frame"""
        # Statistics panel
        panel_x = frame.shape[1] - 300
        panel_y = 30
        panel_width = 280
        panel_height = 150
        
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Draw statistics
        stats_text = [
            "TRAFFIC STATISTICS",
            f"Total IN:  {traffic_stats['total_count_in']}",
            f"Total OUT: {traffic_stats['total_count_out']}",
            f"Net Count: {traffic_stats['net_count']}",
            f"Current:   {traffic_stats['current_occupancy']}",
            f"Total:     {traffic_stats['total_traffic']}"
        ]
        
        text_y = panel_y + 25
        for i, text in enumerate(stats_text):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            font_scale = 0.6 if i == 0 else 0.5
            cv2.putText(frame, text, (panel_x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            text_y += 20
        
        # Draw flow direction indicator
        if traffic_stats['net_count'] != 0:
            arrow_start_x = panel_x + panel_width - 50
            arrow_y = panel_y + 75
            
            if traffic_stats['net_count'] > 0:
                # More people coming in
                cv2.arrowedLine(frame, (arrow_start_x, arrow_y + 15), 
                              (arrow_start_x, arrow_y - 15), (0, 255, 0), 3, tipLength=0.3)
                cv2.putText(frame, "NET IN", (arrow_start_x - 40, arrow_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # More people going out
                cv2.arrowedLine(frame, (arrow_start_x, arrow_y - 15), 
                              (arrow_start_x, arrow_y + 15), (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(frame, "NET OUT", (arrow_start_x - 40, arrow_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def get_current_count(self) -> Dict[str, Any]:
        """Get current counting statistics"""
        return self._calculate_traffic_stats()
    
    def reset_counters(self):
        """Reset all counters"""
        self.total_count_in = 0
        self.total_count_out = 0
        self.current_occupancy = 0
        
        for zone_id in self.zone_counters:
            self.zone_counters[zone_id].update({
                'count_in': 0,
                'count_out': 0,
                'current_occupancy': 0,
                'people_inside': set()
            })
        
        self.track_positions.clear()
        self.track_counted.clear()
        
        self.logger.info("People counters reset")


# Export main class
__all__ = ['PeopleCounter']
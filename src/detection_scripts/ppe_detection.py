#!/usr/bin/env python3
"""
detection_scripts/ppe_detection.py
PPE Detection Script

This script detects Personal Protective Equipment (PPE) compliance
including hard hats, safety vests, safety glasses, and gloves.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from camera_model_base import CameraModelBase
from config import EventTypes

class PPEDetector(CameraModelBase):
    """PPE compliance detection for safety monitoring"""
    
    def __init__(self, camera_id: int, zones: Optional[Dict] = None, 
                 settings: Optional[Dict] = None, db=None, db_writer=None, 
                 frames_base_dir: str = 'output_frames'):
        """
        Initialize PPE detector
        
        Args:
            camera_id: Camera identifier
            zones: Zone definitions (should include restricted_zones)
            settings: Detection settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Frame output directory
        """
        super().__init__(camera_id, zones, settings, db, db_writer, frames_base_dir)
        
        # PPE detection parameters
        self.required_ppe = settings.get('required_ppe', ['hard_hat', 'safety_vest'])
        self.ppe_confidence_threshold = settings.get('ppe_confidence_threshold', 0.7)
        self.grace_period = settings.get('grace_period', 30)  # seconds
        self.detection_enabled = settings.get('ppe_detection_enabled', True)
        
        # PPE tracking state
        self.person_ppe_status = {}  # Track PPE compliance per person
        self.ppe_violation_timers = {}  # Track violation duration
        self.entry_timestamps = {}  # When people entered zones
        
        # Get restricted zones where PPE is required
        self.ppe_required_zones = self.get_zones_by_type('restricted_zone')
        
        # PPE detection classes mapping
        self.ppe_classes = {
            'hard_hat': ['helmet', 'hard_hat', 'hardhat'],
            'safety_vest': ['vest', 'safety_vest', 'hi_vis'],
            'safety_glasses': ['glasses', 'safety_glasses', 'goggles'],
            'safety_gloves': ['gloves', 'safety_gloves']
        }
        
        # PPE colors for visualization
        self.ppe_colors = {
            'hard_hat': (0, 255, 255),      # Yellow
            'safety_vest': (0, 165, 255),   # Orange
            'safety_glasses': (255, 0, 255), # Magenta
            'safety_gloves': (128, 255, 0),  # Lime
            'compliant': (0, 255, 0),       # Green
            'violation': (0, 0, 255),       # Red
            'warning': (0, 255, 255)        # Yellow
        }
        
        self.logger.info(f"PPE detector initialized - Required PPE: {self.required_ppe}")
        self.logger.info(f"Monitoring {len(self.ppe_required_zones)} PPE zones")
    
    def _detect_ppe_items(self, frame: np.ndarray, model_manager) -> List[Dict[str, Any]]:
        """
        Detect PPE items in frame using PPE detection model
        
        Args:
            frame: Input video frame
            model_manager: Model manager instance
            
        Returns:
            List of PPE detections
        """
        if not hasattr(model_manager, 'detect_ppe'):
            # Fallback: simulate PPE detection for demo
            return self._simulate_ppe_detection(frame)
        
        try:
            ppe_detections = model_manager.detect_ppe(frame, self.ppe_confidence_threshold)
            
            # Normalize PPE class names
            normalized_detections = []
            for detection in ppe_detections:
                normalized_detection = detection.copy()
                class_name = detection['class_name'].lower()
                
                # Map to standard PPE classes
                for standard_class, variants in self.ppe_classes.items():
                    if any(variant in class_name for variant in variants):
                        normalized_detection['class_name'] = standard_class
                        break
                
                normalized_detections.append(normalized_detection)
            
            return normalized_detections
            
        except Exception as e:
            self.logger.error(f"Error in PPE detection: {e}")
            return []
    
    def _simulate_ppe_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Simulate PPE detection for demo purposes
        This should be replaced with actual PPE model inference
        """
        # Simple simulation: randomly generate some PPE detections
        import random
        
        simulated_detections = []
        h, w = frame.shape[:2]
        
        # Simulate 0-3 PPE items per frame
        num_items = random.randint(0, 3)
        
        for _ in range(num_items):
            ppe_type = random.choice(list(self.ppe_classes.keys()))
            
            # Random bounding box
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h//2)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(30, 80)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': random.uniform(0.5, 0.95),
                'class_name': ppe_type,
                'class_id': list(self.ppe_classes.keys()).index(ppe_type)
            }
            simulated_detections.append(detection)
        
        return simulated_detections
    
    def _match_ppe_to_people(self, people: List[Dict[str, Any]], 
                           ppe_detections: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Match PPE detections to people based on spatial proximity
        
        Args:
            people: List of tracked people
            ppe_detections: List of PPE detections
            
        Returns:
            Dictionary mapping track_id to list of PPE items
        """
        person_ppe = {}
        
        for person in people:
            track_id = person['track_id']
            person_bbox = person['bbox']
            person_center = self._get_bbox_center(person_bbox)
            
            # Find PPE items near this person
            nearby_ppe = []
            proximity_threshold = 100  # pixels
            
            for ppe_item in ppe_detections:
                ppe_center = self._get_bbox_center(ppe_item['bbox'])
                distance = self._calculate_distance(person_center, ppe_center)
                
                if distance <= proximity_threshold:
                    # Check if PPE overlaps with person
                    if self._calculate_overlap_ratio(person_bbox, ppe_item['bbox']) > 0.1:
                        nearby_ppe.append(ppe_item)
            
            person_ppe[track_id] = nearby_ppe
        
        return person_ppe
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def _calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection / min(area1, area2)
    
    def _check_ppe_compliance(self, track_id: int, ppe_items: List[Dict[str, Any]], 
                            timestamp: float) -> Dict[str, Any]:
        """
        Check PPE compliance for a person
        
        Args:
            track_id: Person track ID
            ppe_items: List of PPE items detected for this person
            timestamp: Current timestamp
            
        Returns:
            Compliance information
        """
        detected_ppe = set(item['class_name'] for item in ppe_items)
        required_ppe = set(self.required_ppe)
        missing_ppe = required_ppe - detected_ppe
        
        is_compliant = len(missing_ppe) == 0
        
        # Update compliance history
        if track_id not in self.person_ppe_status:
            self.person_ppe_status[track_id] = {
                'compliant': is_compliant,
                'missing_ppe': list(missing_ppe),
                'detected_ppe': list(detected_ppe),
                'last_update': timestamp,
                'violation_start': None if is_compliant else timestamp
            }
        else:
            status = self.person_ppe_status[track_id]
            previous_compliant = status['compliant']
            
            # Update status
            status.update({
                'compliant': is_compliant,
                'missing_ppe': list(missing_ppe),
                'detected_ppe': list(detected_ppe),
                'last_update': timestamp
            })
            
            # Track violation start time
            if not is_compliant and previous_compliant:
                status['violation_start'] = timestamp
            elif is_compliant:
                status['violation_start'] = None
        
        return {
            'track_id': track_id,
            'is_compliant': is_compliant,
            'missing_ppe': list(missing_ppe),
            'detected_ppe': list(detected_ppe),
            'ppe_items': ppe_items,
            'confidence': min([item['confidence'] for item in ppe_items]) if ppe_items else 0.0
        }
    
    def _check_zone_ppe_violations(self, tracked_people: List[Dict[str, Any]], 
                                 compliance_results: Dict[int, Dict[str, Any]], 
                                 timestamp: float) -> List[Dict[str, Any]]:
        """Check for PPE violations in restricted zones"""
        violations = []
        
        for zone in self.ppe_required_zones:
            people_in_zone = self._get_people_in_zone(tracked_people, zone)
            zone_id = zone.get('zone_id', 0)
            
            for person in people_in_zone:
                track_id = person['track_id']
                
                if track_id not in compliance_results:
                    continue
                
                compliance = compliance_results[track_id]
                
                if not compliance['is_compliant']:
                    # Check grace period
                    status = self.person_ppe_status.get(track_id, {})
                    violation_start = status.get('violation_start')
                    
                    if violation_start and (timestamp - violation_start) > self.grace_period:
                        violation = {
                            'zone_id': zone_id,
                            'zone_name': zone.get('name', f'Zone {zone_id}'),
                            'track_id': track_id,
                            'missing_ppe': compliance['missing_ppe'],
                            'detected_ppe': compliance['detected_ppe'],
                            'violation_duration': timestamp - violation_start,
                            'timestamp': timestamp,
                            'person_bbox': person['bbox']
                        }
                        violations.append(violation)
        
        return violations
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process frame for PPE detection
        
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
        
        # Detect PPE items (this would use the actual PPE model)
        # For now, we'll simulate PPE detection
        from model_manager import ModelManager  # Import here to avoid circular imports
        try:
            model_manager = ModelManager()
            ppe_detections = self._detect_ppe_items(frame, model_manager)
        except:
            ppe_detections = self._simulate_ppe_detection(frame)
        
        # Match PPE to people
        person_ppe_mapping = self._match_ppe_to_people(tracked_people, ppe_detections)
        
        # Check PPE compliance for each person
        compliance_results = {}
        for person in tracked_people:
            track_id = person['track_id']
            ppe_items = person_ppe_mapping.get(track_id, [])
            compliance = self._check_ppe_compliance(track_id, ppe_items, timestamp)
            compliance_results[track_id] = compliance
        
        # Check for zone violations
        ppe_violations = self._check_zone_ppe_violations(tracked_people, compliance_results, timestamp)
        
        # Create violation events
        for violation in ppe_violations:
            if not self._check_event_cooldown('ppe_violation', violation['zone_id']):
                event = self._create_event(
                    event_type='ppe_violation',
                    description=f"PPE violation: Missing {', '.join(violation['missing_ppe'])} in {violation['zone_name']}",
                    detection_data={
                        'zone_id': violation['zone_id'],
                        'zone_name': violation['zone_name'],
                        'track_id': violation['track_id'],
                        'missing_ppe': violation['missing_ppe'],
                        'detected_ppe': violation['detected_ppe'],
                        'violation_duration': violation['violation_duration'],
                        'person_bbox': violation['person_bbox'],
                        'confidence': 0.8
                    },
                    zone_id=violation['zone_id'],
                    severity='medium'
                )
                events.append(event)
                self._save_event(event)
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracked_people, ppe_detections, compliance_results, ppe_violations
        )
        
        # Update statistics
        self._update_statistics(len(tracked_people))
        
        return annotated_frame, events
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                               tracked_people: List[Dict[str, Any]],
                               ppe_detections: List[Dict[str, Any]],
                               compliance_results: Dict[int, Dict[str, Any]],
                               violations: List[Dict[str, Any]]) -> np.ndarray:
        """Create annotated frame with PPE visualization"""
        annotated_frame = frame.copy()
        
        # Draw PPE required zones
        for zone in self.ppe_required_zones:
            zone_id = zone.get('zone_id', 0)
            has_violations = any(v['zone_id'] == zone_id for v in violations)
            
            color = (0, 0, 255) if has_violations else (255, 255, 0)  # Red if violations, yellow otherwise
            alpha = 0.3 if has_violations else 0.2
            
            label = f"{zone.get('name', f'PPE Zone {zone_id}')} - Required: {', '.join(self.required_ppe)}"
            self._draw_zone(annotated_frame, zone, color, alpha, label)
        
        # Draw PPE detections
        for ppe_item in ppe_detections:
            bbox = ppe_item['bbox']
            ppe_class = ppe_item['class_name']
            confidence = ppe_item['confidence']
            
            color = self.ppe_colors.get(ppe_class, (255, 255, 255))
            
            # Draw PPE bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw PPE label
            label = f"{ppe_class} {confidence:.2f}"
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracked people with PPE status
        for person in tracked_people:
            track_id = person['track_id']
            bbox = person['bbox']
            
            # Get compliance status
            compliance = compliance_results.get(track_id, {})
            is_compliant = compliance.get('is_compliant', True)
            missing_ppe = compliance.get('missing_ppe', [])
            detected_ppe = compliance.get('detected_ppe', [])
            
            # Color based on compliance
            if is_compliant:
                color = self.ppe_colors['compliant']  # Green
                thickness = 2
            else:
                # Check if in grace period
                status = self.person_ppe_status.get(track_id, {})
                violation_start = status.get('violation_start')
                in_grace_period = (violation_start and 
                                 (time.time() - violation_start) <= self.grace_period)
                
                if in_grace_period:
                    color = self.ppe_colors['warning']  # Yellow
                    thickness = 2
                else:
                    color = self.ppe_colors['violation']  # Red
                    thickness = 3
            
            # Draw person bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, thickness)
            
            # Create status label
            if is_compliant:
                label = f"ID:{track_id} PPE OK"
                label_color = self.ppe_colors['compliant']
            else:
                label = f"ID:{track_id} Missing: {', '.join(missing_ppe)}"
                label_color = color
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(annotated_frame,
                         (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                         (int(bbox[0]) + label_size[0] + 10, int(bbox[1])),
                         label_color, -1)
            
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]) + 5, int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw detected PPE list
            if detected_ppe:
                ppe_text = f"Has: {', '.join(detected_ppe)}"
                cv2.putText(annotated_frame, ppe_text,
                           (int(bbox[0]), int(bbox[2]) + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw violation alerts
        alert_y = 60
        for violation in violations:
            alert_text = f"PPE VIOLATION: {violation['zone_name']} - ID:{violation['track_id']} - Missing: {', '.join(violation['missing_ppe'])}"
            cv2.putText(annotated_frame, alert_text, (10, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 30
        
        # Draw main info
        info_text = f"Camera {self.camera_id} - PPE Detection | People: {len(tracked_people)} | Violations: {len(violations)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw PPE requirements
        requirements_text = f"Required PPE: {', '.join(self.required_ppe)}"
        cv2.putText(annotated_frame, requirements_text, (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw compliance statistics
        total_people = len(tracked_people)
        compliant_people = sum(1 for c in compliance_results.values() if c.get('is_compliant', True))
        compliance_rate = (compliant_people / total_people * 100) if total_people > 0 else 100
        
        stats_text = f"Compliance Rate: {compliance_rate:.1f}% ({compliant_people}/{total_people})"
        cv2.putText(annotated_frame, stats_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame


# Export main class
__all__ = ['PPEDetector']
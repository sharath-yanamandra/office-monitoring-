#!/usr/bin/env python3
"""
kalman_tracker.py
Object Tracking using Kalman Filter

This module handles:
1. Multi-object tracking using Kalman filters
2. Track assignment and lifecycle management
3. Person tracking for video monitoring
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import logging

class KalmanBoxTracker:
    """Individual object tracker using Kalman filter"""
    
    count = 0  # Class variable for unique track IDs
    
    def __init__(self, bbox: List[float], track_id: int = None):
        """
        Initialize Kalman tracker for a bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID, auto-generated if None
        """
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # s (scale)
            [0, 0, 0, 1, 0, 0, 0],  # r (aspect ratio)
            [0, 0, 0, 0, 1, 0, 0],  # x velocity
            [0, 0, 0, 0, 0, 1, 0],  # y velocity
            [0, 0, 0, 0, 0, 0, 1]   # s velocity
        ])
        
        # Observation matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # Track properties
        self.time_since_update = 0
        self.id = track_id if track_id is not None else KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Additional properties for video monitoring
        self.last_detection_confidence = 0.0
        self.track_confidence = 0.0
        self.detection_class = 'person'
        self.first_detection_time = None
        self.last_update_time = None

    def _convert_bbox_to_z(self, bbox: List[float]) -> np.ndarray:
        """
        Convert bounding box to Kalman filter state representation
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            State vector [x, y, s, r] where:
            - x, y: center coordinates
            - s: scale (area)
            - r: aspect ratio (w/h)
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale is area
        r = w / float(h) if h != 0 else 1.0  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x: np.ndarray) -> List[float]:
        """
        Convert state vector back to bounding box
        
        Args:
            x: State vector [x, y, s, r, ...]
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 1.0
        x1 = x[0] - w / 2.0
        y1 = x[1] - h / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

    def update(self, bbox: List[float], confidence: float = 1.0, detection_time: float = None):
        """
        Update tracker with new detection
        
        Args:
            bbox: New bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            detection_time: Timestamp of detection
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update detection properties
        self.last_detection_confidence = confidence
        self.last_update_time = detection_time
        
        # Update track confidence (running average)
        alpha = 0.3  # Learning rate
        self.track_confidence = alpha * confidence + (1 - alpha) * self.track_confidence
        
        # Kalman filter update
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        """
        Predict next state and advance tracker
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Handle negative scale
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        # Kalman prediction
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # Store prediction in history
        bbox = self._convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        
        return bbox

    def get_state(self) -> List[float]:
        """
        Get current bounding box estimate
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._convert_x_to_bbox(self.kf.x)

    def get_center(self) -> Tuple[float, float]:
        """Get center coordinates of current estimate"""
        return float(self.kf.x[0]), float(self.kf.x[1])

    def get_velocity(self) -> Tuple[float, float]:
        """Get velocity estimate"""
        return float(self.kf.x[4]), float(self.kf.x[5])

    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (has enough hits)"""
        return self.hits >= min_hits

    def is_tentative(self, max_age: int = 1) -> bool:
        """Check if track is tentative (new, unconfirmed)"""
        return self.age <= max_age and self.hits < 3

    def is_deleted(self, max_age: int = 30) -> bool:
        """Check if track should be deleted"""
        return self.time_since_update > max_age


class MultiObjectTracker:
    """Multi-object tracker using Kalman filters"""
    
    def __init__(self, max_disappeared: int = 30, min_hits: int = 3):
        """
        Initialize multi-object tracker
        
        Args:
            max_disappeared: Maximum frames before deleting track
            min_hits: Minimum hits before confirming track
        """
        self.logger = logging.getLogger('tracker')
        
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        
        # Active trackers
        self.trackers: List[KalmanBoxTracker] = []
        
        # Tracking parameters
        self.iou_threshold = 0.3  # IoU threshold for matching
        self.distance_threshold = 50.0  # Distance threshold for matching
        
        # Statistics
        self.frame_count = 0
        self.total_tracks_created = 0
        
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_center_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate distance between box centers"""
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _associate_detections_to_trackers(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing trackers
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Tuple of (matches, unmatched_dets, unmatched_trks)
        """
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Get current tracker predictions
        tracker_boxes = [trk.get_state() for trk in self.trackers]
        detection_boxes = [det['bbox'] for det in detections]
        
        # Calculate IoU matrix
        if len(detection_boxes) > 0 and len(tracker_boxes) > 0:
            iou_matrix = np.zeros((len(detection_boxes), len(tracker_boxes)))
            
            for d, det_box in enumerate(detection_boxes):
                for t, trk_box in enumerate(tracker_boxes):
                    iou_matrix[d, t] = self._calculate_iou(det_box, trk_box)
        else:
            iou_matrix = np.empty((0, len(tracker_boxes)))
        
        # Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            # Use negative IoU for Hungarian algorithm (maximizes IoU)
            cost_matrix = -iou_matrix
            from scipy.optimize import linear_sum_assignment
            det_indices, trk_indices = linear_sum_assignment(cost_matrix)
            
            matches = []
            for d, t in zip(det_indices, trk_indices):
                if iou_matrix[d, t] >= self.iou_threshold:
                    matches.append((d, t))
            
            matches = np.array(matches)
        else:
            matches = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections and trackers
        unmatched_dets = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_dets.append(d)
        
        unmatched_trks = []
        for t in range(len(self.trackers)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trks.append(t)
        
        # Filter matches by IoU threshold
        if len(matches) > 0:
            good_matches = []
            for d, t in matches:
                if iou_matrix[d, t] >= self.iou_threshold:
                    good_matches.append((d, t))
            matches = good_matches
        
        return matches, unmatched_dets, unmatched_trks
    
    def update(self, detections: List[Dict[str, Any]], current_time: float = None) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections with 'bbox' and optional 'confidence'
            current_time: Current timestamp
            
        Returns:
            List of confirmed tracks with tracking information
        """
        self.frame_count += 1
        
        # Predict next state for all trackers
        for tracker in self.trackers:
            tracker.predict()
        
        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections)
        
        # Update matched trackers
        for det_idx, trk_idx in matches:
            detection = detections[det_idx]
            confidence = detection.get('confidence', 1.0)
            self.trackers[trk_idx].update(detection['bbox'], confidence, current_time)
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            tracker = KalmanBoxTracker(detection['bbox'])
            tracker.last_detection_confidence = detection.get('confidence', 1.0)
            tracker.detection_class = detection.get('class_name', 'person')
            tracker.first_detection_time = current_time
            tracker.last_update_time = current_time
            tracker.track_confidence = detection.get('confidence', 1.0)
            
            self.trackers.append(tracker)
            self.total_tracks_created += 1
        
        # Remove dead trackers
        active_trackers = []
        for tracker in self.trackers:
            if not tracker.is_deleted(self.max_disappeared):
                active_trackers.append(tracker)
        
        self.trackers = active_trackers
        
        # Return confirmed tracks
        confirmed_tracks = []
        for tracker in self.trackers:
            if tracker.is_confirmed(self.min_hits):
                bbox = tracker.get_state()
                center = tracker.get_center()
                velocity = tracker.get_velocity()
                
                track_info = {
                    'track_id': tracker.id,
                    'bbox': bbox,
                    'center': center,
                    'velocity': velocity,
                    'confidence': tracker.track_confidence,
                    'detection_confidence': tracker.last_detection_confidence,
                    'class_name': tracker.detection_class,
                    'hits': tracker.hits,
                    'age': tracker.age,
                    'time_since_update': tracker.time_since_update,
                    'first_detection_time': tracker.first_detection_time,
                    'last_update_time': tracker.last_update_time
                }
                confirmed_tracks.append(track_info)
        
        return confirmed_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get specific track by ID"""
        for tracker in self.trackers:
            if tracker.id == track_id and tracker.is_confirmed(self.min_hits):
                bbox = tracker.get_state()
                center = tracker.get_center()
                velocity = tracker.get_velocity()
                
                return {
                    'track_id': tracker.id,
                    'bbox': bbox,
                    'center': center,
                    'velocity': velocity,
                    'confidence': tracker.track_confidence,
                    'detection_confidence': tracker.last_detection_confidence,
                    'class_name': tracker.detection_class,
                    'hits': tracker.hits,
                    'age': tracker.age,
                    'time_since_update': tracker.time_since_update,
                    'first_detection_time': tracker.first_detection_time,
                    'last_update_time': tracker.last_update_time
                }
        return None
    
    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks (including unconfirmed)"""
        tracks = []
        for tracker in self.trackers:
            bbox = tracker.get_state()
            center = tracker.get_center()
            velocity = tracker.get_velocity()
            
            track_info = {
                'track_id': tracker.id,
                'bbox': bbox,
                'center': center,
                'velocity': velocity,
                'confidence': tracker.track_confidence,
                'detection_confidence': tracker.last_detection_confidence,
                'class_name': tracker.detection_class,
                'hits': tracker.hits,
                'age': tracker.age,
                'time_since_update': tracker.time_since_update,
                'first_detection_time': tracker.first_detection_time,
                'last_update_time': tracker.last_update_time,
                'is_confirmed': tracker.is_confirmed(self.min_hits),
                'is_tentative': tracker.is_tentative()
            }
            tracks.append(track_info)
        
        return tracks
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get tracker statistics"""
        confirmed_tracks = sum(1 for t in self.trackers if t.is_confirmed(self.min_hits))
        tentative_tracks = sum(1 for t in self.trackers if t.is_tentative())
        
        return {
            'frame_count': self.frame_count,
            'total_tracks_created': self.total_tracks_created,
            'active_trackers': len(self.trackers),
            'confirmed_tracks': confirmed_tracks,
            'tentative_tracks': tentative_tracks,
            'max_disappeared': self.max_disappeared,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold
        }
    
    def reset(self):
        """Reset tracker state"""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0  # Reset global counter
        self.logger.info("Tracker reset completed")


# Export main classes
__all__ = ['KalmanBoxTracker', 'MultiObjectTracker']
        
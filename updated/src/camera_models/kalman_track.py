#!/usr/bin/env python3
"""
Script 10: kalman_track.py
File Path: src/camera_models/kalman_track.py

Datacenter Monitoring System - Object Tracking Implementation

This module handles:
1. Kalman filter-based object tracking for people and equipment
2. Hungarian algorithm for data association
3. Track lifecycle management (creation, update, deletion)
4. Multi-object tracking with unique IDs
5. Enhanced tracking for datacenter-specific scenarios
"""

from scipy.optimize import linear_sum_assignment
import scipy.linalg
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any

# Set random seed for reproducibility
np.random.seed(123)

class DatacenterKalmanFilter:
    """
    Enhanced Kalman filter for datacenter object tracking
    
    State vector: [x, y, aspect_ratio, height, vx, vy, va, vh]
    - x, y: center coordinates
    - aspect_ratio: width/height ratio
    - height: bounding box height
    - vx, vy, va, vh: velocities
    """
    
    count = 0  # Global counter for unique track IDs
    
    def __init__(self, measurement: np.ndarray, object_class: str = 'person', debug: bool = False):
        """
        Initialize Kalman filter for object tracking
        
        Args:
            measurement: Initial detection [x, y, aspect_ratio, height, confidence]
            object_class: Type of object being tracked ('person', 'equipment', etc.)
            debug: Enable debug logging
        """
        self.object_class = object_class
        self.debug = debug
        
        # Tracking state
        self.age = 0  # Frames since creation
        self.hits = 0  # Successful updates
        self.time_since_update = 0  # Frames since last update
        self.measurement_association = 0  # Consecutive successful associations
        
        # Unique track ID
        self.id = DatacenterKalmanFilter.count
        DatacenterKalmanFilter.count += 1
        
        # Datacenter-specific tracking attributes
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.zone_history = []  # Track zone transitions
        self.stationary_time = 0  # Time spent stationary (for loitering detection)
        self.last_position = None
        self.movement_history = []  # Track movement for PPE compliance areas
        
        # State dimensions
        ndim, dt = 4, 1  # 4D measurement, 1 time unit
        
        if self.debug:
            print(f"KalmanFilter.__init__() received measurement with shape: {measurement.shape}")
        
        # Ensure measurement has correct format [x, y, aspect_ratio, height]
        if len(measurement) > 4:
            if self.debug:
                print(f"WARNING: Measurement has {len(measurement)} elements, using first 4")
            measurement = measurement[:4]
        elif len(measurement) < 4:
            if self.debug:
                print(f"WARNING: Measurement has only {len(measurement)} elements, padding with defaults")
            measurement = np.pad(measurement, (0, 4 - len(measurement)), 'constant', constant_values=1.0)
        
        # Motion model: constant velocity
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Observation model: we observe position and size
        self._transform_mat = np.eye(ndim, 2 * ndim)
        
        # Noise parameters - tuned for datacenter environments
        self._std_weight_position = 1. / 20  # Position uncertainty
        self._std_weight_velocity = 1. / 160  # Velocity uncertainty
        
        # Initialize state vector [position, velocity]
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        self.mean = np.r_[mean_pos, mean_vel]
        
        # Initialize covariance matrix
        std = [
            self._std_weight_position * measurement[3],  # x uncertainty
            self._std_weight_position * measurement[3],  # y uncertainty  
            1e-2,  # aspect ratio uncertainty
            self._std_weight_position * measurement[3],  # height uncertainty
            self._std_weight_velocity * measurement[3],   # vx uncertainty
            self._std_weight_velocity * measurement[3],   # vy uncertainty
            1e-5,  # aspect ratio velocity uncertainty
            self._std_weight_velocity * measurement[3]    # height velocity uncertainty
        ]
        
        self.covariance = np.diag(np.square(std))
        
        if self.debug:
            print(f"Initialized track {self.id} for {self.object_class}")
    
    def predict(self) -> np.ndarray:
        """
        Predict next state using motion model
        
        Returns:
            Predicted observation [x, y, aspect_ratio, height]
        """
        self.age += 1
        if self.time_since_update > 0:
            self.measurement_association = 0
        
        self.time_since_update += 1
        
        # Process noise - adaptive based on object class
        if self.object_class == 'person':
            # People move more unpredictably
            position_noise = self._std_weight_position * 1.2
            velocity_noise = self._std_weight_velocity * 1.5
        else:
            # Equipment/stationary objects
            position_noise = self._std_weight_position * 0.8
            velocity_noise = self._std_weight_velocity * 0.5
        
        std_pos = [
            position_noise * self.mean[3],  # x noise
            position_noise * self.mean[3],  # y noise
            1e-2,  # aspect ratio noise
            position_noise * self.mean[3]   # height noise
        ]
        
        std_vel = [
            velocity_noise * self.mean[3],  # vx noise
            velocity_noise * self.mean[3],  # vy noise
            1e-5,  # aspect ratio velocity noise
            velocity_noise * self.mean[3]   # height velocity noise
        ]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # Predict state: x(k+1) = F * x(k)
        self.mean = np.dot(self._motion_mat, self.mean)
        
        # Predict covariance: P(k+1) = F * P(k) * F^T + Q
        self.covariance = np.linalg.multi_dot((
            self._motion_mat, self.covariance, self._motion_mat.T
        )) + motion_cov
        
        return np.dot(self._transform_mat, self.mean)
    
    def project(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state to measurement space
        
        Returns:
            Tuple of (projected_mean, projected_covariance)
        """
        # Measurement noise - adaptive based on detection confidence
        std = [
            self._std_weight_position * self.mean[3],  # x measurement noise
            self._std_weight_position * self.mean[3],  # y measurement noise
            1e-1,  # aspect ratio measurement noise
            self._std_weight_position * self.mean[3]   # height measurement noise
        ]
        
        measurement_cov = np.diag(np.square(std))
        
        projected_mean = np.dot(self._transform_mat, self.mean)
        projected_cov = np.linalg.multi_dot((
            self._transform_mat, self.covariance, self._transform_mat.T
        )) + measurement_cov
        
        return projected_mean, projected_cov
    
    def update(self, measurement: np.ndarray) -> None:
        """
        Update filter with new measurement
        
        Args:
            measurement: New observation [x, y, aspect_ratio, height]
        """
        self.time_since_update = 0
        self.hits += 1
        self.measurement_association += 1
        self.last_seen_time = time.time()
        
        # Update movement tracking for datacenter scenarios
        current_pos = measurement[:2]
        if self.last_position is not None:
            movement = np.linalg.norm(current_pos - self.last_position)
            self.movement_history.append(movement)
            
            # Keep only recent movement history
            if len(self.movement_history) > 10:
                self.movement_history.pop(0)
            
            # Update stationary time for loitering detection
            if movement < 2.0:  # Less than 2 pixels movement
                self.stationary_time += 1
            else:
                self.stationary_time = 0
        
        self.last_position = current_pos.copy()
        
        # Kalman filter update
        projected_mean, projected_cov = self.project()
        
        # Compute Kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.covariance, self._transform_mat.T).T,
            check_finite=False
        ).T
        
        # Update state
        innovation = measurement - projected_mean
        self.mean = self.mean + np.dot(innovation, kalman_gain.T)
        
        # Update covariance
        self.covariance = self.covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
    
    def get_state(self) -> np.ndarray:
        """
        Get current state in observation format
        
        Returns:
            Current state [x, y, aspect_ratio, height]
        """
        try:
            if self._transform_mat.shape[1] != self.mean.shape[0]:
                if self.debug:
                    print(f"ERROR: Dimension mismatch in get_state: {self._transform_mat.shape[1]} != {self.mean.shape[0]}")
                
                # Fix mean vector if needed
                if self.mean.shape[0] > 8:
                    self.mean = self.mean[:8]
                elif self.mean.shape[0] < 8:
                    self.mean = np.pad(self.mean, (0, 8 - self.mean.shape[0]), 'constant')
            
            state = np.dot(self._transform_mat, self.mean)
            
            if self.debug:
                print(f"Track {self.id} state: {state}")
            
            return state
            
        except Exception as e:
            if self.debug:
                print(f"Error in get_state for track {self.id}: {str(e)}")
            # Return default state
            return np.array([0, 0, 1, 100])
    
    def get_tracking_info(self) -> Dict[str, Any]:
        """
        Get comprehensive tracking information for datacenter monitoring
        
        Returns:
            Dictionary with tracking details
        """
        current_time = time.time()
        total_time = current_time - self.first_seen_time
        
        return {
            'track_id': self.id,
            'object_class': self.object_class,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'total_tracking_time': total_time,
            'stationary_time': self.stationary_time,
            'zone_history': self.zone_history.copy(),
            'movement_pattern': np.mean(self.movement_history) if self.movement_history else 0,
            'is_stationary': self.stationary_time > 10,  # 10 frames stationary
            'last_position': self.last_position.copy() if self.last_position is not None else None
        }


def calculate_iou_distance(det_bbox: np.ndarray, trk_bbox: np.ndarray) -> float:
    """
    Calculate IoU-based distance between detection and track
    
    Args:
        det_bbox: Detection bounding box [x, y, aspect_ratio, height]
        trk_bbox: Track bounding box [x, y, aspect_ratio, height]
        
    Returns:
        IoU distance (lower is better match)
    """
    # Convert center-based format to corner format
    def convert_to_corners(bbox):
        x, y, aspect_ratio, height = bbox
        width = aspect_ratio * height
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        return np.array([x1, y1, x2, y2])
    
    det_corners = convert_to_corners(det_bbox)
    trk_corners = convert_to_corners(trk_bbox)
    
    # Calculate intersection
    xx1 = np.maximum(det_corners[0], trk_corners[0])
    yy1 = np.maximum(det_corners[1], trk_corners[1])
    xx2 = np.minimum(det_corners[2], trk_corners[2])
    yy2 = np.minimum(det_corners[3], trk_corners[3])
    
    intersection_width = np.maximum(0., xx2 - xx1)
    intersection_height = np.maximum(0., yy2 - yy1)
    intersection_area = intersection_width * intersection_height
    
    # Calculate union
    det_area = (det_corners[2] - det_corners[0]) * (det_corners[3] - det_corners[1])
    trk_area = (trk_corners[2] - trk_corners[0]) * (trk_corners[3] - trk_corners[1])
    union_area = det_area + trk_area - intersection_area
    
    # Calculate IoU
    if union_area <= 0:
        return 1.0  # Maximum distance for no overlap
    
    iou = intersection_area / union_area
    return 1.0 - iou  # Convert to distance (lower is better)


def associate_detections_to_tracks(detections: np.ndarray, tracks: np.ndarray, 
                                 iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to existing tracks using Hungarian algorithm
    
    Args:
        detections: Array of detections [N, 4] or [N, 5]
        tracks: Array of track predictions [M, 4]
        iou_threshold: IoU threshold for valid associations
        
    Returns:
        Tuple of (matched_indices, unmatched_detections, unmatched_tracks)
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(tracks))
    
    # Compute distance matrix
    distance_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    
    for d_idx, detection in enumerate(detections):
        for t_idx, track in enumerate(tracks):
            distance_matrix[d_idx, t_idx] = calculate_iou_distance(
                detection[:4], track[:4]
            )
    
    # Solve assignment problem using Hungarian algorithm
    if distance_matrix.size > 0:
        det_indices, trk_indices = linear_sum_assignment(distance_matrix)
        matched_indices = np.column_stack((det_indices, trk_indices))
    else:
        matched_indices = np.empty((0, 2), dtype=int)
    
    # Find unmatched detections
    unmatched_detections = []
    for d_idx in range(len(detections)):
        if d_idx not in matched_indices[:, 0]:
            unmatched_detections.append(d_idx)
    
    # Find unmatched tracks
    unmatched_tracks = []
    for t_idx in range(len(tracks)):
        if t_idx not in matched_indices[:, 1]:
            unmatched_tracks.append(t_idx)
    
    # Filter out matches with high distance (low IoU)
    valid_matches = []
    for match in matched_indices:
        d_idx, t_idx = match
        if distance_matrix[d_idx, t_idx] <= (1.0 - iou_threshold):
            valid_matches.append(match)
        else:
            unmatched_detections.append(d_idx)
            unmatched_tracks.append(t_idx)
    
    if len(valid_matches) == 0:
        matched_indices = np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.array(valid_matches)
    
    return matched_indices, np.array(unmatched_detections), np.array(unmatched_tracks)


class DatacenterTracker:
    """
    Enhanced SORT tracker for datacenter monitoring
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3, debug: bool = False):
        """
        Initialize datacenter tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for association
            debug: Enable debug logging
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.debug = debug
        
        self.trackers = []  # List of active trackers
        self.confirmed_track_ids = set()  # IDs of confirmed tracks
        self.total_tracks_created = 0
        
        if self.debug:
            print(f"Initialized DatacenterTracker with max_age={max_age}, min_hits={min_hits}")
    
    def update(self, detections: np.ndarray = None, object_classes: List[str] = None) -> Tuple[np.ndarray, int]:
        """
        Update tracker with new detections
        
        Args:
            detections: Array of detections [N, 4] or [N, 5] (x, y, aspect_ratio, height, [confidence])
            object_classes: List of object classes for each detection
            
        Returns:
            Tuple of (tracked_objects, total_unique_tracks)
        """
        if detections is None:
            detections = np.empty((0, 4))
        
        if object_classes is None:
            object_classes = ['person'] * len(detections)
        
        if self.debug:
            print(f"Tracker update: {len(detections)} detections, {len(self.trackers)} active tracks")
        
        # Get predictions from existing trackers
        predicted_tracks = []
        tracks_to_delete = []
        
        for i, tracker in enumerate(self.trackers):
            try:
                predicted_state = tracker.predict()
                if np.any(np.isnan(predicted_state)):
                    tracks_to_delete.append(i)
                    continue
                predicted_tracks.append(predicted_state)
            except Exception as e:
                if self.debug:
                    print(f"Error predicting track {i}: {str(e)}")
                tracks_to_delete.append(i)
        
        # Remove failed trackers
        for i in reversed(tracks_to_delete):
            self.trackers.pop(i)
        
        predicted_tracks = np.array(predicted_tracks) if predicted_tracks else np.empty((0, 4))
        
        # Associate detections to tracks
        if len(predicted_tracks) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
                detections, predicted_tracks, self.iou_threshold
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(detections))
            unmatched_trks = np.arange(len(predicted_tracks))
        
        # Update matched trackers
        for match in matched:
            det_idx, trk_idx = match
            try:
                detection = detections[det_idx][:4]  # Only use position/size
                self.trackers[trk_idx].update(detection)
            except Exception as e:
                if self.debug:
                    print(f"Error updating tracker {trk_idx}: {str(e)}")
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            try:
                detection = detections[det_idx][:4]
                object_class = object_classes[det_idx] if det_idx < len(object_classes) else 'person'
                new_tracker = DatacenterKalmanFilter(detection, object_class, self.debug)
                self.trackers.append(new_tracker)
                self.total_tracks_created += 1
                
                if self.debug:
                    print(f"Created new track {new_tracker.id} for {object_class}")
                    
            except Exception as e:
                if self.debug:
                    print(f"Error creating new tracker for detection {det_idx}: {str(e)}")
        
        # Prepare output
        tracked_objects = []
        tracks_to_remove = []
        
        for i, tracker in enumerate(self.trackers):
            try:
                state = tracker.get_state()
                
                # Add to confirmed tracks if it meets criteria
                if tracker.hits >= self.min_hits:
                    self.confirmed_track_ids.add(tracker.id)
                
                # Only output confirmed tracks
                if tracker.id in self.confirmed_track_ids:
                    # Format: [x, y, aspect_ratio, height, track_id]
                    tracked_objects.append(np.concatenate([state, [tracker.id]]))
                
                # Mark for removal if too old
                if tracker.time_since_update > self.max_age:
                    tracks_to_remove.append(i)
                    if tracker.id in self.confirmed_track_ids:
                        self.confirmed_track_ids.remove(tracker.id)
                    
                    if self.debug:
                        print(f"Removing track {tracker.id} (age: {tracker.time_since_update})")
                        
            except Exception as e:
                if self.debug:
                    print(f"Error processing tracker {i}: {str(e)}")
                tracks_to_remove.append(i)
        
        # Remove old trackers
        for i in reversed(tracks_to_remove):
            self.trackers.pop(i)
        
        # Convert to numpy array
        if tracked_objects:
            tracked_objects = np.array(tracked_objects)
        else:
            tracked_objects = np.empty((0, 5))
        
        return tracked_objects, len(self.confirmed_track_ids)
    
    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific track
        
        Args:
            track_id: ID of the track
            
        Returns:
            Track information dictionary or None if not found
        """
        for tracker in self.trackers:
            if tracker.id == track_id:
                return tracker.get_tracking_info()
        return None
    
    def get_all_tracks_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all active tracks
        
        Returns:
            List of track information dictionaries
        """
        return [tracker.get_tracking_info() for tracker in self.trackers]


# Backward compatibility alias (same interface as bank system)
class Sort(DatacenterTracker):
    """Backward compatibility alias for DatacenterTracker"""
    
    def __init__(self, max_age: int = 30, min_ma: int = 3, debug: bool = False):
        super().__init__(max_age=max_age, min_hits=min_ma, debug=debug)
        self.object_count = 0  # For compatibility
    
    def update(self, detected_bbox: np.ndarray = None):
        """Backward compatible update method"""
        if detected_bbox is None:
            detected_bbox = np.empty((0, 5))
        
        # Convert from [x1, y1, x2, y2, conf] to [x, y, aspect_ratio, height, conf]
        if len(detected_bbox) > 0 and detected_bbox.shape[1] >= 4:
            converted_detections = []
            for detection in detected_bbox:
                if len(detection) >= 4:
                    x1, y1, x2, y2 = detection[:4]
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 1.0
                    
                    converted = [x, y, aspect_ratio, height]
                    if len(detection) > 4:
                        converted.append(detection[4])  # confidence
                    
                    converted_detections.append(converted)
            
            detected_bbox = np.array(converted_detections)
        
        tracked_objects, self.object_count = super().update(detected_bbox)
        return tracked_objects, self.object_count
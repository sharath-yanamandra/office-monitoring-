#!/usr/bin/env python3
"""
model_manager.py
AI Model Management for Video Monitoring

This module handles:
1. Loading and managing YOLO models
2. GPU optimization and batch processing
3. Person detection and PPE detection
4. Model inference coordination
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from ultralytics import YOLO
import threading
import queue
import time

from config import Config

class ModelManager:
    """Manages AI models for video monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('model_manager')
        self.logger.info("Initializing ModelManager")
        
        # Model instances
        self.person_model = None
        self.ppe_model = None
        self.pose_model = None
        
        # GPU setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Batch processing
        self.batch_size = Config.BATCH_SIZE
        self.batch_timeout = Config.BATCH_TIMEOUT
        
        # Processing queues
        self.inference_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        
        # Model loading
        self._load_models()
        
    def _load_models(self):
        """Load YOLO models"""
        try:
            # Person detection model
            self.logger.info(f"Loading person detection model: {Config.DETECTION_MODEL_PATH}")
            self.person_model = YOLO(Config.DETECTION_MODEL_PATH)
            
            if self.device == 'cuda':
                self.person_model.to(self.device)
            
            # PPE detection model (optional)
            try:
                self.logger.info(f"Loading PPE detection model: {Config.PPE_DETECTION_MODEL_PATH}")
                self.ppe_model = YOLO(Config.PPE_DETECTION_MODEL_PATH)
                if self.device == 'cuda':
                    self.ppe_model.to(self.device)
            except Exception as e:
                self.logger.warning(f"PPE model not loaded: {e}")
                
            # Pose estimation model (optional)
            try:
                self.logger.info(f"Loading pose estimation model: {Config.POSE_ESTIMATION_MODEL_PATH}")
                self.pose_model = YOLO(Config.POSE_ESTIMATION_MODEL_PATH)
                if self.device == 'cuda':
                    self.pose_model.to(self.device)
            except Exception as e:
                self.logger.warning(f"Pose model not loaded: {e}")
                
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray, confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect persons in frame
        
        Args:
            frame: Input image frame
            confidence: Detection confidence threshold
            
        Returns:
            List of person detections
        """
        if self.person_model is None:
            return []
        
        if confidence is None:
            confidence = Config.PERSON_DETECTION_CONFIDENCE
        
        try:
            results = self.person_model(frame, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (class 0 in COCO)
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_name': 'person',
                                'class_id': 0
                            })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {e}")
            return []
    
    def detect_ppe(self, frame: np.ndarray, confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect PPE in frame
        
        Args:
            frame: Input image frame
            confidence: Detection confidence threshold
            
        Returns:
            List of PPE detections
        """
        if self.ppe_model is None:
            self.logger.warning("PPE model not available")
            return []
        
        if confidence is None:
            confidence = Config.PPE_CONFIDENCE_THRESHOLD
        
        try:
            results = self.ppe_model(frame, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Map class IDs to PPE names (adjust based on your model)
                        ppe_classes = {
                            0: 'hard_hat',
                            1: 'safety_vest',
                            2: 'safety_glasses',
                            3: 'safety_gloves'
                        }
                        
                        class_name = ppe_classes.get(cls_id, f'ppe_class_{cls_id}')
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_name': class_name,
                            'class_id': cls_id
                        })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in PPE detection: {e}")
            return []
    
    def detect_objects(self, frame: np.ndarray, confidence: float = None) -> List[Dict[str, Any]]:
        """
        General object detection
        
        Args:
            frame: Input image frame
            confidence: Detection confidence threshold
            
        Returns:
            List of object detections
        """
        if self.person_model is None:
            return []
        
        if confidence is None:
            confidence = Config.GENERAL_DETECTION_CONFIDENCE
        
        try:
            results = self.person_model(frame, conf=confidence, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # COCO class names (subset)
                        coco_classes = {
                            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                            5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog'
                        }
                        
                        class_name = coco_classes.get(cls_id, f'object_{cls_id}')
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_name': class_name,
                            'class_id': cls_id
                        })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []
    
    def batch_detect_persons(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Batch person detection for multiple frames
        
        Args:
            frames: List of image frames
            
        Returns:
            List of detection results for each frame
        """
        if not frames or self.person_model is None:
            return [[] for _ in frames]
        
        try:
            results = self.person_model(frames, conf=Config.PERSON_DETECTION_CONFIDENCE, verbose=False)
            
            all_detections = []
            for result in results:
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_name': 'person',
                                'class_id': 0
                            })
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            self.logger.error(f"Error in batch person detection: {e}")
            return [[] for _ in frames]
    
    def estimate_pose(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Estimate human poses in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of pose estimations
        """
        if self.pose_model is None:
            self.logger.warning("Pose model not available")
            return []
        
        try:
            results = self.pose_model(frame, verbose=False)
            
            poses = []
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy()
                    
                    for i, (kpts, confs) in enumerate(zip(keypoints, confidences)):
                        poses.append({
                            'person_id': i,
                            'keypoints': kpts.tolist(),
                            'confidences': confs.tolist()
                        })
            
            return poses
            
        except Exception as e:
            self.logger.error(f"Error in pose estimation: {e}")
            return []
    
    def start_batch_processing(self):
        """Start batch processing thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._batch_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Batch processing started")
    
    def stop_batch_processing(self):
        """Stop batch processing thread"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        self.logger.info("Batch processing stopped")
    
    def _batch_processing_loop(self):
        """Main batch processing loop"""
        batch_frames = []
        batch_metadata = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Try to get frame from queue
                try:
                    frame_data = self.inference_queue.get(timeout=0.1)
                    batch_frames.append(frame_data['frame'])
                    batch_metadata.append(frame_data['metadata'])
                except queue.Empty:
                    pass
                
                current_time = time.time()
                
                # Process batch if conditions are met
                should_process = (
                    len(batch_frames) >= self.batch_size or 
                    (len(batch_frames) > 0 and (current_time - last_batch_time) >= self.batch_timeout)
                )
                
                if should_process:
                    # Process batch
                    results = self.batch_detect_persons(batch_frames)
                    
                    # Put results back
                    for i, (result, metadata) in enumerate(zip(results, batch_metadata)):
                        self.result_queue.put({
                            'detections': result,
                            'metadata': metadata
                        })
                    
                    # Reset batch
                    batch_frames = []
                    batch_metadata = []
                    last_batch_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                batch_frames = []
                batch_metadata = []
    
    def submit_for_batch_processing(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Submit frame for batch processing"""
        try:
            frame_data = {
                'frame': frame,
                'metadata': metadata
            }
            self.inference_queue.put(frame_data, timeout=1.0)
        except queue.Full:
            self.logger.warning("Inference queue full, dropping frame")
    
    def get_batch_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get result from batch processing"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'device': self.device,
            'person_model_available': self.person_model is not None,
            'ppe_model_available': self.ppe_model is not None,
            'pose_model_available': self.pose_model is not None,
            'batch_size': self.batch_size,
            'is_batch_processing': self.is_running
        }


# Export main class
__all__ = ['ModelManager']
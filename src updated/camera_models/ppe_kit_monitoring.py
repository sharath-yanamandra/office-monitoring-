"""
Use Case: PPE Violation Detection
Detects people not wearing safety gear like hardhats, vests, and masks.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

from .camera_model_base import CameraModelBase
#from config import Config
from ..config import Config
from ..utils import draw_text_with_background
from .kalman_track import Sort # Assuming kalman_track.py is in the same folder

class PPEDetector(CameraModelBase):
    """
    Use Case: PPE Violation Detection
    Detects people not wearing safety gear like hardhats, vests, and masks.
    """

    def __init__(self, camera_id, zones=None, rules=None, settings=None, db=None, 
                 db_writer=None, frames_base_dir='frames', camera_manager=None):
        super().__init__(
            camera_id=camera_id,
            zones=zones,
            rules=rules,
            settings=settings,
            db=db,
            db_writer=db_writer,
            frames_base_dir=frames_base_dir,
            camera_manager=camera_manager
        )
        self.logger.info("Initializing PPEDetector")

        # Override the model from the base class with a specific PPE model
        self.model = YOLO("models/PPE_Detection.pt")
        self.logger.info("Loaded PPE specific YOLO model.")

        # Define PPE-relevant classes
        self.class_names = self.model.names
        self.violation_classes = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
        
        # We need to track people to associate violations with them.
        # We also track violations themselves to see them on screen.
        self.person_tracker = Sort(max_age=30, min_hits=3)
        self.violation_tracker = Sort(max_age=30, min_hits=3)

        self.stats.update({
            "ppe_violations": 0,
            "frames_processed": 0,
            "people_detected": 0,
            "events_detected": 0,
            "frames_saved": 0,
            "videos_saved": 0,
            'start_time': time.time(),
        })

        # For aggregator
        self.enable_individual_events = False
        self.current_people_count = 0
        self.current_violation_count = 0

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled
        self.logger.info(f"Individual camera events {'enabled' if enabled else 'disabled'}")

    def get_current_people_count(self):
        """Get current total people count for aggregator"""
        return self.current_people_count

    def get_current_violation_count(self):
        """Get current PPE violation count for aggregator"""
        return self.current_violation_count

    def detect_people(self, frame, detection_result):
        """
        This method is the standardized entry point. For PPE, the detection_result
        from the base model is ignored, and we run our specific PPE model.
        """
        # We ignore the upstream detection_result and run our own model
        annotated_frame, people_count, violation_count, violations_by_class = self.detect_ppe_violations(frame)
        
        return annotated_frame, people_count, violation_count, violations_by_class

    def detect_ppe_violations(self, frame):
        """
        Run PPE detection and tracking on a frame.
        Returns the annotated frame, people count, violation count, and violation details.
        """
        person_detections_for_tracker = []
        violation_detections_for_tracker = []
        all_detections = []

        try:
            results = self.model(frame, stream=False, verbose=False) # stream=False for single image
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5: # Confidence threshold
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = self.class_names[cls_id]
                    
                    detection = {'bbox': [x1, y1, x2, y2], 'label': label, 'conf': conf}
                    all_detections.append(detection)

                    # Prepare data for trackers
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w / 2, y1 + h / 2
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if label == 'Person':
                        person_detections_for_tracker.append([cx, cy, aspect_ratio, h, conf])
                    elif label in self.violation_classes:
                        violation_detections_for_tracker.append([cx, cy, aspect_ratio, h, conf])

        except Exception as e:
            self.logger.error(f"Error during PPE detection: {e}", exc_info=True)

        # Update trackers
        tracked_people = self.person_tracker.update(np.array(person_detections_for_tracker) if person_detections_for_tracker else np.empty((0, 5)))
        tracked_violations = self.violation_tracker.update(np.array(violation_detections_for_tracker) if violation_detections_for_tracker else np.empty((0, 5)))

        annotated_frame = frame.copy()
        
        people_count = len(tracked_people)
        violation_count = len(tracked_violations)
        self.current_people_count = people_count
        self.current_violation_count = violation_count

        violations_by_class = {v_class: [] for v_class in self.violation_classes}

        # Draw tracked people
        for person in tracked_people:
            x1, y1, x2, y2, track_id = map(int, person)
            label = f"Person ID:{track_id}"
            color = (0, 255, 0) # Green for people
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            draw_text_with_background(annotated_frame, label, (x1, y1 - 10), color)

        # Draw tracked violations and populate violation data
        for i, violation in enumerate(tracked_violations):
            x1, y1, x2, y2, track_id = map(int, violation)
            # Find original label for this bbox - this is a simplification
            # A more robust way would be to associate violations to people
            label = "Violation"
            if i < len(all_detections):
                # This is an approximation, matching by index
                det = all_detections[i]
                if det['label'] in self.violation_classes:
                    label = det['label']
                    violations_by_class[label].append({'bbox': [x1, y1, x2, y2], 'track_id': track_id})

            color = (0, 0, 255) # Red for violations
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            draw_text_with_background(annotated_frame, f"{label} ID:{track_id}", (x1, y1 - 10), color)

        if self.enable_individual_events and violation_count > 0:
            self._handle_individual_camera_events(violations_by_class, annotated_frame)

        return annotated_frame, people_count, violation_count, violations_by_class

    def _handle_individual_camera_events(self, violations_by_class, annotated_frame):
        """Handle events for PPE violations."""
        for violation_type, detections in violations_by_class.items():
            if not detections:
                continue

            self.logger.info(f"Event: {len(detections)} '{violation_type}' violations detected.")
            
            for detection in detections:
                track_id = detection.get('track_id')
                if track_id and self._should_record_event(violation_type, 'ppe_zone', track_id):
                    ist_timestamp = time.time()
                    event_id = self._save_event_media(violation_type, [detection], annotated_frame, ist_timestamp, 'ppe_zone')
                    self.logger.info(f"Saved PPE violation event ID: {event_id}")
                    self.stats['ppe_violations'] += 1
                    self.stats['events_detected'] += 1

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Process a frame for PPE violations.
        Note: detection_result from base class is ignored.
        """
        self.logger.debug(f"Processing frame at timestamp {timestamp}")
        
        # The main logic is in detect_ppe_violations, which is called by detect_people
        annotated_frame, people_count, violation_count, violations_by_class = self.detect_people(frame, None)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected'] = people_count
        self.stats['ppe_violations'] = violation_count # Keep track of current frame violations
        
        return annotated_frame, violations_by_class

"""
    def process_video(self, input_path, output_path=None, skip_frames=2):
        
        # Testing wrapper to run detection on video.
        
        self.logger.info(f"Starting video processing for {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {input_path}")
            return

        writer = None
        frame_id = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip_frames == 0:
                # We call _process_frame_impl directly for testing
                processed, _ = self._process_frame_impl(frame, time.time(), None)

                if output_path:
                    if writer is None:
                        h, w = processed.shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                    writer.write(processed)

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        self.logger.info(f"Completed processing {frame_id} frames from {input_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PPE Detection")
    parser.add_argument('--input', type=str, required=True, help='Path to video file or 0 for webcam')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--camera', type=str, default="ppe_cam", help='Camera ID')
    parser.add_argument('--skip-frames', type=int, default=2, help='Process every Nth frame')
    args = parser.parse_args()

    monitor = PPEDetector(camera_id=args.camera)
    monitor.set_individual_events_enabled(True)
    monitor.process_video(args.input, args.output, skip_frames=args.skip_frames)

    """

"""
Use Case: Tailgating Detection
- Detects when a person follows another through an access point without valid authorization.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

from camera_models.camera_model_base import CameraModelBase
from config import Config
from utils import draw_text_with_background, draw_zone
from camera_models.kalman_track import Sort

class TailgatingZoneMonitor(CameraModelBase):
    """
    Use Case: Tailgating Detection
    - Monitors a zone to detect if people follow each other too closely.
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
        self.logger.info("Initializing TailgatingZoneMonitor")

        # --- Model and Tracker Initialization ---
        # NOTE: The original script used 'best.pt'. If this is a person detector,
        # the base model might suffice. If it's a special model, specify its path.
        # We will use the base class model for now for consistency.
        # self.model = YOLO('best.pt') 
        #self.tracker = Sort(max_age=50, min_hits=4)
        self.tracker = Sort(max_age=50, min_ma=4)



        # --- Rules and Configuration ---
        self.tailgating_time_limit = rules.get('tailgating_time_limit', 2.0) if rules else 2.0
        self.detection_confidence_threshold = rules.get('conf_threshold', 0.4) if rules else 0.4

        # --- Zone Configuration ---
        self.entry_zone = self._get_entry_zone()
        if not self.entry_zone:
            self.logger.warning("No entry zone defined for tailgating monitor. Using a default.")
            # Define a default zone if none is provided in the config
            self.entry_zone = {'name': 'default_entry', 'coordinates': [[100, 100], [700, 100], [700, 600], [100, 600]]}
        
        self.zone_polygon = np.array(self.entry_zone['coordinates'], np.int32)
        self.zone_colors = {"entry": (255, 0, 0)} # Blue for entry zone

        # --- State Tracking ---
        self.object_states = {}  # {id: {"in_zone": bool}}
        self.entry_log = deque(maxlen=10) # Stores timestamps of entries
        self.tailgating_ids = set()

        # --- Stats and Aggregator ---
        self.stats.update({
            "tailgating_violations": 0,
            "frames_processed": 0,
            "people_detected": 0,
            "events_detected": 0,
            "frames_saved": 0,
            "videos_saved": 0,
            'start_time': time.time(),
        })
        self.enable_individual_events = False
        self.current_people_count = 0
        self.current_tailgating_count = 0

    def _get_entry_zone(self):
        """Extracts the first 'entry' zone from the configuration."""
        try:
            if self.zones and 'entry' in self.zones and self.zones['entry']:
                return self.zones['entry'][0]
        except Exception as e:
            self.logger.error(f"Error extracting entry zone: {e}", exc_info=True)
        return None

    def set_individual_events_enabled(self, enabled: bool):
        self.enable_individual_events = enabled

    def get_current_people_count(self):
        return self.current_people_count

    def get_current_tailgating_count(self):
        return self.current_tailgating_count

    def _draw_zones(self, frame):
        zone_info = {'coordinates': self.zone_polygon, 'name': self.entry_zone.get('name', 'Entry Zone')}
        draw_zone(frame, zone_info, self.zone_colors["entry"], alpha=0.2, label=zone_info['name'])
        return frame

    def detect_people(self, frame, detection_result):
        """
        Process a frame for tailgating detection.
        """
        detections_for_tracker = []
        if detection_result:
            for box in detection_result.boxes:
                class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                class_name = detection_result.names[class_id] if class_id in detection_result.names else "unknown"
                
                if class_name == 'person' and float(box.conf[0]) > self.detection_confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w / 2, y1 + h / 2
                    aspect_ratio = w / h if h > 0 else 0
                    detections_for_tracker.append([cx, cy, aspect_ratio, h, float(box.conf[0])])

        tracked_objects = self.tracker.update(np.array(detections_for_tracker) if detections_for_tracker else np.empty((0, 5)))
        
        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        current_tracked_ids = set()
        tailgating_events_this_frame = []

        for obj in tracked_objects:
            cx, cy, a, h, obj_id = obj
            obj_id = int(obj_id)
            current_tracked_ids.add(obj_id)

            h_bbox, w_bbox = int(h), int(a * h)
            x1, y1 = int(cx - w_bbox / 2), int(cy - h_bbox / 2)
            x2, y2 = x1 + w_bbox, y1 + h_bbox
            bottom_center_point = (int(cx), y2)

            was_in_zone = self.object_states.get(obj_id, {}).get("in_zone", False)
            is_in = cv2.pointPolygonTest(self.zone_polygon, bottom_center_point, False) >= 0
            
            if is_in and not was_in_zone:
                current_time = time.time()
                self.logger.info(f"Person {obj_id} entered the zone.")
                if self.entry_log and current_time - self.entry_log[-1] < self.tailgating_time_limit:
                    self.logger.warning(f"Alert: Tailgating detected! Person {obj_id} entered too soon.")
                    self.tailgating_ids.add(obj_id)
                    event_data = {'track_id': obj_id, 'timestamp': current_time, 'bbox': [x1, y1, x2, y2]}
                    tailgating_events_this_frame.append(event_data)
                self.entry_log.append(current_time)
            
            self.object_states[obj_id] = {"in_zone": is_in}

            color = (0, 0, 255) if obj_id in self.tailgating_ids else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            draw_text_with_background(annotated_frame, f"ID: {obj_id}", (x1, y1 - 10), color)
            cv2.circle(annotated_frame, bottom_center_point, 5, color, -1)

        # --- Cleanup and update state ---
        for track_id in list(self.object_states.keys()):
            if track_id not in current_tracked_ids:
                del self.object_states[track_id]
        self.tailgating_ids.intersection_update(current_tracked_ids)

        self.current_people_count = len(tracked_objects)
        self.current_tailgating_count = len(self.tailgating_ids)

        if self.enable_individual_events and tailgating_events_this_frame:
            self._handle_individual_camera_events(tailgating_events_this_frame, annotated_frame)

        return annotated_frame, self.current_people_count, self.current_tailgating_count, tailgating_events_this_frame

    def _handle_individual_camera_events(self, tailgating_events, annotated_frame):
        self.stats['tailgating_violations'] += len(tailgating_events)
        for event in tailgating_events:
            track_id = event['track_id']
            if self._should_record_event('tailgating', self.entry_zone['name'], track_id):
                self.logger.info(f"Recording tailgating event for track ID {track_id}")
                event_id = self._save_event_media('tailgating', [event], annotated_frame, event['timestamp'], self.entry_zone['name'])
                self.logger.info(f"Saved tailgating event with ID: {event_id}")
                self.stats['events_detected'] += 1

    def _process_frame_impl(self, frame, timestamp, detection_result):
        annotated_frame, _, _, tailgating_events = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected'] = self.current_people_count
        
        return annotated_frame, {'tailgating_events': tailgating_events}

"""
    def process_video(self, input_path, output_path=None, skip_frames=1):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {input_path}")
            return

        writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_id % skip_frames == 0:
                # Use the base class detect method
                result = self.detect(frame)
                processed_frame, _, _, _ = self.detect_people(frame, result)

                if output_path:
                    if writer is None:
                        h, w = processed_frame.shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                    writer.write(processed_frame)

            frame_id += 1
        
        cap.release()
        if writer: writer.release()
        self.logger.info(f"Video processing complete for {input_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tailgating Detection Testing")
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--camera", type=str, default="tailgating_cam", help="Camera ID")
    args = parser.parse_args()

    # For testing, we can define a dummy zone and rules
    test_zones = {'entry': [{'name': 'front_door', 'coordinates': [[100, 100], [700, 100], [700, 600], [100, 600]]}]}
    test_rules = {'tailgating_time_limit': 2.0, 'conf_threshold': 0.4}

    monitor = TailgatingZoneMonitor(camera_id=args.camera, zones=test_zones, rules=test_rules)
    monitor.set_individual_events_enabled(True)
    
    monitor.process_video(args.input, args.output)

    """
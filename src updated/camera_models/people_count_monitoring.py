"""
Use Case: People Counting
- Count the number of people present in defined zones
"""

import cv2
import numpy as np
import time

from .camera_model_base import CameraModelBase
from ..config import Config
from ..utils import draw_text_with_background, draw_zone

class PeopleCountingMonitor(CameraModelBase):
    """
    Use Case: People Counting
    - Count the number of people present in defined zones
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
        self.logger.info("Initializing PeopleCountingMonitor with detailed logging")

        self.zone_colors = {"counting": (0, 255, 0)}  # Green for counting zones
        
        # Threshold might be used for event triggering if count exceeds a limit
        self.people_count_threshold = rules.get('people_count_threshold', 9999) if rules else 9999

        self.counting_zones = self._get_zones_coordinates()

        self.logger.info(f"Loaded {len(self.counting_zones)} counting zones")
        for zone in self.counting_zones:
            self.logger.info(f"Zone: {zone.get('name', 'Unnamed')} with {len(zone.get('coordinates', []))} coordinates")

        self.stats.update({
            "frames_processed": 0,
            "people_detected": 0, # This will be total people in frame
            "people_in_zone": 0,  # This will be people inside counting zones
            "events_detected": 0,
            "frames_saved": 0,
            'start_time': time.time()
        })

        # For aggregator
        self.enable_individual_events = False
        self.current_people_count = 0 # Total people detected in the frame
        self.current_people_in_zone_count = 0 # People inside a counting zone

    def _get_zones_coordinates(self):
        """Extract counting zones from configuration"""
        counting_zones = []
        try:
            for zone_type in self.zones:
                if zone_type == 'counting':
                    for zone in self.zones[zone_type]:
                        if "coordinates" in zone:
                            counting_zones.append(zone)
        except Exception as e:
            self.logger.error(f"Error extracting counting zones: {e}", exc_info=True)
        return counting_zones

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled
        self.logger.info(f"Individual camera events {'enabled' if enabled else 'disabled'}")

    def get_current_people_count(self):
        """Get current total people count for aggregator"""
        return self.current_people_count
        
    def get_current_people_in_zone_count(self):
        """Get current people count inside zones for aggregator"""
        return self.current_people_in_zone_count

    def _draw_zones(self, frame):
        """Draw counting zones on the frame"""
        try:
            for zone in self.counting_zones:
                if 'coordinates' in zone:
                    name = zone.get("name", "counting")
                    coordinates = np.array(zone['coordinates'], dtype=np.int32)
                    zone_info = {
                        'coordinates': coordinates,
                        'name': name
                    }
                    draw_zone(frame, zone_info, self.zone_colors["counting"], alpha=0.3, label=name)
        except Exception as e:
            self.logger.error(f"Error drawing counting zones: {e}", exc_info=True)
        return frame

    def detect_people(self, frame, detection_result):
        """
        Process a frame for people counting using a tracker for stability.
        Args:
            frame: The input video frame
            detection_result: Detection output from model
        Returns:
            Annotated frame, total people count, count of people in zones, and people by zone.
        """
        people_detections = []
        tracking_data = []

        if detection_result:
            try:
                for box in detection_result.boxes:
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                    class_name = detection_result.names[class_id] if class_id in detection_result.names else "unknown"

                    if class_name == 'person':
                        confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        height = y2 - y1
                        aspect_ratio = (x2 - x1) / height if height > 0 else 1.0

                        person_detection = {
                            'bbox': [x1, y1, x2, y2],
                            'center': (center_x, center_y),
                            'confidence': confidence
                        }
                        people_detections.append(person_detection)
                        tracking_data.append([center_x, center_y, aspect_ratio, int(height), confidence])
            except Exception as e:
                self.logger.error(f"Error parsing detection result: {e}", exc_info=True)

        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        tracked_objects = self.update_tracker(np.array(tracking_data) if tracking_data else np.empty((0, 5)))
        
        total_people = len(tracked_objects)
        self.current_people_count = total_people

        people_in_zone_count = 0
        people_by_zone = {zone.get('name', 'Unnamed'): [] for zone in self.counting_zones}

        for detection in tracked_objects:
            track_id = detection.get('track_id')
            if not track_id:
                continue
            
            # The bbox from the tracker is what we use
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            is_in_any_zone = False
            for zone in self.counting_zones:
                if self.is_in_zone((center_x, center_y), zone):
                    is_in_any_zone = True
                    zone_name = zone.get("name", "counting")
                    if zone_name in people_by_zone:
                        people_by_zone[zone_name].append(detection)
                    
                    # Draw on frame
                    label = f"ID:{track_id}"
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    draw_text_with_background(annotated_frame, label, (int(x1), int(y1) - 10), (0, 255, 0))
                    break # Count person only once even if in multiple zones

            if is_in_any_zone:
                people_in_zone_count += 1

        self.current_people_in_zone_count = people_in_zone_count
        self.stats['people_in_zone'] = people_in_zone_count

        if self.enable_individual_events:
            self._handle_individual_camera_events(people_by_zone, annotated_frame)

        # Add overall count text to the frame
        count_text = f"People in Zones: {people_in_zone_count}"
        draw_text_with_background(annotated_frame, count_text, (10, 30), font_color=(255, 255, 255))

        return annotated_frame, total_people, people_in_zone_count, people_by_zone

    def _handle_individual_camera_events(self, people_by_zone, annotated_frame):
        """Handle events for people counting, e.g., if a threshold is exceeded."""
        for zone_name, detections in people_by_zone.items():
            current_count = len(detections)
            if current_count > self.people_count_threshold:
                # This is a simple event trigger; could be made more sophisticated
                # (e.g., only trigger once until count drops below threshold)
                self.logger.warning(f"Event: People count in '{zone_name}' ({current_count}) exceeds threshold ({self.people_count_threshold})")
                
                # Use a representative track_id for event cooldown, e.g., the first one
                track_id_for_cooldown = detections[0].get('track_id') if detections else None

                if track_id_for_cooldown and self._should_record_event('people_count_exceeded', zone_name, track_id_for_cooldown):
                    ist_timestamp = time.time()
                    event_id = self._save_event_media('people_count_exceeded', detections, annotated_frame, ist_timestamp, zone_name)
                    self.logger.info(f"Saved people count event with ID: {event_id}")
                    self.stats['events_detected'] += 1

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Process a frame for people counting.
        """
        self.logger.debug(f"Processing frame at timestamp {timestamp}")
        
        annotated_frame, people_count, _, people_by_zone = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected'] = people_count
        
        return annotated_frame, people_by_zone

"""
    def process_video(self, input_path, output_path=None, skip_frames=2):
        
        # Testing wrapper to run detection on video.
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {input_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip_frames == 0:
                result = self.detect(frame)
                processed_frame, _, _, _ = self.detect_people(frame, result)

                if output_path:
                    if writer is None:
                        h, w = processed_frame.shape[:2]
                        writer = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                    writer.write(processed_frame)

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        self.logger.info(f"Video processing complete. {frame_id} frames processed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="People Counting Monitor")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--camera", type=str, default="people_counter_cam", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")

    args = parser.parse_args()

    monitor = PeopleCountingMonitor(camera_id=args.camera, db=None, db_writer=None)
    monitor.set_individual_events_enabled(True)

    if args.input:
        monitor.process_video(args.input, args.output, skip_frames=args.skip_frames)
    else:
        print("No input video specified. Use --input to specify a video file.")

        """
"""
Use Case: Intrusion Detection
- Detect people entering restricted/intrusion zones
"""

import cv2
import numpy as np
import time

from .camera_model_base import CameraModelBase
from config import Config
from utils import draw_text_with_background, draw_zone

class IntrusionZoneMonitor(CameraModelBase):
    """
    Use Case: Intrusion Detection
    - Detect people entering restricted/intrusion zones
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
        self.logger.info("Initializing IntrusionZoneMonitor with detailed logging")

        self.zone_colors = {"intrusion": (0, 0, 255)}  # Red for restricted zones
        self.intrusion_threshold = Config.INTRUSION_THRESHOLD
        self.recent_events = {}
        self.tracking_threshold = Config.TRACKING_THRESHOLD
        self.event_cooldown = Config.EVENT_COOLDOWN
        self.auto_recording_enabled = Config.AUTO_RECORDING_ENABLED

        self.intrusion_zones = self._get_zones_coordinates()

        self.logger.info(f"Loaded {len(self.intrusion_zones)} intrusion zones")
        for zone in self.intrusion_zones:
            self.logger.info(f"Zone: {zone.get('name', 'Unnamed')} with {len(zone.get('coordinates', []))} coordinates")

        self.stats.update({
            "intrusion_violations": 0,
            "frames_processed": 0,
            "events_detected": 0,
            "people_detected": 0,
            "frames_saved": 0,
            "videos_saved": 0,
            'object_count': 0,
            'last_processed_time': None,
            'start_time': time.time()
        })

        self.start_time = time.time()

        # For aggregator
        self.enable_individual_events = False
        self.current_people_count = 0
        self.current_intrusion_count = 0

    def _get_zones_coordinates(self):
        """Extract intrusion zones from configuration"""
        intrusion_zones = []
        try:
            # Assuming zones are structured like: {"intrusion": [...]}
            for zone_type in self.zones:
                if zone_type == 'intrusion':
                    for zone in self.zones[zone_type]:
                        if "coordinates" in zone:
                            intrusion_zones.append(zone)
        except Exception as e:
            self.logger.error(f"Error extracting intrusion zones: {e}", exc_info=True)
        return intrusion_zones

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled
        self.logger.info(f"Individual camera events {'enabled' if enabled else 'disabled'}")

    def get_current_people_count(self):
        """Get current total people count for aggregator"""
        return self.current_people_count

    def get_current_intrusion_count(self):
        """Get current intrusion people count for aggregator"""
        return self.current_intrusion_count

    def _draw_zones(self, frame):
        """Draw intrusion (restricted) zones on the frame"""
        try:
            for zone in self.intrusion_zones:
                if 'coordinates' in zone:
                    name = zone.get("name", "intrusion")
                    coordinates = np.array(zone['coordinates'], dtype=np.int32)
                    zone_info = {
                        'coordinates': coordinates,
                        'name': name
                    }
                    draw_zone(frame, zone_info, self.zone_colors["intrusion"], alpha=0.3, label=name)
        except Exception as e:
            self.logger.error(f"Error drawing intrusion zones: {e}", exc_info=True)
        return frame

    def detect_people(self, frame, detection_result):
        """
        Process a frame for intrusion detection
        Args:
            frame: The input video frame
            detection_result: Detection output from model
        Returns:
            Annotated frame, total people count, intrusion count, and people by zone
        """
        people_detections = []
        tracking_data = []

        if detection_result:
            try:
                self.logger.debug(f"Processing detection with {len(detection_result.boxes)} boxes")
                for i, box in enumerate(detection_result.boxes):
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                    class_name = detection_result.names[class_id] if class_id in detection_result.names else "unknown"

                    if class_name == 'person':
                        confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                        x1, y1, x2, y2 = map(float, box.xyxy[0])

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 1.0

                        in_intrusion = False
                        intrusion_name = None

                        for zone in self.intrusion_zones:
                            if self.is_in_zone((center_x, center_y), zone):
                                in_intrusion = True
                                intrusion_name = zone.get("name", "Restricted")
                                break

                        person_detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': (center_x, center_y),
                            'in_intrusion': in_intrusion,
                            'intrusion_name': intrusion_name
                        }

                        people_detections.append(person_detection)
                        tracking_data.append([center_x, center_y, aspect_ratio, int(height), confidence])

                        self.logger.debug(f"Person @({center_x},{center_y}) Conf:{confidence:.2f}, Intrusion:{in_intrusion}, Zone:{intrusion_name}")
            except Exception as e:
                self.logger.error(f"Error parsing detection result: {e}", exc_info=True)

        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        if tracking_data:
            tracking_data = np.array(tracking_data)
            self.logger.debug(f"Sending {len(tracking_data)} detections to tracker")
        else:
            self.logger.debug("No detections to track")

        tracked_objects = self.update_tracker(tracking_data)
        self.logger.debug(f"Tracker returned {len(tracked_objects)} tracked objects")

        total_people = len(tracked_objects)
        self.current_people_count = total_people
        
        intrusion_count = 0
        people_by_zone = {zone.get('name', 'Unnamed'): [] for zone in self.intrusion_zones}
        all_intruding_detections = []

        for i, detection in enumerate(tracked_objects):
            track_id = detection.get('track_id')
            if not track_id:
                continue

            # Associate tracker output with original detection
            if i < len(people_detections):
                person_detection = people_detections[i]
                detection.update(person_detection) # Merge data

                if track_id in self.tracked_objects:
                    in_intrusion = person_detection.get('in_intrusion', False)
                    intrusion_name = person_detection.get('intrusion_name', 'unknown')

                    self.tracked_objects[track_id]['in_intrusion'] = in_intrusion
                    self.tracked_objects[track_id]['intrusion_name'] = intrusion_name

                    if in_intrusion:
                        intrusion_count += 1
                        all_intruding_detections.append(detection)
                        if intrusion_name in people_by_zone:
                            people_by_zone[intrusion_name].append(detection)

                        bbox = detection['bbox']
                        color = (0, 0, 255)
                        label = f"ID:{track_id} ({intrusion_name})"
                        cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        draw_text_with_background(annotated_frame, label, (int(bbox[0]), int(bbox[1]) - 10), color)

        self.current_intrusion_count = intrusion_count

        if self.enable_individual_events:
            self._handle_individual_camera_events(people_by_zone, annotated_frame)

        return annotated_frame, total_people, intrusion_count, people_by_zone

    def _handle_individual_camera_events(self, people_by_zone, annotated_frame):
        """Handle events at individual camera level."""
        for zone_name, detections in people_by_zone.items():
            if len(detections) >= self.intrusion_threshold:
                self.logger.info(f"Individual camera event: {len(detections)} people in intrusion zone {zone_name}")
                track_ids = [p.get('track_id') for p in detections if 'track_id' in p]

                if track_ids and self._should_record_event('intrusion', zone_name, track_ids[0]):
                    self.logger.info(f"Recording individual camera event with {len(track_ids)} people in {zone_name}")
                    ist_timestamp = time.time()
                    event_id = self._save_event_media('intrusion', detections, annotated_frame, ist_timestamp, zone_name)
                    self.logger.info(f"Saved individual camera event with ID: {event_id}")
                    
                    if Config.SMS_ENABLED:
                        sms_sent = self._send_sms_alert(len(track_ids), ist_timestamp, zone_name)
                        if sms_sent:
                            self.logger.info("SMS alert sent successfully")
                    
                    self.stats['intrusion_violations'] += 1

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Process a frame to detect intrusion events.
        """
        self.logger.debug(f"Processing frame at timestamp {timestamp}")
        
        annotated_frame, people_count, intrusion_count, people_by_zone = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected'] = people_count
        
        self.logger.debug(f"Processed frame: {people_count} people detected, {intrusion_count} in intrusion zones.")
        
        return annotated_frame, people_by_zone

"""
    def process_video(self, input_path, output_path=None, skip_frames=2):
        
      #  Testing wrapper to run detection on video (for development/testing only).
      #  This method should be removed or disabled for backend integration.
        
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
                # In a real scenario, detection_result would come from a model
                result = self.detect(frame) # Assuming base class has detect()
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

    parser = argparse.ArgumentParser(description="Intrusion Zone Monitoring")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--camera", type=str, default="intrusion_cam", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")

    args = parser.parse_args()

    # For testing, DB and writer are None
    monitor = IntrusionZoneMonitor(camera_id=args.camera, db=None, db_writer=None)
    monitor.set_individual_events_enabled(True) # Enable events for standalone test

    if args.input:
        monitor.process_video(args.input, args.output, skip_frames=args.skip_frames)
    else:
        print("No input video specified. Use --input to specify a video file.")

    """
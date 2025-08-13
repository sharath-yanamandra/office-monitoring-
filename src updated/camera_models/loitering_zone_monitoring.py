"""
Use Case: Loitering Detection
- Detect people loitering in sensitive zones beyond a certain time threshold
"""

import cv2
import numpy as np
import time

from .camera_model_base import CameraModelBase
from ..config import Config
from ..utils import draw_text_with_background, draw_zone

class LoiteringZoneMonitor(CameraModelBase):
    """
    Use Case: Loitering Detection
    - Detect people loitering in sensitive zones beyond a certain time threshold
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
        self.logger.info("Initializing LoiteringZoneMonitor with detailed logging")

        self.zone_colors = {"loitering": (0, 255, 255)}  # Yellow for loitering zones
        self.loitering_threshold = Config.LOITERING_THRESHOLD
        self.recent_events = {}
        self.tracking_threshold = Config.TRACKING_THRESHOLD
        self.event_cooldown = Config.EVENT_COOLDOWN
        self.auto_recording_enabled = Config.AUTO_RECORDING_ENABLED

        self.loitering_zones = self._get_zones_coordinates()

        self.logger.info(f"Loaded {len(self.loitering_zones)} loitering zones")
        for zone in self.loitering_zones:
            self.logger.info(f"Zone: {zone.get('name', 'Unnamed')} with {len(zone.get('coordinates', []))} coordinates")

        self.stats.update({
            "loitering_events": 0,
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
        self.current_loitering_count = 0

    def _get_zones_coordinates(self):
        """Extract loitering zones from configuration"""
        zones = []
        try:
            for zone_type in self.zones:
                if zone_type == 'loitering':
                    for zone in self.zones[zone_type]:
                        if "coordinates" in zone:
                            zones.append(zone)
        except Exception as e:
            self.logger.error(f"Error extracting loitering zones: {e}", exc_info=True)
        return zones

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled
        self.logger.info(f"Individual camera events {'enabled' if enabled else 'disabled'}")

    def get_current_people_count(self):
        """Get current total people count for aggregator"""
        return self.current_people_count

    def get_current_loitering_count(self):
        """Get current loitering people count for aggregator"""
        return self.current_loitering_count

    def _draw_zones(self, frame):
        """Draw loitering zones on the frame"""
        try:
            for zone in self.loitering_zones:
                if 'coordinates' in zone:
                    name = zone.get("name", "loitering")
                    coordinates = np.array(zone['coordinates'], dtype=np.int32)
                    zone_info = {
                        'coordinates': coordinates,
                        'name': name
                    }
                    draw_zone(frame, zone_info, self.zone_colors["loitering"], alpha=0.3, label=name)
        except Exception as e:
            self.logger.error(f"Error drawing loitering zones: {e}", exc_info=True)
        return frame

    def detect_people(self, frame, detection_result):
        """
        Process a frame for loitering detection
        Args:
            frame: The input video frame
            detection_result: Detection output from model
        Returns:
            Annotated frame, total people count, loitering count, and people by zone
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

                        in_loitering_zone = False
                        loitering_name = None

                        for zone in self.loitering_zones:
                            if self.is_in_zone((center_x, center_y), zone):
                                in_loitering_zone = True
                                loitering_name = zone.get("name", "Loitering")
                                break

                        person_detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': (center_x, center_y),
                            'in_loitering_zone': in_loitering_zone,
                            'loitering_name': loitering_name
                        }

                        people_detections.append(person_detection)
                        tracking_data.append([center_x, center_y, aspect_ratio, int(height), confidence])

                        self.logger.debug(f"Person @({center_x},{center_y}) Conf:{confidence:.2f}, InZone:{in_loitering_zone}, Zone:{loitering_name}")
            except Exception as e:
                self.logger.error(f"Error parsing detection result: {e}", exc_info=True)

        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        if tracking_data:
            tracking_data = np.array(tracking_data)
        
        tracked_objects = self.update_tracker(tracking_data)
        self.logger.debug(f"Tracker returned {len(tracked_objects)} tracked objects")

        total_people = len(tracked_objects)
        self.current_people_count = total_people

        loitering_count = 0
        people_by_zone = {zone.get('name', 'Unnamed'): [] for zone in self.loitering_zones}
        
        for i, detection in enumerate(tracked_objects):
            track_id = detection.get('track_id')
            if not track_id:
                continue

            if i < len(people_detections):
                person_detection = people_detections[i]
                detection.update(person_detection)

                if track_id in self.tracked_objects:
                    in_zone = person_detection.get('in_loitering_zone', False)
                    zone_name = person_detection.get('loitering_name', 'unknown')

                    self.tracked_objects[track_id]['in_loitering_zone'] = in_zone
                    self.tracked_objects[track_id]['loitering_name'] = zone_name

                    if in_zone:
                        entry_time = self.tracked_objects[track_id].get('entry_time')
                        current_time = time.time()

                        if not entry_time:
                            self.tracked_objects[track_id]['entry_time'] = current_time
                            self.logger.debug(f"Track {track_id} entered zone {zone_name} at {current_time}")
                        
                        loitering_duration = current_time - self.tracked_objects[track_id]['entry_time']
                        
                        if loitering_duration >= self.loitering_threshold:
                            loitering_count += 1
                            if zone_name in people_by_zone:
                                people_by_zone[zone_name].append(detection)

                            bbox = detection['bbox']
                            color = (0, 255, 255)
                            label = f"Loitering ID:{track_id} ({zone_name})"
                            cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                            draw_text_with_background(annotated_frame, label, (int(bbox[0]), int(bbox[1]) - 10), color)
                    else:
                        # Reset entry time if object is out of the zone
                        if 'entry_time' in self.tracked_objects[track_id]:
                            del self.tracked_objects[track_id]['entry_time']

        self.current_loitering_count = loitering_count

        if self.enable_individual_events:
            self._handle_individual_camera_events(people_by_zone, annotated_frame)

        return annotated_frame, total_people, loitering_count, people_by_zone

    def _handle_individual_camera_events(self, people_by_zone, annotated_frame):
        """Handle events at individual camera level."""
        for zone_name, detections in people_by_zone.items():
            # Here, the event is triggered for each person loitering, not based on a count threshold
            if detections:
                self.logger.info(f"Individual camera event: {len(detections)} people loitering in zone {zone_name}")
                track_ids = [p.get('track_id') for p in detections if 'track_id' in p]

                for track_id in track_ids:
                    if self._should_record_event('loitering', zone_name, track_id):
                        self.logger.info(f"Recording loitering event for track ID {track_id} in {zone_name}")
                        ist_timestamp = time.time()
                        # Find the specific detection for this track_id to save
                        loitering_person = next((d for d in detections if d.get('track_id') == track_id), None)
                        if loitering_person:
                            event_id = self._save_event_media('loitering', [loitering_person], annotated_frame, ist_timestamp, zone_name)
                            self.logger.info(f"Saved loitering event with ID: {event_id}")
                            
                            if Config.SMS_ENABLED:
                                self._send_sms_alert(1, ist_timestamp, zone_name)
                            
                            self.stats['loitering_events'] += 1

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Process a frame to detect loitering events.
        """
        self.logger.debug(f"Processing frame at timestamp {timestamp}")
        
        annotated_frame, people_count, loitering_count, people_by_zone = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected'] = people_count
        
        self.logger.debug(f"Processed frame: {people_count} people detected, {loitering_count} loitering.")
        
        return annotated_frame, people_by_zone

"""
    def process_video(self, input_path, output_path=None, skip_frames=2):
        
       # Testing wrapper to run detection on video (for development/testing only).

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

    parser = argparse.ArgumentParser(description="Loitering Zone Monitoring")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--camera", type=str, default="loitering_cam", help="Camera identifier in config")
    parser.add_argument("--skip-frames", type=int, default=2, help="Process every Nth frame")

    args = parser.parse_args()

    monitor = LoiteringZoneMonitor(camera_id=args.camera, db=None, db_writer=None)
    monitor.set_individual_events_enabled(True)

    if args.input:
        monitor.process_video(args.input, args.output, skip_frames=args.skip_frames)
    else:
        print("No input video specified. Use --input to specify a video file.")

        """
-- db_setup.sql
-- Simplified Video Monitoring System Database Setup

-- Create database
CREATE DATABASE IF NOT EXISTS video_monitoring;
USE video_monitoring;

-- Sample cameras data
INSERT INTO cameras (name, stream_url, description, location, status, metadata) VALUES
('Main Entrance Camera', 'rtsp://admin:password@192.168.1.101:554/ch0_0.264', 'Main entrance monitoring', 'Building A - Main Entrance', 'active', 
 '{"resolution": "1920x1080", "fps": 30, "night_vision": true, "ptz": false}'),

('Server Room Camera 1', 'rtsp://admin:password@192.168.1.102:554/ch0_0.264', 'Server room monitoring', 'Building A - Server Room 1', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true}'),

('Corridor Camera', 'rtsp://admin:password@192.168.1.103:554/ch0_0.264', 'Corridor monitoring', 'Building A - Main Corridor', 'active',
 '{"resolution": "1920x1080", "fps": 10, "night_vision": true, "ptz": false}'),

('Parking Camera', 'rtsp://admin:password@192.168.1.104:554/ch0_0.264', 'Parking area monitoring', 'Building A - Parking Lot', 'active',
 '{"resolution": "1920x1080", "fps": 15, "night_vision": true, "ptz": true}'),

('Reception Camera', 'rtsp://admin:password@192.168.1.105:554/ch0_0.264', 'Reception area monitoring', 'Building A - Reception', 'active',
 '{"resolution": "1920x1080", "fps": 10, "night_vision": false, "ptz": false}'),

('Emergency Exit Camera', 'rtsp://admin:password@192.168.1.106:554/ch0_0.264', 'Emergency exit monitoring', 'Building A - Emergency Exit', 'active',
 '{"resolution": "1920x1080", "fps": 20, "night_vision": true, "ptz": false}');

-- Get camera IDs for zone setup
SET @cam1 = (SELECT camera_id FROM cameras WHERE name = 'Main Entrance Camera');
SET @cam2 = (SELECT camera_id FROM cameras WHERE name = 'Server Room Camera 1');
SET @cam3 = (SELECT camera_id FROM cameras WHERE name = 'Corridor Camera');
SET @cam4 = (SELECT camera_id FROM cameras WHERE name = 'Parking Camera');
SET @cam5 = (SELECT camera_id FROM cameras WHERE name = 'Reception Camera');
SET @cam6 = (SELECT camera_id FROM cameras WHERE name = 'Emergency Exit Camera');

-- Sample zones data
INSERT INTO zones (camera_id, name, zone_type, coordinates, description) VALUES
-- Main entrance zones
(@cam1, 'Entry Zone', 'entry_zone', '[[100, 100], [500, 100], [500, 400], [100, 400]]', 'Main entry point for tailgating detection'),
(@cam1, 'Counting Line', 'counting_zone', '[[300, 0], [320, 0], [320, 480], [300, 480]]', 'People counting line at entrance'),

-- Server room zones
(@cam2, 'Server Room Area', 'restricted_zone', '[[0, 0], [640, 0], [640, 480], [0, 480]]', 'Restricted server room area'),
(@cam2, 'Critical Equipment Zone', 'restricted_zone', '[[200, 150], [440, 150], [440, 330], [200, 330]]', 'Critical server equipment area'),

-- Corridor zones
(@cam3, 'Corridor Area', 'common_zone', '[[50, 50], [590, 50], [590, 430], [50, 430]]', 'Main corridor for loitering detection'),

-- Parking zones
(@cam4, 'Parking Area', 'perimeter_zone', '[[0, 0], [640, 0], [640, 480], [0, 480]]', 'Parking lot perimeter'),
(@cam4, 'Restricted Parking', 'restricted_zone', '[[100, 100], [540, 100], [540, 380], [100, 380]]', 'VIP parking area'),

-- Reception zones
(@cam5, 'Reception Area', 'common_zone', '[[80, 80], [560, 80], [560, 400], [80, 400]]', 'Reception waiting area'),

-- Emergency exit zones
(@cam6, 'Emergency Exit', 'entry_zone', '[[150, 150], [490, 150], [490, 330], [150, 330]]', 'Emergency exit monitoring');

-- Sample events for demonstration
INSERT INTO events (camera_id, zone_id, event_type, severity, description, detection_data) VALUES
(@cam1, 1, 'tailgating', 'high', 'Two people detected entering simultaneously', 
 '{"people_count": 2, "confidence": 0.89, "timestamp": "2024-01-15T10:30:15"}'),

(@cam2, 3, 'unauthorized_access', 'critical', 'Person detected in server room without authorization', 
 '{"person_id": "unknown", "confidence": 0.92, "timestamp": "2024-01-15T14:22:30"}'),

(@cam2, 4, 'ppe_violation', 'medium', 'Person in server room without required PPE', 
 '{"missing_ppe": ["hard_hat", "safety_vest"], "confidence": 0.85, "timestamp": "2024-01-15T09:15:45"}'),

(@cam3, 5, 'loitering', 'medium', 'Person loitering in corridor for extended period', 
 '{"duration_minutes": 8, "confidence": 0.78, "timestamp": "2024-01-15T16:45:20"}'),

(@cam4, 6, 'intrusion', 'high', 'Vehicle detected in restricted parking area', 
 '{"object_type": "vehicle", "confidence": 0.94, "timestamp": "2024-01-15T20:15:10"}'),

(@cam1, 2, 'people_counting', 'info', 'People count update at entrance', 
 '{"count": 15, "direction": "in", "timestamp": "2024-01-15T12:00:00"}');

-- Sample detection sessions
INSERT INTO detection_sessions (camera_id, models_used, session_name, status, events_detected) VALUES
(@cam1, '["tailgating", "people_counting"]', 'Entrance Monitoring Session', 'completed', 5),
(@cam2, '["unauthorized_access", "ppe_detection"]', 'Server Room Security Check', 'completed', 3),
(@cam3, '["loitering_detection"]', 'Corridor Monitoring', 'running', 1),
(@cam4, '["intrusion_detection"]', 'Parking Security', 'completed', 2);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type_severity ON events(event_type, severity);
CREATE INDEX IF NOT EXISTS idx_sessions_camera_status ON detection_sessions(camera_id, status);
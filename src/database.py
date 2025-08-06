#!/usr/bin/env python3
"""
database.py
Simplified Video Monitoring Database

This module handles:
1. Database connection and pooling
2. Simple table creation for cameras, events, and zones
3. CRUD operations
"""

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class Database:
    _instance: Optional['Database'] = None
    _pool: Optional[MySQLConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database connection pool"""
        self.logger = logging.getLogger('database')
        self.logger.info("Initializing database connection")
        
        self._create_pool()
        if self._pool:
            self._create_tables()
        else:
            self.logger.error("Failed to create database connection pool")

    def _create_pool(self):
        """Create MySQL connection pool"""
        try:
            self.logger.info(f"Creating connection pool to MySQL at {os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}")
            
            pool_config = {
                'pool_name': 'video_monitoring_pool',
                'pool_size': int(os.getenv('DB_POOL_SIZE', 16)),
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'user': os.getenv('MYSQL_USER'),
                'password': os.getenv('MYSQL_PASSWORD'),
                'database': os.getenv('MYSQL_DATABASE', 'video_monitoring'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci',
                'autocommit': True,
                'time_zone': '+00:00'
            }
            
            self._pool = MySQLConnectionPool(**pool_config)
            self.logger.info("Database connection pool created successfully")
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        connection = None
        try:
            connection = self._pool.get_connection()
            yield connection
        except mysql.connector.Error as e:
            self.logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()

    def _create_tables(self):
        """Create necessary tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Cameras table
                self.logger.info("Creating cameras table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cameras (
                        camera_id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        stream_url VARCHAR(500) NOT NULL,
                        description TEXT,
                        location VARCHAR(255),
                        status ENUM('active', 'inactive', 'maintenance') NOT NULL DEFAULT 'active',
                        metadata JSON DEFAULT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_camera_status (status),
                        INDEX idx_camera_location (location)
                    )
                """)

                # Zones table - simplified
                self.logger.info("Creating zones table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS zones (
                        zone_id INT AUTO_INCREMENT PRIMARY KEY,
                        camera_id INT NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        zone_type ENUM('entry_zone', 'restricted_zone', 'common_zone', 'counting_zone', 'perimeter_zone') NOT NULL,
                        coordinates JSON NOT NULL COMMENT 'Array of x,y coordinates defining the zone',
                        description TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        INDEX idx_camera_zones (camera_id),
                        INDEX idx_zone_type (zone_type),
                        UNIQUE KEY unq_camera_zone_name (camera_id, name)
                    )
                """)

                # Events table - simplified
                self.logger.info("Creating events table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        event_id INT AUTO_INCREMENT PRIMARY KEY,
                        camera_id INT NOT NULL,
                        zone_id INT DEFAULT NULL,
                        event_type ENUM('tailgating', 'unauthorized_access', 'people_counting', 
                                       'ppe_violation', 'intrusion', 'loitering') NOT NULL,
                        severity ENUM('info', 'low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'medium',
                        description TEXT,
                        detection_data JSON DEFAULT NULL COMMENT 'Detection results and metadata',
                        media_path VARCHAR(500) DEFAULT NULL,
                        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        processed BOOLEAN NOT NULL DEFAULT FALSE,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE SET NULL,
                        INDEX idx_camera_events (camera_id),
                        INDEX idx_event_type (event_type),
                        INDEX idx_event_timestamp (timestamp),
                        INDEX idx_event_severity (severity),
                        INDEX idx_processed (processed)
                    )
                """)

                # Detection sessions table - track user's detection runs
                self.logger.info("Creating detection_sessions table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_sessions (
                        session_id INT AUTO_INCREMENT PRIMARY KEY,
                        camera_id INT NOT NULL,
                        models_used JSON NOT NULL COMMENT 'List of detection models applied',
                        session_name VARCHAR(255),
                        start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP NULL,
                        status ENUM('running', 'completed', 'failed', 'stopped') NOT NULL DEFAULT 'running',
                        events_detected INT DEFAULT 0,
                        metadata JSON DEFAULT NULL,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        INDEX idx_camera_sessions (camera_id),
                        INDEX idx_session_status (status),
                        INDEX idx_session_time (start_time)
                    )
                """)

                conn.commit()
                self.logger.info("All tables created successfully")

        except mysql.connector.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query, params)
                results = cursor.fetchall()
                return results
        except mysql.connector.Error as e:
            self.logger.error(f"Error executing query: {e}")
            raise

    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except mysql.connector.Error as e:
            self.logger.error(f"Error executing update: {e}")
            raise

    def execute_script(self, script: str):
        """Execute multi-statement SQL script"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Split script into individual statements
                statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        cursor.execute(statement)
                
                conn.commit()
                self.logger.info(f"Script executed successfully - {len(statements)} statements")
        except mysql.connector.Error as e:
            self.logger.error(f"Error executing script: {e}")
            raise

    # Convenience methods for common operations
    def add_camera(self, name: str, stream_url: str, description: str = None, 
                   location: str = None, metadata: dict = None) -> int:
        """Add new camera"""
        query = """
            INSERT INTO cameras (name, stream_url, description, location, metadata) 
            VALUES (%s, %s, %s, %s, %s)
        """
        params = (name, stream_url, description, location, 
                 None if metadata is None else str(metadata).replace("'", '"'))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_cameras(self, status: str = 'active') -> List[Dict[str, Any]]:
        """Get cameras by status"""
        query = "SELECT * FROM cameras WHERE status = %s ORDER BY name"
        return self.execute_query(query, (status,))

    def add_zone(self, camera_id: int, name: str, zone_type: str, 
                 coordinates: list, description: str = None) -> int:
        """Add zone to camera"""
        query = """
            INSERT INTO zones (camera_id, name, zone_type, coordinates, description) 
            VALUES (%s, %s, %s, %s, %s)
        """
        import json
        params = (camera_id, name, zone_type, json.dumps(coordinates), description)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def get_camera_zones(self, camera_id: int) -> List[Dict[str, Any]]:
        """Get zones for specific camera"""
        query = "SELECT * FROM zones WHERE camera_id = %s ORDER BY name"
        return self.execute_query(query, (camera_id,))

    def add_event(self, camera_id: int, event_type: str, severity: str = 'medium',
                  description: str = None, detection_data: dict = None, 
                  media_path: str = None, zone_id: int = None) -> int:
        """Add detection event"""
        query = """
            INSERT INTO events (camera_id, zone_id, event_type, severity, 
                              description, detection_data, media_path) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        import json
        params = (camera_id, zone_id, event_type, severity, description,
                 None if detection_data is None else json.dumps(detection_data),
                 media_path)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def search_events(self, camera_id: int = None, event_type: str = None, 
                      severity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search events with filters"""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if camera_id:
            query += " AND camera_id = %s"
            params.append(camera_id)
        
        if event_type:
            query += " AND event_type = %s"
            params.append(event_type)
            
        if severity:
            query += " AND severity = %s"
            params.append(severity)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        return self.execute_query(query, tuple(params))

    def start_detection_session(self, camera_id: int, models_used: list, 
                               session_name: str = None) -> int:
        """Start new detection session"""
        query = """
            INSERT INTO detection_sessions (camera_id, models_used, session_name) 
            VALUES (%s, %s, %s)
        """
        import json
        params = (camera_id, json.dumps(models_used), session_name)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def end_detection_session(self, session_id: int, status: str = 'completed'):
        """End detection session"""
        query = """
            UPDATE detection_sessions 
            SET end_time = CURRENT_TIMESTAMP, status = %s,
                events_detected = (
                    SELECT COUNT(*) FROM events 
                    WHERE camera_id = (
                        SELECT camera_id FROM detection_sessions WHERE session_id = %s
                    ) AND timestamp >= (
                        SELECT start_time FROM detection_sessions WHERE session_id = %s
                    )
                )
            WHERE session_id = %s
        """
        self.execute_update(query, (status, session_id, session_id, session_id))

    def get_session_events(self, session_id: int) -> List[Dict[str, Any]]:
        """Get events from specific detection session"""
        query = """
            SELECT e.* FROM events e
            JOIN detection_sessions ds ON e.camera_id = ds.camera_id
            WHERE ds.session_id = %s 
            AND e.timestamp >= ds.start_time
            AND (ds.end_time IS NULL OR e.timestamp <= ds.end_time)
            ORDER BY e.timestamp DESC
        """
        return self.execute_query(query, (session_id,))


# Export main class
__all__ = ['Database']
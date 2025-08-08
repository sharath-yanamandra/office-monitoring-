#!/usr/bin/env python3
"""
Script 1: database.py
File Path: src/database.py

Datacenter Monitoring System - Database Foundation

This module handles:
1. Database connection and pooling
2. Datacenter-specific table creation
3. CRUD operations for datacenters, zones, cameras, events
4. User and project management for multi-tenant datacenter monitoring
"""

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from contextlib import contextmanager
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class DatacenterDatabase:
    _instance: Optional['DatacenterDatabase'] = None
    _pool: Optional[MySQLConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatacenterDatabase, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database connection pool and logger"""
        from logger import setup_logger
        self.logger = setup_logger('datacenter_database', 'datacenter_database.log')
        self.logger.info("Initializing datacenter database connection")
        
        # Create pool first
        self._create_pool()
        
        # Then create tables using the pool
        if self._pool:
            self._create_datacenter_tables()
        else:
            self.logger.error("Failed to create database connection pool")

    def _create_pool(self):
        """Create MySQL connection pool for datacenter monitoring"""
        try:
            self.logger.info(f"Creating connection pool to MySQL at {os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}")
            
            pool_config = {
                'pool_name': 'datacenter_pool',
                'pool_size': int(os.getenv('DB_POOL_SIZE', 32)),
                'host': os.getenv('MYSQL_HOST', 'localhost'),
                'user': os.getenv('MYSQL_USER'),
                'password': os.getenv('MYSQL_PASSWORD'),
                'database': os.getenv('MYSQL_DATABASE'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'pool_reset_session': True,
                'connect_timeout': 10,
                'use_pure': True,
                'buffered': True,
                'consume_results': True
            }
            
            self._pool = MySQLConnectionPool(**pool_config)
            self.logger.info("Datacenter database connection pool created successfully")
        
        except mysql.connector.Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for getting a connection from the pool"""
        if not self._pool:
            self.logger.error("Connection pool not initialized")
            raise RuntimeError("Database connection pool not initialized")
        
        conn = None
        try:
            conn = self._pool.get_connection()
            yield conn
        except mysql.connector.Error as e:
            self.logger.error(f"Error getting connection from pool: {str(e)}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()
    
    def _create_datacenter_tables(self):
        """Create database tables for datacenter monitoring system"""
        with self.get_connection() as connection:
            cursor = None
            try:
                self.logger.info("Creating database tables for datacenter monitoring")
                cursor = connection.cursor()

                # Users table - Datacenter administrators and operators
                self.logger.info("Creating users table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(36) PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        full_name VARCHAR(255),
                        hashed_password VARCHAR(255),
                        role ENUM('admin', 'operator', 'viewer') NOT NULL DEFAULT 'viewer',
                        is_active BOOLEAN DEFAULT TRUE,
                        is_verified BOOLEAN DEFAULT FALSE,
                        reset_token VARCHAR(255),
                        reset_token_expires DATETIME,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_email (email),
                        INDEX idx_role (role)
                    )
                """)

                # Datacenters table (equivalent to projects in bank system)
                self.logger.info("Creating datacenters table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datacenters (
                        datacenter_id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(36) NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        location VARCHAR(255) NOT NULL,
                        address TEXT,
                        coordinates VARCHAR(255) COMMENT 'GPS coordinates',
                        facility_type ENUM('tier1', 'tier2', 'tier3', 'tier4') DEFAULT 'tier3',
                        capacity_info JSON DEFAULT NULL COMMENT 'Rack count, power capacity, etc.',
                        contact_info JSON DEFAULT NULL COMMENT 'Emergency contacts, facility manager',
                        status ENUM('active', 'maintenance', 'inactive') NOT NULL DEFAULT 'active',
                        metadata JSON DEFAULT NULL COMMENT 'Additional datacenter properties',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                        INDEX idx_user_datacenters (user_id),
                        INDEX idx_datacenter_status (status),
                        INDEX idx_location (location),
                        UNIQUE KEY unq_user_datacenter_name (user_id, name)
                    )
                """)

                # Cameras table - Updated for datacenter monitoring
                self.logger.info("Creating cameras table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cameras (
                        camera_id INT AUTO_INCREMENT PRIMARY KEY,
                        datacenter_id INT NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        stream_url VARCHAR(255) NOT NULL,
                        camera_type ENUM('dc_entry_monitor', 'dc_server_room', 'dc_corridor', 
                                        'dc_perimeter', 'dc_critical_zone', 'dc_common_area') NOT NULL,
                        location_details JSON DEFAULT NULL COMMENT 'Floor, rack row, specific location',
                        status ENUM('active', 'inactive', 'maintenance', 'fault') NOT NULL DEFAULT 'active',
                        metadata JSON DEFAULT NULL COMMENT 'Camera properties (resolution, fps, etc)',
                        installation_date DATE,
                        last_maintenance DATE,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (datacenter_id) REFERENCES datacenters(datacenter_id) ON DELETE CASCADE,
                        INDEX idx_datacenter_cameras (datacenter_id),
                        INDEX idx_camera_status (status),
                        INDEX idx_camera_type (camera_type),
                        UNIQUE KEY unq_datacenter_camera_name (datacenter_id, name)
                    )
                """)

                # Zones table - Datacenter specific zones
                self.logger.info("Creating zones table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS zones (
                        zone_id INT AUTO_INCREMENT PRIMARY KEY,
                        camera_id INT NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        zone_type ENUM('entry_zone', 'server_zone', 'restricted_zone', 
                                      'common_zone', 'perimeter_zone', 'critical_zone') NOT NULL,
                        polygon_coordinates JSON NOT NULL COMMENT 'Array of x,y coordinates defining the zone',
                        security_level ENUM('public', 'restricted', 'high_security', 'critical') DEFAULT 'restricted',
                        access_requirements JSON DEFAULT NULL COMMENT 'Required PPE, badges, etc.',
                        monitoring_rules JSON DEFAULT NULL COMMENT 'Occupancy limits, time restrictions',
                        metadata JSON DEFAULT NULL COMMENT 'Additional zone properties',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        INDEX idx_camera_zones (camera_id),
                        INDEX idx_zone_type (zone_type),
                        INDEX idx_security_level (security_level),
                        UNIQUE KEY unq_camera_zone_name (camera_id, name)
                    )
                """)

                # Rules table - Datacenter monitoring rules
                self.logger.info("Creating rules table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rules (
                        rule_id INT AUTO_INCREMENT PRIMARY KEY,
                        camera_id INT NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        event_type ENUM('tailgating', 'intrusion', 'ppe_violation', 'loitering', 
                                       'people_counting', 'camera_tamper', 'unauthorized_access') NOT NULL,
                        severity ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'medium',
                        parameters JSON DEFAULT NULL COMMENT 'Rule-specific parameters',
                        schedule JSON DEFAULT NULL COMMENT 'When this rule is active',
                        notification_settings JSON DEFAULT NULL COMMENT 'Who to notify and how',
                        enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        INDEX idx_camera_rules (camera_id),
                        INDEX idx_event_type (event_type),
                        INDEX idx_severity (severity),
                        UNIQUE KEY unq_camera_rule_event (camera_id, event_type)
                    )
                """)

                # Events table - Datacenter security events
                self.logger.info("Creating events table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        event_id VARCHAR(36) PRIMARY KEY,
                        rule_id INT NOT NULL,
                        camera_id INT NOT NULL,
                        zone_id INT,
                        timestamp TIMESTAMP NOT NULL,
                        event_type ENUM('tailgating', 'intrusion', 'ppe_violation', 'loitering', 
                                       'people_counting', 'camera_tamper', 'unauthorized_access') NOT NULL,
                        severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
                        detection_data JSON DEFAULT NULL COMMENT 'Detection details, person count, etc.',
                        snapshot_url VARCHAR(255) DEFAULT NULL COMMENT 'Path to saved image',
                        video_clip_url VARCHAR(255) DEFAULT NULL COMMENT 'Path to saved video clip',
                        status ENUM('new', 'acknowledged', 'resolved', 'false_positive') NOT NULL DEFAULT 'new',
                        acknowledged_by VARCHAR(36) DEFAULT NULL,
                        acknowledged_at TIMESTAMP NULL,
                        resolution_notes TEXT DEFAULT NULL,
                        resolved_by VARCHAR(36) DEFAULT NULL,
                        resolved_at TIMESTAMP NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                        FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE SET NULL,
                        FOREIGN KEY (acknowledged_by) REFERENCES users(user_id) ON DELETE SET NULL,
                        FOREIGN KEY (resolved_by) REFERENCES users(user_id) ON DELETE SET NULL,
                        INDEX idx_camera_events (camera_id, timestamp),
                        INDEX idx_status (status),
                        INDEX idx_event_type (event_type),
                        INDEX idx_severity (severity),
                        INDEX idx_timestamp (timestamp)
                    )
                """)

                # Access logs table - Track personnel access
                self.logger.info("Creating access_logs table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS access_logs (
                        log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        datacenter_id INT NOT NULL,
                        camera_id INT,
                        zone_id INT,
                        person_id VARCHAR(255) COMMENT 'Badge ID or person identifier',
                        access_type ENUM('entry', 'exit', 'denied', 'tailgating') NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        detection_confidence FLOAT DEFAULT NULL,
                        additional_data JSON DEFAULT NULL COMMENT 'PPE status, duration, etc.',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (datacenter_id) REFERENCES datacenters(datacenter_id) ON DELETE CASCADE,
                        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL,
                        FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE SET NULL,
                        INDEX idx_datacenter_access (datacenter_id, timestamp),
                        INDEX idx_person_access (person_id, timestamp),
                        INDEX idx_access_type (access_type)
                    )
                """)

                # Create views for common queries
                self.logger.info("Creating views")
                
                # Camera status view
                cursor.execute("""
                    CREATE OR REPLACE VIEW camera_status_view AS
                    SELECT 
                        c.camera_id,
                        c.datacenter_id,
                        d.name AS datacenter_name,
                        c.name AS camera_name,
                        c.camera_type,
                        c.status,
                        c.stream_url,
                        (SELECT COUNT(*) FROM events e WHERE e.camera_id = c.camera_id) AS total_events,
                        (
                            SELECT COUNT(*) FROM events e 
                            WHERE e.camera_id = c.camera_id 
                            AND e.timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                        ) AS events_last_24h,
                        (
                            SELECT COUNT(*) FROM events e 
                            WHERE e.camera_id = c.camera_id 
                            AND e.status = 'new'
                        ) AS unresolved_events,
                        c.last_maintenance,
                        c.created_at,
                        c.updated_at
                    FROM cameras c
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                """)

                # Active events view
                cursor.execute("""
                    CREATE OR REPLACE VIEW active_events_view AS
                    SELECT 
                        e.event_id,
                        e.timestamp,
                        e.event_type,
                        e.severity,
                        e.status,
                        c.datacenter_id,
                        d.name AS datacenter_name,
                        c.name AS camera_name,
                        c.camera_type,
                        r.name AS rule_name,
                        z.name AS zone_name,
                        z.security_level,
                        e.snapshot_url,
                        e.video_clip_url,
                        e.detection_data
                    FROM events e
                    JOIN cameras c ON e.camera_id = c.camera_id
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                    JOIN rules r ON e.rule_id = r.rule_id
                    LEFT JOIN zones z ON e.zone_id = z.zone_id
                    WHERE e.status IN ('new', 'acknowledged')
                    ORDER BY e.severity DESC, e.timestamp DESC
                """)
                
                self.logger.info("Datacenter database tables created successfully")
                
            except mysql.connector.Error as e:
                self.logger.error(f"Error creating tables: {str(e)}", exc_info=True)
                connection.rollback()
                raise
            finally:
                if cursor:
                    cursor.close()
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a query and return results"""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    connection.commit()
                    return cursor.rowcount
            finally:
                cursor.close()
    
    def execute_script(self, script: str):
        """Execute a multi-statement SQL script"""
        with self.get_connection() as connection:
            cursor = None
            try:
                cursor = connection.cursor()
                
                # Manual parsing approach to handle semicolons properly
                current_statement = []
                
                for line in script.split('\n'):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('--'):
                        continue
                    
                    # Add to current statement
                    current_statement.append(line)
                    
                    # Check if line ends with semicolon
                    if line.rstrip().endswith(';'):
                        # Join and execute the statement
                        full_statement = ' '.join(current_statement)
                        try:
                            self.logger.debug(f"Executing SQL: {full_statement}")
                            cursor.execute(full_statement)
                        except mysql.connector.Error as e:
                            self.logger.error(f"Error executing statement: {e}")
                            self.logger.error(f"Failed statement: {full_statement}")
                            raise
                        # Reset for next statement
                        current_statement = []
                
                connection.commit()
                return True
            except mysql.connector.Error as e:
                self.logger.error(f"Error executing script: {str(e)}", exc_info=True)
                connection.rollback()
                raise
            finally:
                if cursor:
                    cursor.close()
    
    # Datacenter-specific helper methods
    def get_user_datacenters(self, user_id: str):
        """Get all datacenters belonging to a specific user"""
        query = """
            SELECT datacenter_id, name, description, location, facility_type, status, created_at
            FROM datacenters 
            WHERE user_id = %s AND status = 'active'
            ORDER BY name
        """
        return self.execute_query(query, (user_id,))
    
    def get_datacenter_cameras(self, datacenter_id: int):
        """Get all cameras in a specific datacenter"""
        query = """
            SELECT camera_id, name, camera_type, stream_url, status, location_details, metadata
            FROM cameras 
            WHERE datacenter_id = %s
            ORDER BY camera_type, name
        """
        return self.execute_query(query, (datacenter_id,))
    
    def verify_datacenter_ownership(self, user_id: str, datacenter_id: int):
        """Verify that a user owns a specific datacenter"""
        query = """
            SELECT 1 
            FROM datacenters 
            WHERE datacenter_id = %s AND user_id = %s
        """
        result = self.execute_query(query, (datacenter_id, user_id))
        return len(result) > 0
    
    def get_recent_events(self, datacenter_id: int, hours: int = 24):
        """Get recent events for a datacenter"""
        query = """
            SELECT e.*, c.name AS camera_name, z.name AS zone_name
            FROM events e
            JOIN cameras c ON e.camera_id = c.camera_id
            LEFT JOIN zones z ON e.zone_id = z.zone_id
            WHERE c.datacenter_id = %s 
            AND e.timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            ORDER BY e.timestamp DESC
            LIMIT 100
        """
        return self.execute_query(query, (datacenter_id, hours))
    
    @classmethod
    def close_all(cls):
        """Class method to safely close all database connections"""
        if cls._instance and cls._instance._pool:
            cls._instance.logger.info("Closing all datacenter database connections")
            cls._instance._pool = None
            cls._instance = None

    def close(self):
        """Instance method to close database connections"""
        self.logger.info("Closing datacenter database connections")
        if self._pool:
            self._pool = None
        DatacenterDatabase._instance = None

if __name__ == '__main__':
    # Test database initialization
    db = DatacenterDatabase()
    try:
        # Test query
        result = db.execute_query("SELECT COUNT(*) as count FROM datacenters")
        print(f"Datacenters in database: {result[0]['count'] if result else 0}")
    except Exception as e:
        print(f"Database test failed: {e}")
    finally:
        db.close()
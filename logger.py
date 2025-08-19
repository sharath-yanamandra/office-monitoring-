#!/usr/bin/env python3
"""
Script 3: logger.py
File Path: src/logger.py

Datacenter Monitoring System - Logging Configuration

This module handles:
1. Centralized logging configuration for all datacenter components
2. Structured logging with different levels and formats
3. Log rotation and file management
4. Component-specific loggers (camera, detection, events, etc.)
5. Performance and security audit logging
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict, Any

class DatacenterLogFormatter(logging.Formatter):
    """Custom formatter for datacenter monitoring logs with structured output"""
    
    def __init__(self):
        # Define color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green  
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # Base format for all logs
        self.base_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(self.base_format)
    
    def format(self, record):
        """Format log record with additional datacenter context"""
        
        # Add custom fields for datacenter monitoring
        if not hasattr(record, 'datacenter_id'):
            record.datacenter_id = getattr(record, 'datacenter_id', 'unknown')
        if not hasattr(record, 'camera_id'):
            record.camera_id = getattr(record, 'camera_id', 'unknown')
        if not hasattr(record, 'event_type'):
            record.event_type = getattr(record, 'event_type', 'general')
        
        # Format the basic message
        formatted_message = super().format(record)
        
        # Add datacenter context if available
        context_parts = []
        if hasattr(record, 'datacenter_id') and record.datacenter_id != 'unknown':
            context_parts.append(f"DC:{record.datacenter_id}")
        if hasattr(record, 'camera_id') and record.camera_id != 'unknown':
            context_parts.append(f"CAM:{record.camera_id}")
        if hasattr(record, 'event_type') and record.event_type != 'general':
            context_parts.append(f"EVENT:{record.event_type}")
        
        if context_parts:
            context_str = f"[{' | '.join(context_parts)}]"
            formatted_message = f"{formatted_message} {context_str}"
        
        return formatted_message

class DatacenterConsoleFormatter(DatacenterLogFormatter):
    """Console formatter with colors for better readability"""
    
    def format(self, record):
        """Format with colors for console output"""
        formatted_message = super().format(record)
        
        # Add colors for console output
        if record.levelname in self.colors:
            color = self.colors[record.levelname]
            reset = self.colors['RESET']
            formatted_message = f"{color}{formatted_message}{reset}"
        
        return formatted_message

class DatacenterJsonFormatter(logging.Formatter):
    """JSON formatter for structured logging and log analysis"""
    
    def format(self, record):
        """Format log record as JSON for structured logging"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add datacenter-specific fields
        if hasattr(record, 'datacenter_id'):
            log_data['datacenter_id'] = record.datacenter_id
        if hasattr(record, 'camera_id'):
            log_data['camera_id'] = record.camera_id
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        if hasattr(record, 'severity'):
            log_data['severity'] = record.severity
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields passed to logger
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                if not key.startswith('_'):
                    log_data[f'extra_{key}'] = value
        
        return json.dumps(log_data, default=str)

def setup_datacenter_logger(name: str, log_file: Optional[str] = None, 
                           level: int = logging.INFO, 
                           datacenter_id: Optional[str] = None,
                           camera_id: Optional[str] = None,
                           json_logging: bool = False) -> logging.Logger:
    """
    Set up a logger for datacenter monitoring components
    
    Args:
        name: Logger name (typically component name)
        log_file: Optional log file name
        level: Logging level
        datacenter_id: Optional datacenter ID for context
        camera_id: Optional camera ID for context  
        json_logging: Whether to use JSON formatting for structured logs
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_logging:
        console_formatter = DatacenterJsonFormatter()
    else:
        console_formatter = DatacenterConsoleFormatter()
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        # Create rotating file handler to manage log size
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        if json_logging:
            file_formatter = DatacenterJsonFormatter()
        else:
            file_formatter = DatacenterLogFormatter()
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add context to logger for datacenter monitoring
    if datacenter_id:
        logger = DatacenterLoggerAdapter(logger, {'datacenter_id': datacenter_id})
    if camera_id:
        logger = DatacenterLoggerAdapter(logger, {'camera_id': camera_id})
    
    return logger

class DatacenterLoggerAdapter(logging.LoggerAdapter):
    """Adapter to add datacenter context to log records"""
    
    def process(self, msg, kwargs):
        """Add datacenter context to log records"""
        
        # Add extra context from adapter
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs

class DatacenterAuditLogger:
    """Specialized logger for security and audit events"""
    
    def __init__(self, log_file: str = "datacenter_audit.log"):
        self.logger = setup_datacenter_logger(
            'datacenter_audit', 
            log_file, 
            level=logging.INFO,
            json_logging=True  # Always use JSON for audit logs
        )
    
    def log_event_detection(self, event_type: str, camera_id: str, 
                           datacenter_id: str, severity: str, 
                           detection_data: Dict[str, Any]):
        """Log security event detection"""
        self.logger.info(
            f"Security event detected: {event_type}",
            extra={
                'event_type': event_type,
                'camera_id': camera_id,
                'datacenter_id': datacenter_id,
                'severity': severity,
                'detection_data': detection_data,
                'audit_category': 'event_detection'
            }
        )
    
    def log_user_action(self, user_id: str, action: str, 
                       resource: str, result: str):
        """Log user actions for audit trail"""
        self.logger.info(
            f"User action: {action} on {resource}",
            extra={
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'result': result,
                'audit_category': 'user_action'
            }
        )
    
    def log_system_event(self, component: str, event: str, 
                        status: str, details: Optional[Dict] = None):
        """Log system events (startup, shutdown, errors)"""
        self.logger.info(
            f"System event: {component} - {event}",
            extra={
                'component': component,
                'system_event': event,
                'status': status,
                'details': details or {},
                'audit_category': 'system_event'
            }
        )
    
    def log_access_event(self, datacenter_id: str, zone_id: str,
                        person_id: Optional[str], access_type: str,
                        result: str):
        """Log access control events"""
        self.logger.info(
            f"Access event: {access_type} to zone {zone_id}",
            extra={
                'datacenter_id': datacenter_id,
                'zone_id': zone_id,
                'person_id': person_id,
                'access_type': access_type,
                'access_result': result,
                'audit_category': 'access_control'
            }
        )

class DatacenterPerformanceLogger:
    """Specialized logger for performance monitoring"""
    
    def __init__(self, log_file: str = "datacenter_performance.log"):
        self.logger = setup_datacenter_logger(
            'datacenter_performance',
            log_file,
            level=logging.INFO,
            json_logging=True
        )
    
    def log_processing_stats(self, camera_id: str, fps: float, 
                           batch_size: int, processing_time: float,
                           queue_size: int):
        """Log camera processing performance"""
        self.logger.info(
            f"Camera processing stats: {camera_id}",
            extra={
                'camera_id': camera_id,
                'fps': fps,
                'batch_size': batch_size,
                'processing_time': processing_time,
                'queue_size': queue_size,
                'metric_type': 'camera_performance'
            }
        )
    
    def log_detection_stats(self, model_type: str, inference_time: float,
                          detections_count: int, confidence_avg: float):
        """Log AI model performance"""
        self.logger.info(
            f"Detection performance: {model_type}",
            extra={
                'model_type': model_type,
                'inference_time': inference_time,
                'detections_count': detections_count,
                'confidence_avg': confidence_avg,
                'metric_type': 'detection_performance'
            }
        )
    
    def log_system_resources(self, cpu_usage: float, memory_usage: float,
                           gpu_usage: float, disk_usage: float):
        """Log system resource usage"""
        self.logger.info(
            "System resource usage",
            extra={
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'gpu_usage': gpu_usage,
                'disk_usage': disk_usage,
                'metric_type': 'system_resources'
            }
        )

# Convenience function for quick logger setup (backward compatibility)
def setup_logger(name: str, log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """
    Backward compatible setup_logger function
    
    Args:
        name: Logger name
        log_file: Optional log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    return setup_datacenter_logger(name, log_file, level)

# Pre-configured loggers for common components
def get_camera_logger(camera_id: str, datacenter_id: Optional[str] = None) -> logging.Logger:
    """Get logger for camera component"""
    return setup_datacenter_logger(
        f'camera_{camera_id}',
        f'camera_{camera_id}.log',
        camera_id=camera_id,
        datacenter_id=datacenter_id
    )

def get_detection_logger() -> logging.Logger:
    """Get logger for detection components"""
    return setup_datacenter_logger('detection', 'detection.log')

def get_database_logger() -> logging.Logger:
    """Get logger for database operations"""
    return setup_datacenter_logger('database', 'database.log')

def get_api_logger() -> logging.Logger:
    """Get logger for API components"""
    return setup_datacenter_logger('api', 'api.log')

def get_main_logger() -> logging.Logger:
    """Get main application logger"""
    return setup_datacenter_logger('datacenter_main', 'datacenter_main.log')

# Global audit and performance loggers
audit_logger = DatacenterAuditLogger()
performance_logger = DatacenterPerformanceLogger()

# Export main functions and classes
__all__ = [
    'setup_datacenter_logger',
    'setup_logger',  # Backward compatibility
    'get_camera_logger',
    'get_detection_logger', 
    'get_database_logger',
    'get_api_logger',
    'get_main_logger',
    'DatacenterAuditLogger',
    'DatacenterPerformanceLogger',
    'audit_logger',
    'performance_logger'
]
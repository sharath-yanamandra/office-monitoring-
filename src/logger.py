#!/usr/bin/env python3
"""
logger.py
Logging Configuration for Video Monitoring System

This module provides:
1. Structured logging setup
2. Multiple log levels and handlers
3. Performance and audit logging
4. Log rotation and management
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
import threading

def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if log_file:
        # Ensure logs directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, name: str = 'performance'):
        self.logger = setup_logger(f'{name}_performance', f'logs/{name}_performance.log')
        self.timers = {}  # Thread-safe timers
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str, context: Dict[str, Any] = None):
        """Start timing an operation"""
        thread_id = threading.get_ident()
        key = f"{operation}_{thread_id}"
        
        with self.lock:
            self.timers[key] = {
                'start_time': datetime.now(),
                'operation': operation,
                'context': context or {}
            }
    
    def end_timer(self, operation: str, additional_data: Dict[str, Any] = None):
        """End timing and log performance"""
        thread_id = threading.get_ident()
        key = f"{operation}_{thread_id}"
        
        with self.lock:
            if key not in self.timers:
                self.logger.warning(f"Timer not found for operation: {operation}")
                return
            
            timer_data = self.timers.pop(key)
            end_time = datetime.now()
            duration = (end_time - timer_data['start_time']).total_seconds()
            
            log_data = {
                'operation': operation,
                'duration_seconds': duration,
                'start_time': timer_data['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                **timer_data['context']
            }
            
            if additional_data:
                log_data.update(additional_data)
            
            self.logger.info(f"Performance: {json.dumps(log_data)}")
    
    def log_metric(self, metric_name: str, value: float, unit: str = '', 
                   context: Dict[str, Any] = None):
        """Log a performance metric"""
        log_data = {
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat(),
            **(context or {})
        }
        
        self.logger.info(f"Metric: {json.dumps(log_data)}")

class AuditLogger:
    """Logger for audit trail and security events"""
    
    def __init__(self, name: str = 'audit'):
        self.logger = setup_logger(f'{name}_audit', f'logs/{name}_audit.log')
    
    def log_event(self, event_type: str, actor: str, target: str, 
                  action: str, status: str, details: Dict[str, Any] = None):
        """
        Log audit event
        
        Args:
            event_type: Type of event (e.g., 'system', 'user', 'detection')
            actor: Who performed the action
            target: What was acted upon
            action: What action was performed
            status: Result status (success, failure, etc.)
            details: Additional event details
        """
        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'actor': actor,
            'target': target,
            'action': action,
            'status': status,
            'details': details or {}
        }
        
        self.logger.info(f"Audit: {json.dumps(audit_data)}")
    
    def log_system_event(self, component: str, event: str, status: str, 
                        details: Dict[str, Any] = None):
        """Log system-level event"""
        self.log_event(
            event_type='system',
            actor='system',
            target=component,
            action=event,
            status=status,
            details=details
        )
    
    def log_detection_event(self, camera_id: int, detection_type: str, 
                           confidence: float, details: Dict[str, Any] = None):
        """Log detection event"""
        self.log_event(
            event_type='detection',
            actor=f'camera_{camera_id}',
            target=detection_type,
            action='detect',
            status='success',
            details={
                'confidence': confidence,
                **(details or {})
            }
        )
    
    def log_user_action(self, user: str, action: str, target: str, 
                       status: str, details: Dict[str, Any] = None):
        """Log user action"""
        self.log_event(
            event_type='user',
            actor=user,
            target=target,
            action=action,
            status=status,
            details=details
        )

class SystemLogger:
    """Logger for system monitoring and health"""
    
    def __init__(self, name: str = 'system'):
        self.logger = setup_logger(f'{name}_system', f'logs/{name}_system.log')
    
    def log_resource_usage(self, cpu_percent: float, memory_percent: float, 
                          gpu_percent: float = None, disk_percent: float = None):
        """Log system resource usage"""
        resource_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_percent': gpu_percent,
            'disk_percent': disk_percent
        }
        
        self.logger.info(f"Resources: {json.dumps(resource_data)}")
    
    def log_health_status(self, component: str, status: str, 
                         metrics: Dict[str, Any] = None):
        """Log component health status"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'status': status,
            'metrics': metrics or {}
        }
        
        self.logger.info(f"Health: {json.dumps(health_data)}")
    
    def log_error(self, component: str, error_type: str, error_message: str, 
                  context: Dict[str, Any] = None):
        """Log system error"""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.logger.error(f"SystemError: {json.dumps(error_data)}")

# Global logger instances
performance_logger = PerformanceLogger()
audit_logger = AuditLogger()
system_logger = SystemLogger()

# Context manager for performance timing
class performance_timer:
    """Context manager for timing operations"""
    
    def __init__(self, operation: str, context: Dict[str, Any] = None):
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        performance_logger.start_timer(self.operation, self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        additional_data = {}
        if exc_type:
            additional_data['error'] = str(exc_val)
            additional_data['error_type'] = exc_type.__name__
        
        performance_logger.end_timer(self.operation, additional_data)

# Utility functions
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def setup_application_logging(log_level: str = 'INFO'):
    """Set up application-wide logging configuration"""
    
    # Convert string level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Set up main application loggers
    main_logger = setup_logger('main', 'logs/main.log', level)
    database_logger = setup_logger('database', 'logs/database.log', level)
    camera_logger = setup_logger('camera_manager', 'logs/camera_manager.log', level)
    model_logger = setup_logger('model_manager', 'logs/model_manager.log', level)
    
    return main_logger

# Export main components
__all__ = [
    'setup_logger',
    'PerformanceLogger',
    'AuditLogger', 
    'SystemLogger',
    'performance_logger',
    'audit_logger',
    'system_logger',
    'performance_timer',
    'log_function_call',
    'setup_application_logging'
]
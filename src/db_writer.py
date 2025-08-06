#!/usr/bin/env python3
"""
db_writer.py
Asynchronous Database Writer for Video Monitoring

This module handles:
1. Asynchronous database operations
2. Batch writing for performance
3. Event queue management
4. Error handling and retry logic
"""

import asyncio
import queue
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from database import Database

class DatabaseWriter:
    """Asynchronous database writer with batching and retry logic"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0):
        """
        Initialize database writer
        
        Args:
            batch_size: Number of events to batch before writing
            flush_interval: Time interval to force flush (seconds)
        """
        self.logger = logging.getLogger('db_writer')
        
        # Database connection
        self.db = Database()
        
        # Batch configuration
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Event queue and batch
        self.event_queue = queue.Queue()
        self.current_batch = []
        self.last_flush_time = time.time()
        
        # Writer thread
        self.writer_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'events_queued': 0,
            'events_written': 0,
            'batches_written': 0,
            'write_errors': 0,
            'queue_overflows': 0
        }
        
        self.logger.info(f"DatabaseWriter initialized - batch_size: {batch_size}, flush_interval: {flush_interval}s")
    
    def start(self):
        """Start the database writer thread"""
        if not self.is_running:
            self.is_running = True
            self.writer_thread = threading.Thread(target=self._writer_loop)
            self.writer_thread.daemon = True
            self.writer_thread.start()
            self.logger.info("DatabaseWriter started")
    
    def stop(self):
        """Stop the database writer and flush remaining events"""
        if self.is_running:
            self.is_running = False
            
            # Wait for thread to finish
            if self.writer_thread and self.writer_thread.is_alive():
                self.writer_thread.join(timeout=10)
            
            # Flush remaining events
            self._flush_batch()
            
            self.logger.info("DatabaseWriter stopped")
    
    def add_event(self, event: Dict[str, Any]):
        """
        Add event to write queue
        
        Args:
            event: Event dictionary to write
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in event:
                event['timestamp'] = datetime.now()
            
            # Add to queue
            self.event_queue.put(event, block=False)
            self.stats['events_queued'] += 1
            
        except queue.Full:
            self.stats['queue_overflows'] += 1
            self.logger.warning("Event queue full, dropping event")
    
    def _writer_loop(self):
        """Main writer loop that processes events"""
        while self.is_running:
            try:
                # Try to get event from queue
                try:
                    event = self.event_queue.get(timeout=1.0)
                    self.current_batch.append(event)
                except queue.Empty:
                    pass
                
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(self.current_batch) >= self.batch_size or
                    (self.current_batch and 
                     (current_time - self.last_flush_time) >= self.flush_interval)
                )
                
                if should_flush:
                    self._flush_batch()
                
            except Exception as e:
                self.logger.error(f"Error in writer loop: {e}")
                time.sleep(1)
    
    def _flush_batch(self):
        """Flush current batch to database"""
        if not self.current_batch:
            return
        
        try:
            self.logger.debug(f"Flushing batch of {len(self.current_batch)} events")
            
            # Write events to database
            for event in self.current_batch:
                self._write_single_event(event)
            
            # Update statistics
            self.stats['events_written'] += len(self.current_batch)
            self.stats['batches_written'] += 1
            
            # Clear batch
            self.current_batch.clear()
            self.last_flush_time = time.time()
            
            self.logger.debug("Batch flushed successfully")
            
        except Exception as e:
            self.stats['write_errors'] += 1
            self.logger.error(f"Error flushing batch: {e}")
            
            # In case of error, we might want to retry or save to file
            # For now, we'll just clear the batch to prevent infinite loops
            self.current_batch.clear()
    
    def _write_single_event(self, event: Dict[str, Any]):
        """
        Write single event to database
        
        Args:
            event: Event dictionary
        """
        try:
            # Prepare event data
            camera_id = event.get('camera_id')
            zone_id = event.get('zone_id')
            event_type = event.get('event_type')
            severity = event.get('severity', 'medium')
            description = event.get('description', '')
            detection_data = event.get('detection_data')
            media_path = event.get('media_path')
            
            # Convert detection_data to JSON string if it's a dict
            if isinstance(detection_data, dict):
                detection_data_json = json.dumps(detection_data)
            else:
                detection_data_json = detection_data
            
            # Insert into database
            self.db.add_event(
                camera_id=camera_id,
                event_type=event_type,
                severity=severity,
                description=description,
                detection_data=detection_data_json,
                media_path=media_path,
                zone_id=zone_id
            )
            
        except Exception as e:
            self.logger.error(f"Error writing event to database: {e}")
            raise
    
    def add_events_batch(self, events: List[Dict[str, Any]]):
        """
        Add multiple events at once
        
        Args:
            events: List of event dictionaries
        """
        for event in events:
            self.add_event(event)
    
    def force_flush(self):
        """Force flush current batch immediately"""
        self._flush_batch()
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.event_queue.qsize()
    
    def get_batch_size(self) -> int:
        """Get current batch size"""
        return len(self.current_batch)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get writer statistics"""
        return {
            **self.stats,
            'queue_size': self.get_queue_size(),
            'current_batch_size': self.get_batch_size(),
            'is_running': self.is_running,
            'batch_size': self.batch_size,
            'flush_interval': self.flush_interval
        }
    
    def reset_statistics(self):
        """Reset writer statistics"""
        self.stats = {
            'events_queued': 0,
            'events_written': 0,
            'batches_written': 0,
            'write_errors': 0,
            'queue_overflows': 0
        }
        self.logger.info("DatabaseWriter statistics reset")


class AsyncDatabaseWriter:
    """Fully asynchronous database writer using asyncio"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0):
        """
        Initialize async database writer
        
        Args:
            batch_size: Number of events to batch before writing
            flush_interval: Time interval to force flush (seconds)
        """
        self.logger = logging.getLogger('async_db_writer')
        
        self.db = Database()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self.event_queue = asyncio.Queue()
        self.current_batch = []
        self.last_flush_time = time.time()
        
        self.writer_task = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'events_queued': 0,
            'events_written': 0,
            'batches_written': 0,
            'write_errors': 0
        }
    
    async def start(self):
        """Start the async database writer"""
        if not self.is_running:
            self.is_running = True
            self.writer_task = asyncio.create_task(self._async_writer_loop())
            self.logger.info("AsyncDatabaseWriter started")
    
    async def stop(self):
        """Stop the async database writer"""
        if self.is_running:
            self.is_running = False
            
            if self.writer_task:
                await self.writer_task
            
            await self._flush_batch_async()
            self.logger.info("AsyncDatabaseWriter stopped")
    
    async def add_event_async(self, event: Dict[str, Any]):
        """
        Add event to async queue
        
        Args:
            event: Event dictionary to write
        """
        try:
            if 'timestamp' not in event:
                event['timestamp'] = datetime.now()
            
            await self.event_queue.put(event)
            self.stats['events_queued'] += 1
            
        except Exception as e:
            self.logger.error(f"Error adding event to async queue: {e}")
    
    async def _async_writer_loop(self):
        """Async writer loop"""
        while self.is_running:
            try:
                # Wait for event or timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    self.current_batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(self.current_batch) >= self.batch_size or
                    (self.current_batch and 
                     (current_time - self.last_flush_time) >= self.flush_interval)
                )
                
                if should_flush:
                    await self._flush_batch_async()
                
            except Exception as e:
                self.logger.error(f"Error in async writer loop: {e}")
                await asyncio.sleep(1)
    
    async def _flush_batch_async(self):
        """Flush batch asynchronously"""
        if not self.current_batch:
            return
        
        try:
            # Write batch to database in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_batch_sync)
            
            # Update statistics
            self.stats['events_written'] += len(self.current_batch)
            self.stats['batches_written'] += 1
            
            self.current_batch.clear()
            self.last_flush_time = time.time()
            
        except Exception as e:
            self.stats['write_errors'] += 1
            self.logger.error(f"Error in async batch flush: {e}")
            self.current_batch.clear()
    
    def _write_batch_sync(self):
        """Synchronous batch write (called in thread pool)"""
        for event in self.current_batch:
            try:
                camera_id = event.get('camera_id')
                zone_id = event.get('zone_id')
                event_type = event.get('event_type')
                severity = event.get('severity', 'medium')
                description = event.get('description', '')
                detection_data = event.get('detection_data')
                media_path = event.get('media_path')
                
                if isinstance(detection_data, dict):
                    detection_data = json.dumps(detection_data)
                
                self.db.add_event(
                    camera_id=camera_id,
                    event_type=event_type,
                    severity=severity,
                    description=description,
                    detection_data=detection_data,
                    media_path=media_path,
                    zone_id=zone_id
                )
                
            except Exception as e:
                self.logger.error(f"Error writing single event: {e}")


# Export main classes
__all__ = ['DatabaseWriter', 'AsyncDatabaseWriter']
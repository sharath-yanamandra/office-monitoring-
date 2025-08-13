#!/usr/bin/env python3


#!/usr/bin/env python3
"""
Datacenter Model Manager - Enhanced GPU Model Management

This module handles:
1. Multi-GPU support for datacenter monitoring workloads
2. Dynamic model loading and GPU distribution
3. Advanced memory management and optimization
4. GPU health monitoring and failover
5. Load balancing across multiple GPUs
6. Performance tracking and optimization
"""

import torch
import threading
import time
import asyncio
import gc
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from .logger import setup_datacenter_logger
from .config import Config

class GPUStatus(Enum):
    """GPU status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

@dataclass
class GPUInfo:
    """GPU information and statistics"""
    device_id: int
    name: str
    memory_total: int
    memory_allocated: int
    memory_reserved: int
    utilization: float
    temperature: Optional[float]
    status: GPUStatus
    last_check: float
    error_count: int
    models_loaded: int

@dataclass
class ModelLoadingResult:
    """Result of model loading operation"""
    success: bool
    model_instance: Optional['DatacenterModelInstance']
    gpu_id: Optional[int]
    load_time: float
    memory_used: int
    error_message: Optional[str]

class DatacenterModelInstance:
    """Enhanced model instance for datacenter monitoring"""
    def __init__(self, model: YOLO, model_type: str, gpu_id: int, instance_id: str):
        self.model = model
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.instance_id = instance_id
        self.in_use = False
        self.lock = threading.Lock()
        self.created_at = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.total_inference_time = 0.0
        self.memory_footprint = self._calculate_memory_footprint()
        self.warmup_completed = False
        
    def _calculate_memory_footprint(self) -> int:
        """Calculate model memory footprint"""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.gpu_id)
                return torch.cuda.memory_allocated(self.gpu_id)
        except:
            pass
        return 0
    
    def update_usage_stats(self, inference_time: float):
        """Update usage statistics"""
        self.usage_count += 1
        self.total_inference_time += inference_time
        self.last_used = time.time()
    
    def get_average_inference_time(self) -> float:
        """Get average inference time"""
        return self.total_inference_time / max(1, self.usage_count)
    
    def warmup(self):
        """Perform model warmup"""
        if self.warmup_completed:
            return
        try:
            # Create dummy input for warmup
            dummy_input = torch.randn(1, 3, 640, 640).to(f'cuda:{self.gpu_id}')
            with torch.no_grad():
                _ = self.model.model(dummy_input)
            self.warmup_completed = True
        except Exception as e:
            print(f"Warmup failed for model {self.instance_id}: {e}")

class DatacenterModelManager:
    """Advanced model manager for datacenter monitoring with multi-GPU support"""
    
    def __init__(self, enable_multi_gpu: bool = True, memory_threshold: float = 0.85, max_instances_per_gpu: int = 2):
        """
        Initialize datacenter model manager
        
        Args:
            enable_multi_gpu: Enable multi-GPU support
            memory_threshold: GPU memory threshold for cleanup
            max_instances_per_gpu: Maximum model instances per GPU
        """
        self.logger = setup_datacenter_logger('datacenter_model_manager', 'datacenter_model_manager.log')
        self.logger.info("Initializing DatacenterModelManager")
        
        # Configuration
        self.enable_multi_gpu = enable_multi_gpu and torch.cuda.device_count() > 1
        self.memory_threshold = memory_threshold
        self.max_instances_per_gpu = max_instances_per_gpu
        self.health_check_interval = 30.0  # seconds
        
        # GPU discovery and initialization
        self.available_gpus = self._discover_gpus()
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.gpu_locks: Dict[int, threading.Lock] = {}
        
        # Model storage
        self.model_instances: Dict[str, List[DatacenterModelInstance]] = {
            'detection': [],
            'pose_estimation': [],
            'ppe_detection': []
        }
        self.model_type_locks: Dict[str, threading.Lock] = {
            'detection': threading.Lock(),
            'pose_estimation': threading.Lock(),
            'ppe_detection': threading.Lock()
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_inferences': 0,
            'total_inference_time': 0.0,
            'gpu_switches': 0,
            'memory_warnings': 0,
            'model_reloads': 0
        }
        
        # Background threads
        self.health_monitor_thread = None
        self.cleanup_thread = None
        self.running = True
        
        # Load balancing
        self.gpu_workload: Dict[int, float] = {}
        self.last_gpu_assignment: Dict[str, int] = {}
        
        # Initialize system
        self._initialize_gpu_info()
        self._initialize_models()
        self._start_background_threads()
        
        self.logger.info(f"DatacenterModelManager initialized with {len(self.available_gpus)} GPUs")
    
    def _discover_gpus(self) -> List[int]:
        """Discover available GPUs"""
        available_gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Test GPU accessibility
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                    available_gpus.append(i)
                    self.logger.info(f"GPU {i} available: {torch.cuda.get_device_name(i)}")
                except Exception as e:
                    self.logger.warning(f"GPU {i} not accessible: {e}")
        else:
            self.logger.warning("CUDA not available, using CPU")
        return available_gpus
    
    def _initialize_gpu_info(self):
        """Initialize GPU information tracking"""
        for gpu_id in self.available_gpus:
            self.gpu_locks[gpu_id] = threading.Lock()
            self.gpu_workload[gpu_id] = 0.0
            
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                self.gpu_info[gpu_id] = GPUInfo(
                    device_id=gpu_id,
                    name=props.name,
                    memory_total=props.total_memory,
                    memory_allocated=0,
                    memory_reserved=0,
                    utilization=0.0,
                    temperature=None,
                    status=GPUStatus.HEALTHY,
                    last_check=time.time(),
                    error_count=0,
                    models_loaded=0
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize GPU {gpu_id} info: {e}")
    
    def _initialize_models(self):
        """Initialize models on available GPUs"""
        self.logger.info("Loading models on GPUs...")
        
        # Load detection models
        self._load_model_type('detection', Config.DETECTION_MODEL_PATH)
        
        # Load PPE detection if available
        if hasattr(Config, 'PPE_DETECTION_MODEL_PATH') and Config.PPE_DETECTION_MODEL_PATH:
            self._load_model_type('ppe_detection', Config.PPE_DETECTION_MODEL_PATH)
        
        # Load pose estimation if enabled
        if Config.POSE_ENABLED and hasattr(Config, 'POSE_ESTIMATION_MODEL_PATH'):
            self._load_model_type('pose_estimation', Config.POSE_ESTIMATION_MODEL_PATH)
    
    def _load_model_type(self, model_type: str, model_path: str):
        """Load a specific model type on optimal GPUs"""
        if not self.available_gpus:
            self.logger.error(f"No GPUs available for loading {model_type}")
            return
        
        target_instances = min(len(self.available_gpus) * self.max_instances_per_gpu, 4)
        
        for i in range(target_instances):
            optimal_gpu = self._select_optimal_gpu_for_loading()
            if optimal_gpu is None:
                self.logger.warning(f"No optimal GPU found for {model_type} instance {i}")
                continue
                
            result = self._load_model_on_gpu(model_type, model_path, optimal_gpu, f"{model_type}_{optimal_gpu}_{i}")
            
            if result.success:
                self.model_instances[model_type].append(result.model_instance)
                self.gpu_info[optimal_gpu].models_loaded += 1
                self.logger.info(f"Loaded {model_type} instance {i} on GPU {optimal_gpu}")
            else:
                self.logger.error(f"Failed to load {model_type} on GPU {optimal_gpu}: {result.error_message}")
    
    def _load_model_on_gpu(self, model_type: str, model_path: str, gpu_id: int, instance_id: str) -> ModelLoadingResult:
        """Load a model on a specific GPU"""
        start_time = time.time()
        
        try:
            with torch.cuda.device(gpu_id):
                # Clear cache before loading
                torch.cuda.empty_cache()
                
                # Load model
                model = YOLO(model_path)
                model.to(f'cuda:{gpu_id}')
                model.model.eval()
                
                # Create instance
                instance = DatacenterModelInstance(model, model_type, gpu_id, instance_id)
                
                # Perform warmup
                instance.warmup()
                
                load_time = time.time() - start_time
                memory_used = instance.memory_footprint
                
                return ModelLoadingResult(
                    success=True,
                    model_instance=instance,
                    gpu_id=gpu_id,
                    load_time=load_time,
                    memory_used=memory_used,
                    error_message=None
                )
                
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = f"Model loading failed on GPU {gpu_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return ModelLoadingResult(
                success=False,
                model_instance=None,
                gpu_id=gpu_id,
                load_time=load_time,
                memory_used=0,
                error_message=error_msg
            )
    
    def _select_optimal_gpu_for_loading(self) -> Optional[int]:
        """Select optimal GPU for loading new model"""
        if not self.available_gpus:
            return None
        
        best_gpu = None
        best_score = float('inf')
        
        for gpu_id in self.available_gpus:
            if self.gpu_info[gpu_id].status != GPUStatus.HEALTHY:
                continue
                
            if self.gpu_info[gpu_id].models_loaded >= self.max_instances_per_gpu:
                continue
            
            # Calculate score based on memory usage and workload
            memory_usage = self.gpu_info[gpu_id].memory_allocated / self.gpu_info[gpu_id].memory_total
            workload = self.gpu_workload[gpu_id]
            score = memory_usage * 0.7 + workload * 0.3
            
            if score < best_score:
                best_score = score
                best_gpu = gpu_id
        
        return best_gpu
    
    def _select_optimal_instance(self, model_type: str) -> Optional[DatacenterModelInstance]:
        """Select optimal model instance for inference"""
        available_instances = [
            inst for inst in self.model_instances.get(model_type, [])
            if not inst.in_use and self.gpu_info[inst.gpu_id].status == GPUStatus.HEALTHY
        ]
        
        if not available_instances:
            return None
        
        # Select instance with lowest workload GPU
        best_instance = min(
            available_instances,
            key=lambda inst: (
                self.gpu_workload[inst.gpu_id],
                inst.get_average_inference_time(),
                inst.usage_count
            )
        )
        
        return best_instance
    
    def _update_gpu_info(self, gpu_id: int):
        """Update GPU information"""
        try:
            with torch.cuda.device(gpu_id):
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                
                self.gpu_info[gpu_id].memory_allocated = allocated
                self.gpu_info[gpu_id].memory_reserved = reserved
                self.gpu_info[gpu_id].last_check = time.time()
                
                # Check memory threshold
                memory_usage = allocated / self.gpu_info[gpu_id].memory_total
                if memory_usage > self.memory_threshold:
                    self.performance_stats['memory_warnings'] += 1
                    self.logger.warning(f"GPU {gpu_id} memory usage high: {memory_usage:.2%}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update GPU {gpu_id} info: {e}")
            self.gpu_info[gpu_id].error_count += 1
            if self.gpu_info[gpu_id].error_count > 5:
                self.gpu_info[gpu_id].status = GPUStatus.DEGRADED
    
    def _start_background_threads(self):
        """Start background monitoring threads"""
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_worker,
            daemon=True,
            name="gpu_health_monitor"
        )
        self.health_monitor_thread.start()
        
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="gpu_cleanup"
        )
        self.cleanup_thread.start()
    
    def _health_monitor_worker(self):
        """Background thread for GPU health monitoring"""
        while self.running:
            try:
                for gpu_id in self.available_gpus:
                    self._update_gpu_info(gpu_id)
                    
                # Log performance stats periodically
                if self.performance_stats['total_inferences'] % 1000 == 0:
                    self._log_performance_stats()
                    
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(5.0)
    
    def _cleanup_worker(self):
        """Background thread for memory cleanup"""
        while self.running:
            try:
                for gpu_id in self.available_gpus:
                    memory_usage = (
                        self.gpu_info[gpu_id].memory_allocated / 
                        self.gpu_info[gpu_id].memory_total
                    )
                    
                    if memory_usage > self.memory_threshold:
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                            gc.collect()
                        self.logger.debug(f"Cleaned up GPU {gpu_id} memory")
                        
                time.sleep(60.0)  # Run every minute
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
                time.sleep(10.0)
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        avg_inference_time = (
            self.performance_stats['total_inference_time'] / 
            max(1, self.performance_stats['total_inferences'])
        )
        
        self.logger.info(
            f"Performance stats: {self.performance_stats['total_inferences']} inferences, "
            f"avg time: {avg_inference_time:.3f}s, "
            f"GPU switches: {self.performance_stats['gpu_switches']}, "
            f"memory warnings: {self.performance_stats['memory_warnings']}"
        )
    
    async def get_available_model(self, model_type: str) -> Optional[DatacenterModelInstance]:
        """Get an available model instance"""
        if model_type not in self.model_instances:
            self.logger.error(f"Model type {model_type} not available")
            return None
        
        max_wait_time = 10.0  # Maximum wait time in seconds
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            with self.model_type_locks[model_type]:
                instance = self._select_optimal_instance(model_type)
                if instance:
                    with instance.lock:
                        if not instance.in_use:
                            instance.in_use = True
                            self.gpu_workload[instance.gpu_id] += 1.0
                            return instance
            
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"No available {model_type} model after {max_wait_time}s wait")
        return None
    
    def get_model(self, model_type: str) -> Optional[DatacenterModelInstance]:
        """Synchronous version of get_available_model"""
        try:
            return asyncio.run(self.get_available_model(model_type))
        except Exception as e:
            self.logger.error(f"Error getting model of type {model_type}: {str(e)}")
            return None
    
    def _check_memory_usage(self) -> float:
        """
        Check current GPU memory usage
        Returns:
            float: Fraction of GPU memory currently in use
        """
        if not torch.cuda.is_available() or not self.available_gpus:
            return 0.0
        
        # Get memory usage from primary GPU
        primary_gpu = self.available_gpus[0]
        allocated = torch.cuda.memory_allocated(primary_gpu)
        total = self.gpu_info[primary_gpu].memory_total
        return allocated / total

    async def release_model(self, instance: DatacenterModelInstance):
        """Release a model instance back to the pool"""
        with instance.lock:
            instance.in_use = False
            self.gpu_workload[instance.gpu_id] = max(0.0, self.gpu_workload[instance.gpu_id] - 1.0)
            
            # Update performance stats
            self.performance_stats['total_inferences'] += 1
    
    def release_model_sync(self, instance: DatacenterModelInstance):
        """Synchronous version of release_model"""
        try:
            asyncio.run(self.release_model(instance))
        except Exception as e:
            self.logger.error(f"Error releasing model instance: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and GPU information"""
        info = {
            'available_models': list(self.model_instances.keys()),
            'cuda_available': torch.cuda.is_available(),
            'multi_gpu_enabled': self.enable_multi_gpu,
            'total_gpus': len(self.available_gpus),
            'performance_stats': self.performance_stats.copy(),
            'gpus': {}
        }
        
        # Add GPU information
        for gpu_id, gpu_info in self.gpu_info.items():
            info['gpus'][gpu_id] = {
                'name': gpu_info.name,
                'memory_total_gb': gpu_info.memory_total / 1e9,
                'memory_allocated_gb': gpu_info.memory_allocated / 1e9,
                'memory_usage_percent': (gpu_info.memory_allocated / gpu_info.memory_total) * 100,
                'models_loaded': gpu_info.models_loaded,
                'status': gpu_info.status.value,
                'workload': self.gpu_workload[gpu_id],
                'error_count': gpu_info.error_count
            }
        
        # Add instance counts
        for model_type, instances in self.model_instances.items():
            info[f'{model_type}_instances'] = len(instances)
            info[f'{model_type}_in_use'] = sum(1 for inst in instances if inst.in_use)
        
        return info
    
    def get_gpu_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed GPU statistics"""
        stats = {}
        for gpu_id in self.available_gpus:
            self._update_gpu_info(gpu_id)
            gpu_info = self.gpu_info[gpu_id]
            
            stats[gpu_id] = {
                'device_name': gpu_info.name,
                'memory_total': gpu_info.memory_total,
                'memory_allocated': gpu_info.memory_allocated,
                'memory_reserved': gpu_info.memory_reserved,
                'memory_usage_percent': (gpu_info.memory_allocated / gpu_info.memory_total) * 100,
                'models_loaded': gpu_info.models_loaded,
                'workload': self.gpu_workload[gpu_id],
                'status': gpu_info.status.value,
                'error_count': gpu_info.error_count,
                'last_check': gpu_info.last_check
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown model manager and cleanup resources"""
        self.logger.info("Shutting down DatacenterModelManager")
        
        self.running = False
        
        # Wait for background threads
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        # Release all model instances
        for model_type, instances in self.model_instances.items():
            for instance in instances:
                with instance.lock:
                    instance.in_use = False
        
        # Clear CUDA cache on all GPUs
        for gpu_id in self.available_gpus:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error clearing cache on GPU {gpu_id}: {e}")
        
        self.logger.info("DatacenterModelManager shutdown complete")


# Legacy compatibility and factory functions
SingleGPUModelManager = DatacenterModelManager  # For SIB compatibility
DatacenterSingleGPUModelManager = DatacenterModelManager  # Alternative alias
ModelInstance = DatacenterModelInstance

def create_model_manager(**kwargs):
    """Factory function to create a datacenter model manager"""
    return DatacenterModelManager(**kwargs)
    
def get_model_info(self) -> Dict[str, any]:
    """Get information about loaded models"""
    info = {
        'available_models': list(self.model_instances.keys()),
        'cuda_available': torch.cuda.is_available(),
        'memory_usage': self._check_memory_usage() if torch.cuda.is_available() else 0.0,
        'total_gpu_memory_gb': self.total_gpu_memory / 1e9 if torch.cuda.is_available() else 0.0
    }
    
    # Add instance counts
    for model_type, instances in self.model_instances.items():
        info[f'{model_type}_instances'] = len(instances)
        info[f'{model_type}_in_use'] = sum(1 for inst in instances if inst.in_use)
    
    return info

def shutdown(self):
    """Shutdown model manager and cleanup resources"""
    self.logger.info("Shutting down DatacenterSingleGPUModelManager")
    
    # Release all model instances
    for model_type, instances in self.model_instances.items():
        for instance in instances:
            with instance.lock:
                instance.in_use = False
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    self.logger.info("Model manager shutdown complete")


# Legacy compatibility - alias the new class names
SingleGPUModelManager = DatacenterSingleGPUModelManager
ModelInstance = DatacenterModelInstance

# Also create a factory function for backwards compatibility
def create_model_manager(**kwargs):
    """Factory function to create a datacenter model manager"""
    return DatacenterSingleGPUModelManager(**kwargs)









"""
Datacenter Model Manager - Single GPU Model Management

This module handles:
1. Loading and managing AI models for datacenter monitoring
2. Single GPU optimization for multiple camera streams
3. Model instance management and memory optimization
4. Batch inference coordination
5. Model lifecycle management
"""
'''
import torch
from ultralytics import YOLO
from typing import Dict, Optional, List
import threading
import asyncio

from logger import setup_datacenter_logger
from config import Config

class DatacenterModelInstance:
    """Model instance wrapper for datacenter monitoring"""
    def __init__(self, model: YOLO, model_type: str):
        self.model = model
        self.model_type = model_type
        self.in_use = False
        self.lock = threading.Lock()
        # track memory usage for this instance
        # self.initial_memory = torch.cuda.memory_allocated()

class DatacenterSingleGPUModelManager:
    """Model manager optimized for single GPU datacenter monitoring"""
    
    def __init__(self, memory_threshold: float = 0.85):
        """
        Initialize model manager for single GPU datacenter monitoring
        Args:
            memory_threshold: GPU memory threshold for cleanup
        """
        self.logger = setup_datacenter_logger('datacenter_gpu_model_manager', 'datacenter_gpu_model_manager.log')
        self.logger.info("Initializing DatacenterSingleGPUModelManager")

        self.memory_threshold = memory_threshold
        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

        self.model_instances: Dict[str, List[DatacenterModelInstance]] = {
            'detection': [],
            'pose_estimation': []
        }
        self.model_locks = {
            'detection': threading.Lock(),
            'pose_estimation': threading.Lock()
        }
        self.initialize_models()
    
    def _check_memory_usage(self) -> float:
        """
        Check current GPU memory usage
        Returns:
            float: Fraction of GPU memory currently in use
        """
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        return allocated / self.total_gpu_memory
    
    def initialize_models(self):
        """Initialize multiple instances of each model type on single GPU"""
        try:
            # Ensure CUDA is available
            if not torch.cuda.is_available():
                self.logger.warning("CUDA is not available, using CPU")
                device = 'cpu'
            else:
                device = 'cuda'

            # Initialize models with memory management
            with torch.cuda.device(0) if torch.cuda.is_available() else torch.device('cpu'):
                # Detection model
                detection_model = YOLO(Config.DETECTION_MODEL_PATH)
                detection_model.to(device)
                self.model_instances['detection'].append(
                    DatacenterModelInstance(detection_model, 'detection')
                )
                self.logger.info(f"Detection Model initialized on {device}")
                
                # PPE detection model (if different from main detection)
                if hasattr(Config, 'PPE_DETECTION_MODEL_PATH') and Config.PPE_DETECTION_MODEL_PATH:
                    try:
                        ppe_model = YOLO(Config.PPE_DETECTION_MODEL_PATH)
                        ppe_model.to(device)
                        self.model_instances['ppe_detection'] = [DatacenterModelInstance(ppe_model, 'ppe_detection')]
                        self.logger.info(f"PPE Detection Model initialized on {device}")
                    except Exception as e:
                        self.logger.warning(f"PPE detection model not available: {e}")
                
                # Pose estimation model (if enabled)
                if Config.POSE_ENABLED:
                    try:
                        pose_estimation_model = YOLO(Config.POSE_ESTIMATION_MODEL_PATH)
                        # Explicitly move model to GPU and set inference mode
                        pose_estimation_model.to(device)
                        pose_estimation_model.model.eval()  # Set to evaluation mode
                        # Log device placement
                        self.logger.info(f"Pose Estimation Model device: {next(pose_estimation_model.model.parameters()).device}")
                        self.model_instances['pose_estimation'].append(
                            DatacenterModelInstance(pose_estimation_model, 'pose_estimation')
                        )
                        self.logger.info(f"Pose estimation Model initialized on {device}")
                    except Exception as e:
                        self.logger.warning(f"Pose estimation model not available: {e}")
                        
            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                self.logger.info(f"GPU Memory: Allocated={allocated/1e9:.2f}GB, Reserved={reserved/1e9:.2f}GB")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}", exc_info=True)
            raise
    
    def _log_gpu_stats(self):
        """Log GPU memory usage statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            self.logger.debug(f"GPU Memory: Allocated={allocated/1e9:.2f}GB, Reserved={reserved/1e9:.2f}GB")

    async def get_available_model(self, model_type: str) -> Optional[DatacenterModelInstance]:
        """Get an available model instance of the specified type"""
        if model_type not in self.model_instances:
            self.logger.error(f"Model type {model_type} not available")
            return None
            
        while True:
            with self.model_locks[model_type]:
                for instance in self.model_instances[model_type]:
                    with instance.lock:
                        if not instance.in_use:
                            instance.in_use = True
                            # Log GPU stats when model is acquired
                            self._log_gpu_stats()
                            # Verify model is on correct device
                            if torch.cuda.is_available():
                                device = next(instance.model.model.parameters()).device
                                self.logger.debug(f"Model {model_type} running on: {device}")
                            return instance
            # If no instance is available, wait briefly before checking again
            await asyncio.sleep(0.1)
    
    def get_model(self, model_type: str) -> Optional[DatacenterModelInstance]:
        """Synchronous version of get_available_model"""
        try:
            # Use asyncio.run to call the async method
            return asyncio.run(self.get_available_model(model_type))
        except Exception as e:
            self.logger.error(f"Error getting model of type {model_type}: {str(e)}")
            return None
    
    async def release_model(self, instance: DatacenterModelInstance):
        """Release a model instance back to the pool"""
        with instance.lock:
            instance.in_use = False
            # Clear CUDA cache if memory usage is high
            if torch.cuda.is_available():
                current_usage = self._check_memory_usage()
                if current_usage > self.memory_threshold:
                    torch.cuda.empty_cache()
                    self.logger.debug(f"GPU cache cleared due to high memory usage: {current_usage:.2%}")
        self.logger.debug(f"Released {instance.model_type} model instance")
    
    def release_model_sync(self, instance: DatacenterModelInstance):
        """Synchronous version of release_model"""
        try:
            asyncio.run(self.release_model(instance))
        except Exception as e:
            self.logger.error(f"Error releasing model instance: {str(e)}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded models"""
        info = {
            'available_models': list(self.model_instances.keys()),
            'cuda_available': torch.cuda.is_available(),
            'memory_usage': self._check_memory_usage() if torch.cuda.is_available() else 0.0,
            'total_gpu_memory_gb': self.total_gpu_memory / 1e9 if torch.cuda.is_available() else 0.0
        }
        
        # Add instance counts
        for model_type, instances in self.model_instances.items():
            info[f'{model_type}_instances'] = len(instances)
            info[f'{model_type}_in_use'] = sum(1 for inst in instances if inst.in_use)
        
        return info
    
    def shutdown(self):
        """Shutdown model manager and cleanup resources"""
        self.logger.info("Shutting down DatacenterSingleGPUModelManager")
        
        # Release all model instances
        for model_type, instances in self.model_instances.items():
            for instance in instances:
                with instance.lock:
                    instance.in_use = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Model manager shutdown complete")


# Legacy compatibility - alias the new class names
SingleGPUModelManager = DatacenterSingleGPUModelManager
ModelInstance = DatacenterModelInstance

# Also create a factory function for backwards compatibility
def create_model_manager(**kwargs):
    """Factory function to create a datacenter model manager"""
    return DatacenterSingleGPUModelManager(**kwargs)

'''
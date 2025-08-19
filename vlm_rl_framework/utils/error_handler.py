"""
错误处理工具模块
"""

import logging
import traceback
import time
from typing import Dict, Any, Callable, Optional
from functools import wraps
import psutil
import torch

logger = logging.getLogger(__name__)

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_count = 0
        self.error_history = []
    
    def handle_vlm_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """处理VLM相关错误"""
        self.error_count += 1
        error_info = {
            "type": "VLM_ERROR",
            "error": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        self.error_history.append(error_info)
        
        logger.error(f"VLM错误 ({context}): {error}")
        
        # 返回备用结果
        return {
            "success": False,
            "error": str(error),
            "fallback_result": {
                "object_type": "unknown",
                "position": [320, 240],
                "confidence": 0.0
            }
        }
    
    def handle_rl_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """处理RL相关错误"""
        self.error_count += 1
        error_info = {
            "type": "RL_ERROR",
            "error": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        self.error_history.append(error_info)
        
        logger.error(f"RL错误 ({context}): {error}")
        
        return {
            "success": False,
            "error": str(error),
            "fallback_action": None
        }
    
    def handle_environment_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """处理环境相关错误"""
        self.error_count += 1
        error_info = {
            "type": "ENVIRONMENT_ERROR",
            "error": str(error),
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        self.error_history.append(error_info)
        
        logger.error(f"环境错误 ({context}): {error}")
        
        return {
            "success": False,
            "error": str(error)
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return {
            "total_errors": self.error_count,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "error_types": self._count_error_types()
        }
    
    def _count_error_types(self) -> Dict[str, int]:
        """统计错误类型"""
        error_types = {}
        for error in self.error_history:
            error_type = error.get("type", "UNKNOWN")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"函数 {func.__name__} 第{attempt+1}次尝试失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            
            logger.error(f"函数 {func.__name__} 在{max_retries}次尝试后仍然失败")
            raise last_error
        return wrapper
    return decorator

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "timestamps": []
        }
    
    def record_metrics(self):
        """记录性能指标"""
        if self.start_time is None:
            return
        
        timestamp = time.time() - self.start_time
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU使用率（如果可用）
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
            except:
                pass
        
        self.metrics["cpu_usage"].append(cpu_percent)
        self.metrics["memory_usage"].append(memory_percent)
        self.metrics["gpu_usage"].append(gpu_percent)
        self.metrics["timestamps"].append(timestamp)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics["timestamps"]:
            return {}
        
        return {
            "duration": self.metrics["timestamps"][-1],
            "avg_cpu": sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"]),
            "max_cpu": max(self.metrics["cpu_usage"]),
            "avg_memory": sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]),
            "max_memory": max(self.metrics["memory_usage"]),
            "avg_gpu": sum(self.metrics["gpu_usage"]) / len(self.metrics["gpu_usage"]),
            "max_gpu": max(self.metrics["gpu_usage"])
        }
    
    def plot_performance(self, save_path: str = None):
        """绘制性能图表"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
            
            # CPU使用率
            ax1.plot(self.metrics["timestamps"], self.metrics["cpu_usage"])
            ax1.set_ylabel("CPU使用率 (%)")
            ax1.set_title("性能监控")
            
            # 内存使用率
            ax2.plot(self.metrics["timestamps"], self.metrics["memory_usage"])
            ax2.set_ylabel("内存使用率 (%)")
            
            # GPU使用率
            ax3.plot(self.metrics["timestamps"], self.metrics["gpu_usage"])
            ax3.set_ylabel("GPU使用率 (%)")
            ax3.set_xlabel("时间 (秒)")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"性能图表已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制性能图表")

# 全局错误处理器和性能监控器
error_handler = ErrorHandler()
performance_monitor = PerformanceMonitor()

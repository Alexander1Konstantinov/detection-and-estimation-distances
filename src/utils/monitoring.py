# src/utils/performance.py
import time
import statistics
from datetime import datetime

class PerformanceMonitor:
    """Мониторинг производительности системы"""
    
    def __init__(self):
        self.inference_times = []
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def start_frame(self):
        """Начало обработки кадра"""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """Завершение обработки кадра"""
        if self.frame_count == 0:
            return
        processing_time = time.time() - self.frame_start_time
        self.inference_times.append(processing_time)
        self.frame_count += 1
        
        current_time = time.time()
        if current_time - self.last_log_time >= 5:
            self._log_performance()
            self.last_log_time = current_time
        
        return processing_time
    
    def _log_performance(self):
        """Логирование статистики производительности"""
        if not self.inference_times:
            return
            
        avg_time = statistics.mean(self.inference_times)
        max_time = max(self.inference_times)
        fps = 1 / avg_time if avg_time > 0 else 0
        
        print(f"[Performance] FPS: {fps:.1f}, Avg: {avg_time*1000:.1f}ms, "
              f"Max: {max_time*1000:.1f}ms, Frames: {self.frame_count}")
        
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-500:]
    
    def get_stats(self):
        """Получение текущей статистики"""
        if not self.inference_times:
            return 0, 0, 0, 0
            
        avg_time = statistics.mean(self.inference_times)
        max_time = max(self.inference_times)
        min_time = min(self.inference_times)
        fps = 1 / avg_time if avg_time > 0 else 0
        
        return avg_time, max_time, min_time, fps
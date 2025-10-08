import unittest
import time
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_detector import YOLODetector
from src.pipeline.processor import VideoProcessor


class TestPerformance(unittest.TestCase):
    """Performance-тесты для проверки скорости работы."""
    
    def setUp(self) -> None:
        """Инициализация перед каждым тестом."""
        self.detector = YOLODetector()
        self.processor = VideoProcessor()
        
        self.test_images = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),   
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),   
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),  
        ]
    
    def measure_inference_time(self, image: np.ndarray, warmup_runs: int = 3, test_runs: int = 10) -> float:
        """
        Измерение времени инференса для одного изображения.
        
        Args:
            image: Тестовое изображение
            warmup_runs: Количество прогонов для разогрева
            test_runs: Количество прогонов для измерения
            
        Returns:
            float: Среднее время инференса в миллисекундах
        """
        
        for _ in range(warmup_runs):
            _ = list(self.detector.model(image, verbose=False))
        
        
        inference_times: List[float] = []
        for _ in range(test_runs):
            start_time = time.perf_counter()
            _ = list(self.detector.model(image, verbose=False))
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000) 
        
        return sum(inference_times) / len(inference_times)
    
    def test_inference_time_single_image(self) -> None:
        """Тест времени инференса для одного изображения."""
        image = self.test_images[1]  
        
        inference_time = self.measure_inference_time(image)
        
        print(f"\nСреднее время инференса: {inference_time:.2f} мс")
        
        self.assertLess(
            inference_time, 
            100.0,
            f"Время инференса {inference_time:.2f} мс превышает лимит 100 мс"
        )
        
        self.assertGreater(inference_time, 5.0, "Время инференса подозрительно мало")
    
    def test_inference_time_different_resolutions(self) -> None:
        """Тест времени инференса для изображений разных разрешений."""
        results: List[Tuple[Tuple[int, int], float]] = []
        
        for image in self.test_images:
            height, width = image.shape[:2]
            inference_time = self.measure_inference_time(image)
            results.append(((width, height), inference_time))
            
            print(f"Разрешение {width}x{height}: {inference_time:.2f} мс")
            
            self.assertLess(
                inference_time, 
                100.0,
                f"Время инференса для {width}x{height} превышает лимит"
            )
        
        for i in range(1, len(results)):
            self.assertGreaterEqual(
                results[i][1], 
                results[i-1][1] * 0.5, 
                "Время инференса не увеличивается с ростом разрешения???"
            )
    
    def test_video_processing_performance(self) -> None:
        """Тест производительности обработки короткого видео."""
        test_video_path = Path(__file__).parent / "data" / "performance_test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (640, 480))
        
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        try:
            start_time = time.perf_counter()
            self.processor.process_video(str(test_video_path), verbose=False)
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000  
            avg_time_per_frame = total_time / 10  
            
            print(f"\nОбщее время обработки 10 кадров: {total_time:.2f} мс")
            print(f"Среднее время на кадр: {avg_time_per_frame:.2f} мс")
            
            self.assertLess(
                avg_time_per_frame,
                100.0,
                f"Среднее время обработки кадра {avg_time_per_frame:.2f} мс превышает лимит"
            )
            
        finally:
            if test_video_path.exists():
                test_video_path.unlink()
    
    def test_memory_usage(self) -> None:
        """Тест использования памяти (качественная проверка)."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024 
        
        for _ in range(5):
            _ = list(self.detector.model(self.test_images[1], verbose=False))
        
        final_memory = process.memory_info().rss / 1024 / 1024 
        memory_increase = final_memory - initial_memory
        
        print(f"\nИспользование памяти:")
        print(f"Начальное: {initial_memory:.2f} МБ")
        print(f"Конечное: {final_memory:.2f} МБ")
        print(f"Увеличение: {memory_increase:.2f} МБ")
        
        self.assertLess(
            memory_increase,
            50.0,
            f"Слшиком большое увеличение использования памяти: {memory_increase:.2f} МБ"
        )


class TestPerformanceBenchmarks(unittest.TestCase):
    """Бенчмарки производительности для различных сценариев."""
    
    def setUp(self) -> None:
        self.detector = YOLODetector()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_batch_processing_performance(self) -> None:
        """Тест производительности при обработке батча изображений."""
        batch_size = 4
        batch = [self.test_image.copy() for _ in range(batch_size)]
        
        start_time = time.perf_counter()
        for image in batch:
            _ = list(self.detector.model(image, verbose=False))
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        avg_time_per_image = total_time / batch_size
        
        print(f"\nБатч из {batch_size} изображений:")
        print(f"Общее время: {total_time:.2f} мс")
        print(f"Среднее время на изображение: {avg_time_per_image:.2f} мс")
        
        self.assertLess(avg_time_per_image, 100.0)


if __name__ == "__main__":
    unittest.main()
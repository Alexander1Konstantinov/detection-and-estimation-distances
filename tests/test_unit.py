import unittest
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_detector import YOLODetector
from src.models.distance_estimator import DistanceEstimator
from src.utils.visualization import draw_detections, draw_performance_stats


class TestYOLODetector(unittest.TestCase):
    """Unit-тесты для YOLO детектора."""
    
    def setUp(self) -> None:
        """Инициализация перед каждым тестом."""
        self.detector = YOLODetector()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self) -> None:
        """Тест инициализации детектора."""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.model.names)
        self.assertIsInstance(self.detector.model.names, dict)
    
    def test_detector_on_test_image(self) -> None:
        """Тест детекции на тестовом изображении."""
        
        results = list(self.detector.model(self.test_image, verbose=False))
        
        
        self.assertGreater(len(results), 0)
        
        detections = results[0]
        
        
        self.assertTrue(hasattr(detections, 'boxes'))
        self.assertTrue(hasattr(detections, 'names'))
        self.assertTrue(hasattr(detections, 'speed'))
        
        
        if len(detections.boxes) > 0:
            box = detections.boxes[0]
            self.assertTrue(hasattr(box, 'xyxy'))
            self.assertTrue(hasattr(box, 'cls'))
            self.assertTrue(hasattr(box, 'conf'))
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)


class TestDistanceEstimator(unittest.TestCase):
    """Unit-тесты для оценщика расстояния."""
    
    def setUp(self) -> None:
        """Инициализация перед каждым тестом."""
        self.estimator = DistanceEstimator()
    
    def test_estimator_initialization(self) -> None:
        """Тест инициализации оценщика расстояния."""
        self.assertIsNotNone(self.estimator.known_widths)
        self.assertIsNotNone(self.estimator.focal_length)
        self.assertIsInstance(self.estimator.known_widths, dict)
        self.assertIsInstance(self.estimator.focal_length, int)
    
    def test_distance_estimation(self) -> None:
        """Тест оценки расстояния для различных классов."""
        test_cases = [
            ("person", 100, 200),  
            ("car", 150, 350),     
        ]
        
        for class_name, x1, x2 in test_cases:
            with self.subTest(class_name=class_name):
                distance = self.estimator.estimate(class_name, x1, x2)
                
                self.assertIsInstance(distance, float)
                self.assertGreater(distance, 0)
                
                pixel_width = x2 - x1
                known_width = self.estimator.known_widths.get(class_name, 0.5)
                expected_distance = (known_width * self.estimator.focal_length) / pixel_width
                
                self.assertAlmostEqual(distance, expected_distance, places=2)
    
    def test_unknown_class_default_behavior(self) -> None:
        """Тест поведения для неизвестных классов."""
        distance = self.estimator.estimate("unknown_class", 100, 200)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)


class TestVisualization(unittest.TestCase):
    """Unit-тесты для функций визуализации."""
    
    def setUp(self) -> None:
        """Инициализация перед каждым тестом."""
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_draw_performance_stats(self) -> None:
        """Тест отрисовки статистики производительности."""
        original_frame = self.test_frame.copy()
        processed_frame = draw_performance_stats(
            self.test_frame, 
            45.5,  
            2      
        )
        
        self.assertIsInstance(processed_frame, np.ndarray)
        
        self.assertEqual(processed_frame.shape, original_frame.shape)


if __name__ == "__main__":
    unittest.main()
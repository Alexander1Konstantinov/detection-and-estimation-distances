import unittest
import cv2
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.processor import VideoProcessor


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты полного пайплайна."""
    
    def setUp(self) -> None:
        """Инициализация перед каждым тестом."""
        self.processor = VideoProcessor()
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = Path(self.temp_dir.name) / "test_video.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, 10.0, (640, 480))
        
        for _ in range(5):  # 5 кадров
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    def tearDown(self) -> None:
        """Очистка после каждого теста."""
        self.temp_dir.cleanup()
    
    def test_full_pipeline_video_processing(self) -> None:
        """Тест полного пайплайна обработки видео."""
        output_path = Path(self.temp_dir.name) / "output_video.avi"
        
        self.processor.process_video(
            video_path=str(self.test_video_path),
            output_path=str(output_path),
            verbose=False
        )
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
        
        cap = cv2.VideoCapture(str(output_path))
        self.assertTrue(cap.isOpened())
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertGreater(frame_count, 0)
        
        cap.release()
    
    def test_pipeline_with_different_settings(self) -> None:
        """Тест пайплайна с различными настройками."""
        test_cases = [
            {"verbose": True},
            {"verbose": False},
        ]
        
        for settings in test_cases:
            with self.subTest(settings=settings):
                output_path = Path(self.temp_dir.name) / f"output_{hash(str(settings))}.avi"
                
                try:
                    self.processor.process_video(
                        video_path=str(self.test_video_path),
                        output_path=str(output_path),
                        **settings
                    )
                    
                    self.assertTrue(output_path.exists())
                    
                except Exception as e:
                    self.fail(f"Пайплайн упал с настройками {settings}: {e}")


if __name__ == "__main__":
    unittest.main()
import pytest
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Any
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.processor import VideoProcessor
from src.models.yolo_detector import YOLODetector


@pytest.fixture
def test_image_path() -> Path:
    """Путь к тестовому изображению."""
    path = Path(__file__).parent / "data" / "test_image.jpg"
    if not path.exists():
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(path), test_image)
    return path


@pytest.fixture
def sample_video_path() -> Path:
    """Путь к тестовому видео (создаем искусственное)."""
    path = Path(__file__).parent / "data" / "test_video.avi"
    if not path.exists():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(path), fourcc, 10.0, (640, 480))
        for _ in range(30):  # 3 секунды видео
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    return path


@pytest.fixture
def yolo_detector() -> Generator[YOLODetector, None, None]:
    """Фикстура для инициализации детектора."""
    detector = YOLODetector()
    yield detector


@pytest.fixture
def video_processor() -> Generator[VideoProcessor, None, None]:
    """Фикстура для инициализации видео процессора."""
    processor = VideoProcessor()
    yield processor

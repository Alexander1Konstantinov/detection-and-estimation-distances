import cv2
import numpy as np
from config.settings import DistanceConfig
from ..models.distance_estimator import DistanceEstimator

def draw_detections(frame, detections, id2name):
    """
    Отрисовка bounding boxes и информации на кадре
    
    Args:
        frame: исходный кадр
        detections: список обнаружений
        
    Returns:
        frame: кадр с отрисованными детекциями
    """
    estimator = DistanceEstimator()
    
    for detection in detections:
        boxes = detection.boxes
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_id = int(box.cls[0])
            class_name = id2name[cls_id]
            conf = box.conf[0]

            distance = estimator.estimate(class_name, x1, x2)

            label = f"{class_name} {conf:.2f} {distance:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=2,
            )
    
    return frame

def draw_performance_stats(frame, cur_time, detections_count):
    """Отрисовка статистики производительности"""
    stats_text = f"Time per frame: {cur_time:.1f}ms || Objects: {detections_count}"
    cv2.putText(frame, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
    if cur_time > 100:
        warning_text = "LOW PERFORMANCE!"
        cv2.putText(frame, warning_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame
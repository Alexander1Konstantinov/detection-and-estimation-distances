import cv2
from src.models.yolo_detector import YOLODetector
from src.models.distance_estimator import DistanceEstimator
from src.utils.visualization import draw_detections, draw_performance_stats


class VideoProcessor:

    def __init__(self):
        self.detector = YOLODetector()
        self.distance_estimator = DistanceEstimator()

    def process_video(self, video_path: str, verbose: bool = False) -> None:

        cap = cv2.VideoCapture(video_path)
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )

        video_writer = cv2.VideoWriter(
            "detected_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        cap.release()
        for detections in self.detector.model(
            video_path,
            stream=True,
            device="cpu",
            verbose=verbose,
            show_labels=False,
            show_conf=False,
            show_boxes=False,
        ):

            frame = detections.plot(boxes=False, labels=False)

            processed_frame = draw_detections(
                frame, detections, self.detector.model.names
            )

            cur_time = sum(detections.speed.values())
            processed_frame = draw_performance_stats(
                processed_frame, cur_time, len(detections)
            )
            video_writer.write(processed_frame)

        del video_writer

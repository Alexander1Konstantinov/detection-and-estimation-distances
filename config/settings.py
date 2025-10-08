class ModelConfig:
    """Настройки модели детекции"""

    MODEL_PATH = "yolo11n.pt"
    QUANTIZATION = True
    IMG_SIZE = 320


class DistanceConfig:
    KNOWN_WIDTHS = {
        "person": 0.5,
        "car": 1.8,
        "bicycle": 0.7,
        "motorcycle": 0.8,
        "bus": 2.5,
        "truck": 2.2,
    }
    FOCAL_LENGTH = 800


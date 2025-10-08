import os
from ultralytics import YOLO
from config.settings import ModelConfig

class YOLODetector:
    
    def __init__(self):
        if ModelConfig.QUANTIZATION:
            if not os.path.exists(ModelConfig.MODEL_PATH.replace('.pt', '') + '.onnx'):
                YOLO('yolo11n.pt').export(format='onnx', imgsz=ModelConfig.IMG_SIZE, half=False)
            self.model = YOLO(ModelConfig.MODEL_PATH.replace('.pt', '') + '.onnx', task='detect')    
        else:
            self.model = YOLO(ModelConfig.MODEL_PATH, task='detect')
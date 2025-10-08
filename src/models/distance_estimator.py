import numpy as np
from config.settings import DistanceConfig


class DistanceEstimator:

    def __init__(self):
        self.known_widths = DistanceConfig.KNOWN_WIDTHS
        self.focal_length = DistanceConfig.FOCAL_LENGTH
        # self.alert_distance = DistanceConfig.ALERT_DISTANCE

    def estimate(self, class_name, x1, x2):
        """
        Оценка расстояния до объекта

        Args:
            detection: словарь с информацией об обнаружении

        Returns:
            float: расстояние в метрах или None если нельзя вычислить
        """
        pixel_width = x2 - x1

        known_width = self.known_widths.get(class_name, 0.5)

        distance = (known_width * self.focal_length) / pixel_width

        return distance
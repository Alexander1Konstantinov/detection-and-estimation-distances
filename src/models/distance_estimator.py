from config.settings import DistanceConfig


class DistanceEstimator:

    def __init__(self):
        self.known_widths = DistanceConfig.KNOWN_WIDTHS
        self.focal_length = DistanceConfig.FOCAL_LENGTH

    def estimate(self, class_name: str, x1: int, x2: int) -> float:
        """
        Оценка расстояния до объекта

        Args:
            detection: словарь с информацией об обнаружении

        Returns:
            float: расстояние в метрах
        """
        pixel_width = x2 - x1

        known_width = self.known_widths.get(class_name, 0.5)

        distance = (known_width * self.focal_length) / pixel_width

        return distance

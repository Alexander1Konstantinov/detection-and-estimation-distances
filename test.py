import cv2
from ultralytics import YOLO
import math

# Загрузка модели YOLO (например, YOLO11n)
model = YOLO("yolo11n.pt")

# Известные реальные ширины объектов в метрах (примерные значения)
KNOWN_WIDTHS = {
    "person": 0.5,  # Ширина человека в плечах
    "car": 1.8,  # Средняя ширина автомобиля
    "bottle": 0.07,  # Ширина бутылки
}

# Фокусное расстояние (требует калибровки для вашей камеры!)
FOCAL_LENGTH = 800  # в пикселях


def calculate_distance(known_width, focal_length, pixel_width):
    """Вычисляет расстояние до объекта по известной ширине."""
    return (known_width * focal_length) / pixel_width


# Захват видео с камеры
cap = cv2.VideoCapture("data/test_video_short.mp4")
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    "distance_output_test.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("not success")
        break

    # Детекция объектов с помощью YOLO
    results = model(frame, save=True, verbose=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Координаты bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Класс объекта и уверенность
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = box.conf[0]

            # Пропускаем объекты с низкой уверенностью
            # if conf < 0.5:
            #     continue

            # Вычисляем ширину bbox в пикселях
            pixel_width = x2 - x1

            # Получаем известную ширину для данного класса
            if class_name in KNOWN_WIDTHS:
                known_width = KNOWN_WIDTHS[class_name]
                # Вычисляем расстояние
            else:
                known_width = 1
            distance = calculate_distance(known_width, FOCAL_LENGTH, pixel_width)

            # Отрисовка bbox и информации
            label = f"{class_name} {distance:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )

    # Показ результата
    cv2.imshow("YOLO Distance Estimation", frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

class FaceDetector:
    def __init__(self, model_type='haar'):

        self.model_type = model_type

        if model_type == 'haar':
            # Загрузка каскада Хаара для обнаружения лиц
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif model_type == 'dnn':
            # Загрузка предобученной DNN модели
            self.net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt.txt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
        else:
            raise ValueError("Неизвестный тип модели. Используйте 'haar' или 'dnn'")

    def detect_faces(self, frame):

        if self.model_type == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        else:
            # Подготовка изображения для DNN модели
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )
            
            # Подача на вход сети
            self.net.setInput(blob)
            detections = self.net.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Фильтрация по порогу уверенности
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX-startX, endY-startY))

            return faces

    def draw_faces(self, frame, faces):

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Добавляем текст с информацией
            cv2.putText(frame, 'Face', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


def main():
    # Инициализация детектора (можно выбрать 'haar' или 'dnn')
    detector = FaceDetector(model_type='haar')

    # Инициализация видеопотока (0 - встроенная камера)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обнаружение лиц
        faces = detector.detect_faces(frame)

        # Отрисовка результатов
        frame_with_faces = detector.draw_faces(frame, faces)

        # Отображение FPS
        cv2.putText(frame_with_faces, f'Faces detected: {len(faces)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Вывод результата
        cv2.imshow('Face Detection', frame_with_faces)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
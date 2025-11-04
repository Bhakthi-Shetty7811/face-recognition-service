import cv2

class Detector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x+w, y+h])
        return boxes

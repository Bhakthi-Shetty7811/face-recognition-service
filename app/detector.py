from insightface.app import FaceAnalysis

class Detector:

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def detect(self, image):
        faces = self.app.get(image)
        boxes = [f.bbox.astype(int).tolist() for f in faces]
        return boxes
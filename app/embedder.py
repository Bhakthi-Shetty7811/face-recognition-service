import os
import urllib.request
import onnxruntime as ort
import numpy as np
import cv2

MODEL_URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true"
MODEL_PATH = "weights/arcface.onnx"


def download_model():
    os.makedirs("weights", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading ArcFace model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")


class Embedder:
    def __init__(self, model_path=MODEL_PATH):
        download_model() 

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_img):
        inp = self.preprocess(face_img)
        emb = self.session.run([self.output_name], {self.input_name: inp})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb
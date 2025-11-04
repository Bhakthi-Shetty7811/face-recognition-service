from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from .detector import Detector
from .aligner import align_face
from .embedder import Embedder
from .matcher import Matcher
from .db import DB
import io

app = FastAPI(title="Face Recognition Service", version="1.0")

detector = Detector()
embedder = Embedder("weights/arcface_resnet100.onnx")
db = DB("data/gallery.db")
matcher = Matcher(db)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    boxes = detector.detect(img)
    results = []
    for box in boxes:
        results.append([int(x) for x in box])

    return {"boxes": results}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...), top_k: int = 1):
    image = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    boxes = detector.detect(img)
    results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = img[y1:y2, x1:x2]
        face = align_face(face)
        emb = embedder.get_embedding(face)
        match = matcher.search(emb, top_k=top_k)
        results.append({"box": [int(x) for x in box], "match": match})
    return {"results": results}

@app.post("/add_identity")
async def add_identity(name: str = Form(...), file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    boxes = detector.detect(img)
    if not boxes:
        return JSONResponse({"error": "No face detected"}, status_code=400)
    x1, y1, x2, y2 = map(int, boxes[0])
    face = img[y1:y2, x1:x2]
    face = align_face(face)
    emb = embedder.get_embedding(face)
    db.add_identity(name, emb)
    return {"status": "added", "name": name}

@app.get("/list_identities")
def list_identities():
    return {"identities": db.list_identities()}

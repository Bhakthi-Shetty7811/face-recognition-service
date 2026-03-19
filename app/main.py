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
from fastapi import HTTPException
from .utils import read_image_from_bytes
import logging
import os
import redis
import hashlib
import json
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

redis_client = redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition Service", version="1.0")
Instrumentator().instrument(app).expose(app)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

detector = None
embedder = None
db = None
matcher = None

@app.on_event("startup")
def load_models():
    global detector, embedder, db, matcher

    logger.info("Loading models...")

    embedder = Embedder() 
    detector = Detector()
    db = DB()
    matcher = Matcher(db)

    logger.info("All models loaded successfully")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    logger.info("Detect request received")
    img = read_image_from_bytes(await file.read())
    if img is None:
       raise HTTPException(status_code=400, detail="Invalid image")
    boxes = detector.detect(img)
    logger.info("Detected %d faces", len(boxes))
    results = []
    for box in boxes:
        results.append([int(x) for x in box])
    return {
       "success": True,
       "data": {"boxes": results}
    }    

@app.post("/recognize")
@limiter.limit("10/minute")
async def recognize(request: Request, file: UploadFile = File(...), top_k: int = 1):
    logger.info("Recognition request received")
    img_bytes = await file.read()
    hash_key = hashlib.md5(img_bytes).hexdigest()
    cached = cache.get(hash_key)
    if cached:
        logger.info("Cache hit for image")
        return json.loads(cached)
    img = read_image_from_bytes(img_bytes)
    if img is None:
        logger.warning("Invalid image uploaded")
        raise HTTPException(status_code=400, detail="Invalid image")
    boxes = detector.detect(img)
    logger.info("Detected %d faces", len(boxes))
    results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        face = img[y1:y2, x1:x2]
        face = align_face(face)
        emb = embedder.get_embedding(face)
        logger.debug("Embedding generated")
        match = matcher.search(emb, top_k=top_k)
        logger.info("Match result: %s", match)
        results.append({
            "box": [x1, y1, x2, y2],
            "match": match
        })
    response = {
        "success": True,
        "data": {"results": results}
    }
    cache.set(hash_key, json.dumps(response), ex=60)
    logger.info("Response cached")
    return response

@app.post("/add_identity")
async def add_identity(name: str = Form(...), file: UploadFile = File(...)):
    logger.info("Add identity request for %s", name)
    img = read_image_from_bytes(await file.read())
    if img is None:
       raise HTTPException(status_code=400, detail="Invalid image")
    boxes = detector.detect(img)
    if not boxes:
        return JSONResponse({"error": "No face detected"}, status_code=400)
    x1, y1, x2, y2 = map(int, boxes[0])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2) 
    face = img[y1:y2, x1:x2]
    face = align_face(face)
    emb = embedder.get_embedding(face)
    db.add_identity(name, emb)
    logger.info("Identity %s added to database", name)
    return {
        "success": True,
        "status": "added", 
        "name": name
    }

@app.get("/list_identities")
def list_identities():
    return {
        "success": True,
        "identities": db.list_identities()
    }

@app.get("/health")
def health():
    return {"status": "ok"}
import time, cv2, numpy as np
from app.embedder import Embedder

model = Embedder("weights/arcface_resnet100.onnx")
img = cv2.imread("sample.jpg")
img = cv2.resize(img, (112,112))
# Warm-up
for _ in range(5):
    model.get_embedding(img)

times = []
for _ in range(50):
    t1 = time.perf_counter()
    model.get_embedding(img)
    times.append(time.perf_counter()-t1)

avg = np.mean(times)
fps = 1/avg
print(f"Avg latency: {avg*1000:.2f} ms,  Throughput: {fps:.2f} FPS")

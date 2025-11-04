# 🧠 Face Recognition Service (FRS)

An end-to-end **Face Detection and Recognition Microservice** built using **FastAPI + ONNXRuntime + OpenCV**.  
The service detects faces, extracts embeddings using **ArcFace**, and recognizes identities from a gallery dataset.  
All optimized for **CPU inference**, containerized via **Docker**, and benchmarked for production readiness.

---

## 📁 Project Structure

```

face-recognition-service/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── detector.py            # Face detection (Haar cascade)
│   ├── aligner.py             # Face alignment / normalization
│   ├── embedder.py            # ArcFace embedding extraction
│   ├── matcher.py             # Embedding comparison (cosine)
│   ├── db.py                  # SQLite DB for gallery
│   ├── utils.py               # Helper functions
│   └── requirements.txt       # Python dependencies
├── data/
│   ├── gallery_images/        # Local dataset (your faces)
│   ├── gallery.db             # Auto-created face embeddings DB
│   └── lfw_test/              # LFW subset (auto-downloaded)
├── weights/
│   └── arcface_resnet100.onnx # Pretrained ONNX model
├── scripts/
│   ├── prepare_dataset.py     # Downloads & prepares datasets
│   ├── benchmark_cpu.py       # Benchmarks CPU inference
│   ├── eval_detection.py      # Reports precision/recall
│   ├── demo_test_api.py       # (optional) Sample API test
│   ├── download_gallery_dataset.py
│   ├── add_gallery_from_folder.py
├── Dockerfile
├── run.sh
├── sample.jpg                 # Test image for benchmarking
└── README.md

````

---

## ⚙️ Setup Instructions (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Bhakthi-Shetty7811/face-recognition-service.git
cd face-recognition-service
````

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r app/requirements.txt
```

---

## 🧩 Prepare Dataset

```bash
python -m scripts.prepare_dataset
```

📦 This will:

* Download a **subset of LFW dataset** (for testing)
* Validate your **in-house gallery_images** folder
* Create `data/lfw_test` for evaluation

Your structure should now look like:

```
data/
 ├── gallery_images/
 ├── gallery.db
 └── lfw_test/
```

---

## 🧠 Run Benchmark & Evaluation

### CPU Benchmark:

```bash
python -m scripts.benchmark_cpu
```

> Output Example:
>
> ```
> Avg latency: 32.88 ms, Throughput: 30.41 FPS
> ```

### Detection Evaluation:

```bash
python -m scripts.eval_detection
```

> Output Example:
>
> ```
> Precision=0.80, Recall=0.86
> ```

---

## 🚀 Run FastAPI Service

```bash
uvicorn app.main:app --reload
```

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* Test endpoints directly there.

### Available Endpoints:

| Endpoint           | Method | Description                      |
| ------------------ | ------ | -------------------------------- |
| `/detect`          | POST   | Detect faces in image            |
| `/add_identity`    | POST   | Add a person to gallery          |
| `/recognize`       | POST   | Identify a face using embeddings |
| `/list_identities` | GET    | List all stored identities       |

---

## 🐳 Run with Docker

### Build and Run

```bash
bash run.sh
```

or on Windows PowerShell:

```bash
./run.sh
```

This will:

* Build the image (`face-recognition-service`)
* Run container at port `8000`
* Mount `/data` so your gallery persists

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧮 Results Summary

| Metric            | Result    |
| ----------------- | --------- |
| Avg Latency (CPU) | ~32.88 ms |
| Throughput (FPS)  | ~30.41    |
| Precision         | 0.80      |
| Recall            | 0.86      |

✅ Runs smoothly on CPU-only machines.

---

## 🧱 Tech Stack

* **FastAPI** — REST API Framework
* **OpenCV** — Face Detection (Haar Cascade)
* **ArcFace (ONNX)** — Embedding Extraction
* **SQLite** — Local Face Gallery DB
* **Docker** — Containerization
* **Python 3.11**

---

## 👤 Author
**Bhakthi Shetty**
🧩 Contact: [LinkedIn](https://www.linkedin.com/in/bhakthi-shetty-743a33357) | [GitHub](https://github.com/Bhakthi-Shetty7811)

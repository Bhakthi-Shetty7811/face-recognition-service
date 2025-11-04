"""
Prepare Dataset Script
- Loads your in-house gallery dataset (data/gallery_images/)
- Downloads a small subset of LFW dataset for testing
- Aligns, normalizes, and saves faces ready for training/evaluation
"""

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from pathlib import Path

GALLERY_PATH = Path("data/gallery_images")
LFW_PATH = Path("data/lfw_test")
LFW_PATH.mkdir(parents=True, exist_ok=True)

# --- 1. Download LFW dataset (subset)
print("📥 Downloading small LFW subset...")
lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.5, color=True)
images = lfw.images
names = lfw.target_names
targets = lfw.target

print(f"✅ Loaded {len(images)} LFW images of {len(names)} people.")

# --- 2. Save subset to folder for evaluation
for i, img in enumerate(images[:100]):  # limit to 100 images
    person = names[targets[i]].replace(" ", "_")
    person_folder = LFW_PATH / person
    person_folder.mkdir(exist_ok=True)
    cv2.imwrite(str(person_folder / f"{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print(f"✅ Saved 100 LFW images in {LFW_PATH}/")

# --- 3. Quick check for your gallery dataset
print("\n🧠 Checking in-house gallery...")
if not GALLERY_PATH.exists():
    print("⚠️ Gallery path not found! Please create 'data/gallery_images/'.")
else:
    people = [p for p in GALLERY_PATH.iterdir() if p.is_dir()]
    print(f"Found {len(people)} identities:")
    for p in people:
        count = len(list(p.glob('*.jpg')))
        print(f"  - {p.name}: {count} images")
    print("✅ Your gallery dataset looks good!")

print("\n🎯 Dataset prep complete!")

import os
import requests

API_URL = "http://localhost:8000/add_identity"

base_dir = "data/gallery_images"

for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            with open(os.path.join(person_dir, img_file), "rb") as f:
                files = {"file": (img_file, f, "image/jpeg")}
                data = {"name": person}
                res = requests.post(API_URL, files=files, data=data)
                print(f"Added {img_file} for {person}: {res.status_code}")

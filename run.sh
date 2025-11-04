#!/bin/bash

# -------------------------------
# Face Recognition Service - Run Script
# -------------------------------

# Exit on error
set -e

# Name of the Docker image
IMAGE_NAME="face-recognition-service"
CONTAINER_NAME="face-recognition-container"

echo "🚀 Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

echo "🧹 Removing any old running container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🧠 Running container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  $IMAGE_NAME

echo "✅ Container started!"
echo "🌐 Visit the API docs at: http://localhost:8000/docs"

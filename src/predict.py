from ultralytics import YOLO
import os
import cv2

# --- CONFIG ---
MODEL_PATH = "/home/miyad/litter_detection_system/models/best.pt"  # The model you just copied
"""TEST_IMAGES_DIR = "/home/miyad/litter_detection_system/data/processed/images/test" # Your test images
OUTPUT_DIR = "/home/miyad/litter_detection_system/runs/detect/predictions"
"""
# --- RUN INFERENCE ---
model = YOLO(MODEL_PATH)

# Run prediction on the entire test folder
# save=True will save images with boxes drawn on them
results = model.predict(source='0', show=True, conf=0.5)

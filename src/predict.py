from ultralytics import YOLO
import os
import cv2

# --- CONFIG ---
MODEL_PATH = "/home/miyad/litter_detection_system/models/taco_v1.pt"  # The model you just copied
TEST_IMAGES_DIR = "/home/miyad/litter_detection_system/data/processed/images/test" # Your test images
OUTPUT_DIR = "/home/miyad/litter_detection_system/runs/detect/predictions"

# --- RUN INFERENCE ---
model = YOLO(MODEL_PATH)

# Run prediction on the entire test folder
# save=True will save images with boxes drawn on them
results = model.predict(source=TEST_IMAGES_DIR, save=True, project="runs/detect", name="test_results", conf=0.25)

print(f"✅ Predictions saved to {results[0].save_dir}")

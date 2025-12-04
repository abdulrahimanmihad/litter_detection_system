import os
import random
import cv2
import yaml

# CONFIG
DATASET_DIR = "/home/miyad/litter_detection_system/data/processed"
NUM_SAMPLES = 5
OUTPUT_DIR = "checked_images"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load class names
with open(f"/home/miyad/litter_detection_system/config/data.yaml", 'r') as f:
    data = yaml.safe_load(f)
    class_names = data['names']

# Get random images
img_dir = f"{DATASET_DIR}/images/train"
label_dir = f"{DATASET_DIR}/labels/train"
all_images = os.listdir(img_dir)
samples = random.sample(all_images, NUM_SAMPLES)

print(f"Checking {NUM_SAMPLES} images...")

for img_file in samples:
    # Load Image
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Load Label
    label_file = img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(label_dir, label_file)
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            # YOLO format is normalized: x_center y_center width height
            x_c, y_c, bw, bh = map(float, parts[1:])
            
            # Convert to pixel coordinates
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            
            # Draw box and name
            color = (0, 255, 0) # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[cls_id], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save to check
    cv2.imwrite(f"{OUTPUT_DIR}/check_{img_file}", img)
    print(f"Saved: {OUTPUT_DIR}/check_{img_file}")

print(f"Go open the '{OUTPUT_DIR}' folder in Windows to see the results!")

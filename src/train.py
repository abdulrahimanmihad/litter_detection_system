from ultralytics import YOLO
import mlflow
import os

# --- CONFIGURATION ---
# Define paths relative to where you run the script (Project Root)
DATA_CONFIG = 'config/data.yaml'
HYPERPARAMS = 'config/params.yaml'
EXPERIMENT_NAME = "Litter_Detection_Project"

if __name__ == '__main__':
    # 1. Set up MLflow Tracking
    # This tells MLflow to save logs to a folder named 'mlruns' in your project
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load the Model
    # Using 'yolov8s.pt' (Small) as discussed for better accuracy than Nano
    model = YOLO('yolov8s.pt')

    # 3. Start Training
    print(f"Starting training experiment: {EXPERIMENT_NAME}...")
    model.train(
        data=DATA_CONFIG,
        epochs=300,             # Long training
        patience=50,            # Stop if no improvement
        imgsz=640,
        device=0,               # Use GPU
        batch=8,                # Safe for laptop memory
        workers=1,              # Safe for WSL
        cache=False,            # Save RAM
        cfg=HYPERPARAMS,        # Use your augmentation settings
        name='run_v1_small',    # Name of this specific run
        project='runs/detect'   # Where YOLO saves its own local logs
    )
    
    print("Training finished! Check MLflow dashboard.")

from ultralytics import YOLO
import mlflow


# --- CONFIGURATION ---

DATA_CONFIG = '/home/miyad/litter_detection_system/config/data.yaml'
HYPERPARAMS = '/home/miyad/litter_detection_system/config/params.yaml'
EXPERIMENT_NAME = "Litter_Detection_Project"

if __name__ == '__main__':
    # 1. Set up MLflow Tracking
    # This tells MLflow to save logs to a folder named 'mlruns' in project
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load the Model
    # Using 'yolov8s.pt' (Small) for better accuracy than Nano
    model = YOLO('yolov8s.pt')

    # 3. Start Training
    print(f"Starting training experiment: {EXPERIMENT_NAME}...")
    model.train(
        data=DATA_CONFIG,
        epochs=300,             # Long training
        patience=50,            # Stop if no improvement
        imgsz=640,
        device=0,               # Use GPU
        batch=16,               
        workers=2,              
        cache=False,            
        cfg=HYPERPARAMS,        # augmentation settings
        name='run_v1_small',    # Name of this specific run
        project='runs/detect'   # Where YOLO saves its own local logs
    )
    
    print("Training finished! Check MLflow dashboard.")

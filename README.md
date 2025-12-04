# ♻️ Real-Time Litter Detection System

An End-to-End MLOps project for detecting litter (Plastic, Metal, Paper, Trash) in real-time using YOLOv8. 

This project demonstrates a full machine learning lifecycle: from data ingestion (TACO dataset) to deployment (Streamlit App), managed with professional MLOps tools.

## 🛠️ Tech Stack
* **Model:** YOLOv8 (Ultralytics)
* **Tracking:** MLflow (Experiment tracking), DVC (Data Version Control)
* **Deployment:** Streamlit, OpenCV
* **Environment:** Ubuntu (WSL2), Python 3.10

## 📂 Project Structure
```text
├── config/          # Configuration files (data paths, hyperparameters)
├── data/            # Data versioned by DVC (raw & processed)
├── models/          # Trained models versioned by DVC
├── src/             # Source code for training and inference
├── .dvc/            # DVC configuration
└── requirements.txt # Python dependencies
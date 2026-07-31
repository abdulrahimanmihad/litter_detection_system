# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time litter detection with YOLOv8 (Ultralytics), trained on the TACO dataset. An end-to-end MLOps setup: DVC versions the data/models, MLflow tracks experiments, and a Streamlit app serves inference. The README describes the target environment as Ubuntu/WSL2 + Python 3.10, but the repo is currently checked out and run on Windows (`D:\litter_detection_system`, `.venv` is a Windows venv).

## Environment & commands

Use the project venv (`.venv\Scripts\python.exe`) for everything. Common actions:

- **Train:** `python src/train.py` — loads `yolov8s.pt`, reads `config/data.yaml` + `config/params.yaml`, logs to `./mlruns`, and writes YOLO's own outputs to `runs/detect/run_v1_small`.
- **Serve app:** `streamlit run src/app.py` — loads `best.pt` and detects on uploaded images at `conf=0.189`.
- **Webcam predict:** `python src/predict.py` — runs live inference on camera source `0`.
- **View experiments:** `mlflow ui` from the repo root (backing store is `file:./mlruns`).
- **DVC:** `dvc pull` / `dvc push` to sync `data/` and `models/` (both tracked via `.dvc` files, not committed to git). `dvc repro` is not configured — there is no `dvc.yaml` pipeline.

There is no test suite, linter, or build step. "Tests" in this repo means the ad-hoc verification scripts (`check_labels.py`) and `runs/detect/test_results`.

## Hardcoded absolute paths (important)

Scripts do **not** use relative paths or a shared config for their base directory. Two incompatible conventions coexist:

- `src/train.py` uses Windows paths (`D:/litter_detection_system/...`).
- `src/predict.py`, `src/fix_and_merge.py`, `src/check_labels.py`, `src/sanitize_files.py`, `src/rename_backgrounds.py`, and `config/data.yaml` (`path:`) use WSL Linux paths (`/home/miyad/litter_detection_system/...`).
- `src/convert_original_coco_to_yolo.py` reads from WSL mounts of Windows dirs (`/mnt/c/Users/miyad/...`).

Before running any script on Windows, check and fix its `DATA_DIR` / `MODEL_PATH` / path constants. When editing, prefer converting these to paths that work in the current environment rather than adding new hardcoded ones.

## Data pipeline (order matters)

The dataset is built by a sequence of one-off scripts, not an automated pipeline. Rough order from raw TACO to trainable data:

1. `convert_original_coco_to_yolo.py` — converts the original TACO COCO `annotations.json` into YOLO `.txt` labels and a CVAT-ready ZIP, collapsing TACO's fine categories into coarse classes via `COARSE_MAP`.
2. (Manual) annotate/correct in CVAT, export as "YOLO 1.0".
3. `convert_to_yolov8.py` — walks CVAT's `obj_train_data`, pairs images with labels, and does a **seeded** 70/20/10 train/val/test split (`random.seed(42)`) into `data/processed/{images,labels}/{train,val,test}`. Files are renamed with their batch-folder prefix to avoid collisions.
4. `fix_and_merge.py` — remaps class IDs (see below) and creates empty `.txt` label files for background images (images with no objects).
5. `sanitize_files.py` / `rename_backgrounds.py` — clean bad filenames and standardize background image naming.

## Class definitions (7 classes)

`config/data.yaml` defines `nc: 7`: `plastic, cigarette, metal, carton, paper, trash, Glass`.

This is the result of a **remap** in `fix_and_merge.py`. The upstream converter produced 8 classes (`0:plastic 1:cig 2:metal 3:carton 4:paper 5:bio_waste 6:unlabeled/trash 7:Glass`). The remap merges `bio_waste(5)` and `unlabeled(6)` into a single `trash(5)` and shifts `Glass` from 7 → 6. If you regenerate labels from the converter, they will be in the old 8-class scheme and must be re-run through `fix_and_merge.py` before they match `data.yaml`.

## Hyperparameters

`config/params.yaml` holds only augmentation settings (mosaic, mixup, degrees, fliplr, scale, copy_paste) and is passed to Ultralytics via `cfg=`. Training-loop settings (epochs, imgsz, batch, device, patience) are hardcoded as `model.train(...)` arguments in `src/train.py`, not in this file.

## Models

- `models/taco_v1.pt`, `models/taco_v2_highres.pt` — trained checkpoints, DVC-tracked.
- `yolov8s.pt` (repo root) — pretrained base weights for training.
- `best.pt` / `src/best.pt` — the checkpoint the app and predict scripts load; keep this in sync with the model you intend to serve.

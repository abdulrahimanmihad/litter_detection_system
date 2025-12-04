import os
import shutil
import random
import yaml

# --- CONFIGURATION ---
# format for yolov8 after cvat export dataset (yolo 1.0)
# This matches the folder name inside your 'final_dataset'
SOURCE_DIR = "obj_train_data" 
OUTPUT_DIR = "data/processed"
OBJ_NAMES_FILE = "obj.names"

# Split: 70% Train, 20% Val, 10% Test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def main():
    # 1. Setup Directories
    print(f"Creating output directory: {OUTPUT_DIR}")
    # Delete old folder if exists to start clean
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    # 2. Collect all matching Image/Label pairs
    print(f"Scanning files in {SOURCE_DIR}...")
    pairs = []
    
    # This walks through batch_1, batch_2, etc.
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                
                # Look for matching .txt file (same name)
                label_name = os.path.splitext(file)[0] + ".txt"
                label_path = os.path.join(root, label_name)
                
                if os.path.exists(label_path):
                    pairs.append((image_path, label_path))

    print(f"Found {len(pairs)} valid image/label pairs.")
    
    if len(pairs) == 0:
        print("❌ Error: No paired files found. Make sure 'obj_train_data' folder is here.")
        return

    # 3. Shuffle and Split
    random.seed(42)
    random.shuffle(pairs)
    
    train_count = int(len(pairs) * TRAIN_RATIO)
    val_count = int(len(pairs) * VAL_RATIO)
    
    train_set = pairs[:train_count]
    val_set = pairs[train_count : train_count + val_count]
    test_set = pairs[train_count + val_count:]
    
    # 4. Move Files Function
    def move_files(file_pairs, split_name):
        print(f"Processing {split_name} set ({len(file_pairs)} files)...")
        for src_img, src_label in file_pairs:
            # We rename files to include batch name to avoid duplicates
            # e.g. batch_1_0000.jpg
            batch_name = os.path.basename(os.path.dirname(src_img))
            filename = os.path.basename(src_img)
            new_name = f"{batch_name}_{filename}"
            
            # Destination paths
            dst_img = os.path.join(OUTPUT_DIR, "images", split_name, new_name)
            dst_lbl = os.path.join(OUTPUT_DIR, "labels", split_name, os.path.splitext(new_name)[0] + ".txt")
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_lbl)

    move_files(train_set, 'train')
    move_files(val_set, 'val')
    move_files(test_set, 'test')

    # 5. Create data.yaml
    print("Generating data.yaml...")
    try:
        with open(OBJ_NAMES_FILE, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"⚠️ Warning: {OBJ_NAMES_FILE} not found. Creating generic classes.")
        classes = ['class_0']

    # Use absolute path so YOLO never gets lost
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n✅ DONE! Dataset ready.")
    print(f"Config file: {os.path.abspath(OUTPUT_DIR)}/data.yaml")

if __name__ == "__main__":
    main()

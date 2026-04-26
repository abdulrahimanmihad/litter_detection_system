import os
import shutil

# --- CONFIGURATION ---
# Path to your training images
IMG_DIR = '/home/miyad/litter_detection_system/data/processed/images/val'
LBL_DIR = '/home/miyad/litter_detection_system/data/processed/labels/val'

# The new "Batch" name you want
NEW_BATCH_NAME = "batch_18"
START_INDEX = 1

def main():
    print(f"🧹 Scanning {IMG_DIR} for new background images...")
    
    # 1. Find images that DO NOT start with "batch_"
    # We assume these are the new ones you added manually
    all_files = os.listdir(IMG_DIR)
    new_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("batch_")]
    
    if not new_images:
        print("✅ No disorganized files found! Everything already looks like 'batch_...'.")
        return

    print(f"Found {len(new_images)} new images to rename.")
    
    count = 0
    for i, filename in enumerate(new_images):
        # 2. Construct the new name: batch_16_000001.jpg
        idx = START_INDEX + i
        extension = os.path.splitext(filename)[1]
        new_name = f"{NEW_BATCH_NAME}_{idx:06d}{extension}"
        
        # Paths
        old_img_path = os.path.join(IMG_DIR, filename)
        new_img_path = os.path.join(IMG_DIR, new_name)
        
        # 3. Rename the Image
        os.rename(old_img_path, new_img_path)
        
        # 4. Create the Empty Label File (Background)
        # We assume these are background images, so we make a blank .txt
        new_label_name = f"{NEW_BATCH_NAME}_{idx:06d}.txt"
        new_label_path = os.path.join(LBL_DIR, new_label_name)
        
        with open(new_label_path, 'w') as f:
            pass # Create empty file
            
        # Optional: If you had an old label file with the weird name, delete/rename it too?
        # Usually for backgrounds you don't have labels yet, so we just create new ones.
        
        print(f"Renamed: {filename} -> {new_name} (and created empty label)")
        count += 1

    print(f"\n✅ Success! Renamed and labeled {count} images as {NEW_BATCH_NAME}.")

if __name__ == "__main__":
    if not os.path.exists(IMG_DIR):
        print(f"❌ Error: Could not find {IMG_DIR}")
    else:
        main()

import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your local processed data
DATA_DIR = '/home/miyad/litter_detection_system/data/processed'

# CLASS MAPPING LOGIC
# Old IDs: 0:Plastic, 1:Cig, 2:Metal, 3:Carton, 4:Paper, 5:Bio, 6:Trash, 7:Glass
# New IDs: 0:Plastic, 1:Cig, 2:Metal, 3:Carton, 4:Paper, 5:Trash, 6:Glass
CLASS_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 
    5: 5, # Bio -> Trash
    6: 5, # Trash -> Trash
    7: 6  # Glass -> Glass (Shifted down)
}

def main():
    print(f"🔧 Starting dataset repair in: {DATA_DIR}")
    
    # Process train, val, and test folders
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(DATA_DIR, 'images', split)
        lbl_dir = os.path.join(DATA_DIR, 'labels', split)
        
        if not os.path.exists(img_dir): continue
        
        print(f"\n📂 Processing {split} set...")
        
        # --- TASK 1: CREATE BACKGROUND LABELS ---
        # Find all images
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        created_count = 0
        
        for img_file in images:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(lbl_dir, label_file)
            
            # If image exists but label does NOT, create an empty label file
            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    pass # Create empty file
                created_count += 1
        
        if created_count > 0:
            print(f"   ✨ Created {created_count} empty label files for background images.")

        # --- TASK 2: REMAP EXISTING LABELS ---
        # Find all text files
        labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        fixed_count = 0
        
        for label_file in tqdm(labels, desc="   Remapping IDs"):
            path = os.path.join(lbl_dir, label_file)
            
            # Read file
            with open(path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            file_changed = False
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                old_id = int(parts[0])
                
                # Check if this ID needs to change
                if old_id in CLASS_MAP:
                    new_id = CLASS_MAP[old_id]
                    
                    if new_id != old_id:
                        file_changed = True
                    
                    # Rebuild line: new_id + x y w h
                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
            
            # Only write to disk if we actually changed a number
            if file_changed:
                with open(path, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
                
        print(f"   🔄 Updated class IDs in {fixed_count} files.")

    print("\n✅ DONE! Bio_waste merged, Glass shifted, and Backgrounds labeled.")

if __name__ == "__main__":
    main()

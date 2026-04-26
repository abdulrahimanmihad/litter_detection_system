import os
import re

# --- CONFIGURATION ---
DATA_DIR = '/home/miyad/litter_detection_system/data/processed'

def sanitize_filename(name):
    # 1. Replace # with nothing (remove it)
    new_name = name.replace("#", "")
    # 2. Replace spaces with underscores
    new_name = new_name.replace(" ", "_")
    # 3. Remove other weird characters if any (keep only letters, numbers, dots, underscores, dashes)
    # This regex keeps alphanumeric, ., _, and -
    new_name = re.sub(r'[^a-zA-Z0-9._-]', '', new_name)
    return new_name

def main():
    print(f"🧹 Scanning {DATA_DIR} for bad filenames...")
    renamed_count = 0

    # Walk through all folders
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            # Check if name needs fixing
            if "#" in filename or " " in filename or not re.match(r'^[a-zA-Z0-9._-]+$', filename):
                
                old_path = os.path.join(root, filename)
                
                # Generate clean name
                clean_name = sanitize_filename(filename)
                new_path = os.path.join(root, clean_name)
                
                # Rename
                # Handle duplicate names if conflict arises
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(clean_name)
                    clean_name = f"{base}_fixed{ext}"
                    new_path = os.path.join(root, clean_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {clean_name}")
                renamed_count += 1

    print(f"\n✅ Finished! Renamed {renamed_count} files.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Could not find {DATA_DIR}")
    else:
        main()

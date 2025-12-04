#!/usr/bin/env python3
"""
convert_original_coco_to_yolo.py

Use the ORIGINAL uploaded COCO JSON and ORIGINAL images (Windows batches).
Outputs a CVAT-ready YOLO ZIP that preserves batch folders and contains:
 - images in their original relative batch folders
 - .txt files (YOLO format) in same relative locations

Input JSON (original upload): /mnt/data/annotations.json
Windows data root (mounted in WSL): /mnt/c/Users/miyad/Downloads/TACO-master/TACO-master/data

Output:
 - /home/miyad/yolo_labels/...   (txt files mirroring image paths)
 - /home/miyad/yolo_for_cvat.zip (images + .txt files, ready to upload to CVAT)
"""

import json, os, sys, zipfile
from pathlib import Path
# === CONFIG ===
IN_JSON = "/mnt/c/Users/miyad/Downloads/TACO-master/TACO-master/data/annotations.json"   # ORIGINAL JSON uploaded in chat
WIN_DATA_DIR = "/mnt/c/Users/miyad/Downloads/TACO-master/TACO-master/data"  # images root (WSL mount)
YOLO_LABEL_ROOT = os.path.expanduser("/home/miyad/yolo_labels")
OUT_ZIP = os.path.expanduser("/home/miyad/yolo_for_cvat.zip")
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
DEFAULT_CLASS = 6  # if category mapping missing, assign this (you can change)
# Define or edit mapping from original category NAMES -> coarse YOLO class index (0-based)
# Example mapping: adjust as needed. If a name isn't in this dict, script will use heuristics or DEFAULT_CLASS.
COARSE_CLASS_ORDER = ["plastic","cigarette","metal","carton","paper","bio_waste","unlabeled_litter","trash"]
# Provide string->index map for deterministic classes (edit if you want different grouping)
COARSE_MAP = {
    # plastic family -> class 0
    "Other plastic bottle": 0, "Clear plastic bottle": 0, "Plastic bottle cap": 0,
    "Other plastic": 0, "Other plastic cup": 0, "Other plastic container": 0,
    "Plastic lid": 0, "Plastic glooves": 0, "Plastic utensils": 0,
    "Other plastic wrapper": 0, "Plastic film": 0, "Garbage bag": 0,
    "Single-use carrier bag": 0, "Polypropylene bag": 0, "Crisp packet": 0,
    "Disposable plastic cup": 0, "Foam cup": 0, "Disposable food container": 0,
    "Foam food container": 0, "Spread tub": 0, "Tupperware": 0, "Plastic straw": 0,
    "Tupperware": 0, "Plastic bottle cap": 0,

    # cigarettes -> class 1
    "Cigarette": 1,

    # metal -> class 2
    "Aluminium foil": 2, "Aluminium blister pack": 2, "Carded blister pack": 2,
    "Battery": 2, "Food Can": 2, "Drink can": 2, "Pop tab": 2, "Scrap metal": 2,
    "Metal bottle cap": 2, "Metal lid": 2,

    # carton -> class 3
    "Other carton": 3, "Drink carton": 3, "Corrugated carton": 3,
    "Meal carton": 3, "Pizza box": 3, "Egg carton": 3,

    # paper -> class 4
    "Magazine paper": 4, "Tissues": 4, "Wrapping paper": 4, "Normal paper": 4,
    "Paper bag": 4, "Plastified paper bag": 4, "Paper cup": 4, "Paper straw": 4, "Toilet tube": 4,

    # bio_waste -> class 5
    "Food waste": 5,

    # unlabeled_litter -> class 6
    "Unlabeled litter": 6,

    # trash (catch-all) -> class 7
    "Broken glass": 7, "Glass bottle": 7, "Glass jar": 7, "Glass cup": 7,
    "Shoe": 7, "Squeezable tube": 7, "Styrofoam piece": 7, "Aerosol": 7, "Other plastic bottle": 0
}
# === end CONFIG ===

def norm_path(p: str) -> str:
    if not isinstance(p, str): return p
    s = p.replace("\\", "/").lstrip("./")
    # strip leading drive like C:/ if present
    if len(s) > 2 and s[1] == ':' and s[2] == '/':
        s = s.split(':',1)[1].lstrip('/')
    # if someone embedded /mnt/c/... keep only the subpath (strip mnt/c)
    if s.startswith("mnt/"):
        parts = s.split('/')
        if len(parts) > 2 and parts[0].lower() == 'mnt':
            s = '/'.join(parts[2:])
    return s

def find_input():
    if os.path.exists(IN_JSON):
        return IN_JSON
    print("[ERROR] Input JSON not found at", IN_JSON)
    sys.exit(2)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def get_img_size_via_pil(imgpath):
    try:
        from PIL import Image
    except Exception:
        return None, None
    try:
        with Image.open(imgpath) as im:
            return im.size  # (width, height)
    except:
        return None, None

def map_category_name_to_class(name):
    if name in COARSE_MAP:
        return COARSE_MAP[name]
    ln = name.lower()
    # heuristics
    if "plastic" in ln or "bag" in ln or "film" in ln or "tupperware" in ln or "cup" in ln:
        return 0
    if "cig" in ln:
        return 1
    if any(k in ln for k in ["aluminium","aluminum","metal","can","battery","pop tab","scrap"]):
        return 2
    if "carton" in ln or "pizza" in ln or "egg" in ln:
        return 3
    if any(k in ln for k in ["paper","magazine","tissue","wrap","normal paper","bag","straw"]):
        return 4
    if "food" in ln or "waste" in ln:
        return 5
    if "unlabeled" in ln or "unlabel" in ln:
        return 6
    return 7  # default -> trash

def main():
    inp = find_input()
    print("[*] Using original JSON:", inp)
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if not images:
        print("[ERROR] no images[] in JSON")
        sys.exit(3)

    # build id->file_name map from JSON (normalize paths)
    id_to_file = {}
    for im in images:
        try:
            iid = int(im.get("id"))
        except:
            continue
        fn = im.get("file_name", "")
        fn = norm_path(fn)
        id_to_file[iid] = fn

    # verify images exist physically under WIN_DATA_DIR
    missing = []
    for iid, rel in id_to_file.items():
        p = Path(WIN_DATA_DIR) / rel
        if not p.exists():
            # try basename fallback
            if (Path(WIN_DATA_DIR) / Path(rel).name).exists():
                # replace mapping to basename
                id_to_file[iid] = Path(rel).name
                continue
            missing.append(str(p))
    if missing:
        print("[ERROR] Missing image files referenced in original JSON. Count:", len(missing))
        print("Sample missing:", missing[:20])
        print("Fix paths or ensure WIN_DATA_DIR is correct, then re-run.")
        sys.exit(4)

    # Map original category id -> name
    id_to_name = {}
    for c in categories:
        try:
            cid = int(c.get("id"))
        except:
            continue
        id_to_name[cid] = c.get("name", "")

    # Build image info map (try to get w,h from JSON if present)
    img_wh = {}
    for im in images:
        iid = int(im.get("id"))
        w = im.get("width"); h = im.get("height")
        img_wh[iid] = {"width": int(w) if w else None, "height": int(h) if h else None, "file_name": id_to_file[iid]}

    # create YOLO labels (accumulate per relpath)
    labels_acc = {}  # relpath -> list of lines
    skipped_ann = 0
    for ann in annotations:
        try:
            img_id = int(ann.get("image_id"))
        except:
            continue
        if img_id not in id_to_file:
            skipped_ann += 1
            continue
        rel = id_to_file[img_id]
        # skip if no bbox (YOLO needs bbox)
        if 'bbox' not in ann:
            skipped_ann += 1
            continue
        bbox = ann['bbox']  # x,y,w,h
        try:
            bx,by,bw,bh = [float(x) for x in bbox]
        except:
            skipped_ann += 1
            continue
        # determine image size
        W = img_wh[img_id]['width']; H = img_wh[img_id]['height']
        if W is None or H is None:
            # attempt to read image size from disk
            img_abs = Path(WIN_DATA_DIR) / rel
            if not img_abs.exists(): img_abs = Path(WIN_DATA_DIR) / Path(rel).name
            w,h = get_img_size_via_pil(str(img_abs))
            if w is None or h is None:
                skipped_ann += 1
                continue
            W, H = w, h
            img_wh[img_id]['width'], img_wh[img_id]['height'] = W, H
        # convert to YOLO normalized center coords
        cx = bx + bw/2.0
        cy = by + bh/2.0
        nx = cx / W
        ny = cy / H
        nw = bw / W
        nh = bh / H
        # determine class index (map from original category name if available)
        oldcid = ann.get('category_id')
        try:
            oldcid = int(oldcid)
            orig_name = id_to_name.get(oldcid, "")
        except:
            orig_name = ""
        class_idx = map_category_name_to_class(orig_name)
        line = f"{class_idx} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
        labels_acc.setdefault(rel, []).append(line)

    print(f"[*] Prepared YOLO lines for {len(labels_acc)} images. Skipped {skipped_ann} annotations (no bbox or invalid).")

    # write labels to YOLO_LABEL_ROOT preserving folder structure
    for rel, lines in labels_acc.items():
        out_txt = Path(YOLO_LABEL_ROOT) / rel
        out_txt = out_txt.with_suffix('.txt')
        ensure_parent = out_txt.parent
        ensure_parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # create zip: include image files (preserve relative paths) and the .txt files (matching rel paths)
    print("[*] Creating ZIP", OUT_ZIP)
    base = Path(WIN_DATA_DIR)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as zf:
        # add images from base (only image extensions)
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                arc = str(p.relative_to(base)).replace("\\","/")
                zf.write(str(p), arc)
        # add label txts from YOLO_LABEL_ROOT
        root = Path(YOLO_LABEL_ROOT)
        for p in root.rglob("*.txt"):
            if p.is_file():
                arc = str(p.relative_to(root)).replace("\\","/")
                zf.write(str(p), arc)
    print("[OK] Wrote ZIP:", OUT_ZIP)
    print("Upload this ZIP to CVAT and import using YOLO format. The structure preserves batch folders and matching .txt files.")
    print("Class index mapping (0-based):")
    for i,name in enumerate(COARSE_CLASS_ORDER):
        print(f" {i}: {name}")
    print("If you need a different mapping, edit COARSE_MAP at top of the script and re-run.")

if __name__ == "__main__":
    main()

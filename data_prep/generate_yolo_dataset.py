import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from itertools import product
from ultralytics.models.sam.predict import SAM3SemanticPredictor
import random
import yaml
from tqdm import tqdm
from pathlib import Path

# --- Config ---
IMAGE_DIR = "raw_images"  # Folder containing partial .tif files
TILE_SIZE = 1024          # Output tile size
OUTPUT_DIR = "datasets/sidewalk_segmentation"
CONF_THRESH = 0.25

def setup_dirs(base_dir):
    for split in ["train", "val"]:
        os.makedirs(f"{base_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{base_dir}/labels/{split}", exist_ok=True)

def pad_to_size(img, target_size=1024):
    """Pads image to (target_size, target_size) with zeros."""
    h, w, c = img.shape
    if h == target_size and w == target_size:
        return img, h, w
    canvas = np.zeros((target_size, target_size, c), dtype=np.uint8)
    canvas[:h, :w, :] = img
    return canvas, h, w

def normalize_poly(poly, valid_w, valid_h, target_w, target_h):
    """Normalize polygon relative to the full target tile size."""
    normalized = poly.astype(float)
    normalized[:, 0] /= target_w
    normalized[:, 1] /= target_h
    return normalized.flatten()

import hashlib
import json

# ... (Config unchanged) ...
PROCESSED_LOG = f"{OUTPUT_DIR}/processed_log.json"

def get_img_hash(path):
    """Returns a hash of the first 16KB of the file to detect duplicates quickly."""
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            buf = f.read(16384) # Read first 16KB
            hasher.update(buf)
            # mixing file size into hash to avoid collisions on identical headers
            f.seek(0, 2)
            hasher.update(str(f.tell()).encode('utf-8'))
    except Exception:
        return None
    return hasher.hexdigest()

def load_processed_log():
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_log(processed_set):
    with open(PROCESSED_LOG, 'w') as f:
        json.dump(list(processed_set), f)

def main():
    print("--- YOLO Dataset Gen (Multi-Image) ---")
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory '{IMAGE_DIR}' not found. Please create it and add .tif images.")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))]
    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'")
        return
        
    setup_dirs(OUTPUT_DIR)
    processed_hashes = load_processed_log()
    
    # Init Model
    print("Loading SAM3...")
    
    # SILENCE Ultralytics Warnings
    import logging
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    
    try:
        # Added imgsz=TILE_SIZE to fix "stride 14" warning and ensure correct resolution
        overrides = dict(conf=CONF_THRESH, task="segment", mode="predict", model="SAM3/sam3.pt", half=True, imgsz=TILE_SIZE)
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"Model Load Failed: {e}")
        return

    total_saved = 0
    
    # Iterate over ALL images in folder
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        img_name = Path(img_file).stem
        
        # 1. Check for Duplicates via Hash
        img_hash = get_img_hash(img_path)
        if img_hash in processed_hashes:
            print(f"Skipping {img_file} (Already Processed - Hash Match)")
            continue
            
        print(f"\nProcessing: {img_file}...")
        
        try:
            with rasterio.open(img_path) as src:
                W, H = src.width, src.height
                tiles = list(product(range(0, W, TILE_SIZE), range(0, H, TILE_SIZE)))
                
                # Progress bar
                pbar = tqdm(tiles, desc=f"Scanning {img_name}", unit="tile", leave=True, position=0)
                
                for col, row in pbar:
                    base_name = f"{img_name}_tile_{col}_{row}"
                    
                    # Check Exists
                    if os.path.exists(f"{OUTPUT_DIR}/labels/train/{base_name}.txt") or \
                       os.path.exists(f"{OUTPUT_DIR}/labels/val/{base_name}.txt"):
                        continue

                    # Read
                    window = Window(col, row, min(TILE_SIZE, W - col), min(TILE_SIZE, H - row))
                    data = src.read(window=window)
                    if data.shape[0] < 3: continue
                    img = np.ascontiguousarray(np.moveaxis(data[:3], 0, -1))
                    if img.max() == 0: continue

                    # Pad
                    img_pad, valid_h, valid_w = pad_to_size(img, TILE_SIZE)

                    # Inference
                    predictor.set_image(img_pad)
                    results = predictor(text=["sidewalk"], save=False, verbose=False)

                    # Save Labels
                    labels = []
                    if results and results[0].masks:
                        for poly in results[0].masks.xy:
                            if len(poly) < 3: continue
                            norm = normalize_poly(poly, valid_w, valid_h, TILE_SIZE, TILE_SIZE)
                            labels.append(f"0 {' '.join(map(str, norm))}")

                    if labels:
                        split = "train" if random.random() < 0.9 else "val"
                        img_out = f"{OUTPUT_DIR}/images/{split}/{base_name}.jpg"
                        lbl_out = f"{OUTPUT_DIR}/labels/{split}/{base_name}.txt"
                        
                        cv2.imwrite(img_out, cv2.cvtColor(img_pad, cv2.COLOR_RGB2BGR))
                        with open(lbl_out, "w") as f:
                            f.write("\n".join(labels))
                        
                        total_saved += 1
                        pbar.set_description(f"Saved: {total_saved}")
            
            # Log completion for this image
            processed_hashes.add(img_hash)
            save_processed_log(processed_hashes)
                        
        except Exception as e:
            print(f"Skipping {img_file} due to error: {e}")

    # Gen YAML
    yaml_lines = dict(path=os.path.abspath(OUTPUT_DIR), train='images/train', val='images/val', names={0: 'sidewalk'})
    with open(f"{OUTPUT_DIR}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_lines, f)

    print(f"\nAll Done! Total Tiles Saved: {total_saved}")

if __name__ == "__main__":
    main()

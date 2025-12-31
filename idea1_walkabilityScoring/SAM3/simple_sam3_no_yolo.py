import cv2
print("Loading heavy libraries (torch, sam, transformers)... please wait.")
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import argparse
from PIL import Image
from ultralytics import SAM
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="SAM3 Segmentation with Class Filtering")
parser.add_argument("--specific", type=str, default=None, help="Specific class to detect (e.g., 'sidewalk'). If not set, segments everything.")
parser.add_argument("--img", type=str, default="datasets/sidewalk_segmentation/images/train/tile_6144_13312.jpg", help="Path to image")
parser.add_argument("--fast", action="store_true", help="Use even faster settings (lower quality)")
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_PATH = args.img
SAM_MODEL_PATH = "SAM3/sam3.pt"   
CONFIDENCE_THRESHOLD = 0.22 

# Optimization: Grid Density
# Default SAM uses 32 (32x32 = 1024 points). 
# standard usage: 16 (256 points) -> 4x Faster
# fast usage: 8 (64 points) -> 16x Faster
POINTS_PER_SIDE = 8 if args.fast else 16

# FORCE GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {DEVICE}")

def main():
    # --- STEP 1: LOAD MODELS ---
    print(f"\n--- STEP 1: LOAD MODELS [Target: '{args.specific if args.specific else 'EVERYTHING'}'] ---")
    try:
        sam = SAM(SAM_MODEL_PATH)
        print(f"SAM loaded.")
    except Exception as e:
        print(f"Error loading SAM: {e}")
        return

    clip_model = None
    clip_processor = None
    
    # Only load CLIP if filtering is requested
    if args.specific:
        print(f"Loading CLIP for text filtering ('{args.specific}')...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- STEP 2: READ IMAGE ---
    img_cv = cv2.imread(IMAGE_PATH)
    if img_cv is None:
        print("Error: Image not found.")
        return
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # --- STEP 3: RUN SAM (TIMED) ---
    print(f"\n--- STEP 2: RUNNING SEGMENTATION (Points Grid: {POINTS_PER_SIDE}x{POINTS_PER_SIDE}) ---")
    start_time = time.time()
    
    # Run SAM with reduced image size for speed
    # imgsz=512 is MUCH faster than 1024 (default)
    # This acts as the "density" control indirectly.
    inference_size = 512 if args.fast else 640
    print(f"Inference Image Size: {inference_size}")
    
    results = sam(img_rgb, 
                  imgsz=inference_size,
                  iou=0.8,
                  device=DEVICE, 
                  verbose=False)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"SUCCESS: Segmentation took {duration:.2f} seconds.")

    if not results[0].masks:
        print("No masks found.")
        return

    all_masks = results[0].masks.data.cpu().numpy()
    print(f"Total Segments Found: {len(all_masks)}")

    # Debug: Save ALL masks to see if Sidewalk even exists
    print(f"Debug: Saving 'debug_all_masks.jpg' with {len(all_masks)} segments...")
    debug_overlay = img_rgb.copy()
    for i, m in enumerate(all_masks):
        np.random.seed(i)
        color = np.random.randint(0, 255, 3)
        debug_overlay[m.astype(bool)] = debug_overlay[m.astype(bool)] * 0.5 + color * 0.5
    cv2.imwrite("debug_all_masks.jpg", cv2.cvtColor(debug_overlay, cv2.COLOR_RGB2BGR))
    
    final_masks = all_masks # Fallback

    # --- STEP 4: FILTERING (Optional) ---
    final_masks = []
    
    if args.specific:
        print(f"\n--- STEP 3: FILTERING for '{args.specific}' ---")
        filter_start = time.time()
        
        # Enhanced Negative Prompts
        negatives = ["background", "noise", "other objects", "car", "vehicle", "tree", "bush", "house", "building", "road", "asphalt"]
        labels = [args.specific] + negatives
        print(f"Labels: {labels}")
        
        count = 0 
        for i, mask in enumerate(all_masks):
            if mask.sum() < 500: continue 
            
            y, x = np.where(mask > 0)
            if len(y) == 0: continue
            y1, y2, x1, x2 = y.min(), y.max(), x.min(), x.max()
            crop = img_rgb[y1:y2+1, x1:x2+1]
            pil_crop = Image.fromarray(crop)
            
            # CLIP Inference
            inputs = clip_processor(
                text=labels, 
                images=pil_crop, 
                return_tensors="pt", 
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # Index 0 is result
            score = probs[0]
            top_idx = probs.argmax()
            top_label = labels[top_idx]
            
            # DEBUG PRINT
            if score > 0.1: # Print anything remotely close
                print(f"Mask {i}: {score:.3f} for '{args.specific}' | Top: '{top_label}' ({probs[top_idx]:.3f})")

            if score > CONFIDENCE_THRESHOLD and top_label == args.specific:
                final_masks.append(mask)
            
            if i % 10 == 0: print(f"Classifying {i}/{len(all_masks)}...", end='\r')
            
        print(f"Filtering took {time.time() - filter_start:.2f}s. Matches: {len(final_masks)}")

    # --- STEP 5: VISUALIZE ---
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    
    if len(final_masks) > 0:
        # Create colorful mask for "everything", or single color for "specific"
        # MANUAL BLENDING (More Robust)
        print("Blending masks into image...")
        blended_img = img_rgb.copy().astype(float)
        
        sorted_masks = sorted(final_masks, key=lambda x: x.sum(), reverse=True)
        
        for i, m in enumerate(sorted_masks):
            if args.specific:
                color = np.array([0, 255, 0]) # Green
            else:
                np.random.seed(i)
                color = np.random.randint(0, 255, 3)
                
            m_bool = m.astype(bool)
            print(f" - Mask {i}: {m_bool.sum()} pixels.")
            
            # Simple alpha blend: 0.6*Image + 0.4*Color
            blended_img[m_bool] = blended_img[m_bool] * 0.6 + color * 0.4
            
        plt.imshow(blended_img.astype(np.uint8))
        title = f"Prompt: '{args.specific}'" if args.specific else "Segment Everything"
        plt.title(f"{title}\nTime: {duration:.2f}s | Count: {len(final_masks)}")
    
    plt.axis('off')
    out_name = f"result_{args.specific if args.specific else 'all'}.jpg"
    plt.savefig(out_name)
    print(f"\nSaved result to: {out_name}")

if __name__ == "__main__":
    main()

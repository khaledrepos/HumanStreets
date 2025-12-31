import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from itertools import product
from ultralytics.models.sam.predict import SAM3SemanticPredictor
import matplotlib.pyplot as plt
import random
import yaml
from pathlib import Path
import folium
from shapely.geometry import Polygon as ShapelyPolygon
import json
from pyproj import Transformer
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm  # Progress bar

# --- Configuration ---
IMAGE_PATH = "./mosiac_rgb_6cmPerPixel.tif"
TILE_SIZE = 1024
OUTPUT_DIR = "datasets/sidewalk_segmentation"
TRAIN_RATIO = 0.90
VISUALIZATION_SAMPLES = 5 

# Specific tiles to debug (col_off, row_off)
DEBUG_TILES = [
    (6144, 13312),
    (1024, 19456), 
    (1024, 20480)
]
FORCE_DEBUG_ONLY = False # ENABLE FULL RUN


# Ensure reproducibility
random.seed(42)

def create_directory_structure():
    """Creates the YOLO dataset directory structure."""
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)
    print(f"Created dataset structure in {OUTPUT_DIR}")

def normalize_polygon(polygon, width, height):
    """
    Normalizes polygon coordinates (0-1) for YOLO format.
    Polygon shape: (N, 2) where columns are x, y.
    """
    normalized = polygon.astype(float)
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    return normalized.flatten()

def save_sample_visualization(samples, filename="sample_masks.png"):
    """
    Saves a plot of sample images with their masks overlayed.
    samples: List of (image, mask) tuples.
    """
    if not samples:
        return
    
    count = len(samples)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    if count == 1:
        axes = [axes]
    
    for ax, (img, mask) in zip(axes, samples):
        ax.imshow(img)
        # Create a colored overlay for the mask
        overlay = np.zeros_like(img)
        overlay[mask == 1] = [0, 255, 0]  # Green for mask
        ax.imshow(overlay, alpha=0.4)
        ax.axis('off')
        ax.set_title("Generated Mask")
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization sample to {filename}")

def main():
    # 1. Setup
    create_directory_structure()
    
    # Initialize models
    print("Loading models...")
    # Initialize SAM3 predictor with configuration
    overrides = dict(
        conf=0.25,              # confidence threshold
        task="segment",         # task i.e. segment
        mode="predict",         # mode i.e. predict
        model="SAM3/sam3.pt",   # model file = sam3.pt
        half=True,              # Use FP16 for faster inference on GPU.
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    # Store all found polygons for the map
    all_map_polygons = []
    map_overlays = [] # List of (base64_img, bounds)

    # 2. Process Image
    with rasterio.open(IMAGE_PATH) as src:
        W, H = src.width, src.height
        transform = src.transform
        crs = src.crs
        print(f"Image Size: {W}x{H}, Tile Size: {TILE_SIZE}")
        print(f"CRS: {crs}")
        
        # Reprojection Transformer
        transformer = None
        if crs.to_string() != "EPSG:4326":
            print("CRS is not EPSG:4326. Will reproject map coords.")
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) # returns (lon, lat)
            
        # DEBUG: Print Center
        x_c, y_c = src.xy(H//2, W//2)
        if transformer:
             lon_c, lat_c = transformer.transform(x_c, y_c)
             print(f"Center Image Coords: {int(x_c)}, {int(y_c)} -> Lat/Lon: {lat_c}, {lon_c}")
        else:
             print(f"Center Image Coords (Lat/Lon): {y_c}, {x_c}")
        
        # Iterate over tiles
        if FORCE_DEBUG_ONLY:
            print(f"DEBUG MODE: Processing only {len(DEBUG_TILES)} specific tiles...")
            tile_indices = DEBUG_TILES
        else:
            # USE ALL TILES (Full Run)
            tile_indices = list(product(range(0, W, TILE_SIZE), range(0, H, TILE_SIZE)))
            print(f"Starting FULL RUN on {len(tile_indices)} tiles...")
        
        processed_count = 0
        
        # --- PROCESS TILES WITH PROGRESS BAR ---
        # Using tqdm to show progress for the long running task
        for min_processed, (col_off, row_off) in enumerate(tqdm(tile_indices, desc="Processing Tiles", unit="tile")):
            window = Window(col_off, row_off, min(TILE_SIZE, W - col_off), min(TILE_SIZE, H - row_off))
            
            # SKIP if already done
            base_filename = f"tile_{col_off}_{row_off}"
            # Check train/val headers
            exists = False
            for split in ['train', 'val']:
                if os.path.exists(f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"):
                    exists = True
                    break
            
            if exists:
                print(f"Skipping {base_filename} (Already Exists)")
                # If we skip, we still might want to add to map if we could read it?
                # For now, simplistic approach: just skip processing. 
                # To ensure map is generated, we might need to read the file? 
                # Let's assume user wants to process 10 NEW or 10 TOTAL?
                # "10 images only, and it has a mask consider it done and skip it"
                # If we skip, we won't add to `all_map_polygons` unless we load it back.
                # So verify: if we skip, we want the map? Yes.
                # Let's simple-load lines back for the map.
                # (Reading back logic)
                for split in ['train', 'val']:
                     p = f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"
                     if os.path.exists(p):
                         with open(p, 'r') as f:
                             lines = f.readlines()
                         for line in lines:
                             parts = list(map(float, line.strip().split()[1:]))
                             # Denormalize
                             poly_pts = np.array(parts).reshape(-1, 2)
                             poly_pts[:, 0] *= window.width
                             poly_pts[:, 1] *= window.height
                             
                             global_pixels = poly_pts + [col_off, row_off]
                             rows = global_pixels[:, 1]
                             cols = global_pixels[:, 0]
                             xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                             if transformer:
                                 lons, lats = transformer.transform(xs, ys)
                                 all_map_polygons.append(list(zip(lats, lons)))
                             else:
                                 all_map_polygons.append(list(zip(ys, xs)))
                 
                # Load image for map overlay (if user wants to see it)
                # Try to find the image file
                img_path = None
                for split in ['train', 'val']:
                    p = f"{OUTPUT_DIR}/images/{split}/{base_filename}.jpg"
                    if os.path.exists(p):
                        img_path = p
                        break
                
                if img_path:
                    with open(img_path, "rb") as img_f:
                        b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                        img_src = f"data:image/jpeg;base64,{b64_data}"
                        
                        # Calculate Bounds
                        # window is defined at top of loop
                        l, b, r, t = src.window_bounds(window)
                        if transformer:
                            lon_min, lat_min = transformer.transform(l, b)
                            lon_max, lat_max = transformer.transform(r, t)
                            # Folium bounds: [[lat_min, lon_min], [lat_max, lon_max]]
                            bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                            map_overlays.append((img_src, bounds))
                
                continue


            # Print progress every tile to confirm liveness
            # Print progress every tile to confirm liveness (superseded by tqdm but kept for log files)
            # print(f"Scanning tile {min_processed+1}/{len(tile_indices)}: {col_off}_{row_off}...", end="\r", flush=True)

            window = Window(col_off, row_off, min(TILE_SIZE, W - col_off), min(TILE_SIZE, H - row_off))
            
            # Read tile
            tile_data = src.read(window=window)
            if tile_data.shape[0] == 3: # Check for 3 channels (RGB)
                 tile_img = np.ascontiguousarray(np.moveaxis(tile_data, 0, -1)) # (C, H, W) -> (H, W, C)
            else:
                 # Handle cases with alpha channel or other formats if necessary, assuming first 3 are RGB
                 tile_img = np.ascontiguousarray(np.moveaxis(tile_data[:3, :, :], 0, -1))

            # Skip empty/black tiles
            if tile_img.max() == 0:
                continue
            
            # Ensure uint8
            if tile_img.dtype != np.uint8:
                 if tile_img.max() > 255:
                     tile_img = (tile_img / tile_img.max() * 255).astype(np.uint8)
                 else:
                     tile_img = tile_img.astype(np.uint8)

            # Need valid RGB
            if tile_img.shape[2] != 3:
                 continue
                 
            # -- NEW PIPELINE: SAM3 Text Prompt --
            
            # Set image for inference
            try:
                predictor.set_image(tile_img)
            except Exception as e:
                print(f"Error setting image for {col_off}_{row_off}: {e}")
                continue

            # Run inference
            import time
            t0 = time.time()
            results = predictor(text=["sidewalk"], save=False)
            t1 = time.time()
            # print(f"SAM3 took {t1-t0:.2f}s")
            
            labels = [] # YOLO labels
            
            if results and results[0].masks is not None:
                # masks.xy gives us the polygons directly
                polygons = results[0].masks.xy 
                
                for poly in polygons:
                    if len(poly) < 3: continue
                    
                    # 1. YOLO Format (Normalized relative to tile)
                    norm_poly = normalize_polygon(poly, window.width, window.height)
                    labels.append(f"0 {' '.join(map(str, norm_poly))}")
                    
                    # 2. Map Format (Global Lat/Lon)
                    global_pixels = poly + [col_off, row_off]
                    
                    rows = global_pixels[:, 1]
                    cols = global_pixels[:, 0]
                    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                    
                    if transformer:
                        lons, lats = transformer.transform(xs, ys)
                        all_map_polygons.append(list(zip(lats, lons)))
                    else:
                        all_map_polygons.append(list(zip(ys, xs)))

            # 4. Save Data (Only if labels found)
            if len(labels) > 0:
                split = "train" if random.random() < TRAIN_RATIO else "val"
                base_filename = f"tile_{col_off}_{row_off}"
                img_path = f"{OUTPUT_DIR}/images/{split}/{base_filename}.jpg"
                lbl_path = f"{OUTPUT_DIR}/labels/{split}/{base_filename}.txt"
                
                cv2.imwrite(img_path, cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR))
                with open(lbl_path, "w") as f:
                    f.write("\n".join(labels))
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} tiles...")

                # Add to map overlays
                pil_img = Image.fromarray(tile_img)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG", quality=95)
                b64_data = base64.b64encode(buff.getvalue()).decode('utf-8')
                img_src = f"data:image/jpeg;base64,{b64_data}"
                
                l, b, r, t = src.window_bounds(window)
                if transformer:
                     lon_min, lat_min = transformer.transform(l, b)
                     lon_max, lat_max = transformer.transform(r, t)
                     bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                     map_overlays.append((img_src, bounds))

    # 5. Finalize
    # save_sample_visualization(visualization_data) # Skip sample viz for full run or keep it? Keep simple.
    
    # Create dataset.yaml
    yaml_content = {'path': os.path.abspath(OUTPUT_DIR), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'sidewalk'}}
    with open(f"{OUTPUT_DIR}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
        
    print(f"\nDataset generation complete! Processed {processed_count} tiles.")
    
    # --- Generate Map ---
    # --- Generate Map ---
    print(f"Generating map with {len(all_map_polygons)} segments...")

    import requests

    def fetch_osm_streets(min_lat, min_lon, max_lat, max_lon):
        """Fetch street data (Highways) from Overpass API."""
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
            [out:json];
            (
              way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
            );
            out geom;
        """
        try:
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Overpass API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Overpass fetch failed: {e}")
            return None

    if len(all_map_polygons) > 0 or transformer:
        # Determine map center and bounds
        if len(all_map_polygons) > 0:
             center = all_map_polygons[0][0]
        elif transformer:
             # Calculate center from image center if no polygons found yet
             # (Reuse vars from earlier print if scope allows, or recalc)
             with rasterio.open(IMAGE_PATH) as s:
                 tx, ty = s.xy(s.height // 2, s.width // 2)
                 t_lon, t_lat = transformer.transform(tx, ty)
                 center = [t_lat, t_lon]
        else:
             center = [0, 0] # Fallback
             
        m = folium.Map(
            location=center, 
            zoom_start=19, 
            max_zoom=28, 
            tiles=None
        )
        folium.TileLayer(
            'CartoDB positron', 
            max_zoom=28, 
            max_native_zoom=19,
            name='CartoDB Positron'
        ).add_to(m)
        
        # --- Layer Groups ---
        fg_imagery = folium.FeatureGroup(name="Satellite Imagery", show=True)
        fg_sidewalks = folium.FeatureGroup(name="Sidewalks (Detected)", show=True)
        fg_streets = folium.FeatureGroup(name="OSM Streets", show=True)
        fg_aoi = folium.FeatureGroup(name="AOI Boundary", show=True)

        # 1. AOI BBox (Total Image Extent)
        if transformer:
             with rasterio.open(IMAGE_PATH) as s:
                 l, b, r, t = s.bounds
                 # Points: BL, TL, TR, BR
                 # transform(x, y) -> (lon, lat)
                 # We need (lat, lon) for folium
                 corners_x = [l, l, r, r, l]
                 corners_y = [b, t, t, b, b]
                 
                 aoi_latlons = []
                 for x, y in zip(corners_x, corners_y):
                     lon, lat = transformer.transform(x, y)
                     aoi_latlons.append([lat, lon])
                 
                 folium.PolyLine(
                     aoi_latlons,
                     color="blue",
                     weight=4,
                     opacity=1.0,
                     popup="AOI Boundary"
                 ).add_to(fg_aoi)

        # 2. Add Sidewalks
        for poly_points in all_map_polygons:
            folium.Polygon(
                locations=poly_points,
                color="#00ff00",
                weight=2,
                fill_opacity=0.4,
                popup="Sidewalk"
            ).add_to(fg_sidewalks)

        # 3. Add Image Overlays
        print(f"Adding {len(map_overlays)} image overlays...")
        for img_src, bounds in map_overlays:
            # Note: ImageOverlay doesn't strictly support adding to FeatureGroup in all versions, 
            # but it should work. Note: It might need to be added to map directly or macro element.
            # Let's try adding to FG.
            folium.raster_layers.ImageOverlay(
                image=img_src,
                bounds=bounds,
                opacity=1.0, # High opacity for visibility, user can toggle
                name="Satellite Tile"
            ).add_to(fg_imagery)

        # 4. Add OSM Streets
        if transformer:
             with rasterio.open(IMAGE_PATH) as s:
                 l, b, r, t = s.bounds
                 lon_min, lat_min = transformer.transform(l, b)
                 lon_max, lat_max = transformer.transform(r, t)
                 
                 min_lat, max_lat = min(lat_min, lat_max), max(lat_min, lat_max)
                 min_lon, max_lon = min(lon_min, lon_max), max(lon_min, lon_max)
                 
                 print(f"Fetching OSM Strees for bounds: {min_lat:.5f}, {min_lon:.5f} to {max_lat:.5f}, {max_lon:.5f}")
                 osm_data = fetch_osm_streets(min_lat, min_lon, max_lat, max_lon)
                 
                 if osm_data and 'elements' in osm_data:
                     print(f"Found {len(osm_data['elements'])} street segments.")
                     for element in osm_data['elements']:
                         if element['type'] == 'way' and 'geometry' in element:
                             line_points = [[pt['lat'], pt['lon']] for pt in element['geometry']]
                             folium.PolyLine(
                                 line_points, 
                                 color="red", 
                                 weight=2, 
                                 opacity=0.7,
                                 popup=f"Street: {element.get('tags', {}).get('name', 'unnamed')}"
                             ).add_to(fg_streets)

        # Add Groups to Map
        fg_imagery.add_to(m)
        fg_sidewalks.add_to(m)
        fg_streets.add_to(m)
        fg_aoi.add_to(m)

        # Add base layers
        folium.TileLayer('openstreetmap').add_to(m)
        
        # Layer Control
        folium.LayerControl(collapsed=False).add_to(m)

        map_path = "sidewalk_map_full.html"
        m.save(map_path)
        print(f"Map saved to {map_path}")
    else:
        print("No segments found and no transformer setup. Cannot generate valid map.")


if __name__ == "__main__":
    main()

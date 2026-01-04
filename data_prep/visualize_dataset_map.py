
import os
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
import folium
import base64
import glob
import numpy as np
from tqdm import tqdm

# --- Config ---
IMAGE_PATH = "./mosiac_rgb_6cmPerPixel.tif"
DATA_DIR = "datasets/sidewalk_segmentation"
MAP_OUT = "sidewalk_dataset_map.html"

def get_base64_img(path):
    with open(path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def main():
    print("--- Map Viz (Simple) ---")
    if not os.path.exists(DATA_DIR):
        print("No dataset found.")
        return

    # 1. Setup Map Center
    with rasterio.open(IMAGE_PATH) as src:
        T, crs = src.transform, src.crs
        # Transformer: Proj -> WGS84 (Lat, Lon)
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # Center map
        cx, cy = src.xy(src.height//2, src.width//2)
        clon, clat = to_wgs84.transform(cx, cy)
        m = folium.Map([clat, clon], zoom_start=18, max_zoom=28, tiles="CartoDB positron")

    # 2. Add Layers
    fg_img = folium.FeatureGroup("Images", show=True)
    fg_mask = folium.FeatureGroup("Masks", show=True)
    
    files = glob.glob(f"{DATA_DIR}/labels/**/*.txt", recursive=True)
    print(f"Mapping {len(files)} tiles...")

    for lbl_path in tqdm(files, desc="Adding to Map"):
        # Parse Tile Info
        try:
            parts = os.path.splitext(os.path.basename(lbl_path))[0].split('_')
            col, row = int(parts[1]), int(parts[2])
        except: continue
        
        img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(img_path): continue

        # A. Image Overlay (Assume 1024x1024 Padded/Stride)
        # Get global coords for TL and BR of the 1024 block
        xs, ys = rasterio.transform.xy(T, [row, row+1024], [col, col+1024], offset='ul')
        lon_min, lat_min = to_wgs84.transform(min(xs), min(ys))
        lon_max, lat_max = to_wgs84.transform(max(xs), max(ys))
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        folium.raster_layers.ImageOverlay(
            get_base64_img(img_path), bounds=bounds, opacity=1.0, name=f"T{col}_{row}"
        ).add_to(fg_img)

        # B. Polygons (Denormalize)
        with open(lbl_path) as f:
            for line in f:
                # YOLO: class x y ... (Normalized 0-1)
                norm_pts = np.array(list(map(float, line.split()[1:]))).reshape(-1, 2)
                # To Pixel relative to tile -> To Global Pixel -> To LatLon
                # norm * 1024 + offset
                global_pix = (norm_pts * 1024) + [col, row]
                pxs, pys = rasterio.transform.xy(T, global_pix[:,1], global_pix[:,0], offset='center')
                plons, plats = to_wgs84.transform(pxs, pys)
                
                folium.Polygon(list(zip(plats, plons)), color="#00ff00", weight=1, fill_opacity=0.4).add_to(fg_mask)

    fg_img.add_to(m)
    fg_mask.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(MAP_OUT)
    print(f"Map saved: {MAP_OUT}")

if __name__ == "__main__":
    main()

# [#] Load a single TIF image
import rasterio
import numpy as np
import time
import psycopg2
from ultralytics import YOLO
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import dumps
from rasterio.windows import Window
from itertools import product
from pyproj import Transformer

# --- Configuration ---
TIF_PATH = r"V:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\raw_images\mosiac_rgb_6cmPerPixel.tif"
MODEL_PATH = r"V:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\runs\segment\train6\weights\best.pt"
TILE_SIZE = 1024
DB_Config = {
    "dbname": "streets_eval",
    "user": "postgres",
    "password": "12345",
    "host": "localhost",
    "port": "5432"
}

def get_db_connection():
    return psycopg2.connect(**DB_Config)

def setup_transformer(src):
    """Setup coordinate transformer and bounds boundaries."""
    # CRS -> WGS84 Transformer
    to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    
    # Transform bounds to WGS84 for validation
    bounds = src.bounds
    min_lon, min_lat = to_wgs84.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = to_wgs84.transform(bounds.right, bounds.top)
    
    return to_wgs84, (min_lat, min_lon, max_lat, max_lon)

def validate_and_correct_poly(poly):
    """Ensure polygon is valid, simple, and not empty."""
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    if poly.is_empty:
        return []
        
    # Handle MultiPolygon by exploding
    if isinstance(poly, MultiPolygon):
        return [p for p in poly.geoms if p.is_valid and not p.is_empty]
    
    return [poly]

def process_image(tif_path, model):
    """Run tiled inference on the large image."""
    geo_polygons = []
    
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        transform_affine = src.transform
        transformer, wgs_bounds = setup_transformer(src)
        min_lat, min_lon, max_lat, max_lon = wgs_bounds

        print(f"Processing Image: {W}x{H} | CRS: {src.crs}")

        # Tile Generation
        tiles = list(product(range(0, W, TILE_SIZE), range(0, H, TILE_SIZE)))
        print(f"Total Tiles: {len(tiles)}")

        for col, row in tiles:
            # Read Tile
            window = Window(col, row, min(TILE_SIZE, W - col), min(TILE_SIZE, H - row))
            img_data = src.read([1, 2, 3], window=window)
            if img_data.shape[0] < 3 or img_data.max() == 0: continue
            
            # Prepare for YOLO
            img = np.transpose(img_data, (1, 2, 0))
            
            # Inference
            results = model.predict(source=img, imgsz=TILE_SIZE, device=0, verbose=False, conf=0.25)
            if not results or results[0].masks is None: continue
            
            # Process Detections
            seg_result = results[0]
            for class_idx, poly_coords in enumerate(seg_result.masks.xy):
                if len(poly_coords) < 3: continue

                # 1. Tile -> Global Pixel
                global_x = poly_coords[:, 0] + col
                global_y = poly_coords[:, 1] + row
                
                # 2. Global Pixel -> Native CRS
                native_x, native_y = rasterio.transform.xy(transform_affine, global_y, global_x)
                
                # 3. Native -> WGS84
                wgs_lon, wgs_lat = transformer.transform(native_x, native_y)
                
                # Validation: Finite & Range
                if not (np.all(np.isfinite(wgs_lon)) and np.all(np.isfinite(wgs_lat))): continue
                if np.any(np.abs(wgs_lat) > 90) or np.any(np.abs(wgs_lon) > 180): continue

                # Create & Validte Polygon
                raw_poly = Polygon(zip(wgs_lon, wgs_lat))
                
                # Bounds Check (Centroid)
                c = raw_poly.centroid
                if not (min_lon <= c.x <= max_lon and min_lat <= c.y <= max_lat): continue

                # Buffer/Explode check
                valid_polys = validate_and_correct_poly(raw_poly)
                
                for p in valid_polys:
                    geo_polygons.append((p, int(seg_result.boxes.cls[class_idx])))

    return geo_polygons

def upload_to_db(polygons):
    """Bulk upload polygons to PostGIS."""
    if not polygons:
        print("No polygons to upload.")
        return

    print(f"Uploading {len(polygons)} polygons...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Reset Table
                cur.execute("DROP TABLE IF EXISTS segmentations;")
                cur.execute("""
                    CREATE TABLE segmentations (
                        id SERIAL PRIMARY KEY,
                        geom GEOMETRY(Polygon, 4326),
                        class_id INTEGER
                    );
                """)
                
                # Prepare Bulk Insert
                records = [(dumps(p), cls) for p, cls in polygons]
                
                args_str = ",".join(cur.mogrify("(ST_GeomFromText(%s, 4326), %s)", x).decode("utf-8") for x in records)
                cur.execute(f"INSERT INTO segmentations (geom, class_id) VALUES {args_str}")
                conn.commit()
        print("Upload Successful!")
    except Exception as e:
        print(f"Database Error: {e}")

def main():
    print("--- Starting Large Image Segmentation Pipeline ---")
    start_time = time.time()
    
    # 1. Load Model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    # 2. Process
    polygons = process_image(TIF_PATH, model)
    
    # 3. Upload
    upload_to_db(polygons)

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()

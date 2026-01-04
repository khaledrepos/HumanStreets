from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # nano = fastest, small VRAM footprint

if __name__ == "__main__":
    model.train(
        data="datasets/sidewalk_segmentation/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,             # Increased batch size for better GPU utilization
        device=0,
        workers=4,           # Enable multiprocessing for data loading
        augment=True,
        patience=15,         # Increased patience
        optimizer="auto",    # Let YOLO decide the best optimizer
        save_period=5,
        cache=True,          # Cache images in RAM
    )
